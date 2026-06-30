from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from activation_utils import (
    extract_layer_activation,
    infer_hook_name,
    load_nla_meta,
    load_target_model,
    resolve_hf_token,
)
from config import ACTIVATION_DIR, NLA_AR, NLA_AV, NLA_DIR, TARGET_LAYER
from data import PILOT_CASES
from io_utils import ensure_dirs, save_vector, write_json


def resolve_checkpoint_path(repo_or_path: str) -> str:
    path = Path(repo_or_path).expanduser()
    if path.exists():
        return str(path)
    return snapshot_download(repo_id=repo_or_path, token=resolve_hf_token())


def build_nla_client(av_checkpoint: str, sglang_url: str) -> Any:
    from nla_inference import NLAClient

    return NLAClient(av_checkpoint, sglang_url=sglang_url)


def build_nla_critic(ar_checkpoint: str, device: str) -> Any:
    from nla_inference import NLACritic

    return NLACritic(ar_checkpoint, device=device)


def call_first_available(obj: Any, names: list[str], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            return fn(*args, **kwargs)
        except TypeError:
            if kwargs:
                return fn(*args)
            raise
    raise AttributeError(f"{type(obj).__name__} has none of: {', '.join(names)}")


def verbalize(client: Any, vector: torch.Tensor) -> str:
    payload = vector.detach().float().cpu()
    result = call_first_available(client, ["verbalize", "describe", "generate"], payload)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("explanation", "text", "description"):
            if key in result:
                return str(result[key])
    return str(result)


def reconstruct(critic: Any, explanation: str) -> torch.Tensor:
    result = call_first_available(critic, ["reconstruct", "embed", "encode"], explanation)
    if isinstance(result, torch.Tensor):
        return result.detach().float().cpu()
    if isinstance(result, np.ndarray):
        return torch.from_numpy(result).float()
    if isinstance(result, dict):
        for key in ("reconstruction", "vector", "activation"):
            if key in result:
                value = result[key]
                return value.detach().float().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value).float()
    return torch.tensor(result).float()


def round_trip_mse(reconstructed: torch.Tensor, original: torch.Tensor) -> float:
    reconstructed = reconstructed.flatten().float()
    original = original.flatten().float()
    return float(torch.mean((reconstructed - original) ** 2).item())


def cosine_mse(reconstructed: torch.Tensor, original: torch.Tensor) -> float:
    reconstructed = reconstructed.flatten().float()
    original = original.flatten().float()
    cos = torch.nn.functional.cosine_similarity(reconstructed, original, dim=0).item()
    return float(2 * (1 - cos))


def selected_cases(groups: set[str] | None) -> list[dict]:
    if not groups:
        return PILOT_CASES
    return [case for case in PILOT_CASES if case["group"] in groups]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLA explanations for layer-32 activations.")
    parser.add_argument("--groups", default="B", help="Comma separated groups to run, e.g. B or A,B,C. Default: B")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-critic", action="store_true")
    parser.add_argument("--av-checkpoint", default=NLA_AV)
    parser.add_argument("--ar-checkpoint", default=NLA_AR)
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--critic-device", default="cpu")
    args = parser.parse_args()

    ensure_dirs(ACTIVATION_DIR, NLA_DIR)
    groups = {part.strip() for part in args.groups.split(",") if part.strip()} if args.groups else None
    cases = selected_cases(groups)
    if args.limit:
        cases = cases[: args.limit]

    meta = load_nla_meta(NLA_AV)
    hook_name = infer_hook_name(meta)
    model, tokenizer = load_target_model()
    av_checkpoint = resolve_checkpoint_path(args.av_checkpoint)
    ar_checkpoint = None if args.skip_critic else resolve_checkpoint_path(args.ar_checkpoint)
    client = build_nla_client(av_checkpoint, args.sglang_url)
    if ar_checkpoint is None and not args.skip_critic:
        raise RuntimeError("AR checkpoint resolution failed.")
    critic = None if args.skip_critic else build_nla_critic(ar_checkpoint, args.critic_device)

    rows = []
    for case in cases:
        record = extract_layer_activation(
            model,
            tokenizer,
            case["prompt"],
            case.get("target_token", "last"),
            hook_name=hook_name,
        )
        activation_path = ACTIVATION_DIR / f"{case['id']}_L{TARGET_LAYER}_tok{record.token_pos}.npy"
        save_vector(activation_path, record.vector)

        explanation = verbalize(client, record.vector)
        reconstructed_path = None
        mse = None
        cos_mse = None
        if critic is not None:
            reconstructed = reconstruct(critic, explanation)
            reconstructed_path = NLA_DIR / f"{case['id']}_reconstructed.npy"
            save_vector(reconstructed_path, reconstructed)
            mse = round_trip_mse(reconstructed, record.vector)
            cos_mse = cosine_mse(reconstructed, record.vector)

        confabulation_candidate = (
            case["group"] == "C"
            or (cos_mse is not None and cos_mse < 0.4 and not any(term.lower() in explanation.lower() for term in case["expected_chain"]))
        )
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "prompt": case["prompt"],
                "expected_chain": case["expected_chain"],
                "target_layer": TARGET_LAYER,
                "target_token_pos": record.token_pos,
                "target_token_text": record.token_text,
                "hook_name": record.hook_name,
                "activation_path": str(activation_path),
                "explanation": explanation,
                "reconstructed_path": str(reconstructed_path) if reconstructed_path else None,
                "round_trip_mse": mse,
                "cosine_mse_2_1_minus_cos": cos_mse,
                "confabulation_candidate": confabulation_candidate,
            }
        )
        print(f"{case['id']}: {explanation} | cosine_mse={cos_mse}")

    out_path = NLA_DIR / "nla_results.json"
    write_json(out_path, rows)
    print(f"Wrote {len(rows)} NLA rows to {out_path}")


if __name__ == "__main__":
    main()
