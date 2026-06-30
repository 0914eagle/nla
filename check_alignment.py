from __future__ import annotations

import argparse
import gc
import os
from typing import Any

import torch

from activation_utils import (
    extract_layer_activation,
    infer_hook_name,
    load_nla_meta,
    load_target_model,
)
from config import (
    ACTIVATION_DIR,
    DEFAULT_CIRCUIT_TRACER_BACKEND,
    DEFAULT_TRANSCODER_SET,
    NLA_AV,
    OUT_DIR,
    TARGET_LAYER,
    TARGET_MODEL,
)
from io_utils import ensure_dirs, save_vector, write_json


def load_replacement_model(transcoder_set: str, backend: str, device: str, dtype: torch.dtype):
    from circuit_tracer import ReplacementModel

    return ReplacementModel.from_pretrained(
        TARGET_MODEL,
        transcoder_set=transcoder_set,
        backend=backend,
        device=device,
        dtype=dtype,
    )


def validate_transcoder_layer(replacement_model: Any) -> dict[str, Any]:
    cfg = getattr(replacement_model, "cfg", None)
    n_layers = getattr(cfg, "n_layers", None)
    if n_layers is not None and TARGET_LAYER >= int(n_layers):
        raise RuntimeError(f"ReplacementModel has {n_layers} layers; layer {TARGET_LAYER} is unavailable.")
    transcoders = getattr(replacement_model, "transcoders", None)
    return {
        "backend": getattr(replacement_model, "backend", None),
        "n_layers": int(n_layers) if n_layers is not None else None,
        "scan_name": getattr(replacement_model, "scan_name", None),
        "transcoder_type": type(transcoders).__name__ if transcoders is not None else None,
        "layer_32_available": n_layers is None or TARGET_LAYER < int(n_layers),
    }


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heavy alignment check. This loads Gemma and optionally circuit-tracer."
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--token", default="last")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-device-map", default=os.environ.get("GEMMA_DEVICE_MAP", "auto"))
    parser.add_argument(
        "--transcoder-set",
        default=os.environ.get("CIRCUIT_TRACER_TRANSCODER_SET", DEFAULT_TRANSCODER_SET),
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("CIRCUIT_TRACER_BACKEND", DEFAULT_CIRCUIT_TRACER_BACKEND),
    )
    parser.add_argument("--skip-transcoder", action="store_true")
    args = parser.parse_args()

    ensure_dirs(OUT_DIR, ACTIVATION_DIR)

    meta = load_nla_meta(NLA_AV)
    hook_name = infer_hook_name(meta)
    model, tokenizer = load_target_model(dtype=torch.bfloat16, device_map=args.model_device_map)
    record = extract_layer_activation(model, tokenizer, args.prompt, args.token, hook_name=hook_name)
    save_vector(ACTIVATION_DIR / "alignment_activation.npy", record.vector)
    del model, tokenizer
    free_cuda()

    result: dict[str, Any] = {
        "target_model": TARGET_MODEL,
        "target_layer": TARGET_LAYER,
        "nla_av": NLA_AV,
        "nla_hook_name": hook_name,
        "prompt": args.prompt,
        "token_pos": record.token_pos,
        "token_text": record.token_text,
        "activation_norm": float(record.vector.norm().item()),
        "hf_vs_nla_hook_alignment": "pass_for_resid_post_hidden_states",
        "replacement_model": None,
    }

    if not args.skip_transcoder:
        replacement_model = load_replacement_model(
            args.transcoder_set, args.backend, args.device, torch.bfloat16
        )
        result["replacement_model"] = validate_transcoder_layer(replacement_model)

    write_json(OUT_DIR / "alignment_check.json", result)
    print(f"Alignment metadata written to {OUT_DIR / 'alignment_check.json'}")
    print(f"Layer {TARGET_LAYER} hook: {hook_name}; token {record.token_pos} {record.token_text!r}")


if __name__ == "__main__":
    main()
