from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

import numpy as np
import torch

from config import (
    ATTRIBUTION_DIR,
    DEFAULT_CIRCUIT_TRACER_BACKEND,
    DEFAULT_TRANSCODER_SET,
    NLA_DIR,
    TARGET_LAYER,
    TARGET_MODEL,
)
from io_utils import ensure_dirs, load_vector, read_json, write_json


def load_replacement_model(transcoder_set: str, backend: str, device: str, dtype: torch.dtype) -> Any:
    from circuit_tracer import ReplacementModel

    return ReplacementModel.from_pretrained(
        model_name=TARGET_MODEL,
        transcoder_set=transcoder_set,
        backend=backend,
        device=device,
        dtype=dtype,
    )


def import_context_class() -> Any:
    candidates = [
        "circuit_tracer.attribution.context_nnsight",
        "circuit_tracer.attribution.context",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in ("AttributionContext", "AttributionContextNNSight"):
            if hasattr(module, name):
                return getattr(module, name)
    raise RuntimeError("Could not find circuit_tracer AttributionContext class.")


def direct_layer32_attribution(
    replacement_model: Any,
    prompt: str,
    token_pos: int,
    target_vector: np.ndarray,
    top_k: int,
) -> dict:
    Context = import_context_class()
    ctx = Context(replacement_model, prompt)
    if hasattr(ctx, "__enter__"):
        with ctx as active_ctx:
            return compute_direct_attribution(active_ctx, token_pos, target_vector, top_k)
    return compute_direct_attribution(ctx, token_pos, target_vector, top_k)


def compute_direct_attribution(ctx: Any, token_pos: int, target_vector: np.ndarray, top_k: int) -> dict:
    vector = torch.tensor(target_vector, dtype=torch.float32)
    vector = vector / (vector.norm() + 1e-8)

    if not hasattr(ctx, "compute_batch"):
        raise RuntimeError("circuit_tracer AttributionContext lacks compute_batch; cannot target layer-32 residual.")

    rows = ctx.compute_batch(
        layers=[TARGET_LAYER],
        positions=[token_pos],
        inject_values=[vector],
    )
    if isinstance(rows, tuple):
        rows = rows[0]
    row = rows[0] if getattr(rows, "ndim", 1) > 1 else rows
    row = row.detach().float().cpu()

    values, indices = torch.topk(row.abs(), k=min(top_k, row.numel()))
    feature_rows = []
    for rank, (abs_value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        signed_value = float(row[index].item())
        feature_rows.append(resolve_feature(ctx, rank, int(index), signed_value, float(abs_value)))

    return {
        "api_mode": "direct_layer_residual_direction",
        "target_layer": TARGET_LAYER,
        "target_token_pos": token_pos,
        "target_direction": "normalized original NLA activation vector",
        "top_features": feature_rows,
    }


def resolve_feature(ctx: Any, rank: int, column_index: int, contribution: float, abs_contribution: float) -> dict:
    feature = {
        "rank": rank,
        "column_index": column_index,
        "contribution": contribution,
        "abs_contribution": abs_contribution,
        "feature_id": None,
        "feature_layer": None,
        "source_token_pos": None,
        "source_token_text": None,
    }
    active_features = getattr(ctx, "active_features", None)
    if active_features is not None:
        try:
            mapped = active_features[column_index]
            if isinstance(mapped, dict):
                feature.update(mapped)
            elif isinstance(mapped, (tuple, list)):
                if len(mapped) > 0:
                    feature["feature_layer"] = int(mapped[0]) if isinstance(mapped[0], (int, np.integer)) else mapped[0]
                if len(mapped) > 1:
                    feature["source_token_pos"] = int(mapped[1]) if isinstance(mapped[1], (int, np.integer)) else mapped[1]
                if len(mapped) > 2:
                    feature["feature_id"] = int(mapped[2]) if isinstance(mapped[2], (int, np.integer)) else mapped[2]
        except Exception:
            pass

    tokens = getattr(ctx, "tokens", None)
    tokenizer = getattr(getattr(ctx, "model", None), "tokenizer", None)
    if feature["source_token_pos"] is not None and tokens is not None and tokenizer is not None:
        try:
            tok_id = int(tokens[feature["source_token_pos"]])
            feature["source_token_text"] = tokenizer.decode([tok_id])
        except Exception:
            pass
    return feature


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute NLA target activations back through circuit-tracer transcoders.")
    parser.add_argument("--nla-results", default=str(NLA_DIR / "nla_results.json"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transcoder-set", default=os.environ.get("CIRCUIT_TRACER_TRANSCODER_SET", DEFAULT_TRANSCODER_SET))
    parser.add_argument("--backend", default=os.environ.get("CIRCUIT_TRACER_BACKEND", DEFAULT_CIRCUIT_TRACER_BACKEND))
    args = parser.parse_args()

    ensure_dirs(ATTRIBUTION_DIR)
    rows = read_json(args.nla_results)
    replacement_model = load_replacement_model(args.transcoder_set, args.backend, args.device, torch.bfloat16)

    outputs = []
    for row in rows:
        activation = load_vector(row["activation_path"])
        result = direct_layer32_attribution(
            replacement_model,
            row["prompt"],
            int(row["target_token_pos"]),
            activation,
            args.top_k,
        )
        result.update(
            {
                "id": row["id"],
                "group": row["group"],
                "prompt": row["prompt"],
                "target_token_text": row["target_token_text"],
                "nla_explanation": row["explanation"],
            }
        )
        outputs.append(result)
        print(f"{row['id']}: collected {len(result['top_features'])} top contributors")

    out_path = ATTRIBUTION_DIR / "attribution_results.json"
    write_json(out_path, outputs)
    print(f"Wrote {len(outputs)} attribution rows to {out_path}")


if __name__ == "__main__":
    main()
