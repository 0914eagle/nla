from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import D_MODEL, NLA_AV, TARGET_LAYER, TARGET_MODEL


@dataclass
class ActivationRecord:
    prompt: str
    token_pos: int
    token_text: str
    vector: torch.Tensor
    hook_name: str


def require_hf_token() -> None:
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. Gemma-3 is gated, so export HF_TOKEN before running."
        )


def load_target_model(dtype: torch.dtype = torch.bfloat16, device_map: str = "auto") -> tuple[Any, Any]:
    require_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_nla_meta(repo_id: str = NLA_AV) -> dict[str, Any]:
    meta_path = hf_hub_download(repo_id=repo_id, filename="nla_meta.yaml", token=os.environ.get("HF_TOKEN"))
    return yaml.safe_load(Path(meta_path).read_text(encoding="utf-8"))


def infer_hook_name(meta: dict[str, Any]) -> str:
    extraction = meta.get("extraction", {})
    for key in ("hook_name", "hook", "activation_name", "site"):
        if extraction.get(key):
            return str(extraction[key])
    # The public NLA sidecar records layer_index but not always a full hook path.
    return f"layers.{TARGET_LAYER}.resid_post"


def hidden_state_index_for_hook(hook_name: str, layer: int = TARGET_LAYER) -> int:
    lowered = hook_name.lower()
    if "resid_mid" in lowered:
        raise NotImplementedError(
            "NLA sidecar appears to request resid_mid. This pilot currently extracts resid_post "
            "through transformers hidden_states; add an architecture hook before continuing."
        )
    # transformers hidden_states[0] is embeddings, hidden_states[layer + 1] is post-block
    # output for zero-based layer indices.
    return layer + 1


def tokenize_prompt(tokenizer: Any, prompt: str) -> torch.Tensor:
    if "gemma-3" in TARGET_MODEL and TARGET_MODEL.endswith("-it"):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return ids
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids


def resolve_token_pos(input_ids: torch.Tensor, selector: str | int) -> int:
    seq_len = int(input_ids.shape[-1])
    if isinstance(selector, int):
        if selector < 0:
            selector = seq_len + selector
        if not 0 <= selector < seq_len:
            raise IndexError(f"token position {selector} out of range for length {seq_len}")
        return selector
    if selector == "last":
        return seq_len - 1
    if selector.startswith("index:"):
        return resolve_token_pos(input_ids, int(selector.split(":", 1)[1]))
    raise ValueError(f"Unsupported token selector: {selector!r}")


@torch.inference_mode()
def extract_layer_activation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    token_selector: str | int = "last",
    hook_name: str | None = None,
) -> ActivationRecord:
    hook_name = hook_name or f"layers.{TARGET_LAYER}.resid_post"
    input_ids = tokenize_prompt(tokenizer, prompt).to(model.device)
    token_pos = resolve_token_pos(input_ids, token_selector)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    idx = hidden_state_index_for_hook(hook_name)
    vector = outputs.hidden_states[idx][0, token_pos].detach().float().cpu()
    if vector.numel() != D_MODEL:
        raise RuntimeError(f"Expected d_model={D_MODEL}, got vector shape {tuple(vector.shape)}")
    token_text = tokenizer.decode([int(input_ids[0, token_pos].detach().cpu())])
    return ActivationRecord(
        prompt=prompt,
        token_pos=token_pos,
        token_text=token_text,
        vector=vector,
        hook_name=hook_name,
    )


def cosine_distance_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return float(2 * (1 - cos))

