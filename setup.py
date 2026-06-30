from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any

from huggingface_hub import get_token, hf_hub_download

from config import (
    ACTIVATION_DIR,
    ATTRIBUTION_DIR,
    DEFAULT_CIRCUIT_TRACER_BACKEND,
    DEFAULT_TRANSCODER_SET,
    NLA_AV,
    NLA_DIR,
    OUT_DIR,
    REPORT_DIR,
    TARGET_LAYER,
    TARGET_MODEL,
)
from io_utils import ensure_dirs, write_json


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "yaml",
    "numpy",
    "circuit_tracer",
    "nla_inference",
]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_hf_access() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token:
        return {
            "ok": False,
            "message": "No Hugging Face token found. Run `huggingface-cli login` or export HF_TOKEN.",
        }

    meta_path = hf_hub_download(repo_id=NLA_AV, filename="nla_meta.yaml", token=token)
    return {
        "ok": True,
        "token_source": "env" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") else "huggingface-cli",
        "nla_meta_path": meta_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight preflight for the NLA grounding pilot.")
    parser.add_argument("--skip-hf-check", action="store_true")
    args = parser.parse_args()

    ensure_dirs(OUT_DIR, ACTIVATION_DIR, NLA_DIR, ATTRIBUTION_DIR, REPORT_DIR)

    modules = {name: module_available(name) for name in REQUIRED_MODULES}
    missing = [name for name, ok in modules.items() if not ok]
    hf_access = None if args.skip_hf_check else check_hf_access()

    result = {
        "python": sys.executable,
        "target_model": TARGET_MODEL,
        "target_layer": TARGET_LAYER,
        "out_dir": str(OUT_DIR),
        "activation_dir": str(ACTIVATION_DIR),
        "nla_dir": str(NLA_DIR),
        "attribution_dir": str(ATTRIBUTION_DIR),
        "report_dir": str(REPORT_DIR),
        "default_transcoder_set": os.environ.get("CIRCUIT_TRACER_TRANSCODER_SET", DEFAULT_TRANSCODER_SET),
        "default_circuit_tracer_backend": os.environ.get(
            "CIRCUIT_TRACER_BACKEND", DEFAULT_CIRCUIT_TRACER_BACKEND
        ),
        "modules": modules,
        "missing_modules": missing,
        "hf_access": hf_access,
    }

    write_json(OUT_DIR / "setup_preflight.json", result)

    if missing:
        raise RuntimeError(f"Missing required Python modules: {', '.join(missing)}")
    if hf_access is not None and not hf_access["ok"]:
        raise RuntimeError(hf_access["message"])

    print(f"Preflight passed. Wrote {OUT_DIR / 'setup_preflight.json'}")


if __name__ == "__main__":
    main()

