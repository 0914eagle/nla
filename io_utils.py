from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # compare.py does not need torch.
    torch = None


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required to convert tensors")
    return tensor.detach().float().cpu().numpy()


def save_vector(path: Path, vector: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor_to_numpy(vector) if torch is not None and isinstance(vector, torch.Tensor) else vector
    np.save(path, array.astype(np.float32))


def load_vector(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)
