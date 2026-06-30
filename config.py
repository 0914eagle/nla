from __future__ import annotations

import os
from pathlib import Path


TARGET_MODEL = "google/gemma-3-12b-it"
TARGET_LAYER = 32
N_LAYERS = 48
D_MODEL = 3840

NLA_AV = "kitft/nla-gemma3-12b-L32-av"
NLA_AR = "kitft/nla-gemma3-12b-L32-ar"

# Gemma-3 PLTs in circuit-tracer-compatible format live under a HF repo subfolder.
# The circuit-tracer README uses this GemmaScope-2 layout:
# mwhanna/gemma-scope-2-27b-pt/transcoder_all/width_262k_l0_small
DEFAULT_TRANSCODER_SET = "mwhanna/gemma-scope-2-12b-it/transcoder_all/width_262k_l0_small"
DEFAULT_CIRCUIT_TRACER_BACKEND = "nnsight"

OUT_DIR = Path(os.environ.get("NLA_OUTPUT_DIR", "outputs")).expanduser()
ACTIVATION_DIR = OUT_DIR / "activations"
NLA_DIR = OUT_DIR / "nla"
ATTRIBUTION_DIR = OUT_DIR / "attribution"
REPORT_DIR = OUT_DIR / "reports"

DEFAULT_TOKEN_SELECTOR = "last"
