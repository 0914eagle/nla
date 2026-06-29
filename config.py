from __future__ import annotations

import os
from pathlib import Path


TARGET_MODEL = "google/gemma-3-12b-it"
TARGET_LAYER = 32
N_LAYERS = 48
D_MODEL = 3840

NLA_AV = "kitft/nla-gemma3-12b-L32-av"
NLA_AR = "kitft/nla-gemma3-12b-L32-ar"

# circuit-tracer's public examples load transcoders by a hub/config name.
# Set this to the exact Gemma-3-12B PLT/GemmaScope-2 transcoder set available
# in your server environment if auto-discovery fails.
DEFAULT_TRANSCODER_SET = "gemma-3-12b-it-gemmascope-2-pt"

OUT_DIR = Path(os.environ.get("NLA_OUTPUT_DIR", "outputs")).expanduser()
ACTIVATION_DIR = OUT_DIR / "activations"
NLA_DIR = OUT_DIR / "nla"
ATTRIBUTION_DIR = OUT_DIR / "attribution"
REPORT_DIR = OUT_DIR / "reports"

DEFAULT_TOKEN_SELECTOR = "last"
