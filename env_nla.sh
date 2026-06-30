# Source this before running the pilot on the server.
# Code can live in /home/eagle0914/nla; heavy caches, outputs, uv-managed
# Python installs, and virtualenvs should live under /data/heejae.

export PATH="/data/heejae/bin:$PATH"

export HF_HOME=/data/heejae/huggingface
export HF_HUB_CACHE=/data/heejae/huggingface/hub
export TRANSFORMERS_CACHE=/data/heejae/huggingface/transformers
export TORCH_HOME=/data/heejae/torch
export XDG_CACHE_HOME=/data/heejae/.cache

export UV_CACHE_DIR=/data/heejae/uv/cache
export UV_PYTHON_INSTALL_DIR=/data/heejae/uv/python
export UV_TOOL_DIR=/data/heejae/uv/tools
export UV_INDEX_STRATEGY=unsafe-best-match

export NLA_OUTPUT_DIR=/data/heejae/nla_outputs
export CIRCUIT_TRACER_BACKEND=nnsight

# Optional. Set this only after confirming the exact HF repo/path for the
# Gemma-3-12B-IT PLT transcoder set on the server.
# export CIRCUIT_TRACER_TRANSCODER_SET=mwhanna/gemma-scope-2-12b-it/transcoder_all/width_262k_l0_small
