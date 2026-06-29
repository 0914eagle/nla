# Source this before running the pilot on the server.
# Code can live in /home/eagle0914/nla; heavy caches, outputs, and conda envs
# will live under /data/heejae.

export HF_HOME=/data/heejae/huggingface
export HF_HUB_CACHE=/data/heejae/huggingface/hub
export TRANSFORMERS_CACHE=/data/heejae/huggingface/transformers
export TORCH_HOME=/data/heejae/torch
export XDG_CACHE_HOME=/data/heejae/.cache

export CONDA_PKGS_DIRS=/data/heejae/conda/pkgs
export CONDA_ENVS_PATH=/data/heejae/conda/envs

export NLA_OUTPUT_DIR=/data/heejae/nla_outputs

# Optional. Set this only after confirming the exact HF repo/path for the
# Gemma-3-12B-IT PLT transcoder set on the server.
# export CIRCUIT_TRACER_TRANSCODER_SET=<exact-transcoder-set-name-or-path>
