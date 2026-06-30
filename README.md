# NLA x Transcoder Grounding Pilot

This workspace implements the pilot described in `PILOT_INSTRUCTION.md`.

## Install

```bash
mkdir -p /data/heejae
git clone https://github.com/0914eagle/nla.git /home/eagle0914/nla
cd /home/eagle0914/nla

cp env_nla.sh /data/heejae/env_nla.sh
source /data/heejae/env_nla.sh

conda create -y -p /data/heejae/conda/envs/nla python=3.11
conda activate /data/heejae/conda/envs/nla
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Authenticate Hugging Face before running gated Gemma checkpoints:

```bash
huggingface-cli login
# or: export HF_TOKEN=...
```

If using `uv pip install`, `env_nla.sh` sets `UV_INDEX_STRATEGY=unsafe-best-match`
so uv can resolve PyTorch CUDA wheels from the PyTorch index instead of stopping
after seeing the CPU/PyPI package name first.

`circuit-tracer` requires Python 3.10 or newer. Python 3.11 is the tested target
for this requirements set.

`nla_inference.py` is vendored from
`kitft/nla-inference@38b802a33d1d317f21b6825a9116f388c2141f86` because that
repository currently ships as a single file and is not pip-installable. Its
license is included in `NLA_INFERENCE_LICENSE`.

If you want to run the NLA AV SGLang server locally, use a separate environment:

```bash
source /data/heejae/env_nla.sh
conda create -y -p /data/heejae/conda/envs/nla-sglang python=3.11
conda activate /data/heejae/conda/envs/nla-sglang
cd /home/eagle0914/nla
python -m pip install -U pip setuptools wheel
pip install -r requirements-sglang-server.txt
```

Then launch SGLang on a private interface with `--disable-radix-cache`.

If circuit-tracer auto-loading cannot find the Gemma-3-12B PLT/GemmaScope-2
transcoder set, set:

```bash
export CIRCUIT_TRACER_BACKEND=nnsight
export CIRCUIT_TRACER_TRANSCODER_SET=mwhanna/gemma-scope-2-12b-it/transcoder_all/width_16k_l0_small
```

## Run

```bash
python setup.py
python check_alignment.py --skip-transcoder
python run_nla.py --groups B
python run_attribution.py
python compare.py
```

Then open `outputs/reports/pilot_report.md` and fill the manual judgment line for
each case.

If `NLA_OUTPUT_DIR=/data/heejae/nla_outputs` is set, the report is written to
`/data/heejae/nla_outputs/reports/pilot_report.md`.

## Persistent Server Environment

Shell `export` values last only for the current login session. To avoid setting
them manually after every SSH login, add this line to the server user's shell rc
file:

```bash
source /data/heejae/env_nla.sh
```

For bash this is usually `~/.bashrc`; for zsh it is `~/.zshrc`.

## Notes

- Layer 32 is hard-coded in `config.py`.
- `setup.py` is intentionally lightweight: it checks paths, imports, and HF access only.
- `check_alignment.py` is the heavy optional validation script that loads Gemma and,
  unless `--skip-transcoder` is passed, circuit-tracer's `ReplacementModel`.
- NLA hook metadata is read from `kitft/nla-gemma3-12b-L32-av/nla_meta.yaml`.
- `run_attribution.py` targets the layer-32 residual direction directly through
  circuit-tracer's attribution context instead of treating the activation as a
  next-token logit target.
