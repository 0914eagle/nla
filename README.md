# NLA x Transcoder Grounding Pilot

This workspace implements the pilot described in `PILOT_INSTRUCTION.md`.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
export HF_TOKEN=...
```

`circuit-tracer` requires Python 3.10 or newer. Python 3.11 is the tested target
for this requirements set.

`nla_inference.py` is vendored from
`kitft/nla-inference@38b802a33d1d317f21b6825a9116f388c2141f86` because that
repository currently ships as a single file and is not pip-installable. Its
license is included in `NLA_INFERENCE_LICENSE`.

If you want to run the NLA AV SGLang server locally, use a separate environment:

```bash
python3.11 -m venv .venv-sglang
source .venv-sglang/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-sglang-server.txt
```

Then launch SGLang on a private interface with `--disable-radix-cache`.

If circuit-tracer auto-loading cannot find the Gemma-3-12B PLT/GemmaScope-2
transcoder set, set:

```bash
export CIRCUIT_TRACER_TRANSCODER_SET=<exact-transcoder-set-name>
```

## Run

```bash
python setup.py
python run_nla.py --groups B
python run_attribution.py
python compare.py
```

Then open `outputs/reports/pilot_report.md` and fill the manual judgment line for
each case.

## Notes

- Layer 32 is hard-coded in `config.py`.
- NLA hook metadata is read from `kitft/nla-gemma3-12b-L32-av/nla_meta.yaml`.
- `run_attribution.py` targets the layer-32 residual direction directly through
  circuit-tracer's attribution context instead of treating the activation as a
  next-token logit target.
