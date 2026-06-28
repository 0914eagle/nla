# NLA x Transcoder Grounding Pilot

This workspace implements the pilot described in `PILOT_INSTRUCTION.md`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...
```

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
