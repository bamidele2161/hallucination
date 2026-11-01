# Medical Hallucination Detectors

**Trusting Hallucination Detectors in Medicine: A Training-Free Analysis of Accuracy and Confidence Calibration**.

This repository contains structure only. Dataset loading, model inference, metrics, calibration, and plotting are intentionally unimplemented. No datasets or model weights have been downloaded.

## Layout

```text
src/medical_hallucinations/
  datasets/       # RAGTruth QA, RAGBench CovidQA/PubMedQA, MedHallu
  detectors/      # HHEM, MiniCheck, Lynx, LettuceDetect adapters
  evaluation/     # F1, AUROC, ECE, Brier score, threshold analysis
  calibration/    # Platt, temperature, isotonic score calibration
  visualization/ # Reliability diagrams and comparisons
  pipelines/     # Inference, evaluation, calibration orchestration
  schemas.py     # Shared records and score convention
  cli.py         # Placeholder command-line entry point
configs/         # Experiment configuration placeholders
data/            # Local raw data, processed records, and split manifests
artifacts/       # Fitted score calibrators and run metadata
results/         # Predictions, metrics, and figures
notebooks/       # Future exploratory analysis
tests/           # Future behavior tests
docs/            # Study scope and source assignment
```

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
medical-hallucinations --help
```

The CLI currently reports scaffold status only. Add runtime dependencies when implementing the adapters.
