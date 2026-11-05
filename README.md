# Medical Hallucination Detectors

**Trusting Hallucination Detectors in Medicine: A Training-Free Analysis of Accuracy and Confidence Calibration**.

Frozen, pretrained hallucination detectors are run in inference mode on general-domain
(RAGTruth QA) and biomedical (RAGBench CovidQA/PubMedQA, MedHallu) question answering, then
evaluated for detection (RQ1), calibration (RQ2), and post-hoc recalibration (RQ3).

Label convention everywhere: `1 = hallucinated`, and every detector score is converted to
`P(hallucinated)`.

## Pipeline

| Stage | Command | Output | Status |
|---|---|---|---|
| 1. Prepare data | `prepare` | `data/processed/<dataset>.jsonl`, `data/splits/<dataset>.json` | done |
| 2. Detector inference | `infer --detector hhem` | `results/predictions/<detector>/<dataset>.jsonl` | done (HHEM) |
| 3. Detection metrics (RQ1) | `evaluate` | `results/metrics/detection.{csv,json}` | done |
| 4. Calibration metrics (RQ2) | | ECE, Brier, reliability diagrams | todo |
| 5. Post-hoc calibration (RQ3) | | Platt, temperature, isotonic | todo |
| 6. Remaining detectors | | MiniCheck, Lynx, LettuceDetect adapters | todo |
| 7. Reporting | | cross-domain comparison figures and tables | todo |

Each dataset is split by `source_id` (answers to the same question never straddle the split)
into a **calibration** partition (20%), used only to fit calibrators and tune thresholds, and an
**evaluation** partition (80%) that all reported metrics use.

| Dataset | Raw file(s) in `data/raw/` | Used split | Labels |
|---|---|---|---|
| `ragtruth_qa` | `ragtruth/{source_info,response}.jsonl` | test (LettuceDetect trained on train) | human spans |
| `ragbench_covidqa` | `ragbench/covidqa/test.parquet` | test | GPT-4o annotator |
| `ragbench_pubmedqa` | `ragbench/pubmedqa/test.parquet` | test | GPT-4-turbo annotator |
| `medhallu` | `medhallu/pqa_labeled.parquet` | all (pairs, 50% hallucinated) | constructed |

## Layout

```text
src/medical_hallucinations/
  schemas.py      # Shared record contract and validation
  datasets/       # Loaders, JSONL helpers, dataset registry, source-grouped splits
  detectors/      # Detector base class and adapters (HHEM implemented)
  evaluation/     # F1, AUROC, grouped bootstrap CIs, threshold tuning
  calibration/    # Platt, temperature, isotonic (stage 5)
  visualization/  # Reliability diagrams and comparisons (stages 4, 7)
  pipelines/      # prepare -> inference -> evaluate (-> calibrate)
  cli.py          # Command-line entry point
```

## Usage

The project uses two environments, since the HHEM pins (`torch==2.2.2`, `numpy<2`) need an
older Python than the data tooling.

```bash
# Data, evaluation, tests
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[data,dev]'
medical-hallucinations prepare
python -m unittest discover -s tests

# HHEM inference (Python 3.11/3.12)
python3.12 -m venv .venv-hhem && .venv-hhem/bin/python -m pip install -e '.[hhem]'
HF_HOME=$PWD/artifacts/huggingface .venv-hhem/bin/medical-hallucinations infer --detector hhem

medical-hallucinations evaluate
```

Inference appends after every batch, so rerunning `infer` resumes an interrupted run. Use
`--limit N` for a quick smoke test; `evaluate` reports `coverage` below 100% for partial runs.
