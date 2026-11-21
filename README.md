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
| 2. Detector inference | `infer --detector <name>` | `results/predictions/<name>/<dataset>.jsonl` | HHEM, LettuceDetect done |
| 3. Detection metrics (RQ1) | `evaluate` | `results/metrics/detection.{csv,json}` | done |
| 4. Calibration metrics (RQ2) | `calibrate`, `plot` | `results/metrics/calibration.{csv,json}` (ECE, Brier), `results/figures/reliability/` | done |
| 5. Post-hoc calibration (RQ3) | `calibrate` | same table, one row per method; `results/metrics/calibrators.json` | done |
| 6. Remaining detectors | | MiniCheck, Lynx adapters (see `detectors/lynx.py`, `minicheck.py`) | todo |
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
  detectors/      # Detector contract, registry, adapters, shared causal-LM scorer
  evaluation/     # F1, AUROC, ECE, Brier, grouped bootstrap CIs, threshold tuning
  calibration/    # Platt, temperature, isotonic calibrators (fit/predict)
  visualization/  # Reliability diagrams and comparisons (stages 4, 7)
  pipelines/      # prepare -> inference -> evaluate -> calibrate
  cli.py          # Command-line entry point
```

## Usage

Each detector gets its own environment, since their pins conflict (HHEM needs
`torch==2.2.2`/`transformers==4.44.2`; LettuceDetect-v2 needs `torch>=2.6`/`transformers>=5.5`).
The data/evaluation environment needs no torch. Predictions are plain JSONL, so inference can
run on a GPU machine and the `results/predictions/` files can be copied back for evaluation.

```bash
# Data, evaluation, tests
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[data,plot,dev]'
medical-hallucinations prepare
python -m unittest discover -s tests

# HHEM inference (Python 3.11/3.12)
python3.12 -m venv .venv-hhem && .venv-hhem/bin/python -m pip install -e '.[hhem]'
HF_HOME=$PWD/artifacts/huggingface .venv-hhem/bin/medical-hallucinations infer --detector hhem

# LettuceDetect on a CUDA GPU (Python 3.11+; v2 is a 2B model, bf16 on GPU)
python3.12 -m venv .venv-lettuce && .venv-lettuce/bin/python -m pip install -e '.[lettucedetect]'
.venv-lettuce/bin/python -m medical_hallucinations.detectors.lettucedetect   # smoke test
.venv-lettuce/bin/medical-hallucinations infer --detector lettucedetect --device cuda

medical-hallucinations evaluate
medical-hallucinations calibrate   # --methods platt temperature isotonic --n-bins 10
medical-hallucinations plot        # reliability diagrams
```

`infer` options: `--device` (`auto` picks cuda, then mps, then cpu), `--batch-size`, `--limit N`,
and repeatable `--option KEY=VALUE` passed to the adapter (e.g. `--option dtype=float16` on GPUs
without bfloat16).

Inference appends after every batch, so rerunning `infer` resumes an interrupted run. Use
`--limit N` for a quick smoke test; `evaluate` reports `coverage` below 100% for partial runs.

Calibration uses the **positive-class** convention: a score is P(hallucinated) and is compared
with the observed hallucination rate in 10 equal-width bins (`[k/10, (k+1)/10)`, with 1.0 in
the last bin; empty bins are skipped). Calibrators are fitted on the calibration partition only
and are skipped (null metrics) when its scored rows contain a single class. Platt and
temperature scaling act on the score's log-odds, clipped at `1e-6`; isotonic regression clips
out-of-range scores to its fitted endpoints.

## Detectors

| CLI name | Model | Input | P(hallucinated) |
|---|---|---|---|
| `hhem` | `vectara/hallucination_evaluation_model` | passage, answer | 1 - P(consistent) |
| `lettucedetect` | `KRLabsOrg/lettucedect-v2-qwen-2b` (generative) | question, passage, answer | 1 - P(model replies `{"hallucinated_spans": []}`) |
| `lettucedetect-v1` | `KRLabsOrg/lettucedect-large-modernbert-en-v1` | question, passage, answer | max token hallucination probability |
| `minicheck`, `lynx` | todo | | |

LettuceDetect-v2 only emits spans, so its score is read from its own token probabilities: one
teacher-forced forward pass gives the probability of the "no hallucinated spans" reply
(`detectors/causal_lm.py`, reusable for other generative judges such as Lynx).

**Adding a detector:** subclass `Detector` (`detectors/base.py` documents the contract), pin the
model revision, return P(hallucinated) in [0, 1], register the class in `detectors/__init__.py`,
and add a conversion test. `infer`, `evaluate`, `calibrate`, and `plot` then work unchanged.
