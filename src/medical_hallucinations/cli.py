"""Run the study stages: prepare data, run detectors, evaluate detection."""
import argparse

from .datasets.base import DATASETS


def _prepare(args: argparse.Namespace) -> None:
    from .pipelines.prepare import prepare_dataset

    for name in args.datasets:
        summary = prepare_dataset(
            name, calibration_fraction=args.calibration_fraction, seed=args.seed
        )
        parts = ", ".join(
            f"{partition}={info['n_examples']} ({info['hallucination_rate']:.1%} hallucinated)"
            for partition, info in summary["partitions"].items()
        )
        print(f"{name}: {parts}")


def _infer(args: argparse.Namespace) -> None:
    from .datasets.base import read_jsonl
    from .detectors import load_detector
    from .pipelines.inference import predictions_path, run_inference

    detector = load_detector(args.detector)
    for name in args.datasets:
        examples = read_jsonl(f"data/processed/{name}.jsonl")
        if args.limit is not None:
            examples = examples[:args.limit]
        output = predictions_path("results/predictions", args.detector, name)
        count = run_inference(detector, examples, output, batch_size=args.batch_size)
        print(f"{args.detector}/{name}: scored {count} new examples -> {output}")


def _evaluate(args: argparse.Namespace) -> None:
    from .pipelines.evaluate import evaluate_all

    for row in evaluate_all():
        auroc = "n/a" if row["auroc"] is None else f"{row['auroc']:.3f}"
        tuned = "n/a" if row["f1@tuned"] is None else (
            f"{row['f1@tuned']:.3f} (t={row['tuned_threshold']:.3f})"
        )
        print(
            f"{row['detector']:>6} {row['dataset']:<18} {row['subset']:<18} "
            f"n={row['n_examples']:<5} AUROC={auroc} F1@0.5={row['f1@0.5']:.3f} F1@tuned={tuned}"
            + ("" if row["coverage"] == 1 else f" coverage={row['coverage']:.0%}")
        )
    print("Saved results/metrics/detection.csv and detection.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Stage 1: build data/processed/*.jsonl")
    prepare.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    prepare.add_argument("--calibration-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(handler=_prepare)

    infer = commands.add_parser("infer", help="Stage 2: cache detector scores")
    infer.add_argument("--detector", required=True)
    infer.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--limit", type=int, help="Score only the first N examples (smoke test)")
    infer.set_defaults(handler=_infer)

    evaluate = commands.add_parser("evaluate", help="Stage 3: detection metrics (RQ1)")
    evaluate.set_defaults(handler=_evaluate)

    args = parser.parse_args()
    args.handler(args)
