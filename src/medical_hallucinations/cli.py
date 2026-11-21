"""Run the study stages: prepare data, run detectors, evaluate detection and calibration."""
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


def _parse_option(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"--option expects KEY=VALUE, got {text!r}")
    return key.replace("-", "_"), value


def _infer(args: argparse.Namespace) -> None:
    from .datasets.base import read_jsonl
    from .detectors import load_detector
    from .pipelines.inference import predictions_path, run_inference

    options = dict(_parse_option(text) for text in args.option)
    detector = load_detector(args.detector, device=args.device, **options)
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


def _calibrate(args: argparse.Namespace) -> None:
    from .pipelines.calibrate import calibrate_all

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    for row in calibrate_all(methods=args.methods, n_bins=args.n_bins):
        print(
            f"{row['detector']:>6} {row['dataset']:<18} {row['subset']:<18} "
            f"{row['method']:<11} n={row['n_examples']:<5} ECE={fmt(row['ece'])} "
            f"Brier={fmt(row['brier'])} AUROC={fmt(row['auroc'])}"
            + ("" if row["coverage"] == 1 else f" coverage={row['coverage']:.0%}")
        )
    print("Saved results/metrics/calibration.{csv,json}, calibrators.json, reliability.json")


def _plot(args: argparse.Namespace) -> None:
    from .visualization.reliability import plot_all

    for path in plot_all():
        print(f"Saved {path}")


def _report(args: argparse.Namespace) -> None:
    from .pipelines.report import report

    rows, figures = report()
    print(f"Saved results/metrics/summary.{{csv,json,md}} ({len(rows)} detector/dataset rows)")
    for path in figures:
        print(f"Saved {path}")
    if not figures:
        print("Install the plot extra (matplotlib) for comparison figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Stage 1: build data/processed/*.jsonl")
    prepare.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    prepare.add_argument("--calibration-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(handler=_prepare)

    infer = commands.add_parser("infer", help="Stage 2: cache detector scores")
    from .detectors import DETECTORS

    infer.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    infer.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--limit", type=int, help="Score only the first N examples (smoke test)")
    infer.add_argument(
        "--device", default="auto", help="cuda, cuda:1, mps, cpu, or auto (default: best available)"
    )
    infer.add_argument(
        "--option", action="append", default=[], metavar="KEY=VALUE",
        help="Extra adapter option, repeatable (e.g. --option dtype=float16)",
    )
    infer.set_defaults(handler=_infer)

    evaluate = commands.add_parser("evaluate", help="Stage 3: detection metrics (RQ1)")
    evaluate.set_defaults(handler=_evaluate)

    from .calibration import CALIBRATORS

    calibrate = commands.add_parser(
        "calibrate", help="Stages 4-5: ECE/Brier of raw and recalibrated scores (RQ2, RQ3)"
    )
    calibrate.add_argument(
        "--methods", nargs="+", choices=sorted(CALIBRATORS), default=list(CALIBRATORS)
    )
    calibrate.add_argument("--n-bins", type=int, default=10)
    calibrate.set_defaults(handler=_calibrate)

    plot = commands.add_parser("plot", help="Reliability diagrams from the calibrate outputs")
    plot.set_defaults(handler=_plot)

    report = commands.add_parser(
        "report", help="Stage 7: summary tables and cross-domain comparison figures"
    )
    report.set_defaults(handler=_report)

    args = parser.parse_args()
    args.handler(args)
