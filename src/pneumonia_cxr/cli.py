"""The `pcxr` command line: one subcommand per stage, plus `all` to run everything.

    pcxr explore    class counts and sample images
    pcxr baseline   fit the logistic regression and save its predictions
    pcxr cnn        train the CNN once per seed and save its predictions
    pcxr report     metrics, confidence intervals, results table, and figures
    pcxr all        all four stages, in order

Training and evaluation only communicate through saved prediction files
(results/predictions/<run>_<split>.csv). So `report` can be re-run, or
changed, without retraining anything, and both models are scored by exactly
the same code.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import __version__
from .data import VALID_SIZES, Split, class_counts, load_data, pixel_stats
from .evaluate import (
    METRICS,
    binary_metrics,
    bootstrap_ci,
    choose_threshold,
    confusion_counts,
    paired_bootstrap_auroc_difference,
    summarize_across_seeds,
)
from .models import count_parameters
from .plots import (
    plot_class_balance,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_sample_grid,
    plot_training_curves,
)
from .train import CNNConfig, fit_cnn, fit_logistic_regression, predict_cnn, predict_logistic_regression
from .utils import environment_info, save_json, timestamp

log = logging.getLogger(__name__)

BASELINE_RUN = "logreg"
CNN_RUN_PATTERN = re.compile(r"cnn_seed(\d+)")


class ResultPaths:
    """Where every output goes, inside the results directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.figures = self.root / "figures"
        self.predictions = self.root / "predictions"
        self.histories = self.root / "histories"
        self.models = self.root / "models"
        self.logs = self.root / "logs"

    def prediction_file(self, run: str, split: str) -> Path:
        return self.predictions / f"{run}_{split}.csv"

    def make_dirs(self) -> None:
        for folder in (self.figures, self.predictions, self.histories, self.models, self.logs):
            folder.mkdir(parents=True, exist_ok=True)


def display_name(run: str) -> str:
    """'logreg' -> 'Logistic regression', 'cnn_seed0' -> 'CNN (seed 0)'."""
    if run == BASELINE_RUN:
        return "Logistic regression"
    match = CNN_RUN_PATTERN.fullmatch(run)
    return f"CNN (seed {match.group(1)})" if match else run


def save_predictions(paths: ResultPaths, run: str, split_name: str, split: Split, probs: np.ndarray) -> None:
    """One row per image: its position in the official split, true label, predicted probability."""
    frame = pd.DataFrame({"index": np.arange(len(split)), "label": split.labels, "prob": probs})
    frame.to_csv(paths.prediction_file(run, split_name), index=False)


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def run_explore(splits: dict[str, Split], paths: ResultPaths) -> None:
    log.info("Exploring the data")
    counts = class_counts(splits)
    counts.to_csv(paths.root / "class_counts.csv")
    log.info("Class counts:\n%s", counts.to_string())
    height, width = splits["train"].images.shape[1:]
    mean, std = pixel_stats(splits["train"].images)
    log.info("Images are %dx%d. Training-set pixel mean %.4f, std %.4f (pixels scaled to 0-1).", height, width, mean, std)
    plot_sample_grid(splits["train"], paths.figures / "sample_images.png")
    plot_class_balance(counts, paths.figures / "class_balance.png")


def run_baseline(splits: dict[str, Split], paths: ResultPaths, c_grid: list[float]) -> None:
    log.info("Fitting the logistic regression baseline (C chosen by validation AUROC)")
    model, best_c, search = fit_logistic_regression(splits["train"], splits["val"], tuple(c_grid))
    pd.DataFrame(search).to_csv(paths.root / "logreg_c_search.csv", index=False)
    log.info("  chose C=%g", best_c)
    for split_name in ("val", "test"):
        probs = predict_logistic_regression(model, splits[split_name])
        save_predictions(paths, BASELINE_RUN, split_name, splits[split_name], probs)
    joblib.dump(model, paths.models / f"{BASELINE_RUN}.joblib")


def run_cnn(splits: dict[str, Split], paths: ResultPaths, args: argparse.Namespace) -> None:
    for seed in args.seeds:
        run = f"cnn_seed{seed}"
        config = CNNConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            max_rotation=args.max_rotation,
            dropout=args.dropout,
            seed=seed,
            device=args.device,
        )
        log.info("Training the CNN: %s", display_name(run))
        result = fit_cnn(splits, config)
        log.info("  %d trainable parameters", count_parameters(result.model))
        pd.DataFrame(result.history).to_csv(paths.histories / f"{run}.csv", index=False)
        for split_name in ("val", "test"):
            probs = predict_cnn(result, splits[split_name], device=args.device)
            save_predictions(paths, run, split_name, splits[split_name], probs)
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in result.model.state_dict().items()},
                "mean": result.mean,
                "std": result.std,
                "best_epoch": result.best_epoch,
                "config": asdict(config),
            },
            paths.models / f"{run}.pt",
        )


def find_runs(paths: ResultPaths) -> list[str]:
    """Every run with saved test predictions: the baseline first, then CNN seeds in order."""
    runs = {p.name[: -len("_test.csv")] for p in paths.predictions.glob("*_test.csv")}

    def sort_key(run: str) -> tuple[int, int, str]:
        if run == BASELINE_RUN:
            return (0, 0, run)
        match = CNN_RUN_PATTERN.fullmatch(run)
        return (1, int(match.group(1)), run) if match else (2, 0, run)

    return sorted(runs, key=sort_key)


def run_report(paths: ResultPaths, n_boot: int, level: float, bootstrap_seed: int) -> str:
    """Score every run's saved predictions and write tables and figures. Returns the markdown table."""
    runs = find_runs(paths)
    if not runs:
        raise SystemExit(f"No predictions in {paths.predictions}. Run `pcxr baseline` or `pcxr cnn` first.")
    log.info("Evaluating %d runs on the test set (%d bootstrap resamples)", len(runs), n_boot)

    rows, test_predictions, counts = [], {}, {}
    reference_labels = None
    for run in runs:
        val = pd.read_csv(paths.prediction_file(run, "val"))
        test = pd.read_csv(paths.prediction_file(run, "test"))
        y_val, p_val = val["label"].to_numpy(), val["prob"].to_numpy()
        y_test, p_test = test["label"].to_numpy(), test["prob"].to_numpy()
        if reference_labels is None:
            reference_labels = y_test
        elif not np.array_equal(reference_labels, y_test):
            raise ValueError(f"{run} was evaluated on a different test set than the other runs.")

        threshold = choose_threshold(y_val, p_val)  # chosen on validation, never on test
        point = binary_metrics(y_test, p_test, threshold)
        ci = bootstrap_ci(y_test, p_test, threshold, n_boot=n_boot, level=level, seed=bootstrap_seed)
        cm = confusion_counts(y_test, (p_test >= threshold).astype(int))

        row: dict = {"run": run, "threshold": threshold}
        for name in METRICS:
            row[name] = point[name]
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci[name]
        row.update(cm)
        rows.append(row)
        test_predictions[run] = (y_test, p_test)
        counts[run] = cm

    metrics = pd.DataFrame(rows).set_index("run")
    metrics.to_csv(paths.root / "metrics.csv")

    cnn_runs = [run for run in runs if CNN_RUN_PATTERN.fullmatch(run)]
    seed_summary = (
        summarize_across_seeds([metrics.loc[run, list(METRICS)].to_dict() for run in cnn_runs]) if cnn_runs else None
    )

    comparisons = []
    if BASELINE_RUN in runs:
        y_test, p_baseline = test_predictions[BASELINE_RUN]
        for run in cnn_runs:
            diff = paired_bootstrap_auroc_difference(
                y_test, test_predictions[run][1], p_baseline, n_boot=n_boot, level=level, seed=bootstrap_seed
            )
            comparisons.append({"comparison": f"{run} minus {BASELINE_RUN}", **diff})
    if comparisons:
        pd.DataFrame(comparisons).to_csv(paths.root / "comparisons.csv", index=False)

    table = results_markdown(metrics, seed_summary, comparisons, n_test=len(reference_labels), n_boot=n_boot, level=level)
    (paths.root / "results_table.md").write_text(table, encoding="utf-8")
    log.info("\n%s", table)

    plot_roc_curves({display_name(r): test_predictions[r] for r in runs}, paths.figures / "roc_curves.png")
    plot_confusion_matrices({display_name(r): counts[r] for r in runs}, paths.figures / "confusion_matrices.png")
    histories = {
        display_name(run): pd.read_csv(paths.histories / f"{run}.csv")
        for run in cnn_runs
        if (paths.histories / f"{run}.csv").exists()
    }
    if histories:
        plot_training_curves(histories, paths.figures / "cnn_training_curves.png")
    return table


def results_markdown(
    metrics: pd.DataFrame,
    seed_summary: dict[str, tuple[float, float]] | None,
    comparisons: list[dict],
    n_test: int,
    n_boot: int,
    level: float,
) -> str:
    """The results as a markdown table, ready to paste into the README."""
    pct = f"{level:.0%}"
    lines = [
        "## Test-set results",
        "",
        f"Each model's decision threshold was chosen on the validation set (Youden's J) and then applied "
        f"unchanged to the {n_test} test images. Brackets show {pct} percentile bootstrap confidence "
        f"intervals ({n_boot:,} resamples of the test set).",
        "",
        "| Model | AUROC | Sensitivity | Specificity | Accuracy | Threshold |",
        "|---|---|---|---|---|---|",
    ]
    for run, row in metrics.iterrows():
        cells = [f"{row[m]:.3f} [{row[m + '_ci_low']:.3f}, {row[m + '_ci_high']:.3f}]" for m in METRICS]
        lines.append(f"| {display_name(run)} | " + " | ".join(cells) + f" | {row['threshold']:.3f} |")
    if seed_summary is not None:
        n_seeds = sum(1 for run in metrics.index if CNN_RUN_PATTERN.fullmatch(run))
        cells = [f"{seed_summary[m][0]:.3f} ± {seed_summary[m][1]:.3f}" for m in METRICS]
        lines.append(f"| **CNN, mean ± SD over {n_seeds} seeds** | " + " | ".join(cells) + " | – |")

    if comparisons:
        lines += [
            "",
            "### CNN vs. logistic regression (paired bootstrap)",
            "",
            f"| Comparison | AUROC difference [{pct} CI] |",
            "|---|---|",
        ]
        for c in comparisons:
            run = c["comparison"].split(" minus ")[0]
            lines.append(
                f"| {display_name(run)} minus Logistic regression | "
                f"{c['difference']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] |"
            )
        lines += ["", "An interval that excludes 0 means the test data support a real difference in AUROC."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command-line plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("data and outputs")
    group.add_argument("--data-dir", type=Path, default=Path("data"), help="download folder (default: data)")
    group.add_argument("--size", type=int, default=28, choices=VALID_SIZES, help="image size in pixels (default: 28)")
    group.add_argument("--npz", type=Path, default=None, help="use this local .npz file instead of downloading")
    group.add_argument("--results-dir", type=Path, default=Path("results"), help="output folder (default: results)")

    baseline = argparse.ArgumentParser(add_help=False)
    group = baseline.add_argument_group("logistic regression")
    group.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0], help="values of C to try (default: %(default)s)"
    )

    defaults = CNNConfig()
    cnn = argparse.ArgumentParser(add_help=False)
    group = cnn.add_argument_group("CNN training")
    group.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="one run per seed (default: 0 1 2)")
    group.add_argument("--epochs", type=int, default=defaults.epochs, help="maximum epochs (default: %(default)s)")
    group.add_argument("--batch-size", type=int, default=defaults.batch_size, help="(default: %(default)s)")
    group.add_argument("--lr", type=float, default=defaults.lr, help="Adam learning rate (default: %(default)s)")
    group.add_argument("--weight-decay", type=float, default=defaults.weight_decay, help="(default: %(default)s)")
    group.add_argument(
        "--patience", type=int, default=defaults.patience, help="early-stopping patience in epochs (default: %(default)s)"
    )
    group.add_argument(
        "--max-rotation", type=float, default=defaults.max_rotation, help="augmentation, degrees (default: %(default)s)"
    )
    group.add_argument("--dropout", type=float, default=defaults.dropout, help="(default: %(default)s)")
    group.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="(default: auto)")

    report = argparse.ArgumentParser(add_help=False)
    group = report.add_argument_group("evaluation")
    group.add_argument("--n-bootstrap", type=int, default=2000, help="bootstrap resamples (default: %(default)s)")
    group.add_argument("--ci-level", type=float, default=0.95, help="confidence level (default: %(default)s)")
    group.add_argument("--bootstrap-seed", type=int, default=0, help="(default: %(default)s)")

    parser = argparse.ArgumentParser(
        prog="pcxr",
        description="Pneumonia vs. normal classification on PneumoniaMNIST chest X-rays.",
        epilog="Run `pcxr <command> --help` for a command's options. Typical use: `pcxr all`.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("explore", parents=[common], help="class counts and sample images")
    commands.add_parser("baseline", parents=[common, baseline], help="fit the logistic regression baseline")
    commands.add_parser("cnn", parents=[common, cnn], help="train the CNN, one run per seed")
    commands.add_parser("report", parents=[common, report], help="metrics, confidence intervals, tables, figures")
    commands.add_parser("all", parents=[common, baseline, cnn, report], help="run every stage in order")
    return parser


def setup_logging(log_file: Path) -> None:
    """Print progress to the screen and also append it to a log file."""
    logger = logging.getLogger("pneumonia_cxr")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):  # avoid duplicated lines if main() runs twice
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ResultPaths(args.results_dir)
    paths.make_dirs()
    setup_logging(paths.logs / "pcxr.log")
    log.info("pcxr %s: %s", __version__, args.command)
    # Record exactly how this run was made: arguments, package versions, git commit.
    save_json(
        {"command": args.command, "arguments": vars(args), "environment": environment_info()},
        paths.logs / f"{timestamp()}_{args.command}.json",
    )

    if args.command == "report":
        run_report(paths, args.n_bootstrap, args.ci_level, args.bootstrap_seed)
        return 0

    splits = load_data(args.data_dir, args.size, args.npz)
    if args.command in ("explore", "all"):
        run_explore(splits, paths)
    if args.command in ("baseline", "all"):
        run_baseline(splits, paths, args.c_grid)
    if args.command in ("cnn", "all"):
        run_cnn(splits, paths, args)
    if args.command == "all":
        run_report(paths, args.n_bootstrap, args.ci_level, args.bootstrap_seed)
    log.info("Done. Outputs are in %s", paths.root.resolve())
    return 0
