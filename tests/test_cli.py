"""End-to-end tests: run the real command line on the small synthetic dataset."""

import pandas as pd
import pytest

from pneumonia_cxr.cli import build_parser, main


@pytest.fixture(scope="module")
def finished_run(synthetic_npz, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    exit_code = main(
        [
            "all",
            "--npz", str(synthetic_npz),
            "--results-dir", str(out),
            "--seeds", "0", "1",
            "--epochs", "3",
            "--batch-size", "32",
            "--c-grid", "0.01", "0.1",
            "--n-bootstrap", "100",
            "--device", "cpu",
        ]
    )
    assert exit_code == 0
    return out


def test_all_writes_every_expected_output(finished_run):
    expected = [
        "class_counts.csv",
        "logreg_c_search.csv",
        "metrics.csv",
        "comparisons.csv",
        "results_table.md",
        "figures/sample_images.png",
        "figures/class_balance.png",
        "figures/roc_curves.png",
        "figures/confusion_matrices.png",
        "figures/cnn_training_curves.png",
        "histories/cnn_seed0.csv",
        "models/logreg.joblib",
        "models/cnn_seed1.pt",
        "logs/pcxr.log",
    ]
    for run in ("logreg", "cnn_seed0", "cnn_seed1"):
        for split in ("val", "test"):
            expected.append(f"predictions/{run}_{split}.csv")
    missing = [name for name in expected if not (finished_run / name).exists()]
    assert not missing
    assert list((finished_run / "logs").glob("*_all.json"))  # the run record


def test_metrics_table_has_one_row_per_run(finished_run):
    metrics = pd.read_csv(finished_run / "metrics.csv", index_col="run")
    assert list(metrics.index) == ["logreg", "cnn_seed0", "cnn_seed1"]
    for column in ("auroc", "auroc_ci_low", "auroc_ci_high", "sensitivity", "specificity", "threshold"):
        assert metrics[column].between(0, 1).all()
    assert (metrics["auroc_ci_low"] <= metrics["auroc"]).all()
    assert (metrics["auroc"] <= metrics["auroc_ci_high"]).all()


def test_results_markdown_mentions_every_model(finished_run):
    table = (finished_run / "results_table.md").read_text(encoding="utf-8")
    assert "| Logistic regression |" in table
    assert "CNN (seed 1)" in table
    assert "mean ± SD over 2 seeds" in table
    assert "minus Logistic regression" in table


def test_report_rebuilds_from_saved_predictions(finished_run):
    (finished_run / "results_table.md").unlink()
    assert main(["report", "--results-dir", str(finished_run), "--n-bootstrap", "50"]) == 0
    assert (finished_run / "results_table.md").exists()


def test_report_without_predictions_explains_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="Run `pcxr baseline` or `pcxr cnn` first"):
        main(["report", "--results-dir", str(tmp_path)])


def test_parser_defaults():
    args = build_parser().parse_args(["all"])
    assert args.seeds == [0, 1, 2]
    assert args.size == 28
    assert args.n_bootstrap == 2000
    assert args.npz is None
