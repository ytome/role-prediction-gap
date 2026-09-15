#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import DATA_DIR, DEFAULT_OUT_DIR, assert_expected, bootstrap_mean, group_rows, read_csv, write_csv


MODEL_ORDER = ["GPT-OSS", "Qwen"]
READOUT_ORDER = ["ML", "MH", "MLrbf", "MHpca4"]
CONTRASTS = [
    ("MLrbf", "ML", "nonlinear_role_gain_vs_linear_role"),
    ("MLrbf", "MH", "nonlinear_role_gap_vs_full_state"),
    ("MLrbf", "MHpca4", "nonlinear_role_vs_same_dim_hidden_pca"),
    ("MHpca4", "ML", "same_dim_hidden_pca_vs_linear_role"),
    ("MHpca4", "MH", "same_dim_hidden_pca_gap_vs_full_state"),
]
DRAWS_BY_MODEL = {"GPT-OSS": 5000, "Qwen": 2000}
BOOTSTRAP_SEED = 20260915


def reproduce(data_dir: Path, out_dir: Path, check: bool) -> None:
    rows = read_csv(data_dir / "readout_case_losses.csv")
    by_model = group_rows(rows, "model")

    summary_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        model_rows = by_model[model]
        n_cases = len(model_rows)
        n_rows = n_cases * 5

        for readout in READOUT_ORDER:
            losses = [float(row[f"loss_{readout}"]) for row in model_rows]
            summary_rows.append(
                {
                    "model": model,
                    "readout": readout,
                    "analysis_identity": "baseline_reused" if readout in {"ML", "MH"} else "reviewer_repair_new",
                    "mean_log_loss": sum(losses) / len(losses),
                    "n_cases": n_cases,
                    "n_rows": n_rows,
                }
            )

        for left, right, meaning in CONTRASTS:
            diffs = [float(row[f"loss_{left}"]) - float(row[f"loss_{right}"]) for row in model_rows]
            mean, ci_low, ci_high = bootstrap_mean(diffs, DRAWS_BY_MODEL[model], BOOTSTRAP_SEED)
            contrast_rows.append(
                {
                    "model": model,
                    "contrast": f"{left}-{right}",
                    "meaning": meaning,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "draws": DRAWS_BY_MODEL[model],
                    "seed": BOOTSTRAP_SEED,
                    "n_cases": n_cases,
                    "bootstrap_unit": "semantic_case",
                    "interval_identity": "marginal_posthoc_conditional_on_fitted_oof",
                }
            )

    write_csv(
        out_dir / "readout_summary.csv",
        summary_rows,
        ["model", "readout", "analysis_identity", "mean_log_loss", "n_cases", "n_rows"],
    )
    write_csv(
        out_dir / "readout_contrasts.csv",
        contrast_rows,
        [
            "model",
            "contrast",
            "meaning",
            "mean",
            "ci_low",
            "ci_high",
            "draws",
            "seed",
            "n_cases",
            "bootstrap_unit",
            "interval_identity",
        ],
    )

    if check:
        assert_expected(
            summary_rows,
            read_csv(data_dir / "expected_readout_summary.csv"),
            ["model", "readout"],
            ["mean_log_loss", "n_cases", "n_rows"],
            tolerance=1e-12,
        )
        assert_expected(
            contrast_rows,
            read_csv(data_dir / "expected_readout_contrasts.csv"),
            ["model", "contrast"],
            ["mean", "ci_low", "ci_high", "draws", "seed", "n_cases"],
            tolerance=1e-12,
        )

    for row in summary_rows:
        if row["readout"] in {"ML", "MLrbf", "MHpca4", "MH"}:
            print(f"{row['model']} {row['readout']}: {float(row['mean_log_loss']):.6f}")
    for row in contrast_rows:
        print(
            f"{row['model']} {row['contrast']}: "
            f"{float(row['mean']):+.6f} "
            f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]"
        )
    if check:
        print("readout check: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    reproduce(args.data_dir, args.out_dir, args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
