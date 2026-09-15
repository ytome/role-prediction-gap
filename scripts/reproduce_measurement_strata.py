#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import DATA_DIR, DEFAULT_OUT_DIR, assert_expected, bootstrap_mean, grouped_case_values, read_csv, write_csv


MODEL_ORDER = ["GPT-OSS", "Qwen"]
ROLE_ORDER = ["system", "user", "tool", "cot", "assistant"]
GROUP_ORDER = ["single_alignment", "low_alignment", "mid_alignment", "high_alignment"]
SEED_BASE = {"GPT-OSS": 2026091501, "Qwen": 2026092501}
DRAWS = 5000


def filter_rows(rows: list[dict[str, str]], **criteria: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row[key] == value for key, value in criteria.items())]


def summarize_stratum(model: str, stratum_type: str, stratum: str, rows: list[dict[str, str]], seed: int) -> dict[str, object]:
    cases = grouped_case_values(rows, "gap_ML_minus_MH")
    gaps = [float(case["value"]) for case in cases]
    mean, ci_low, ci_high = bootstrap_mean(gaps, DRAWS, seed)
    local_vals = [float(case["local_alignment_overall"]) for case in cases]
    global_vals = [float(case["global_alignment_overall"]) for case in cases]
    return {
        "model": model,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "n_cases": len(cases),
        "n_rows": len(rows),
        "gap_ML_minus_MH_mean": mean,
        "gap_ML_minus_MH_ci95_low": ci_low,
        "gap_ML_minus_MH_ci95_high": ci_high,
        "gap_MH_minus_ML_mean": -mean,
        "gap_MH_minus_ML_ci95_low": -ci_high,
        "gap_MH_minus_ML_ci95_high": -ci_low,
        "local_alignment_mean": sum(local_vals) / len(local_vals),
        "local_alignment_min": min(local_vals),
        "local_alignment_max": max(local_vals),
        "global_alignment_mean": sum(global_vals) / len(global_vals),
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": seed,
    }


def high_low_contrast(model: str, rows: list[dict[str, str]], seed: int) -> dict[str, object] | None:
    cases = grouped_case_values(rows, "gap_ML_minus_MH")
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        grouped[str(case["alignment_group"])].append(float(case["value"]))
    if "high_alignment" not in grouped or "low_alignment" not in grouped:
        return None
    high = grouped["high_alignment"]
    low = grouped["low_alignment"]

    import numpy as np

    rng = np.random.default_rng(seed)
    boot = np.empty(DRAWS, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    for draw_ix in range(DRAWS):
        hi = high_arr[rng.integers(0, len(high_arr), size=len(high_arr))].mean()
        lo = low_arr[rng.integers(0, len(low_arr), size=len(low_arr))].mean()
        boot[draw_ix] = hi - lo
    return {
        "model": model,
        "contrast": "high_alignment_minus_low_alignment",
        "mean": float(high_arr.mean() - low_arr.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_cases_high": len(high),
        "n_cases_low": len(low),
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": seed,
        "bootstrap_unit": "semantic_case_within_alignment_group",
    }


def reproduce(data_dir: Path, out_dir: Path, check: bool) -> None:
    all_rows = read_csv(data_dir / "measurement_row_losses.csv")
    result_rows: list[dict[str, object]] = []
    high_low_rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        model_rows = filter_rows(all_rows, model=model)
        base = SEED_BASE[model]
        result_rows.append(summarize_stratum(model, "overall", "all_cases", model_rows, base))

        for offset, role in enumerate(ROLE_ORDER, start=1):
            role_rows = filter_rows(model_rows, style_role=role)
            result_rows.append(summarize_stratum(model, "wording_role", role, role_rows, base + offset))

        present_groups = {row["alignment_group"] for row in model_rows}
        ordered_groups = [group for group in GROUP_ORDER if group in present_groups]
        for offset, group in enumerate(ordered_groups, start=10):
            group_rows = filter_rows(model_rows, alignment_group=group)
            result_rows.append(summarize_stratum(model, "benign_alignment_bin", group, group_rows, base + offset))

        contrast = high_low_contrast(model, model_rows, base + 99)
        if contrast is not None:
            high_low_rows.append(contrast)

    write_csv(
        out_dir / "measurement_strata.csv",
        result_rows,
        [
            "model",
            "stratum_type",
            "stratum",
            "n_cases",
            "n_rows",
            "gap_ML_minus_MH_mean",
            "gap_ML_minus_MH_ci95_low",
            "gap_ML_minus_MH_ci95_high",
            "gap_MH_minus_ML_mean",
            "gap_MH_minus_ML_ci95_low",
            "gap_MH_minus_ML_ci95_high",
            "local_alignment_mean",
            "local_alignment_min",
            "local_alignment_max",
            "global_alignment_mean",
            "bootstrap_draws",
            "bootstrap_seed",
        ],
    )
    write_csv(
        out_dir / "measurement_high_low.csv",
        high_low_rows,
        [
            "model",
            "contrast",
            "mean",
            "ci95_low",
            "ci95_high",
            "n_cases_high",
            "n_cases_low",
            "bootstrap_draws",
            "bootstrap_seed",
            "bootstrap_unit",
        ],
    )

    if check:
        assert_expected(
            result_rows,
            read_csv(data_dir / "expected_measurement_strata.csv"),
            ["model", "stratum_type", "stratum"],
            [
                "n_cases",
                "n_rows",
                "gap_ML_minus_MH_mean",
                "gap_ML_minus_MH_ci95_low",
                "gap_ML_minus_MH_ci95_high",
                "gap_MH_minus_ML_mean",
                "gap_MH_minus_ML_ci95_low",
                "gap_MH_minus_ML_ci95_high",
                "local_alignment_mean",
                "local_alignment_min",
                "local_alignment_max",
                "global_alignment_mean",
                "bootstrap_draws",
                "bootstrap_seed",
            ],
            tolerance=1e-12,
        )
        assert_expected(
            high_low_rows,
            read_csv(data_dir / "expected_measurement_high_low.csv"),
            ["model", "contrast"],
            ["mean", "ci95_low", "ci95_high", "n_cases_high", "n_cases_low", "bootstrap_draws", "bootstrap_seed"],
            tolerance=1e-12,
        )

    overall = [row for row in result_rows if row["stratum_type"] == "overall"]
    for row in overall:
        print(
            f"{row['model']} ML-MH overall: "
            f"{float(row['gap_ML_minus_MH_mean']):+.6f} "
            f"[{float(row['gap_ML_minus_MH_ci95_low']):+.6f}, "
            f"{float(row['gap_ML_minus_MH_ci95_high']):+.6f}]"
        )
    for row in high_low_rows:
        print(
            f"{row['model']} high-low alignment: "
            f"{float(row['mean']):+.6f} "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]"
        )
    if check:
        print("measurement strata check: PASS")


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
