from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: Iterable[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return dict(out)


def grouped_case_values(rows: Iterable[dict[str, str]], value_col: str) -> list[dict[str, float | str | int]]:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    case_values: list[dict[str, float | str | int]] = []
    for case_id in sorted(by_case):
        case_rows = by_case[case_id]
        vals = [float(row[value_col]) for row in case_rows]
        case_values.append(
            {
                "case_id": case_id,
                "value": float(sum(vals) / len(vals)),
                "n_rows": len(case_rows),
                "local_alignment_overall": float(case_rows[0].get("local_alignment_overall", "nan")),
                "global_alignment_overall": float(case_rows[0].get("global_alignment_overall", "nan")),
                "alignment_group": case_rows[0].get("alignment_group", ""),
            }
        )
    return case_values


def bootstrap_mean(values: list[float], draws: int, seed: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for draw_ix in range(draws):
        ix = rng.integers(0, len(vals), size=len(vals))
        boot[draw_ix] = vals[ix].mean()
    return float(vals.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def assert_expected(
    produced: list[dict[str, object]],
    expected: list[dict[str, str]],
    key_cols: list[str],
    numeric_cols: list[str],
    tolerance: float,
) -> None:
    actual_by_key = index_by_unique_key(produced, key_cols, "produced")
    expected_by_key = index_by_unique_key(expected, key_cols, "expected")
    if set(actual_by_key) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(actual_by_key))
        extra = sorted(set(actual_by_key) - set(expected_by_key))
        raise AssertionError(f"key mismatch: missing={missing}, extra={extra}")
    for key in sorted(expected_by_key):
        actual = actual_by_key[key]
        exp = expected_by_key[key]
        for col in numeric_cols:
            got = float(actual[col])
            want = float(exp[col])
            if not math.isfinite(got) or not math.isfinite(want):
                raise AssertionError(f"{key} {col}: non-finite value got={got}, expected={want}")
            if abs(got - want) > tolerance:
                raise AssertionError(f"{key} {col}: got {got}, expected {want}")


def index_by_unique_key(
    rows: Iterable[dict[str, object]],
    key_cols: list[str],
    label: str,
) -> dict[tuple[str, ...], dict[str, object]]:
    indexed: dict[tuple[str, ...], dict[str, object]] = {}
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(str(row[col]) for col in key_cols)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        raise AssertionError(f"duplicate {label} keys: {duplicates}")
    return indexed
