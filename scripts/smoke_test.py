#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import assert_expected


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_ROOT / "outputs" / "smoke"


def run_script(script_name: str) -> None:
    cmd = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / script_name),
        "--out-dir",
        str(OUT_DIR),
        "--check",
    ]
    try:
        completed = subprocess.run(cmd, cwd=PACKAGE_ROOT, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise
    print(completed.stdout.strip())


def main() -> int:
    verify_assert_expected_guards()
    run_script("reproduce_readout_checks.py")
    run_script("reproduce_measurement_strata.py")
    print(f"smoke test: PASS; outputs written to {OUT_DIR}")
    return 0


def verify_assert_expected_guards() -> None:
    checks = [
        (
            [{"id": "a", "value": 1.0}, {"id": "a", "value": 1.0}],
            [{"id": "a", "value": 1.0}],
            "duplicate produced keys",
        ),
        (
            [{"id": "a", "value": 1.0}],
            [{"id": "a", "value": 1.0}, {"id": "a", "value": 1.0}],
            "duplicate expected keys",
        ),
        (
            [{"id": "a", "value": float("nan")}],
            [{"id": "a", "value": 1.0}],
            "non-finite produced value",
        ),
        (
            [{"id": "a", "value": 1.0}],
            [{"id": "a", "value": float("inf")}],
            "non-finite expected value",
        ),
    ]
    for produced, expected, label in checks:
        try:
            assert_expected(produced, expected, ["id"], ["value"], tolerance=1e-12)
        except AssertionError:
            continue
        raise AssertionError(f"verification guard failed to reject {label}")
    print("verification guard check: PASS")


if __name__ == "__main__":
    raise SystemExit(main())
