from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pyarrow.parquet as pq


DEFAULT_ML_ROOT = r"C:\projects\ValueServices\data_ml\phase_3_1"


@dataclass
class CheckResult:
    ok: bool
    message: str


def _normalize_path(p: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(p))
    return Path(expanded).resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hard-fail DoD validator for Phase 3.1")
    ap.add_argument("--out-root", default=DEFAULT_ML_ROOT, help="Phase 3.1 output directory.")
    ap.add_argument("--split-mode", default="time", choices=["time", "none"], help="Expected split mode.")
    ap.add_argument("--min-rows", type=int, default=1, help="Minimum acceptable row count for dataset.")
    return ap.parse_args()


def check_exists_and_rows(path: Path, min_rows: int) -> CheckResult:
    if not path.exists():
        return CheckResult(False, f"Missing file: {path}")
    table = pq.read_table(str(path))
    if table.num_rows < min_rows:
        return CheckResult(False, f"Too few rows in {path}: rows={table.num_rows} < {min_rows}")
    return CheckResult(True, f"OK: {path} rows={table.num_rows}")


def check_target_return_nonempty(dataset_path: Path) -> CheckResult:
    t = pq.read_table(str(dataset_path))
    if "target_return" not in t.column_names:
        return CheckResult(False, "Column missing: target_return")

    col = t.column("target_return")
    null_count = col.null_count
    total = t.num_rows

    if total == 0:
        return CheckResult(False, "Dataset has 0 rows")
    if null_count == total:
        return CheckResult(False, "target_return is entirely NULL/NaN (no usable labels)")
    return CheckResult(True, f"OK: target_return non-null rows={total - null_count} / {total}")


def check_time_split_order(train_path: Path, valid_path: Path) -> CheckResult:
    train = pq.read_table(str(train_path)).to_pandas()
    valid = pq.read_table(str(valid_path)).to_pandas()

    if "date" not in train.columns or "date" not in valid.columns:
        return CheckResult(False, "Missing 'date' column in train/valid (required for time split validation)")

    if train.empty or valid.empty:
        return CheckResult(False, f"Train/valid must be non-empty. train_rows={len(train)} valid_rows={len(valid)}")

    max_train = train["date"].max()
    min_valid = valid["date"].min()

    if not (max_train < min_valid):
        return CheckResult(
            False,
            f"Time split invalid (overlap/leakage): max(train.date)={max_train} >= min(valid.date)={min_valid}"
        )

    return CheckResult(True, f"OK: time split order max(train.date)={max_train} < min(valid.date)={min_valid}")


def run_checks(out_root: Path, split_mode: str, min_rows: int) -> List[CheckResult]:
    dataset_path = out_root / "dataset.parquet"
    train_path = out_root / "train.parquet"
    valid_path = out_root / "valid.parquet"

    results: List[CheckResult] = []
    results.append(check_exists_and_rows(dataset_path, min_rows=min_rows))
    if results[-1].ok:
        results.append(check_target_return_nonempty(dataset_path))

    if split_mode == "time":
        results.append(check_exists_and_rows(train_path, min_rows=1))
        results.append(check_exists_and_rows(valid_path, min_rows=1))
        if results[-1].ok and results[-2].ok:
            results.append(check_time_split_order(train_path, valid_path))

    return results


def main() -> None:
    args = parse_args()
    out_root = _normalize_path(args.out_root)

    try:
        results = run_checks(out_root=out_root, split_mode=args.split_mode, min_rows=args.min_rows)
    except Exception as e:
        print(f"[PHASE3.1][ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    ok = all(r.ok for r in results)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[PHASE3.1][{status}] {r.message}")

    if ok:
        print("[PHASE3.1] DONE")
        sys.exit(0)
    else:
        print("[PHASE3.1] NOT DONE")
        sys.exit(2)


if __name__ == "__main__":
    main()
