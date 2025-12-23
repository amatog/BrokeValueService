# backtest/export_dataset.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# -----------------------------
# Config / Defaults (Phase 3.1)
# -----------------------------

DEFAULT_RAW_ROOT = r"C:\projects\ValueServices\data"
DEFAULT_ML_ROOT = r"C:\projects\ValueServices\data_ml\phase_3_1"

DATASET_FILE = "dataset.parquet"
TRAIN_FILE = "train.parquet"
VALID_FILE = "valid.parquet"


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    out_root: Path
    dataset_path: Path
    train_path: Path
    valid_path: Path


def _normalize_path(p: str) -> Path:
    """
    Normalize Windows/Unix-ish user input to an absolute Path.
    Supports environment variable expansion and ~.
    """
    expanded = os.path.expandvars(os.path.expanduser(p))
    return Path(expanded).resolve()


def build_paths(raw_root: str, out_root: str) -> Paths:
    rr = _normalize_path(raw_root)
    oroot = _normalize_path(out_root)

    dataset = oroot / DATASET_FILE
    train = oroot / TRAIN_FILE
    valid = oroot / VALID_FILE

    return Paths(
        raw_root=rr,
        out_root=oroot,
        dataset_path=dataset,
        train_path=train,
        valid_path=valid,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_partition_layout(raw_root: Path) -> str:
    """
    Detect whether raw store is date-partitioned (date=YYYY-MM-DD) or symbol-partitioned (symbol=XYZ).
    Hard-fails if neither is detected.
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    children = [p for p in raw_root.iterdir() if p.is_dir()]
    if any(p.name.startswith("date=") for p in children):
        return "date"
    if any(p.name.startswith("symbol=") for p in children):
        return "symbol"

    raise RuntimeError(
        "Could not detect partition layout under raw_root. "
        "Expected folders like 'date=YYYY-MM-DD' or 'symbol=XYZ'. "
        f"raw_root={raw_root}"
    )


def list_parquet_files(raw_root: Path) -> list[Path]:
    """
    Enumerate parquet files under the partitioned store.
    Assumes files are stored in partition directories (e.g. date=.../part-0000.parquet).
    """
    files = sorted(raw_root.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found under raw_root: {raw_root}")
    return files


# -----------------------------
# Robust Loader (fixes symbol type mismatch)
# -----------------------------

def _ensure_symbol_string(t: pa.Table) -> pa.Table:
    """
    Force column 'symbol' to be string, to normalize mixed Arrow types:
    string vs dictionary<values=string,...>.
    """
    if "symbol" not in t.column_names:
        return t

    sym = t["symbol"]
    if pa.types.is_dictionary(sym.type):
        sym_str = pc.cast(sym, pa.string())
        t = t.set_column(t.schema.get_field_index("symbol"), "symbol", sym_str)
    elif pa.types.is_large_string(sym.type):
        sym_str = pc.cast(sym, pa.string())
        t = t.set_column(t.schema.get_field_index("symbol"), "symbol", sym_str)

    return t


def load_raw_table_from_partitions(parquet_files: list[Path], chunk_size: int = 200) -> pa.Table:
    """
    Robust loader:
    - reads parquet files one by one (avoids schema merge errors across files)
    - normalizes 'symbol' column to string (fixes dictionary vs string mismatch)
    - concatenates in chunks to control memory overhead
    """
    if not parquet_files:
        raise RuntimeError("No parquet files provided.")

    chunks: list[pa.Table] = []
    acc: list[pa.Table] = []

    for i, p in enumerate(parquet_files, start=1):
        try:
            pf = pq.ParquetFile(str(p))
            t = pf.read()
        except Exception as e:
            raise RuntimeError(f"Failed reading parquet file: {p} ({e})") from e

        t = _ensure_symbol_string(t)
        acc.append(t)

        if len(acc) >= chunk_size:
            chunks.append(pa.concat_tables(acc, promote=True))
            acc.clear()

        if i % 1000 == 0:
            print(f"[EXPORT] read_files={i}/{len(parquet_files)}")

    if acc:
        chunks.append(pa.concat_tables(acc, promote=True))

    return pa.concat_tables(chunks, promote=True)


# -----------------------------
# Feature + Target (minimal)
# -----------------------------

def build_features_and_target(raw: pa.Table, target_horizon: int) -> pa.Table:
    """
    Minimal but robust Phase 3.1 builder:
    - requires: date, symbol, price column (adjusted_close/adj_close/close)
    - computes: target_return = (future_price / price) - 1, shifted by target_horizon within each symbol
    - keeps: date, symbol, price, target_return + numeric columns already present
    """
    import pandas as pd

    cols = set(raw.column_names)

    if "date" not in cols:
        raise RuntimeError("Raw table missing required column: 'date'")
    if "symbol" not in cols:
        raise RuntimeError("Raw table missing required column: 'symbol'")

    price_candidates = ["adjusted_close", "adj_close", "close"]
    price_col = next((c for c in price_candidates if c in cols), None)
    if price_col is None:
        raise RuntimeError(f"Raw table missing price column. Expected one of: {price_candidates}")

    df = raw.to_pandas()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise RuntimeError("Column 'date' could not be parsed to datetime (all NaT).")

    df["symbol"] = df["symbol"].astype(str)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=["date", "symbol", price_col]).copy()
    if df.empty:
        raise RuntimeError("No usable rows after dropping NA date/symbol/price.")

    df = df.sort_values(["symbol", "date"], kind="mergesort")

    future_price = df.groupby("symbol", sort=False)[price_col].shift(-target_horizon)
    df["target_return"] = (future_price / df[price_col]) - 1.0

    keep_base = ["date", "symbol", price_col, "target_return"]

    numeric_cols = []
    for c in df.columns:
        if c in keep_base:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    keep_cols = keep_base + numeric_cols
    out = df[keep_cols].copy()

    if out["target_return"].notna().sum() == 0:
        raise RuntimeError(
            "target_return is entirely null. "
            f"This usually means target_horizon={target_horizon} is too large for your per-symbol history."
        )

    return pa.Table.from_pandas(out, preserve_index=False)


# -----------------------------
# Leakage-safe time split (date-level)
# -----------------------------

def time_split(table: pa.Table, split_ratio: float) -> Tuple[pa.Table, pa.Table]:
    """
    Deterministic, leakage-safe time split on DATE LEVEL (not row level).
    Ensures: max(train.date) < min(valid.date)
    """
    import pandas as pd

    if "date" not in table.column_names:
        raise RuntimeError("time_split requires a 'date' column in the dataset.")

    df = table.to_pandas()
    if df.empty:
        raise RuntimeError("Dataset is empty; cannot split.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise RuntimeError("date column could not be parsed to datetime (all NaT).")

    df = df.sort_values(["date", "symbol"] if "symbol" in df.columns else ["date"], kind="mergesort")

    dates = pd.Series(df["date"].unique()).sort_values().to_list()
    if len(dates) < 2:
        raise RuntimeError(f"Not enough unique dates for time split. unique_dates={len(dates)}")

    cut_idx = int(len(dates) * split_ratio) - 1
    cut_idx = max(0, min(cut_idx, len(dates) - 2))  # ensure at least one date in valid
    cut_date = dates[cut_idx]

    train_df = df[df["date"] <= cut_date].copy()
    valid_df = df[df["date"] > cut_date].copy()

    if train_df.empty or valid_df.empty:
        raise RuntimeError(
            f"Invalid split produced empty side: train_rows={len(train_df)} valid_rows={len(valid_df)} "
            f"(cut_date={cut_date}, unique_dates={len(dates)})"
        )

    return (
        pa.Table.from_pandas(train_df, preserve_index=False),
        pa.Table.from_pandas(valid_df, preserve_index=False),
    )


# -----------------------------
# Main CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export ML dataset (Phase 3.1) from Java Parquet store.")
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Java Parquet store root (A1/A2/A3).")
    ap.add_argument("--out-root", default=DEFAULT_ML_ROOT, help="Phase 3.1 output directory (ML store).")
    ap.add_argument("--split-mode", default="time", choices=["time", "none"], help="Dataset split strategy.")
    ap.add_argument("--split-ratio", type=float, default=0.8, help="Train ratio for time split (0..1).")
    ap.add_argument("--target-horizon", type=int, default=21, help="Horizon for target_return computation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(args.raw_root, args.out_root)

    ensure_dir(paths.out_root)

    layout = detect_partition_layout(paths.raw_root)
    print(f"[EXPORT] raw_root={paths.raw_root}")
    print(f"[EXPORT] layout={layout}")
    print(f"[EXPORT] out_root={paths.out_root}")

    parquet_files = list_parquet_files(paths.raw_root)
    print(f"[EXPORT] parquet_files={len(parquet_files)}")

    raw_table = load_raw_table_from_partitions(parquet_files)
    print(f"[EXPORT] raw_rows={raw_table.num_rows} raw_cols={len(raw_table.column_names)}")

    dataset = build_features_and_target(raw_table, target_horizon=args.target_horizon)
    print(f"[EXPORT] dataset_rows={dataset.num_rows} dataset_cols={len(dataset.column_names)}")

    # Write full dataset (may include unlabeled tail rows)
    pq.write_table(dataset, str(paths.dataset_path))
    print(f"[EXPORT] wrote={paths.dataset_path}")

    # Optional split (IMPORTANT: split on labeled rows to avoid valid=0 in Phase 3.2)
    if args.split_mode == "time":
        if "target_return" not in dataset.column_names:
            raise RuntimeError("Split requested but dataset has no 'target_return' column.")

        labeled = dataset.filter(pc.is_valid(dataset["target_return"]))
        print(f"[EXPORT] labeled_rows={labeled.num_rows} (target_return not null)")

        if labeled.num_rows == 0:
            raise RuntimeError(
                "No labeled rows available (target_return all null). "
                "Reduce target_horizon or extend data coverage."
            )

        train, valid = time_split(labeled, split_ratio=args.split_ratio)

        if valid.num_rows == 0:
            raise RuntimeError(
                "Valid split is empty after labeling filter. "
                "Reduce split_ratio (e.g. 0.7) or extend data coverage."
            )

        pq.write_table(train, str(paths.train_path))
        pq.write_table(valid, str(paths.valid_path))
        print(f"[EXPORT] wrote={paths.train_path} rows={train.num_rows}")
        print(f"[EXPORT] wrote={paths.valid_path} rows={valid.num_rows}")

    print("[EXPORT] completed")


if __name__ == "__main__":
    main()
