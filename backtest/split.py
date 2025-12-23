# split.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


_DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class TimeSplit:
    """Globaler, deterministischer Time-Series Split über Trading-Days (date partitions)."""
    train_dates: List[str]
    val_dates: List[str]
    test_dates: List[str]

    @property
    def train_min(self) -> Optional[str]:
        return self.train_dates[0] if self.train_dates else None

    @property
    def train_max(self) -> Optional[str]:
        return self.train_dates[-1] if self.train_dates else None

    @property
    def val_min(self) -> Optional[str]:
        return self.val_dates[0] if self.val_dates else None

    @property
    def val_max(self) -> Optional[str]:
        return self.val_dates[-1] if self.val_dates else None

    @property
    def test_min(self) -> Optional[str]:
        return self.test_dates[0] if self.test_dates else None

    @property
    def test_max(self) -> Optional[str]:
        return self.test_dates[-1] if self.test_dates else None

    def to_dict(self) -> dict:
        return {
            "train_dates": self.train_dates,
            "val_dates": self.val_dates,
            "test_dates": self.test_dates,
            "train_min": self.train_min,
            "train_max": self.train_max,
            "val_min": self.val_min,
            "val_max": self.val_max,
            "test_min": self.test_min,
            "test_max": self.test_max,
        }


def _parse_date(s: str) -> Date:
    # YYYY-MM-DD
    y, m, d = s.split("-")
    return Date(int(y), int(m), int(d))


def list_partition_dates(primary_root: str | Path) -> List[str]:
    """
    Listet vorhandene date=YYYY-MM-DD Partitionen aus dem A.2 Primary Store.
    Rückgabe: sortierte Liste von ISO-Dates (YYYY-MM-DD).
    """
    root = Path(primary_root)
    if not root.exists():
        raise FileNotFoundError(f"Primary root not found: {root}")

    dates: List[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _DATE_DIR_RE.match(p.name)
        if not m:
            continue
        dates.append(m.group(1))

    # Sortierung deterministisch nach Datum
    dates.sort(key=_parse_date)
    return dates


def make_time_split(
        dates: Sequence[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_train_days: int = 252,
        min_val_days: int = 63,
        min_test_days: int = 63,
) -> TimeSplit:
    """
    Erzeugt einen globalen Time-Series Split über die Date-Partitions.

    - Kein Shuffle
    - ratios werden auf Index-Cuts abgebildet
    - Mindesttage erzwingen Fail-Fast bei zu wenig Daten
    """
    if not dates:
        raise ValueError("No dates provided for split.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n = len(dates)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Sicherstellen dass test nicht leer ist
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_dates = list(dates[:train_end])
    val_dates = list(dates[train_end:val_end])
    test_dates = list(dates[val_end:])

    # Minimum constraints
    if len(train_dates) < min_train_days:
        raise ValueError(
            f"Insufficient train days: {len(train_dates)} < {min_train_days} "
            f"(total dates={n})"
        )
    if len(val_dates) < min_val_days:
        raise ValueError(
            f"Insufficient val days: {len(val_dates)} < {min_val_days} "
            f"(train_end={train_end}, val_end={val_end}, total dates={n})"
        )
    if len(test_dates) < min_test_days:
        raise ValueError(
            f"Insufficient test days: {len(test_dates)} < {min_test_days} "
            f"(train_end={train_end}, val_end={val_end}, total dates={n})"
        )

    return TimeSplit(train_dates=train_dates, val_dates=val_dates, test_dates=test_dates)


def resolve_split_from_primary_root(
        primary_root: str | Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_train_days: int = 252,
        min_val_days: int = 63,
        min_test_days: int = 63,
) -> TimeSplit:
    """
    Convenience: liest Partition-Dates aus primary_root und erzeugt TimeSplit.
    """
    dates = list_partition_dates(primary_root)
    return make_time_split(
        dates=dates,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_train_days=min_train_days,
        min_val_days=min_val_days,
        min_test_days=min_test_days,
    )


def save_split_json(split: TimeSplit, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, ensure_ascii=False, indent=2)


def load_split_json(path: str | Path) -> TimeSplit:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return TimeSplit(
        train_dates=list(d.get("train_dates", [])),
        val_dates=list(d.get("val_dates", [])),
        test_dates=list(d.get("test_dates", [])),
    )


def main() -> None:
    """
    CLI (optional):
      python -m ml.train.split --primaryRoot C:\\projects\\ValueServices\\data\\data_primary --out ml/artifacts/split.json
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--primaryRoot", required=True, help="A.2 Primary Store root with date=YYYY-MM-DD partitions")
    ap.add_argument("--out", required=True, help="Output JSON path for split metadata")
    ap.add_argument("--trainRatio", type=float, default=0.70)
    ap.add_argument("--valRatio", type=float, default=0.15)
    ap.add_argument("--testRatio", type=float, default=0.15)
    ap.add_argument("--minTrainDays", type=int, default=252)
    ap.add_argument("--minValDays", type=int, default=63)
    ap.add_argument("--minTestDays", type=int, default=63)
    args = ap.parse_args()

    split = resolve_split_from_primary_root(
        primary_root=args.primaryRoot,
        train_ratio=args.trainRatio,
        val_ratio=args.valRatio,
        test_ratio=args.testRatio,
        min_train_days=args.minTrainDays,
        min_val_days=args.minValDays,
        min_test_days=args.minTestDays,
    )

    save_split_json(split, args.out)

    print("[SPLIT] written:", args.out)
    print("[SPLIT] train:", split.train_min, "->", split.train_max, "days=", len(split.train_dates))
    print("[SPLIT]   val:", split.val_min, "->", split.val_max, "days=", len(split.val_dates))
    print("[SPLIT]  test:", split.test_min, "->", split.test_max, "days=", len(split.test_dates))


if __name__ == "__main__":
    main()
