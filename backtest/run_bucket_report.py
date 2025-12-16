from __future__ import annotations

from typing import Any, Dict, List
import os
import yaml

from backtest.buckets import compute_bucket_report, BucketRow
from backtest.db_join import fetch_joined_snapshot_returns, fetch_distinct_sectors
from backtest.storage_bucket import insert_postgres_bucket_report


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    run_id = cfg["run"]["run_id"]
    db_cfg = cfg.get("db") or {}
    returns_cfg = cfg.get("returns") or {}
    horizons = returns_cfg.get("horizons_months") or [12]

    bcfg = cfg.get("buckets") or {}
    scheme = (bcfg.get("scheme") or "quintiles").strip().lower()
    by_sector = bool(bcfg.get("by_sector", False))
    min_rows_per_sector = int(bcfg.get("min_rows_per_sector", 30))

    snapshot_table = (db_cfg.get("table") or "bt_snapshot").strip()
    # returns table is fixed in our setup
    return_table = "bt_return"

    all_rows: List[BucketRow] = []

    for h in horizons:
        h = int(h)

        # Overall buckets (sector=None)
        joined = fetch_joined_snapshot_returns(
            db_cfg,
            run_id=run_id,
            horizon_months=h,
            snapshot_table=snapshot_table,
            return_table=return_table,
            sector=None,
        )
        #rep = compute_bucket_report(rows=joined, run_id=run_id, horizon_months=h, scheme=scheme, sector=None)

        rep = compute_bucket_report(
            rows=joined,
            run_id=run_id,
            horizon_months=h,
            scheme=scheme,
            sector="__ALL__",   # <-- PK-sicherer Overall-Wert
        )

        all_rows.extend(rep)

        # Optional: sector-split buckets
        if by_sector:
            sectors = fetch_distinct_sectors(db_cfg, run_id=run_id, snapshot_table=snapshot_table)
            for sec in sectors:
                joined_sec = fetch_joined_snapshot_returns(
                    db_cfg,
                    run_id=run_id,
                    horizon_months=h,
                    snapshot_table=snapshot_table,
                    return_table=return_table,
                    sector=sec,
                )
                if len(joined_sec) < min_rows_per_sector:
                    continue
                rep_sec = compute_bucket_report(rows=joined_sec, run_id=run_id, horizon_months=h, scheme=scheme, sector=sec)
                all_rows.extend(rep_sec)

        print(f"[OK] horizon={h} -> rows(overall)={len(joined)} buckets={len(rep)}")

    insert_postgres_bucket_report(db_cfg, all_rows, table="bt_bucket_report")
    print(f"\nDONE 2.1.4: wrote {len(all_rows)} bucket rows into bt_bucket_report for run_id={run_id}")


if __name__ == "__main__":
    main()
