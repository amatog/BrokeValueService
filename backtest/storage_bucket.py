from __future__ import annotations

from typing import Any, Dict, List, Optional
import psycopg2

from backtest.buckets import BucketRow


def _pg_connect(db_cfg: Dict[str, Any]):
    url = db_cfg.get("url")
    if url:
        return psycopg2.connect(url)

    return psycopg2.connect(
        host=db_cfg.get("host", "localhost"),
        port=int(db_cfg.get("port", 5432)),
        dbname=db_cfg["database"],
        user=db_cfg["username"],
        password=db_cfg["password"],
    )


def insert_postgres_bucket_report(
        db_cfg: Dict[str, Any],
        rows: List[BucketRow],
        table: str = "bt_bucket_report",
) -> None:
    if not db_cfg.get("enabled") or not rows:
        return

    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            sql = f"""
                INSERT INTO {table}
                  (run_id, horizon_months, bucket_scheme, bucket_name, sector,
                   n, mean_return, median_return, hit_rate, std_return,
                   min_score, max_score)
                VALUES
                  (%s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s,
                   %s, %s)
                ON CONFLICT (run_id, horizon_months, bucket_scheme, bucket_name, sector)
                DO UPDATE SET
                  n = EXCLUDED.n,
                  mean_return = EXCLUDED.mean_return,
                  median_return = EXCLUDED.median_return,
                  hit_rate = EXCLUDED.hit_rate,
                  std_return = EXCLUDED.std_return,
                  min_score = EXCLUDED.min_score,
                  max_score = EXCLUDED.max_score
            """
            for r in rows:
                cur.execute(
                    sql,
                    (
                        r.run_id,
                        r.horizon_months,
                        r.bucket_scheme,
                        r.bucket_name,
                        r.sector,
                        r.n,
                        r.mean_return,
                        r.median_return,
                        r.hit_rate,
                        r.std_return,
                        r.min_score,
                        r.max_score,
                    ),
                )
        conn.commit()
    finally:
        conn.close()