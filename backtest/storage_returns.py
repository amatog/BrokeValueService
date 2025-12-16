from __future__ import annotations

from typing import Any, Dict, List
import psycopg2


def _pg_connect(db_cfg: Dict[str, Any]):
    url = db_cfg.get("url")
    if url:
        return psycopg2.connect(url)

    host = db_cfg.get("host", "localhost")
    port = int(db_cfg.get("port", 5432))
    database = db_cfg.get("database")
    username = db_cfg.get("username")
    password = db_cfg.get("password")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=username,
        password=password,
    )


def insert_postgres_returns(db_cfg: Dict[str, Any], rows: List[Dict[str, Any]], table: str = "bt_return") -> None:
    if not db_cfg.get("enabled") or not rows:
        return

    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            sql = f"""
                INSERT INTO {table}
                  (run_id, asof_date, symbol, horizon_months,
                   price_t, price_t_h, return_value, source)
                VALUES
                  (%s, %s, %s, %s,
                   %s, %s, %s, %s)
                ON CONFLICT (run_id, asof_date, symbol, horizon_months)
                DO UPDATE SET
                  price_t = EXCLUDED.price_t,
                  price_t_h = EXCLUDED.price_t_h,
                  return_value = EXCLUDED.return_value,
                  source = EXCLUDED.source
            """
            for r in rows:
                cur.execute(
                    sql,
                    (
                        r["run_id"],
                        r["asof_date"],
                        r["symbol"],
                        r["horizon_months"],
                        r["price_t"],
                        r["price_t_h"],
                        r["return_value"],
                        r.get("source", "CSV"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
