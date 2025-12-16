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


def fetch_snapshots_minimal(db_cfg: Dict[str, Any], run_id: str, table: str = "bt_snapshot") -> List[Dict[str, str]]:
    """
    Returns list of dicts:
      [{"symbol": "...", "asof_date": "YYYY-MM-DD"}, ...]
    """
    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT symbol, to_char(asof_date, 'YYYY-MM-DD') AS asof_date
                FROM {table}
                WHERE run_id = %s
                ORDER BY asof_date, symbol
                """,
                (run_id,),
            )
            out: List[Dict[str, str]] = []
            for sym, asof in cur.fetchall():
                out.append({"symbol": sym, "asof_date": asof})
            return out
    finally:
        conn.close()
