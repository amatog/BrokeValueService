from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pg_connect(db_cfg: Dict[str, Any]):
    """
    Supports either:
      - db_cfg["url"] as SQLAlchemy-like URL (postgresql://user:pass@host:port/db)
      - or host/port/database/username/password
    """
    url = db_cfg.get("url")
    if url:
        return psycopg2.connect(url)

    host = db_cfg.get("host", "localhost")
    port = int(db_cfg.get("port", 5432))
    database = db_cfg.get("database")
    username = db_cfg.get("username")
    password = db_cfg.get("password")

    if not database or not username:
        raise RuntimeError("Postgres DB config incomplete: database/username missing.")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=username,
        password=password,
    )


def insert_postgres_snapshots(
        *,
        db_cfg: Dict[str, Any],
        rows: List[Dict[str, Any]],
        raw_by_key: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Inserts snapshot rows into PostgreSQL table (default: bt_snapshot).

    Erwartete Spalten (wie von Ihnen erweitert):
      run_id TEXT
      asof_date DATE
      symbol TEXT

      engine_mode TEXT
      engine_base_url TEXT
      sector TEXT
      currency TEXT

      value_score DOUBLE PRECISION
      value_rating TEXT

      features_json JSONB
      raw_json JSONB

    Primärschlüssel:
      (run_id, asof_date, symbol)
    """
    if not db_cfg.get("enabled"):
        return

    table = db_cfg.get("table", "bt_snapshot")

    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            sql = f"""
                INSERT INTO {table}
                    (run_id, asof_date, symbol,
                     engine_mode, engine_base_url,
                     sector, currency,
                     value_score, value_rating,
                     features_json, raw_json)
                VALUES
                    (%s, %s, %s,
                     %s, %s,
                     %s, %s,
                     %s, %s,
                     %s::jsonb, %s::jsonb)
                ON CONFLICT (run_id, asof_date, symbol)
                DO UPDATE SET
                    engine_mode     = EXCLUDED.engine_mode,
                    engine_base_url = EXCLUDED.engine_base_url,
                    sector          = EXCLUDED.sector,
                    currency        = EXCLUDED.currency,
                    value_score     = EXCLUDED.value_score,
                    value_rating    = EXCLUDED.value_rating,
                    features_json   = EXCLUDED.features_json,
                    raw_json        = EXCLUDED.raw_json
            """

            for r in rows:
                key = f"{r.get('symbol')}|{r.get('asof')}"
                raw = raw_by_key.get(key) if raw_by_key else None

                features_json = json.dumps(r, ensure_ascii=False)
                raw_json = json.dumps(raw, ensure_ascii=False) if raw is not None else None

                # --- execute(...) komplett ---
                cur.execute(
                    sql,
                    (
                        r.get("run_id"),
                        r.get("asof"),          # psycopg2 akzeptiert 'YYYY-MM-DD' für DATE
                        r.get("symbol"),

                        r.get("engine_mode"),
                        r.get("engine_base_url"),

                        r.get("sector"),
                        r.get("currency"),

                        r.get("value_score"),
                        r.get("value_rating"),

                        features_json,
                        raw_json,
                    ),
                )

        conn.commit()
    finally:
        conn.close()
