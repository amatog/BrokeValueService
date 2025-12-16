from __future__ import annotations

from typing import Any, Dict, List
import psycopg2


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


def fetch_joined_snapshot_returns(
        db_cfg: Dict[str, Any],
        *,
        run_id: str,
        horizon_months: int,
        snapshot_table: str = "bt_snapshot",
        return_table: str = "bt_return",
        sector: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with:
      symbol, asof_date, sector, value_score, return_value
    """
    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            if sector:
                sql = f"""
                    SELECT s.symbol,
                           to_char(s.asof_date, 'YYYY-MM-DD') AS asof_date,
                           s.sector,
                           s.value_score,
                           r.return_value
                    FROM {snapshot_table} s
                    JOIN {return_table} r
                      ON r.run_id = s.run_id
                     AND r.asof_date = s.asof_date
                     AND r.symbol = s.symbol
                    WHERE s.run_id = %s
                      AND r.horizon_months = %s
                      AND s.value_score IS NOT NULL
                      AND r.return_value IS NOT NULL
                      AND s.sector = %s
                """
                cur.execute(sql, (run_id, horizon_months, sector))
            else:
                sql = f"""
                    SELECT s.symbol,
                           to_char(s.asof_date, 'YYYY-MM-DD') AS asof_date,
                           s.sector,
                           s.value_score,
                           r.return_value
                    FROM {snapshot_table} s
                    JOIN {return_table} r
                      ON r.run_id = s.run_id
                     AND r.asof_date = s.asof_date
                     AND r.symbol = s.symbol
                    WHERE s.run_id = %s
                      AND r.horizon_months = %s
                      AND s.value_score IS NOT NULL
                      AND r.return_value IS NOT NULL
                """
                cur.execute(sql, (run_id, horizon_months))

            rows: List[Dict[str, Any]] = []
            for sym, asof, sec, score, ret in cur.fetchall():
                rows.append(
                    {
                        "symbol": sym,
                        "asof_date": asof,
                        "sector": sec,
                        "value_score": score,
                        "return_value": ret,
                    }
                )
            return rows
    finally:
        conn.close()


def fetch_distinct_sectors(db_cfg: Dict[str, Any], *, run_id: str, snapshot_table: str = "bt_snapshot") -> List[str]:
    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT sector
                FROM {snapshot_table}
                WHERE run_id = %s
                  AND sector IS NOT NULL
                  AND sector <> ''
                ORDER BY sector
                """,
                (run_id,),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
