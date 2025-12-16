from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import yaml

from backtest.prices import CsvPriceProvider
from backtest.returns import forward_return
from backtest.db_read import fetch_snapshots_minimal
from backtest.storage_returns import insert_postgres_returns


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config("backtest/config.yaml")

    run_id = cfg["run"]["run_id"]
    db_cfg = cfg.get("db") or {}
    returns_cfg = cfg.get("returns") or {}
    horizons = returns_cfg.get("horizons_months") or [12]

    ps = returns_cfg.get("price_source") or {}
    if str(ps.get("mode", "")).upper() != "CSV":
        raise RuntimeError("Dieser Runner ist für CSV konzipiert. Setze returns.price_source.mode = CSV.")

    provider = CsvPriceProvider(
        path=ps["csv_path"],
        date_col=ps.get("date_col", "date"),
        symbol_col=ps.get("symbol_col", "symbol"),
        price_col=ps.get("price_col", "close"),
    )

    snapshots = fetch_snapshots_minimal(db_cfg, run_id, table=db_cfg.get("table", "bt_snapshot"))
    if not snapshots:
        raise RuntimeError(f"Keine Snapshots gefunden für run_id={run_id}. Erst 2.1.2 laufen lassen.")

    rows: List[Dict[str, Any]] = []
    for s in snapshots:
        sym = s["symbol"].upper().strip()
        asof = datetime.strptime(s["asof_date"], "%Y-%m-%d").date()

        for h in horizons:
            h = int(h)
            res = forward_return(provider=provider, symbol=sym, asof=asof, horizon_months=h)
            if not res:
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "asof_date": s["asof_date"],
                    "symbol": sym,
                    "horizon_months": h,
                    "price_t": res["price_t"],
                    "price_t_h": res["price_t_h"],
                    "return_value": res["return_value"],
                    "source": "CSV",
                }
            )

    insert_postgres_returns(db_cfg, rows, table="bt_return")
    print(f"DONE 2.1.3: wrote {len(rows)} forward returns into bt_return for run_id={run_id}")


if __name__ == "__main__":
    main()
