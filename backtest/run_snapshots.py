from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import yaml

from backtest.engines import ValueRestEngine, DirectStrategyEngine
from backtest.flatten import flatten_bundle
from backtest.universe import get_symbols
from backtest.schedule import get_asof_dates
from backtest.storage import (
    ensure_dir,
    write_csv,
    write_jsonl,
    insert_postgres_snapshots,
)

from data_layer import DataLayer


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_engine(cfg: Dict[str, Any]):
    e = cfg.get("engine") or {}
    mode = str(e.get("mode") or "DIRECT").strip().upper()

    if mode == "REST":
        base_url = str(e.get("value_base_url") or "http://localhost:8001").strip()
        timeout = int(e.get("timeout_seconds") or 25)
        return ValueRestEngine(base_url=base_url, timeout_seconds=timeout)

    if mode == "DIRECT":
        dl = DataLayer()
        return DirectStrategyEngine(data_layer=dl)

    raise ValueError(f"Unknown engine.mode: {mode}")


def main() -> None:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    run_cfg = cfg.get("run") or {}
    run_id = str(run_cfg.get("run_id") or f"run_{int(time.time())}")

    symbols = get_symbols(cfg)
    asof_dates = get_asof_dates(cfg)

    if not symbols:
        raise RuntimeError("Universe ist leer. Bitte backtest/config.yaml -> universe.symbols setzen.")
    if not asof_dates:
        raise RuntimeError("Schedule ist leer. Bitte backtest/config.yaml -> run.asof_dates setzen.")

    engine_cfg = cfg.get("engine") or {}
    engine_mode = str(engine_cfg.get("mode") or "DIRECT").strip().upper()
    engine_base_url = str(engine_cfg.get("value_base_url") or "").strip() if engine_mode == "REST" else ""

    engine = build_engine(cfg)

    out_cfg = cfg.get("output") or {}
    out_dir = str(out_cfg.get("out_dir") or "backtest_runs")
    run_dir = os.path.join(out_dir, run_id)
    ensure_dir(run_dir)

    rows: List[Dict[str, Any]] = []
    raw_by_key: Dict[str, Any] = {}

    for asof in asof_dates:
        for sym in symbols:
            sym = sym.upper().strip()
            if not sym:
                continue

            # 1) Snapshot (REST oder DIRECT)
            raw = engine.snapshot(sym, asof=asof)

            # 2) Engine-Metadaten injizieren (damit Flattening/DB es speichern kann)
            raw["_engine_mode"] = engine_mode
            raw["_engine_base_url"] = engine_base_url

            raw_by_key[f"{sym}|{asof}"] = raw

            # 3) Flatten
            row = flatten_bundle(raw, symbol=sym, asof=asof, run_id=run_id)
            rows.append(row)

            print(
                f"[OK] snapshot {sym} @ {asof} -> value_score={row.get('value_score')} rating={row.get('value_rating')}"
            )

    # 4) Files schreiben
    if bool(out_cfg.get("write_csv", True)):
        write_csv(os.path.join(run_dir, "snapshots.csv"), rows)

    if bool(out_cfg.get("write_jsonl", True)):
        write_jsonl(os.path.join(run_dir, "snapshots.jsonl"), rows)

    # 5) Optional: PostgreSQL Insert
    db_cfg = cfg.get("db") or {}
    if bool(db_cfg.get("enabled", False)):
        insert_postgres_snapshots(db_cfg=db_cfg, rows=rows, raw_by_key=raw_by_key)

    print(f"\nDONE 2.1.2: wrote {len(rows)} snapshots to: {run_dir}")

    print("[CFG] loaded keys:", list(cfg.keys()))
    print("[CFG] universe:", (cfg.get("universe") or {}))
    print("[CFG] engine:", (cfg.get("engine") or {}))



if __name__ == "__main__":
    main()
