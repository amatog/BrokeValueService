from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
import psycopg2


# --------------------------------------------------------------------------------------
# Calibration Runner
# --------------------------------------------------------------------------------------
# - Reads:
#     backtest/config.yaml                  (infrastructure: DB, engine mode, universe, returns, buckets)
#     configs/weights_v1_baseline.yaml      (baseline weights)
#     configs/calibration_search_v1.yaml    (search ranges)
#
# - Generates candidate configs:
#     configs/generated/weights_candidate_<calibration_id>_<idx>.yaml
#
# - Runs pipeline per candidate (new run_id each time):
#     python -m backtest.run_snapshots
#     python -m backtest.run_returns
#     python -m backtest.run_bucket_report
#
# - Writes metrics into Postgres:
#     bt_calibration_result
#
# IMPORTANT:
#   This runner passes the candidate weights file to the scoring engine via:
#       ENV VALUE_WEIGHTS_CONFIG=<candidate_yaml_path>
#   Your strategies/combined_value_score must read that file to apply weights.
# --------------------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent.parent  # project root (ValueServices/)
BACKTEST_DIR = BASE_DIR / "backtest"
CONFIG_PATH = BACKTEST_DIR / "config.yaml"

BASELINE_PATH = BASE_DIR / "configs" / "weights_v1_baseline.yaml"
SEARCH_PATH = BASE_DIR / "configs" / "calibration_search_v1.yaml"
GENERATED_DIR = BASE_DIR / "configs" / "generated"

ENV_WEIGHTS_KEY = "VALUE_WEIGHTS_CONFIG"


@dataclass
class Candidate:
    idx: int
    path: Path
    config_version: str
    run_id: str
    weights: Dict[str, float]
    insufficient_data_factor: float


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


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


def _ensure_calibration_table(db_cfg: Dict[str, Any]) -> None:
    """
    Check-only: In production-like setups the application user (broker_user)
    typically does NOT have CREATE privilege on schema public.

    Therefore we do not create/migrate tables here. The table must exist and
    the app user must have SELECT/INSERT/UPDATE rights.
    """
    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'bt_calibration_result'
                """
            )
            ok = cur.fetchone()
            if not ok:
                raise RuntimeError(
                    "Missing table public.bt_calibration_result. "
                    "Create it once via pgAdmin as owner/superuser, then GRANT rights to broker_user."
                )
    finally:
        conn.close()



def _fetch_bucket_extremes(
        db_cfg: Dict[str, Any],
        *,
        run_id: str,
        horizon_months: int,
        bucket_scheme: str,
) -> Optional[Dict[str, Any]]:
    """
    Reads bt_bucket_report (sector='__ALL__') and returns a dict (metrics_raw) containing:
      - n_joined
      - top_mean_return, bottom_mean_return, spread_mean
      - top_hit_rate, bottom_hit_rate

    NOTE:
      We rely on bucket labels containing "Top" / "Bottom" as produced by run_bucket_report.
      Overall rows must use sector='__ALL__' (PK-safe).
    """
    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bucket_name, n, mean_return, hit_rate
                FROM public.bt_bucket_report
                WHERE run_id = %s
                  AND horizon_months = %s
                  AND bucket_scheme = %s
                  AND sector = '__ALL__'
                """,
                (run_id, horizon_months, bucket_scheme),
            )
            rows = cur.fetchall()
            if not rows:
                return None

            top = None
            bottom = None
            n_joined = 0

            for bucket_name, n, mean_return, hit_rate in rows:
                n_joined += int(n or 0)

                if bucket_name and "Top" in bucket_name:
                    top = {
                        "bucket_name": bucket_name,
                        "n": int(n or 0),
                        "mean_return": float(mean_return) if mean_return is not None else None,
                        "hit_rate": float(hit_rate) if hit_rate is not None else None,
                    }

                if bucket_name and "Bottom" in bucket_name:
                    bottom = {
                        "bucket_name": bucket_name,
                        "n": int(n or 0),
                        "mean_return": float(mean_return) if mean_return is not None else None,
                        "hit_rate": float(hit_rate) if hit_rate is not None else None,
                    }

            if not top or not bottom:
                # bucket labels missing or incomplete
                return {
                    "n_joined": n_joined,
                    "top_mean_return": None,
                    "bottom_mean_return": None,
                    "spread_mean": None,
                    "top_hit_rate": None,
                    "bottom_hit_rate": None,
                }

            top_mean = top["mean_return"]
            bottom_mean = bottom["mean_return"]
            spread = (top_mean - bottom_mean) if (top_mean is not None and bottom_mean is not None) else None

            return {
                "n_joined": n_joined,
                "top_mean_return": top_mean,
                "bottom_mean_return": bottom_mean,
                "spread_mean": spread,
                "top_hit_rate": top["hit_rate"],
                "bottom_hit_rate": bottom["hit_rate"],
            }
    finally:
        conn.close()


def _upsert_calibration_result(
        db_cfg: Dict[str, Any],
        *,
        calibration_id: str,
        run_id: str,
        horizon_months: int,
        config_version: str,
        metrics: Dict[str, Any],
) -> None:
    """
    Writes metric rows compatible with the existing PK:
      PRIMARY KEY (calibration_id, run_id, horizon_months, config_version, metric)

    Each metric is a separate row with metric_value.
    """
    conn = _pg_connect(db_cfg)
    try:
        with conn.cursor() as cur:
            sql = """
                  INSERT INTO public.bt_calibration_result
                  (calibration_id, run_id, horizon_months, config_version,
                   metric, metric_value)
                  VALUES
                      (%s, %s, %s, %s,
                       %s, %s)
                      ON CONFLICT (calibration_id, run_id, horizon_months, config_version, metric)
                DO UPDATE SET
                      metric_value = EXCLUDED.metric_value; \
                  """

            # map metrics dict -> rows
            for metric_name, metric_value in metrics.items():
                cur.execute(
                    sql,
                    (
                        calibration_id,
                        run_id,
                        horizon_months,
                        config_version,
                        str(metric_name),
                        metric_value,
                    ),
                )

        conn.commit()
    finally:
        conn.close()



def _generate_candidates(
        *,
        baseline: Dict[str, Any],
        search: Dict[str, Any],
        calibration_id: str,
        max_candidates: int = 1000,
) -> List[Candidate]:
    """
    Creates candidates from cartesian product of the search space.
    Each candidate writes a full weights-config file that your engine can load.
    """
    base_agg = (baseline.get("aggregation") or {})
    base_weights: Dict[str, float] = dict(base_agg.get("weights") or {})
    base_idf = float(base_agg.get("insufficient_data_factor", 0.25))

    s = (search.get("search") or {})
    s_weights = (s.get("weights") or {})
    s_idf_list = (s.get("insufficient_data_factor") or [])

    # Build grid list: each entry is (param_name, values)
    grid_items: List[Tuple[str, List[Any]]] = []
    for k, v in s_weights.items():
        grid_items.append((f"w::{k}", list(v)))
    if s_idf_list:
        grid_items.append(("idf", list(s_idf_list)))

    if not grid_items:
        raise RuntimeError("Search space empty: configs/calibration_search_v1.yaml -> search.* is missing/empty.")

    keys = [k for k, _ in grid_items]
    values_list = [vals for _, vals in grid_items]

    out: List[Candidate] = []
    for idx, combo in enumerate(itertools.product(*values_list), start=1):
        if idx > max_candidates:
            break

        weights = dict(base_weights)
        idf = base_idf

        for k, v in zip(keys, combo):
            if k == "idf":
                idf = float(v)
            elif k.startswith("w::"):
                name = k[3:]
                weights[name] = float(v)

        config_version = f"candidate_{calibration_id}_{idx:04d}"
        run_id = f"{calibration_id}_{idx:04d}"

        candidate_doc = {
            "version": config_version,
            "description": f"Auto-generated candidate {idx} from calibration {calibration_id}",
            "aggregation": {
                "insufficient_data_factor": idf,
                "weights": weights,
            },
        }

        path = GENERATED_DIR / f"weights_candidate_{calibration_id}_{idx:04d}.yaml"
        _dump_yaml(path, candidate_doc)

        out.append(
            Candidate(
                idx=idx,
                path=path,
                config_version=config_version,
                run_id=run_id,
                weights=weights,
                insufficient_data_factor=idf,
            )
        )

    return out


def _run_module(module_name: str, *, env: Dict[str, str]) -> None:
    """
    Executes: python -m <module_name>
    Raises on non-zero exit.
    """
    cmd = [sys.executable, "-m", module_name]
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Module failed: {module_name} (exit={p.returncode})")


def _patch_backtest_config_run_id(config_path: Path, new_run_id: str) -> None:
    """
    Minimal, reversible patch:
    updates run.run_id inside backtest/config.yaml for the duration of a candidate pipeline run.
    """
    cfg = _load_yaml(config_path)
    cfg.setdefault("run", {})
    cfg["run"]["run_id"] = new_run_id
    _dump_yaml(config_path, cfg)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing backtest config: {CONFIG_PATH}")
    if not BASELINE_PATH.exists():
        raise RuntimeError(f"Missing baseline weights file: {BASELINE_PATH}")
    if not SEARCH_PATH.exists():
        raise RuntimeError(f"Missing search file: {SEARCH_PATH}")

    infra_cfg = _load_yaml(CONFIG_PATH)
    db_cfg = infra_cfg.get("db") or {}
    if not db_cfg.get("enabled"):
        raise RuntimeError("DB must be enabled for calibration (db.enabled=true).")

    _ensure_calibration_table(db_cfg) # Erstellt unter anderem DB Tabelle. Kann unter anderem deaktiviert werden nachdem Tabelle existiert

    baseline = _load_yaml(BASELINE_PATH)
    search = _load_yaml(SEARCH_PATH)

    calibration_id = datetime.now().strftime("cal_%Y%m%d_%H%M%S")
    max_candidates = int((infra_cfg.get("calibration") or {}).get("max_candidates", 200))

    candidates = _generate_candidates(
        baseline=baseline,
        search=search,
        calibration_id=calibration_id,
        max_candidates=max_candidates,
    )

    print(f"[CAL] calibration_id={calibration_id} candidates={len(candidates)}")
    print(f"[CAL] baseline={BASELINE_PATH}")
    print(f"[CAL] search={SEARCH_PATH}")
    print(f"[CAL] generated_dir={GENERATED_DIR}")

    # Keep original config.yaml so we can restore run_id afterwards
    original_cfg = _load_yaml(CONFIG_PATH)
    original_run_id = str((original_cfg.get("run") or {}).get("run_id") or "run_original")

    # Bucket scheme (needs to match your run_bucket_report config)
    bucket_scheme = ((infra_cfg.get("buckets") or {}).get("scheme") or "quintiles").strip().lower()
    horizons = (infra_cfg.get("returns") or {}).get("horizons_months") or [12]
    horizons = [int(h) for h in horizons]

    # Environment baseline
    base_env = os.environ.copy()

    try:
        for c in candidates:
            print(f"\n[CAL] candidate={c.idx}/{len(candidates)} run_id={c.run_id} weights_file={c.path.name}")

            # 1) Patch run_id (so tables are keyed per candidate)
            _patch_backtest_config_run_id(CONFIG_PATH, c.run_id)

            # 2) Provide weights to the engine via ENV
            env = dict(base_env)
            env[ENV_WEIGHTS_KEY] = str(c.path)

            # 3) Run pipeline
            t0 = time.time()
            _run_module("backtest.run_snapshots", env=env)
            _run_module("backtest.run_returns", env=env)
            _run_module("backtest.run_bucket_report", env=env)
            dt = time.time() - t0
            print(f"[CAL] pipeline OK in {dt:.1f}s")

            # 4) Read bucket extremes & write calibration metrics
            for h in horizons:
                # 1) raw metrics from bucket table (top/bottom)
                metrics_raw = _fetch_bucket_extremes(
                    db_cfg,
                    run_id=c.run_id,
                    horizon_months=h,
                    bucket_scheme=bucket_scheme,
                )

            # 2) fallback if no data (e.g. missing prices for this horizon)
            if metrics_raw is None:
                metrics_raw = {
                    "n_joined": 0,
                    "top_mean_return": None,
                    "bottom_mean_return": None,
                    "spread_mean": None,
                    "top_hit_rate": None,
                    "bottom_hit_rate": None,
                }

                # 3) flatten to metric_name -> metric_value (compatible with PK including 'metric')
                metrics = {
                    "n_joined": float(metrics_raw.get("n_joined", 0)),
                    "top_mean_return": metrics_raw.get("top_mean_return"),
                    "bottom_mean_return": metrics_raw.get("bottom_mean_return"),
                    "spread_mean": metrics_raw.get("spread_mean"),
                    "top_hit_rate": metrics_raw.get("top_hit_rate"),
                    "bottom_hit_rate": metrics_raw.get("bottom_hit_rate"),
                }

        _upsert_calibration_result(
            db_cfg,
            calibration_id=calibration_id,
            run_id=c.run_id,
            horizon_months=h,
            config_version=c.config_version,
            metrics=metrics,
        )

        print(
            f"[CAL] h={h} n={metrics.get('n_joined')} "
            f"spread={metrics.get('spread_mean')} "
            f"top_hit={metrics.get('top_hit_rate')}"
        )


    finally:
        # Restore original run_id
        _patch_backtest_config_run_id(CONFIG_PATH, original_run_id)

    print(f"\nDONE 2.1.5: calibration_id={calibration_id} (results in bt_calibration_result)")


if __name__ == "__main__":
    main()
