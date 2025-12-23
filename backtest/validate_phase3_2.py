from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


DEFAULT_MODELS_ROOT = r"C:\projects\ValueServices\models"


@dataclass
class CheckResult:
    ok: bool
    message: str


def _normalize_path(p: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(p))
    return Path(expanded).resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hard-fail DoD validator for Phase 3.2")
    ap.add_argument("--run-dir", default="", help="Exact run directory path. If empty, uses --models-root + --run-id.")
    ap.add_argument("--models-root", default=DEFAULT_MODELS_ROOT, help="Models root directory.")
    ap.add_argument("--run-id", default="", help="Run id folder name under models-root.")
    ap.add_argument("--max-valid-rmse", type=float, default=float("inf"), help="Optional quality gate (RMSE).")
    return ap.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_exists(path: Path) -> CheckResult:
    return CheckResult(path.exists(), f"{'OK' if path.exists() else 'Missing'}: {path}")


def check_metrics(metrics_path: Path, max_valid_rmse: float) -> CheckResult:
    if not metrics_path.exists():
        return CheckResult(False, f"Missing metrics.json: {metrics_path}")

    m = _load_json(metrics_path)

    # Required keys
    required = ["valid_rmse", "valid_mae", "train_rows", "valid_rows", "n_features"]
    missing = [k for k in required if k not in m]
    if missing:
        return CheckResult(False, f"metrics.json missing keys: {missing}")

    # Validate numeric
    for k in ["valid_rmse", "valid_mae"]:
        v = m.get(k)
        if v is None or not np.isfinite(v):
            return CheckResult(False, f"Invalid metric {k}={v}")

    if int(m.get("train_rows", 0)) <= 0 or int(m.get("valid_rows", 0)) <= 0:
        return CheckResult(False, f"Invalid row counts train={m.get('train_rows')} valid={m.get('valid_rows')}")

    if int(m.get("n_features", 0)) <= 0:
        return CheckResult(False, f"Invalid n_features={m.get('n_features')}")

    if float(m["valid_rmse"]) > max_valid_rmse:
        return CheckResult(False, f"Quality gate failed: valid_rmse={m['valid_rmse']} > {max_valid_rmse}")

    return CheckResult(True, f"OK: metrics valid (valid_rmse={m['valid_rmse']}, valid_mae={m['valid_mae']})")


def check_model_loadable(model_path: Path) -> CheckResult:
    if not model_path.exists():
        return CheckResult(False, f"Missing model file: {model_path}")
    try:
        _ = joblib.load(model_path)
    except Exception as e:
        return CheckResult(False, f"Model not loadable: {e}")
    return CheckResult(True, f"OK: model loadable {model_path}")


def main() -> None:
    args = parse_args()

    try:
        if args.run_dir.strip():
            run_dir = _normalize_path(args.run_dir)
        else:
            if not args.run_id.strip():
                raise RuntimeError("Provide either --run-dir or --run-id.")
            run_dir = _normalize_path(args.models_root) / args.run_id.strip()

        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        features_path = run_dir / "features.json"
        sig_path = run_dir / "data_signature.json"

        results = [
            check_exists(run_dir),
            check_exists(model_path),
            check_exists(metrics_path),
            check_exists(config_path),
            check_exists(features_path),
            check_exists(sig_path),
        ]

        # Only run deeper checks if files exist
        if model_path.exists():
            results.append(check_model_loadable(model_path))
        if metrics_path.exists():
            results.append(check_metrics(metrics_path, args.max_valid_rmse))

        ok = all(r.ok for r in results)
        for r in results:
            status = "PASS" if r.ok else "FAIL"
            print(f"[PHASE3.2][{status}] {r.message}")

        if ok:
            print("[PHASE3.2] DONE")
            sys.exit(0)
        else:
            print("[PHASE3.2] NOT DONE")
            sys.exit(2)

    except Exception as e:
        print(f"[PHASE3.2][ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
