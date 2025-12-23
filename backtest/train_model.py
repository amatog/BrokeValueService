# backtest/train_model.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PHASE3_1_ROOT = r"C:\projects\ValueServices\data_ml\phase_3_1"
DEFAULT_MODELS_ROOT = r"C:\projects\ValueServices\models"


@dataclass(frozen=True)
class TrainPaths:
    in_root: Path
    models_root: Path
    run_dir: Path
    model_path: Path
    metrics_path: Path
    config_path: Path
    features_path: Path
    data_signature_path: Path


def _normalize_path(p: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(p))
    return Path(expanded).resolve()


def _utc_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}"


def _load_parquet_df(path: Path) -> pd.DataFrame:
    t = pq.read_table(str(path))
    return t.to_pandas()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _data_signature(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> Dict:
    sig = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "date_min": str(df["date"].min()) if "date" in df.columns and len(df) else None,
        "date_max": str(df["date"].max()) if "date" in df.columns and len(df) else None,
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns and len(df) else None,
        "label_non_null": int(df[label_col].notna().sum()) if label_col in df.columns else None,
    }
    return sig


# -----------------------------
# Feature rules (Allowlist/Blocklist)
# -----------------------------

def _build_block_cols(train_df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """
    Blocklist rules:
      - Always block identifiers: date, symbol
      - Block anything starting with label_  (label_return, label_up, ...)
      - Block engine metadata: engine_* and _engine_*
      - Block json.features_json.* entirely (feature export meta)
      - Block json.raw_json.* meta fields (checks/method/rating/thresholds_used/components/aggregation/etc.)
        but DO NOT block json.raw_json.*.metrics.* (we want to allow metrics)
    """
    cols = list(train_df.columns)

    blocked: List[str] = []
    blocked_reasons: Dict[str, int] = {
        "id_cols": 0,
        "label_prefix": 0,
        "engine_meta": 0,
        "features_json_meta": 0,
        "raw_json_meta": 0,
    }

    # Always blocked identifiers
    for c in ("date", "symbol"):
        if c in train_df.columns:
            blocked.append(c)
            blocked_reasons["id_cols"] += 1

    for c in cols:
        if c in ("date", "symbol"):
            continue

        # label_* (hard block)
        if c.startswith("label_"):
            blocked.append(c)
            blocked_reasons["label_prefix"] += 1
            continue

        # engine_* or _engine_* (hard block)
        if c.startswith("engine_") or c.startswith("_engine_") or "_engine_" in c:
            blocked.append(c)
            blocked_reasons["engine_meta"] += 1
            continue

        # json.features_json.* (hard block)
        if c.startswith("json.features_json."):
            blocked.append(c)
            blocked_reasons["features_json_meta"] += 1
            continue

        # json.raw_json.* meta fields (block), but allow metrics.* explicitly
        if c.startswith("json.raw_json."):
            # allow metrics.* (we will keep only numeric anyway)
            if ".metrics." in c:
                continue

            # block common meta payloads
            meta_markers = (
                ".checks",
                ".method",
                ".rating",
                ".thresholds_used",
                ".missing_data",
                ".components",
                ".aggregation",
                "._engine_",
                "._engine",
            )
            if any(m in c for m in meta_markers):
                blocked.append(c)
                blocked_reasons["raw_json_meta"] += 1
                continue

            # If it's raw_json but not metrics.* and not in the meta markers,
            # keep it for now; numeric-only filtering later will prune strings/objects.

    # De-dup while preserving order
    seen = set()
    blocked_unique = []
    for c in blocked:
        if c not in seen:
            blocked_unique.append(c)
            seen.add(c)

    return blocked_unique, blocked_reasons


def _select_features(
        df: pd.DataFrame,
        label_col: str,
        block_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Select numeric feature columns excluding blocked columns and excluding label.
    """
    if label_col not in df.columns:
        raise RuntimeError(f"Missing label column: {label_col}")

    cols = [c for c in df.columns if c not in ([label_col] + block_cols)]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].copy()
    y = df[label_col].copy()
    return X, y, numeric_cols


def _drop_all_nan_features(
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], int]:
    """
    Drop columns that are entirely missing in train; keep valid aligned.
    """
    all_nan_cols = [c for c in X_train.columns if X_train[c].notna().sum() == 0]
    if all_nan_cols:
        X_train = X_train.drop(columns=all_nan_cols)
        X_valid = X_valid.drop(columns=all_nan_cols, errors="ignore")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    return X_train, X_valid, feature_cols, len(all_nan_cols)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 3.2 - Train baseline model on Phase 3.1 dataset")
    ap.add_argument("--in-root", default=DEFAULT_PHASE3_1_ROOT, help="Phase 3.1 output root (train/valid parquet).")
    ap.add_argument("--models-root", default=DEFAULT_MODELS_ROOT, help="Root directory for model artifacts.")
    ap.add_argument("--run-id", default="", help="Optional run id; if empty, auto-generated.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--label-col", default="target_return", help="Label column name.")
    ap.add_argument("--model", choices=["ridge"], default="ridge", help="Baseline model family.")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength.")
    return ap.parse_args()


def build_paths(in_root: str, models_root: str, run_id: str) -> TrainPaths:
    in_root_p = _normalize_path(in_root)
    models_root_p = _normalize_path(models_root)
    rid = run_id.strip() or _utc_run_id("run")

    run_dir = models_root_p / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    return TrainPaths(
        in_root=in_root_p,
        models_root=models_root_p,
        run_dir=run_dir,
        model_path=run_dir / "model.joblib",
        metrics_path=run_dir / "metrics.json",
        config_path=run_dir / "config.json",
        features_path=run_dir / "features.json",
        data_signature_path=run_dir / "data_signature.json",
    )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    paths = build_paths(args.in_root, args.models_root, args.run_id)

    train_path = paths.in_root / "train.parquet"
    valid_path = paths.in_root / "valid.parquet"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Missing Phase 3.1 train/valid. train={train_path} valid={valid_path}")

    train_df = _load_parquet_df(train_path)
    valid_df = _load_parquet_df(valid_path)

    # Hard gate: label must exist and be usable
    if args.label_col not in train_df.columns:
        raise RuntimeError(f"Train missing label col: {args.label_col}")
    if args.label_col not in valid_df.columns:
        raise RuntimeError(f"Valid missing label col: {args.label_col}")

    # Use only non-null labels
    train_df = train_df[train_df[args.label_col].notna()].copy()
    valid_df = valid_df[valid_df[args.label_col].notna()].copy()

    if train_df.empty or valid_df.empty:
        raise RuntimeError(
            f"Empty train/valid after label filtering. train_rows={len(train_df)} valid_rows={len(valid_df)}"
        )

    # Build blocklist with allowlist behavior for json.raw_json.*.metrics.*
    block_cols, block_stats = _build_block_cols(train_df)
    print(f"[TRAIN] blocked columns: {len(block_cols)} {block_stats}")

    # Feature selection (numeric-only) with blocklist applied
    X_train, y_train, feature_cols = _select_features(train_df, label_col=args.label_col, block_cols=block_cols)
    X_valid, y_valid, _ = _select_features(valid_df, label_col=args.label_col, block_cols=block_cols)

    # Align valid columns to train columns
    X_valid = X_valid.reindex(columns=feature_cols)

    # Drop all-NaN features (train-side)
    X_train, X_valid, feature_cols, dropped_all_nan = _drop_all_nan_features(X_train, X_valid, feature_cols)
    if dropped_all_nan:
        print(f"[TRAIN] dropping all-NaN features: {dropped_all_nan}")

    if not feature_cols:
        raise RuntimeError("No usable numeric feature columns after blocklist + numeric filter.")

    # Baseline pipeline: impute -> scale -> ridge
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

    if args.model == "ridge":
        model = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    else:
        raise RuntimeError(f"Unsupported model: {args.model}")

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    print(f"[TRAIN] in_root={paths.in_root}")
    print(f"[TRAIN] run_dir={paths.run_dir}")
    print(f"[TRAIN] rows train={len(train_df)} valid={len(valid_df)} features={len(feature_cols)} label={args.label_col}")

    pipe.fit(X_train, y_train)

    pred_valid = pipe.predict(X_valid)
    pred_train = pipe.predict(X_train)

    metrics = {
        "model": args.model,
        "seed": args.seed,
        "label_col": args.label_col,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "n_features": int(len(feature_cols)),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "valid_mae": float(mean_absolute_error(y_valid, pred_valid)),
        "train_rmse": _rmse(y_train.to_numpy(), pred_train),
        "valid_rmse": _rmse(y_valid.to_numpy(), pred_valid),
    }

    # Persist artifacts
    joblib.dump(pipe, paths.model_path)

    with open(paths.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cfg = {
        "run_id": paths.run_dir.name,
        "in_root": str(paths.in_root),
        "models_root": str(paths.models_root),
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": args.seed,
        "model": args.model,
        "ridge_alpha": args.ridge_alpha,
        "label_col": args.label_col,
        "blocked_cols_count": len(block_cols),
        "blocked_cols_stats": block_stats,
        "allow_rule": "json.raw_json.*.metrics.* allowed; json.features_json.* blocked; json.raw_json meta blocked",
    }
    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with open(paths.features_path, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    sig = {
        "train": _data_signature(train_df, feature_cols, args.label_col),
        "valid": _data_signature(valid_df, feature_cols, args.label_col),
    }
    with open(paths.data_signature_path, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)

    print(f"[TRAIN] wrote model={paths.model_path}")
    print(f"[TRAIN] wrote metrics={paths.metrics_path}")
    print(f"[TRAIN] valid_rmse={metrics['valid_rmse']:.6f} valid_mae={metrics['valid_mae']:.6f}")
    print("[TRAIN] completed")


if __name__ == "__main__":
    main()
