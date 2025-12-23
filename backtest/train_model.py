# backtest/train_model.py
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Local modules (must be in same package: backtest/)
from .split import resolve_split_from_primary_root, save_split_json
from .evaluate import evaluate_predictions, save_eval_artifacts


# -----------------------------
# Config / Defaults
# -----------------------------

DEFAULT_LABEL_COL = "label_return"

DEFAULT_SYMBOL_COL_CANDIDATES = [
    "json.features_json.symbol",
    "symbol",
]

DEFAULT_DATE_COL_CANDIDATES = [
    "json.features_json.asof",
    "asof",
    "date",
]

# If your engineered feature set is not present in parquet yet, we fall back to OHLCV baseline features:
OHLCV_FALLBACK_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]

ARTIFACT_FILENAMES = {
    "split": "split.json",
    "model": "model.pkl",
    "feature_importance": "feature_importance.json",
    "training_metadata": "training_metadata.json",
    "predictions_test": "predictions_test.csv",
    "evaluation_report": "evaluation_report.json",
    "evaluation_table": "evaluation_table.csv",
}


@dataclass(frozen=True)
class RunPaths:
    artifacts_dir: Path

    @property
    def split_json(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["split"]

    @property
    def model_pkl(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["model"]

    @property
    def feature_importance_json(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["feature_importance"]

    @property
    def training_metadata_json(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["training_metadata"]

    @property
    def predictions_test_csv(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["predictions_test"]

    @property
    def evaluation_report_json(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["evaluation_report"]

    @property
    def evaluation_table_csv(self) -> Path:
        return self.artifacts_dir / ARTIFACT_FILENAMES["evaluation_table"]


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_feature_cols_from_json(path: str | Path) -> List[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both {"feature_cols":[...]} and direct list [...]
    if isinstance(data, dict) and "feature_cols" in data:
        cols = data["feature_cols"]
    else:
        cols = data

    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError(f"Invalid feature columns format in {p}. Expected list of strings.")
    return cols


def _detect_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _partition_paths_for_dates(primary_root: Path, dates: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for d in dates:
        part_dir = primary_root / f"date={d}"
        if not part_dir.exists():
            raise FileNotFoundError(f"Missing partition directory: {part_dir}")
        for f in part_dir.glob("*.parquet"):
            files.append(f)
    return files


def _load_parquet_files(files: Sequence[Path], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Loads parquet files into a single DataFrame.

    Important:
    - PyArrow raises when 'columns' contains fields not present in the parquet schema.
    - We therefore fallback to reading full file and then selecting intersection.
    """
    if not files:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    wanted = list(columns) if columns else None

    for f in files:
        if wanted is None:
            df = pd.read_parquet(f)
            dfs.append(df)
            continue

        try:
            # Fast path: read only requested columns (works only if all exist)
            df = pd.read_parquet(f, columns=wanted)
            dfs.append(df)
        except Exception:
            # Safe path: read full file, then keep only intersection
            df_full = pd.read_parquet(f)
            existing = [c for c in wanted if c in df_full.columns]
            if not existing:
                # If none of the requested columns exist, keep full to allow later fallback logic
                dfs.append(df_full)
            else:
                dfs.append(df_full[existing])

    return pd.concat(dfs, ignore_index=True)



def _ensure_label_return(
        df: pd.DataFrame,
        label_col: str,
        symbol_col: str,
        date_col: str,
        price_col_candidates: Sequence[str] = ("adjusted_close", "close"),
        horizon_days: int = 1,
) -> pd.DataFrame:
    """
    Ensure df has label_col. If missing, compute forward return per symbol:
      label[t] = price[t+h] / price[t] - 1
    Requires symbol_col + date_col + a price column.
    """
    if label_col in df.columns:
        return df

    price_col = None
    for c in price_col_candidates:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise KeyError(
            f"Cannot compute {label_col}: none of price columns found: {list(price_col_candidates)}"
        )
    if symbol_col not in df.columns:
        raise KeyError(f"Cannot compute {label_col}: missing symbol column '{symbol_col}'")
    if date_col not in df.columns:
        raise KeyError(f"Cannot compute {label_col}: missing date column '{date_col}'")

    dd = df.copy()
    dd[date_col] = pd.to_datetime(dd[date_col], errors="coerce")
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna(subset=[symbol_col, date_col, price_col])
    dd = dd.sort_values([symbol_col, date_col], ascending=True)

    dd[label_col] = (
            dd.groupby(symbol_col, sort=False)[price_col].shift(-horizon_days) / dd[price_col] - 1.0
    )
    return dd


def _select_training_columns(
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        symbol_col: Optional[str],
        date_col: Optional[str],
) -> pd.DataFrame:
    keep: List[str] = []
    for c in feature_cols:
        if c in df.columns:
            keep.append(c)

    if label_col in df.columns:
        keep.append(label_col)

    if symbol_col and symbol_col in df.columns:
        keep.append(symbol_col)
    if date_col and date_col in df.columns:
        keep.append(date_col)

    # Deduplicate while preserving order
    seen = set()
    cols: List[str] = []
    for c in keep:
        if c not in seen:
            cols.append(c)
            seen.add(c)

    return df[cols].copy()


def _coerce_numeric_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    effective: List[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            effective.append(c)
        else:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().any():
                df[c] = converted
                effective.append(c)
    return df, effective


def _write_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _feature_importance_from_model(model: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    reg = model.named_steps.get("regressor")
    if reg is None:
        return {}
    if hasattr(reg, "feature_importances_"):
        arr = getattr(reg, "feature_importances_")
        if arr is None:
            return {}
        return {feature_cols[i]: float(arr[i]) for i in range(min(len(feature_cols), len(arr)))}
    return {}


def _fallback_features_if_needed(df: pd.DataFrame, requested_features: List[str], label_col: str) -> List[str]:
    """
    If requested engineered features are not present yet, fall back to OHLCV.
    """
    present = [c for c in requested_features if c in df.columns]
    if present:
        return present

    fallback = [c for c in OHLCV_FALLBACK_FEATURES if c in df.columns and c != label_col]
    return fallback


# -----------------------------
# Main training pipeline
# -----------------------------

def train_and_evaluate(
        primary_root: Path,
        artifacts_dir: Path,
        feature_cols: List[str],
        label_col: str = DEFAULT_LABEL_COL,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: Optional[int] = 6,
        min_train_days: int = 252,
        min_val_days: int = 63,
        min_test_days: int = 63,
) -> None:
    rp = RunPaths(artifacts_dir=artifacts_dir)
    _ensure_dir(rp.artifacts_dir)

    t0 = time.time()

    # 1) Split
    split = resolve_split_from_primary_root(
        primary_root=primary_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_train_days=min_train_days,
        min_val_days=min_val_days,
        min_test_days=min_test_days,
    )
    save_split_json(split, rp.split_json)

    # 2) Load parquet split slices
    # IMPORTANT:
    # - label_return may not exist in parquet yet -> we compute it after load
    # - ensure we always load symbol/date + price columns for label computation
    meta_candidates = list(set(DEFAULT_SYMBOL_COL_CANDIDATES + DEFAULT_DATE_COL_CANDIDATES))
    mandatory_for_label = ["adjusted_close", "close"]  # if one exists it's enough
    read_cols = list(dict.fromkeys([*feature_cols, *meta_candidates, *mandatory_for_label]))

    train_files = _partition_paths_for_dates(primary_root, split.train_dates)
    val_files = _partition_paths_for_dates(primary_root, split.val_dates)
    test_files = _partition_paths_for_dates(primary_root, split.test_dates)

    if not train_files or not val_files or not test_files:
        raise RuntimeError(
            f"Unexpected empty split file list: train={len(train_files)} val={len(val_files)} test={len(test_files)}"
        )

    df_train = _load_parquet_files(train_files, columns=read_cols)
    df_val = _load_parquet_files(val_files, columns=read_cols)
    df_test = _load_parquet_files(test_files, columns=read_cols)

    if df_train.empty or df_val.empty or df_test.empty:
        raise RuntimeError(
            f"Empty split dataframe: train={df_train.shape} val={df_val.shape} test={df_test.shape}"
        )

    # Detect meta columns
    symbol_col = _detect_first_existing_col(df_train, DEFAULT_SYMBOL_COL_CANDIDATES)
    date_col = _detect_first_existing_col(df_train, DEFAULT_DATE_COL_CANDIDATES)

    if symbol_col is None:
        raise RuntimeError(f"Cannot proceed: no symbol column found in {DEFAULT_SYMBOL_COL_CANDIDATES}")
    if date_col is None:
        raise RuntimeError(f"Cannot proceed: no date column found in {DEFAULT_DATE_COL_CANDIDATES}")

    # 3) Ensure label exists (compute forward return if missing)
    df_train = _ensure_label_return(df_train, label_col, symbol_col, date_col)
    df_val = _ensure_label_return(df_val, label_col, symbol_col, date_col)
    df_test = _ensure_label_return(df_test, label_col, symbol_col, date_col)

    # 4) Feature selection (fallback to OHLCV if engineered features are not present yet)
    requested_or_fallback = _fallback_features_if_needed(df_train, feature_cols, label_col)
    if not requested_or_fallback:
        raise RuntimeError(
            "No usable feature columns found. Neither requested features nor OHLCV fallback features exist."
        )

    # Narrow to required columns
    df_train = _select_training_columns(df_train, requested_or_fallback, label_col, symbol_col, date_col)
    df_val = _select_training_columns(df_val, requested_or_fallback, label_col, symbol_col, date_col)
    df_test = _select_training_columns(df_test, requested_or_fallback, label_col, symbol_col, date_col)

    # Coerce numeric features
    df_train, effective_features = _coerce_numeric_features(df_train, requested_or_fallback)
    df_val, _ = _coerce_numeric_features(df_val, requested_or_fallback)
    df_test, _ = _coerce_numeric_features(df_test, requested_or_fallback)

    if not effective_features:
        # last-resort: auto-pick numeric columns (excluding label/meta)
        candidates = [c for c in OHLCV_FALLBACK_FEATURES if c in df_train.columns and c != label_col]
        df_train, effective_features = _coerce_numeric_features(df_train, candidates)
        df_val, _ = _coerce_numeric_features(df_val, candidates)
        df_test, _ = _coerce_numeric_features(df_test, candidates)

    if not effective_features:
        raise RuntimeError("No usable numeric feature columns found after coercion and fallback.")

    # Drop rows with missing label (includes last row per symbol due to forward shift)
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col])
    df_val = df_val.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col])
    df_test = df_test.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col])

    X_train = df_train[effective_features]
    y_train = df_train[label_col].astype(float).to_numpy()

    X_val = df_val[effective_features]
    y_val = df_val[label_col].astype(float).to_numpy()

    X_test = df_test[effective_features]
    y_test = df_test[label_col].astype(float).to_numpy()

    # 5) Model pipeline (impute -> regressor)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    random_state=random_state,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred_val = model.predict(X_val)  # kept for future use; not persisted here
    y_pred_test = model.predict(X_test)

    # Persist model
    with rp.model_pkl.open("wb") as f:
        pickle.dump(model, f)

    # Feature importance artifact (may be empty for this estimator; still persisted)
    fi = _feature_importance_from_model(model, effective_features)
    _write_json(rp.feature_importance_json, {"feature_importance": fi, "feature_cols": effective_features})

    # Predictions CSV for test
    pred_out = pd.DataFrame(
        {
            label_col: y_test,
            "y_pred": y_pred_test,
        }
    )
    if symbol_col in df_test.columns:
        pred_out[symbol_col] = df_test[symbol_col].values
    if date_col in df_test.columns:
        pred_out[date_col] = df_test[date_col].values

    pred_out.to_csv(rp.predictions_test_csv, index=False)

    # Evaluation on test split
    eval_res = evaluate_predictions(pred_out, y_true_col=label_col, y_pred_col="y_pred")
    save_eval_artifacts(
        eval_res,
        out_json_path=rp.evaluation_report_json,
        out_csv_path=rp.evaluation_table_csv,
    )

    # Training metadata
    elapsed_s = float(time.time() - t0)
    meta = {
        "primary_root": str(primary_root),
        "artifacts_dir": str(artifacts_dir),
        "label_col": label_col,
        "symbol_col": symbol_col,
        "date_col": date_col,
        "feature_cols_requested": feature_cols,
        "feature_cols_used": effective_features,
        "split": {
            "train_days": len(split.train_dates),
            "val_days": len(split.val_dates),
            "test_days": len(split.test_dates),
            "train_min": split.train_min,
            "train_max": split.train_max,
            "val_min": split.val_min,
            "val_max": split.val_max,
            "test_min": split.test_min,
            "test_max": split.test_max,
        },
        "rows": {
            "train": int(len(df_train)),
            "val": int(len(df_val)),
            "test": int(len(df_test)),
        },
        "model": {
            "type": "HistGradientBoostingRegressor",
            "params": {
                "random_state": random_state,
                "max_iter": max_iter,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "early_stopping": True,
            },
        },
        "evaluation_test": eval_res.to_dict(),
        "elapsed_seconds": elapsed_s,
        "created_at_epoch_ms": int(time.time() * 1000),
    }
    _write_json(rp.training_metadata_json, meta)

    # Console summary
    print("[TRAIN] artifacts_dir:", rp.artifacts_dir)
    print("[TRAIN] split.json:", rp.split_json)
    print("[TRAIN] model.pkl:", rp.model_pkl)
    print("[TRAIN] predictions_test.csv:", rp.predictions_test_csv)
    print("[TRAIN] evaluation_report.json:", rp.evaluation_report_json)
    print("[TRAIN] evaluation_table.csv:", rp.evaluation_table_csv)
    print("[TRAIN] features_used:", effective_features)
    print("[EVAL][TEST] n_rows:", eval_res.n_rows)
    print("[EVAL][TEST] rmse:", eval_res.rmse)
    print("[EVAL][TEST] mae :", eval_res.mae)
    print("[EVAL][TEST] r2  :", eval_res.r2)
    print("[EVAL][TEST] spearman_ic:", eval_res.spearman_ic)
    print("[EVAL][TEST] directional_accuracy:", eval_res.directional_accuracy)
    print("[EVAL][TEST] quintile_spread:", eval_res.quintile_spread)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase 3.3: Train baseline model + evaluate (global time split).")
    ap.add_argument("--primaryRoot", required=True, help="A.2 Primary Store root (contains date=YYYY-MM-DD folders)")
    ap.add_argument("--artifactsDir", default="backtest/artifacts", help="Output directory for artifacts")
    ap.add_argument(
        "--featureColsJson",
        required=True,
        help="Path to JSON containing feature columns (either {'feature_cols':[...]} or a list).",
    )
    ap.add_argument("--labelCol", default=DEFAULT_LABEL_COL, help=f"Label column name (default: {DEFAULT_LABEL_COL})")
    ap.add_argument("--trainRatio", type=float, default=0.70)
    ap.add_argument("--valRatio", type=float, default=0.15)
    ap.add_argument("--testRatio", type=float, default=0.15)
    ap.add_argument("--minTrainDays", type=int, default=252)
    ap.add_argument("--minValDays", type=int, default=63)
    ap.add_argument("--minTestDays", type=int, default=63)

    # Baseline model knobs
    ap.add_argument("--randomState", type=int, default=42)
    ap.add_argument("--maxIter", type=int, default=300)
    ap.add_argument("--learningRate", type=float, default=0.05)
    ap.add_argument("--maxDepth", type=int, default=6)

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    primary_root = Path(args.primaryRoot)
    artifacts_dir = Path(args.artifactsDir)

    feature_cols = _read_feature_cols_from_json(args.featureColsJson)

    train_and_evaluate(
        primary_root=primary_root,
        artifacts_dir=artifacts_dir,
        feature_cols=feature_cols,
        label_col=args.labelCol,
        train_ratio=args.trainRatio,
        val_ratio=args.valRatio,
        test_ratio=args.testRatio,
        min_train_days=args.minTrainDays,
        min_val_days=args.minValDays,
        min_test_days=args.minTestDays,
        random_state=args.randomState,
        max_iter=args.maxIter,
        learning_rate=args.learningRate,
        max_depth=args.maxDepth,
    )


if __name__ == "__main__":
    # IMPORTANT: run as module for reliable relative imports:
    #   python -m backtest.train_model --primaryRoot ... --featureColsJson ...
    main()
