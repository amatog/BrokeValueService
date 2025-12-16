from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# ----------------------------
# Paths / Defaults
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

DEFAULT_HORIZON = int(os.getenv("FEATURE_EXPORT_HORIZON", "12"))
DEFAULT_MODEL_TAG = os.getenv("BASELINE_TAG", "baseline_v1")

DEFAULT_TARGET_CLASS = os.getenv("BASELINE_TARGET_CLASS", "label_up")
DEFAULT_TARGET_REG = os.getenv("BASELINE_TARGET_REG", "label_return")


# ----------------------------
# Config / Spec
# ----------------------------
@dataclass
class TrainSpec:
    input_path: Path
    model_tag: str
    run_id: Optional[str]
    horizon_months: int
    target_class: str
    target_reg: str
    do_classification: bool
    do_regression: bool
    test_size: float
    seed: int
    prefer_time_split: bool


def _env_bool(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y")


def build_spec() -> TrainSpec:
    """
    Inputs via ENV:
      - BASELINE_INPUT: explicit path to parquet/csv
      - RUN_ID: used to auto-pick data/features_<run_id>_h12*_v1.parquet
      - FEATURE_EXPORT_HORIZON: horizon months (default 12)
      - BASELINE_TAG: output tag for model/report files
      - BASELINE_DO_CLASS: 1/0
      - BASELINE_DO_REG: 1/0
      - BASELINE_TEST_SIZE: e.g. 0.3
      - BASELINE_SEED: e.g. 42
      - BASELINE_TIME_SPLIT: 1/0 (prefer asof_date split)
    """
    horizon = int(os.getenv("FEATURE_EXPORT_HORIZON", str(DEFAULT_HORIZON)))
    run_id = os.getenv("RUN_ID")
    model_tag = os.getenv("BASELINE_TAG", DEFAULT_MODEL_TAG).strip() or DEFAULT_MODEL_TAG

    do_class = _env_bool("BASELINE_DO_CLASS", "1")
    do_reg = _env_bool("BASELINE_DO_REG", "1")

    test_size = float(os.getenv("BASELINE_TEST_SIZE", "0.3"))
    seed = int(os.getenv("BASELINE_SEED", "42"))
    prefer_time_split = _env_bool("BASELINE_TIME_SPLIT", "1")

    explicit = os.getenv("BASELINE_INPUT", "").strip()
    if explicit:
        input_path = Path(explicit)
    else:
        if not run_id:
            # fallback: pick newest matching features_*_h{horizon}*_v1.parquet
            pattern = f"features_*_h{horizon}*_v1.parquet"
            matches = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not matches:
                # allow non-v1
                pattern2 = f"features_*_h{horizon}.parquet"
                matches = sorted(DATA_DIR.glob(pattern2), key=lambda p: p.stat().st_mtime, reverse=True)
            if not matches:
                raise FileNotFoundError(
                    f"No features parquet found in {DATA_DIR}. "
                    f"Looked for features_*_h{horizon}*_v1.parquet or features_*_h{horizon}.parquet. "
                    f"Set BASELINE_INPUT or RUN_ID."
                )
            input_path = matches[0]
        else:
            # prefer v1
            p1 = DATA_DIR / f"features_{run_id}_h{horizon}_v1.parquet"
            p2 = DATA_DIR / f"features_{run_id}_h{horizon}.parquet"
            if p1.exists():
                input_path = p1
            elif p2.exists():
                input_path = p2
            else:
                # fallback: any match for this run_id
                matches = sorted(
                    DATA_DIR.glob(f"features_{run_id}_h{horizon}*.parquet"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not matches:
                    raise FileNotFoundError(
                        f"Missing features parquet for run_id={run_id} horizon={horizon}. "
                        f"Expected {p1} or {p2} or wildcard features_{run_id}_h{horizon}*.parquet"
                    )
                input_path = matches[0]

    return TrainSpec(
        input_path=input_path,
        model_tag=model_tag,
        run_id=run_id,
        horizon_months=horizon,
        target_class=DEFAULT_TARGET_CLASS,
        target_reg=DEFAULT_TARGET_REG,
        do_classification=do_class,
        do_regression=do_reg,
        test_size=test_size,
        seed=seed,
        prefer_time_split=prefer_time_split,
    )


# ----------------------------
# Loading / Feature selection
# ----------------------------
ID_LIKE_COLS = {
    "run_id",
    "asof_date",
    "symbol",
    "created_at",
    "engine_mode",
    "engine_base_url",
    "sector",
    "currency",
}


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # parquet
    return pd.read_parquet(path)


def pick_feature_columns(df: pd.DataFrame, *, target_cols: List[str]) -> List[str]:
    """
    Keep only numeric columns (float/int/bool) except id-like and target columns.
    Also allow numeric-ish object columns converted later via to_numeric coercion.
    """
    cols = []
    for c in df.columns:
        if c in ID_LIKE_COLS:
            continue
        if c in target_cols:
            continue
        # drop obvious text categoricals
        if c.startswith("json.") and (
                c.endswith(".symbol")
                or c.endswith(".sector")
                or c.endswith(".currency")
                or c.endswith(".run_id")
                or c.endswith(".asof")
        ):
            continue
        cols.append(c)
    return cols


def _coerce_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    # Best-effort numeric coercion for object cols
    for c in feature_cols:
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif out[c].dtype == bool:
            out[c] = out[c].astype("int64")
    return out


# ----------------------------
# Splitting
# ----------------------------
def time_split(
        df: pd.DataFrame,
        *,
        test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split using asof_date (ascending).
    """
    if "asof_date" not in df.columns:
        raise ValueError("asof_date missing for time split")

    tmp = df.copy()
    tmp["asof_date"] = pd.to_datetime(tmp["asof_date"], errors="coerce")
    tmp = tmp.sort_values(["asof_date", "symbol"], ascending=[True, True])

    n = len(tmp)
    if n < 5:
        # too small: fall back to simple split
        cut = max(1, int(round(n * (1 - test_size))))
    else:
        cut = int(np.floor(n * (1 - test_size)))
        cut = max(1, min(cut, n - 1))

    train = tmp.iloc[:cut].copy()
    test = tmp.iloc[cut:].copy()
    return train, test


def random_split(
        df: pd.DataFrame,
        *,
        test_size: float,
        seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(tmp)
    cut = int(np.floor(n * (1 - test_size)))
    cut = max(1, min(cut, n - 1))
    return tmp.iloc[:cut].copy(), tmp.iloc[cut:].copy()


# ----------------------------
# Training
# ----------------------------
def make_numeric_pipeline(model) -> Pipeline:
    """
    Simple numeric preprocessing:
      - impute missing with median
      - standardize
      - model
    """
    pre = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    # ColumnTransformer in case we want to extend later
    transformer = ColumnTransformer(
        transformers=[
            ("num", pre, slice(0, 10_000_000)),  # all columns passed as numpy array
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("prep", transformer),
            ("model", model),
        ]
    )


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman correlation (Information Coefficient proxy).
    """
    try:
        s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        if pd.isna(s):
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def train_classification(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        seed: int,
) -> Tuple[Pipeline, Dict[str, Any]]:
    # ---- PATCH: single-class fallback (DummyClassifier) ----
    y_unique = np.unique(y_train[~pd.isna(y_train)])
    if len(y_unique) < 2:
        const_class = int(y_unique[0]) if len(y_unique) == 1 else 0
        model = DummyClassifier(strategy="constant", constant=const_class)
        pipe = make_numeric_pipeline(model)
        pipe.fit(X_train, y_train)

        # Evaluate (will be somewhat meaningless but stable)
        pred = pipe.predict(X_test)
        metrics: Dict[str, Any] = {
            "warning": "single_class_training_data",
            "train_unique_classes": [int(x) for x in y_unique.tolist()] if len(y_unique) else [],
            "train_rows": int(len(y_train)),
            "accuracy": float(accuracy_score(y_test, pred)) if len(y_test) else float("nan"),
            "f1": float(f1_score(y_test, pred, zero_division=0)) if len(y_test) else float("nan"),
        }
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["auc"] = float("nan")

        return pipe, metrics
    # ---- END PATCH ----

    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        n_jobs=None,
        random_state=seed,
    )
    pipe = make_numeric_pipeline(model)
    pipe.fit(X_train, y_train)

    # Predict proba if available
    proba = None
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None

    pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    if proba is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["auc"] = float("nan")

    return pipe, metrics


def train_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> Tuple[Pipeline, Dict[str, Any]]:
    model = Ridge(alpha=1.0, random_state=42)
    pipe = make_numeric_pipeline(model)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    metrics: Dict[str, Any] = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "spearman_ic": float(spearman_ic(y_test, pred)),
    }
    return pipe, metrics


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    spec = build_spec()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features(spec.input_path)

    # Basic sanity: require label columns
    missing_labels = []
    if spec.do_classification and spec.target_class not in df.columns:
        missing_labels.append(spec.target_class)
    if spec.do_regression and spec.target_reg not in df.columns:
        missing_labels.append(spec.target_reg)
    if missing_labels:
        raise RuntimeError(f"Missing label columns in dataset: {missing_labels}")

    # Drop rows with missing labels
    if spec.do_classification:
        df = df[df[spec.target_class].notna()]
    if spec.do_regression:
        df = df[df[spec.target_reg].notna()]

    if len(df) < 2:
        raise RuntimeError("Not enough rows after label filtering to train a model.")

    # Prefer time split if possible
    if spec.prefer_time_split and "asof_date" in df.columns and df["asof_date"].notna().any():
        train_df, test_df = time_split(df, test_size=spec.test_size)
        split_mode = "time(asof_date)"
    else:
        train_df, test_df = random_split(df, test_size=spec.test_size, seed=spec.seed)
        split_mode = "random"

    # Feature selection
    targets = []
    if spec.do_classification:
        targets.append(spec.target_class)
    if spec.do_regression:
        targets.append(spec.target_reg)

    feature_cols = pick_feature_columns(df, target_cols=targets)
    if not feature_cols:
        raise RuntimeError("No feature columns selected. Check your features export / schema.")

    # Coerce to numeric where possible (object -> numeric)
    train_df = _coerce_features(train_df, feature_cols)
    test_df = _coerce_features(test_df, feature_cols)

    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    report: Dict[str, Any] = {
        "input_path": str(spec.input_path),
        "model_tag": spec.model_tag,
        "run_id": spec.run_id,
        "horizon_months": spec.horizon_months,
        "n_rows_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "split_mode": split_mode,
        "feature_count": int(len(feature_cols)),
        "targets": targets,
    }

    # Train + save
    out: Dict[str, Any] = {}

    if spec.do_classification:
        y_train = train_df[spec.target_class].astype("int64").to_numpy()
        y_test = test_df[spec.target_class].astype("int64").to_numpy()

        clf, m = train_classification(X_train, y_train, X_test, y_test, seed=spec.seed)
        out["classification"] = {"metrics": m}

        clf_path = MODELS_DIR / f"{spec.model_tag}_clf.joblib"
        joblib.dump(clf, clf_path)
        out["classification"]["model_path"] = str(clf_path)

    if spec.do_regression:
        y_train = pd.to_numeric(train_df[spec.target_reg], errors="coerce").to_numpy()
        y_test = pd.to_numeric(test_df[spec.target_reg], errors="coerce").to_numpy()

        reg, m = train_regression(X_train, y_train, X_test, y_test)
        out["regression"] = {"metrics": m}

        reg_path = MODELS_DIR / f"{spec.model_tag}_reg.joblib"
        joblib.dump(reg, reg_path)
        out["regression"]["model_path"] = str(reg_path)

    # Save feature list (critical for inference consistency)
    feat_path = MODELS_DIR / f"{spec.model_tag}_features.json"
    feat_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    report["features_path"] = str(feat_path)

    report.update(out)

    report_path = MODELS_DIR / f"{spec.model_tag}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print("[OK] Baseline training complete")
    print(f"[OK] input: {spec.input_path}")
    print(f"[OK] split: {split_mode} train={len(train_df)} test={len(test_df)}")
    print(f"[OK] features: {len(feature_cols)}")
    if "classification" in out:
        print(f"[OK] clf metrics: {out['classification']['metrics']}")
        print(f"[OK] clf model: {out['classification']['model_path']}")
    if "regression" in out:
        print(f"[OK] reg metrics: {out['regression']['metrics']}")
        print(f"[OK] reg model: {out['regression']['model_path']}")
    print(f"[OK] report: {report_path}")
    print(f"[OK] features: {feat_path}")


if __name__ == "__main__":
    main()
