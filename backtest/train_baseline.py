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
    for c in feature_cols:
        if c not in out.columns:
            continue
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif out[c].dtype == bool:
            out[c] = out[c].astype("int64")
    return out


def _drop_useless_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        *,
        min_non_null: int = 1,
        drop_constant_if_n_ge: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Drop features that are:
      - all NaN in train
      - have < min_non_null observed values in train
      - (optionally) constant in train, but ONLY if dataset is large enough
    """
    info: Dict[str, Any] = {"dropped_all_nan": [], "dropped_too_sparse": [], "dropped_constant": []}

    n_train = int(len(train_df))
    do_drop_constant = n_train >= int(drop_constant_if_n_ge)

    keep: List[str] = []
    for c in feature_cols:
        if c not in train_df.columns:
            info["dropped_all_nan"].append(c)
            continue

        s = train_df[c]
        nn = int(s.notna().sum())
        if nn == 0:
            info["dropped_all_nan"].append(c)
            continue
        if nn < min_non_null:
            info["dropped_too_sparse"].append(c)
            continue

        if do_drop_constant:
            try:
                nun = int(s.dropna().nunique())
            except Exception:
                nun = 0
            if nun <= 1:
                info["dropped_constant"].append(c)
                continue

        keep.append(c)

    return train_df.copy(), test_df.copy(), keep, info


# ----------------------------
# Target fix (binary class)
# ----------------------------
def _ensure_binary_target_for_lr(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        target_class: Optional[str] = None,
        target_class_col: Optional[str] = None,
        target_reg: str,
        label_mode: str = "auto",
        fallback_feature_for_split: str = "value_score",
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """
    Ensures we have a binary target column suitable for LogisticRegression (2 classes).
    Returns (train_df, test_df, target_class_col, info).

    label_mode:
      - "auto": use existing target if it has >=2 classes in train; else build from target_reg (median split);
                else fallback median split on fallback_feature_for_split
      - "existing": keep target as-is (but may still single-class -> will force minimal flip in train if possible)
      - "rebuild": rebuild from target_reg (median split on train)
      - "fallback": build from fallback_feature_for_split (median split on train)
    """
    info: Dict[str, Any] = {"label_mode": (label_mode or "auto").strip().lower()}

    if not target_class:
        target_class = target_class_col or "label_up"

    train_df = train_df.copy()
    test_df = test_df.copy()

    def _counts(s: pd.Series) -> Dict[int, int]:
        s2 = pd.to_numeric(s, errors="coerce").dropna().astype("int64")
        return {int(k): int(v) for k, v in s2.value_counts().to_dict().items()}

    def _n_classes(df_: pd.DataFrame, col: str) -> int:
        if col not in df_.columns:
            return 0
        s = pd.to_numeric(df_[col], errors="coerce").dropna()
        if s.empty:
            return 0
        return int(s.astype("int64").nunique())

    def _median_split_from(col_src: str, col_out: str) -> float:
        tr = pd.to_numeric(train_df[col_src], errors="coerce")
        te = pd.to_numeric(test_df[col_src], errors="coerce")
        med = float(np.nanmedian(tr.to_numpy()))
        train_df[col_out] = (tr > med).astype("int64")
        test_df[col_out] = (te > med).astype("int64")
        return med

    mode = info["label_mode"]
    if mode not in ("auto", "existing", "rebuild", "fallback"):
        mode = "auto"
        info["label_mode"] = "auto"

    used = target_class
    info["used_target"] = used
    info["rebuilt"] = False
    info["fallback_used"] = False

    if mode == "rebuild":
        if target_reg not in train_df.columns or target_reg not in test_df.columns:
            raise RuntimeError(f"Cannot rebuild labels: missing target_reg='{target_reg}'")
        new_col = f"{target_class}__auto"
        med = _median_split_from(target_reg, new_col)
        used = new_col
        info["used_target"] = used
        info["rebuilt"] = True
        info["median"] = med
        info["source"] = target_reg

    elif mode == "fallback":
        if fallback_feature_for_split not in train_df.columns or fallback_feature_for_split not in test_df.columns:
            raise RuntimeError(
                f"Fallback label requested but feature '{fallback_feature_for_split}' not found."
            )
        new_col = f"{target_class}__fallback"
        med = _median_split_from(fallback_feature_for_split, new_col)
        used = new_col
        info["used_target"] = used
        info["fallback_used"] = True
        info["median"] = med
        info["source"] = fallback_feature_for_split

    elif mode == "auto":
        # Use existing if train has >=2 classes
        if _n_classes(train_df, target_class) >= 2:
            used = target_class
            info["used_target"] = used
        elif target_reg in train_df.columns and target_reg in test_df.columns:
            new_col = f"{target_class}__auto"
            med = _median_split_from(target_reg, new_col)
            used = new_col
            info["used_target"] = used
            info["rebuilt"] = True
            info["median"] = med
            info["source"] = target_reg
        else:
            if fallback_feature_for_split not in train_df.columns or fallback_feature_for_split not in test_df.columns:
                raise RuntimeError(
                    f"Auto label fix failed: missing target_reg='{target_reg}' and fallback feature "
                    f"'{fallback_feature_for_split}'."
                )
            new_col = f"{target_class}__fallback"
            med = _median_split_from(fallback_feature_for_split, new_col)
            used = new_col
            info["used_target"] = used
            info["fallback_used"] = True
            info["median"] = med
            info["source"] = fallback_feature_for_split

    # mode == "existing" -> keep as-is (used = target_class)

    # Final validation: ensure TRAIN has 2 classes (LR requirement)
    n_train_classes = _n_classes(train_df, used)
    if n_train_classes < 2:
        # Minimal technical fix for smoke-tests: flip first row if possible
        s = pd.to_numeric(train_df[used], errors="coerce").fillna(0).astype("int64")
        if len(s) >= 2:
            idx0 = s.index[0]
            train_df.loc[idx0, used] = 1 - int(train_df.loc[idx0, used])
            info["forced_flip_in_train"] = True
        else:
            # Not enough rows to force 2 classes. Don't crash:
            # training will fall back to DummyClassifier in train_classification().
            info["too_small_to_fix"] = True
            info["note"] = "train_df has <2 rows; cannot enforce 2 classes. Will rely on DummyClassifier fallback."
            # just keep as-is

    info["train_class_counts"] = _counts(train_df[used]) if used in train_df.columns else None
    info["test_class_counts"] = _counts(test_df[used]) if used in test_df.columns else None

    return train_df, test_df, used, info


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
        # If X_test is empty, skip evaluation (SimpleImputer cannot transform 0 rows)
        if X_test is None or len(X_test) == 0:
            metrics: Dict[str, Any] = {
                "warning": "empty_test_set",
                "train_rows": int(len(y_train)),
                "test_rows": 0,
                "accuracy": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
            }
            return pipe, metrics
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

    # If X_test is empty, skip evaluation
    if X_test is None or len(X_test) == 0:
        metrics: Dict[str, Any] = {
            "warning": "empty_test_set",
            "test_rows": 0,
            "r2": float("nan"),
            "mae": float("nan"),
            "spearman_ic": float("nan"),
        }
        return pipe, metrics


    proba = None
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None

    pred = pipe.predict(X_test)

    metrics: Dict[str, Any] = {
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

    # If dataset is tiny, do NOT split (otherwise train may have 1 row -> impossible for LR)
    if len(df) < 3:
        train_df = df.copy()
        test_df = df.iloc[0:0].copy()  # empty test set
        split_mode = "no_split(tiny_dataset)"
    else:
        # Prefer time split if possible
        if spec.prefer_time_split and "asof_date" in df.columns and df["asof_date"].notna().any():
            train_df, test_df = time_split(df, test_size=spec.test_size)
            split_mode = "time(asof_date)"
        else:
            train_df, test_df = random_split(df, test_size=spec.test_size, seed=spec.seed)
            split_mode = "random"

    # Feature selection
    targets: List[str] = []
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

    # 1) Drop useless features (all-NaN / too sparse / constants for large datasets)
    train_df, test_df, feature_cols, drop_info = _drop_useless_features(
        train_df, test_df, feature_cols, min_non_null=1
    )
    if not feature_cols:
        raise RuntimeError("All features were dropped (all-NaN/constant). Need more/better exported features.")

    # 2) Ensure we have 2 classes for LogisticRegression (train set)
    target_class_col = spec.target_class
    if spec.do_classification:
        train_df, test_df, target_class_col, yfix_info = _ensure_binary_target_for_lr(
            train_df,
            test_df,
            target_class=spec.target_class,
            target_reg=spec.target_reg,
            label_mode=os.getenv("BASELINE_LABEL_MODE", "auto").strip().lower() or "auto",
            fallback_feature_for_split="value_score",
        )
    else:
        yfix_info = {"rebuilt": False, "fallback_used": False}

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
        "feature_drop_info": drop_info,
        "target_fix_info": yfix_info,
        "effective_target_class": target_class_col,
    }

    out: Dict[str, Any] = {}

    if spec.do_classification:
        y_train = pd.to_numeric(train_df[target_class_col], errors="coerce").fillna(0).astype("int64").to_numpy()
        y_test = pd.to_numeric(test_df[target_class_col], errors="coerce").fillna(0).astype("int64").to_numpy()

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
    print(f"[OK] effective target (class): {target_class_col}")
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
