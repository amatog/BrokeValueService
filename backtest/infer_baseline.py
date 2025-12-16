# backtest/infer_baseline.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import yaml
import pandas as pd
import psycopg2
import joblib


# ----------------------------
# Paths / Defaults
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
BACKTEST_DIR = BASE_DIR / "backtest"
CONFIG_PATH = BACKTEST_DIR / "config.yaml"

DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"

DEFAULT_HORIZON_MONTHS = 12
DEFAULT_SCHEMA_PATH = str(BASE_DIR / "configs" / "feature_schema_v1.yaml")


# ----------------------------
# Config / DB
# ----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pg_connect(db_cfg: Dict[str, Any]):
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


# ----------------------------
# Feature schema helpers
# ----------------------------
def _load_feature_schema(schema_path: str) -> Dict[str, Any]:
    p = Path(schema_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing feature schema file: {schema_path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure all cols exist. Missing -> 0.0.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[cols] = out[cols].fillna(0.0)
    return out


def _extract_id_cols(schema: Dict[str, Any]) -> List[str]:
    return list(schema.get("id_cols") or ["run_id", "asof_date", "symbol"])


def _extract_label_cols(schema: Dict[str, Any]) -> List[str]:
    return list(schema.get("label_cols") or ["label_return", "label_up"])


# ----------------------------
# Model artifact helpers
# ----------------------------
def _default_features_path(run_id: str, horizon_months: int) -> Path:
    return DEFAULT_DATA_DIR / f"features_{run_id}_h{horizon_months}.parquet"


def _default_model_path(run_id: str, horizon_months: int) -> Path:
    return DEFAULT_MODELS_DIR / f"baseline_model_{run_id}_h{horizon_months}.joblib"


def _default_model_meta_path(run_id: str, horizon_months: int) -> Path:
    return DEFAULT_MODELS_DIR / f"baseline_model_{run_id}_h{horizon_months}.meta.json"


def _load_model_bundle(model_path: Path, meta_path: Optional[Path]) -> Tuple[Any, Dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)

    meta: Dict[str, Any] = {}
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return model, meta


# ----------------------------
# Postgres writer (optional)
# ----------------------------
def _ensure_prediction_table(conn, table: str = "bt_ml_prediction") -> None:
    """
    Optional convenience (falls du Rechte hast). Wenn nicht: einfach db_write=false lassen.
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS public.{table} (
      run_id         text        NOT NULL,
      asof_date      date        NOT NULL,
      symbol         text        NOT NULL,
      horizon_months int         NOT NULL,
      model_version  text        NOT NULL,
      proba_up       double precision,
      pred_up        int,
      created_at     timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY (run_id, asof_date, symbol, horizon_months, model_version)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def upsert_predictions(
        conn,
        *,
        df_pred: pd.DataFrame,
        table: str,
        model_version: str,
        horizon_months: int,
) -> None:
    need_cols = ["run_id", "asof_date", "symbol", "proba_up", "pred_up"]
    for c in need_cols:
        if c not in df_pred.columns:
            raise RuntimeError(f"Missing required prediction column: {c}")

    rows = []
    for _, r in df_pred.iterrows():
        rows.append(
            (
                str(r["run_id"]),
                r["asof_date"],
                str(r["symbol"]),
                int(horizon_months),
                str(model_version),
                None if pd.isna(r["proba_up"]) else float(r["proba_up"]),
                None if pd.isna(r["pred_up"]) else int(r["pred_up"]),
            )
        )

    sql = f"""
    INSERT INTO public.{table}
      (run_id, asof_date, symbol, horizon_months, model_version, proba_up, pred_up)
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (run_id, asof_date, symbol, horizon_months, model_version)
    DO UPDATE SET
      proba_up = EXCLUDED.proba_up,
      pred_up  = EXCLUDED.pred_up,
      created_at = now()
    ;
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()


# ----------------------------
# Spec
# ----------------------------
@dataclass
class InferSpec:
    run_id: str
    horizon_months: int
    features_path: Path
    model_path: Path
    model_meta_path: Path
    schema_path: str
    out_dir: Path
    export_csv: bool
    db_write: bool
    db_table: str
    model_version: str  # tag


def build_spec(cfg: Dict[str, Any]) -> InferSpec:
    run_id = str((cfg.get("run") or {}).get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("backtest/config.yaml -> run.run_id is missing/empty.")

    horizon = int(os.getenv("FEATURE_EXPORT_HORIZON", str(DEFAULT_HORIZON_MONTHS)))

    features_path = Path(os.getenv("INFER_FEATURES_PATH", str(_default_features_path(run_id, horizon))))
    model_path = Path(os.getenv("INFER_MODEL_PATH", str(_default_model_path(run_id, horizon))))
    model_meta_path = Path(os.getenv("INFER_MODEL_META_PATH", str(_default_model_meta_path(run_id, horizon))))

    schema_path = os.getenv("FEATURE_SCHEMA_PATH", DEFAULT_SCHEMA_PATH)

    out_dir = Path(os.getenv("INFER_OUTDIR", str(DEFAULT_OUTPUT_DIR)))
    export_csv = os.getenv("INFER_EXPORT_CSV", "1").strip().lower() in ("1", "true", "yes")

    db_write = os.getenv("INFER_DB_WRITE", "0").strip().lower() in ("1", "true", "yes")
    db_table = os.getenv("INFER_DB_TABLE", "bt_ml_prediction").strip()

    # model version: prefer env, else from meta, else fallback
    model_version = os.getenv("INFER_MODEL_VERSION", "").strip() or f"baseline_{run_id}_h{horizon}"

    return InferSpec(
        run_id=run_id,
        horizon_months=horizon,
        features_path=features_path,
        model_path=model_path,
        model_meta_path=model_meta_path,
        schema_path=schema_path,
        out_dir=out_dir,
        export_csv=export_csv,
        db_write=db_write,
        db_table=db_table,
        model_version=model_version,
    )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    spec = build_spec(cfg)

    if not spec.features_path.exists():
        raise FileNotFoundError(f"Missing features parquet: {spec.features_path}")

    # Load schema + data
    schema = _load_feature_schema(spec.schema_path)
    id_cols = _extract_id_cols(schema)
    label_cols = _extract_label_cols(schema)

    df = pd.read_parquet(spec.features_path)

    # Load model + meta
    model, meta = _load_model_bundle(spec.model_path, spec.model_meta_path)

    # Determine expected feature columns:
    # - if training saved meta["feature_cols"], use it (best!)
    # - else: use "all cols except id/label" as fallback
    feature_cols = meta.get("feature_cols")
    if isinstance(feature_cols, list) and feature_cols:
        feature_cols = [str(c) for c in feature_cols]
    else:
        drop = set(id_cols + label_cols)
        feature_cols = [c for c in df.columns if c not in drop]
        feature_cols.sort()

    # Align columns (missing -> 0)
    df = _ensure_columns(df, id_cols + label_cols + feature_cols)
    X = df[feature_cols]
    X = _coerce_numeric(X, feature_cols)

    # Predict proba (binary classifier)
    proba_up = None
    pred_up = None

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        # convention: class 1 is "up"
        if p.shape[1] >= 2:
            proba_up = p[:, 1]
        else:
            proba_up = p[:, 0]
        pred_up = (proba_up >= 0.5).astype("int64")
    else:
        # fallback: predict -> cast to int
        y = model.predict(X)
        pred_up = pd.Series(y).astype("int64").to_numpy()
        proba_up = None

    # Build output frame (keep identifiers + optional true label)
    out_cols = [c for c in id_cols if c in df.columns]
    if "label_return" in df.columns:
        out_cols.append("label_return")
    if "label_up" in df.columns:
        out_cols.append("label_up")

    df_out = df[out_cols].copy()
    if proba_up is not None:
        df_out["proba_up"] = proba_up
    else:
        df_out["proba_up"] = None
    df_out["pred_up"] = pred_up

    # Write outputs
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{spec.run_id}_h{spec.horizon_months}"
    out_parquet = spec.out_dir / f"pred_{tag}.parquet"
    out_csv = spec.out_dir / f"pred_{tag}.csv"

    df_out.to_parquet(out_parquet, index=False)
    if spec.export_csv:
        df_out.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] inference rows={len(df_out)}")
    print(f"[OK] model: {spec.model_path.name}")
    print(f"[OK] features: {spec.features_path.name}")
    print(f"[OK] out parquet: {out_parquet}")
    if spec.export_csv:
        print(f"[OK] out csv: {out_csv}")

    # Optional DB write
    if spec.db_write:
        db_cfg = (cfg.get("db") or {})
        if not db_cfg.get("enabled"):
            raise RuntimeError("DB write requested but db.enabled=false in backtest/config.yaml")

        conn = pg_connect(db_cfg)
        try:
            # If you don't have privileges, disable this or pre-create table as admin
            if os.getenv("INFER_DB_ENSURE_TABLE", "0").strip().lower() in ("1", "true", "yes"):
                _ensure_prediction_table(conn, table=spec.db_table)

            upsert_predictions(
                conn,
                df_pred=df_out,
                table=spec.db_table,
                model_version=spec.model_version,
                horizon_months=spec.horizon_months,
            )
            print(f"[OK] wrote predictions to public.{spec.db_table} model_version={spec.model_version}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
