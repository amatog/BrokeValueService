from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import psycopg2
import pandas as pd
import yaml


# ----------------------------
# Paths / Defaults
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
BACKTEST_DIR = BASE_DIR / "backtest"
CONFIG_PATH = BACKTEST_DIR / "config.yaml"

DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
DEFAULT_HORIZON_MONTHS = 12


# ----------------------------
# Config helpers
# ----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------------
# Feature schema helpers (3.1.2)
# ----------------------------
def _load_feature_schema(schema_path: str) -> Dict[str, Any]:
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Missing feature schema file: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compile_drop_patterns(schema: Dict[str, Any]) -> List[re.Pattern]:
    pats = []
    for rx in (schema.get("drop_regex") or []):
        pats.append(re.compile(rx))
    return pats


def _should_drop(col: str, drop_exact: set, drop_patterns: List[re.Pattern]) -> bool:
    if col in drop_exact:
        return True
    for p in drop_patterns:
        if p.search(col):
            return True
    return False


def _select_feature_columns(
        df: "pd.DataFrame",
        schema: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns: (keep_id, keep_label, keep_meta, keep_features)
    """
    id_cols = schema.get("id_cols") or []
    label_cols = schema.get("label_cols") or []
    meta_cols = schema.get("meta_cols") or []

    allow_prefixes = schema.get("allow_prefixes") or []
    drop_exact = set(schema.get("drop_exact") or [])
    drop_patterns = _compile_drop_patterns(schema)

    cols = list(df.columns)

    keep_id = [c for c in id_cols if c in cols]
    keep_label = [c for c in label_cols if c in cols]
    keep_meta = [c for c in meta_cols if c in cols]

    # Allowed feature columns by prefix
    feature_candidates = []
    for c in cols:
        if any(c.startswith(pfx) for pfx in allow_prefixes):
            feature_candidates.append(c)

    # Drop filters
    keep_features = []
    for c in feature_candidates:
        if not _should_drop(c, drop_exact, drop_patterns):
            keep_features.append(c)

    # Deterministic ordering
    keep_id.sort()
    keep_label.sort()
    keep_meta.sort()
    keep_features.sort()

    return keep_id, keep_label, keep_meta, keep_features


def _force_numeric(
        df: "pd.DataFrame",
        cols: List[str],
) -> "pd.DataFrame":
    """
    Enforces numeric dtype for given columns (errors -> NaN).
    """
    if not cols:
        return df
    coerced = {c: pd.to_numeric(df[c], errors="coerce") for c in cols if c in df.columns}
    df = df.copy()
    for c, ser in coerced.items():
        df[c] = ser
    return df


def _apply_missing_policy(
        df: "pd.DataFrame",
        feature_cols: List[str],
        schema: Dict[str, Any],
) -> "pd.DataFrame":
    missing_cfg = schema.get("missing") or {}
    strategy = (missing_cfg.get("strategy") or "fillna_zero").strip()
    add_ind = bool(missing_cfg.get("add_missing_indicators", True))

    df = df.copy()

    if add_ind:
        miss_flags = {f"{c}__is_missing": df[c].isna().astype("int64") for c in feature_cols if c in df.columns}
        if miss_flags:
            df = pd.concat([df, pd.DataFrame(miss_flags, index=df.index)], axis=1)

    if feature_cols:
        if strategy == "fillna_zero":
            df[feature_cols] = df[feature_cols].fillna(0.0)
        elif strategy == "median_by_column":
            med = df[feature_cols].median(numeric_only=True)
            df[feature_cols] = df[feature_cols].fillna(med)
        else:
            raise ValueError(f"Unknown missing.strategy: {strategy}")

    return df


def _schema_version_tag(schema: Dict[str, Any]) -> str:
    out = schema.get("output") or {}
    tag = out.get("version_tag") or "v1"
    return str(tag).strip() or "v1"


# ----------------------------
# DB helpers
# ----------------------------
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


def table_columns(conn, table: str, schema: str = "public") -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [r[0] for r in cur.fetchall()]


def _safe_json_load(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8")
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flattens nested dicts to one level: {"a":{"b":1}} -> {"a.b":1}
    Lists are stored as JSON strings (stable).
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort: convert columns that look numeric to float.
    (No errors='ignore' to avoid FutureWarning)
    """
    keep_as_text = {
        "symbol", "sector", "industry", "currency",
        "run_id", "asof_date",
        "engine_mode", "engine_base_url",
        "value_rating", "value_level", "rating",
    }
    df = df.copy()
    for col in df.columns:
        if col in keep_as_text:
            continue
        if df[col].dtype == object:
            # one deterministic conversion path
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ----------------------------
# Core export
# ----------------------------
@dataclass
class ExportSpec:
    run_id: str
    horizon_months: int
    snapshot_table: str
    return_table: str
    output_dir: Path
    export_csv: bool
    json_feature_columns: List[str]  # columns to parse/flatten if present


def build_export_spec(cfg: Dict[str, Any]) -> ExportSpec:
    run_id = str((cfg.get("run") or {}).get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("backtest/config.yaml -> run.run_id is missing/empty.")

    returns_cfg = cfg.get("returns") or {}
    horizon_months = int(os.getenv(
        "FEATURE_EXPORT_HORIZON",
        str(returns_cfg.get("export_horizon_months", DEFAULT_HORIZON_MONTHS))
    ))

    db_cfg = cfg.get("db") or {}
    snapshot_table = str(db_cfg.get("table") or "bt_snapshot").strip()
    return_table = str(os.getenv("FEATURE_EXPORT_RETURN_TABLE", "bt_return")).strip()

    out_dir = Path(os.getenv("FEATURE_EXPORT_OUTDIR", str(DEFAULT_OUTPUT_DIR)))
    export_csv = os.getenv("FEATURE_EXPORT_CSV", "0").strip() in ("1", "true", "True", "YES", "yes")

    # Try common json columns used in pipelines; only parsed if actually present
    json_cols = [
        "features_json",
        "raw_json",
        "bundle_json",
        "snapshot_json",
        "value_json",
        "quality_json",
    ]
    env_cols = os.getenv("FEATURE_EXPORT_JSON_COLS")
    if env_cols:
        json_cols = [c.strip() for c in env_cols.split(",") if c.strip()]

    return ExportSpec(
        run_id=run_id,
        horizon_months=horizon_months,
        snapshot_table=snapshot_table,
        return_table=return_table,
        output_dir=out_dir,
        export_csv=export_csv,
        json_feature_columns=json_cols,
    )


def fetch_joined_rows(
        conn,
        *,
        run_id: str,
        horizon_months: int,
        snapshot_table: str,
        return_table: str,
        snapshot_cols: List[str],
) -> List[Tuple[Any, ...]]:
    """
    Joins snapshot + return for given run_id and horizon.
    Selects all snapshot columns + return_value.
    """
    s_cols_sql = ", ".join([f"s.{c}" for c in snapshot_cols])
    sql = f"""
        SELECT
            {s_cols_sql},
            r.return_value AS label_return
        FROM public.{snapshot_table} s
        JOIN public.{return_table} r
          ON r.run_id = s.run_id
         AND r.asof_date = s.asof_date
         AND r.symbol = s.symbol
        WHERE s.run_id = %s
          AND r.horizon_months = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (run_id, horizon_months))
        return cur.fetchall()


def assemble_dataframe(
        *,
        rows: List[Tuple[Any, ...]],
        snapshot_cols: List[str],
        json_feature_cols_present: List[str],
) -> pd.DataFrame:
    """
    Creates a dataframe with:
      - base columns from snapshot table
      - label_return
      - plus flattened features from json columns (if present)
    """
    all_cols = snapshot_cols + ["label_return"]
    df = pd.DataFrame(rows, columns=all_cols)

    extra_frames: List[pd.DataFrame] = []
    for jc in json_feature_cols_present:
        parsed_rows: List[Dict[str, Any]] = []
        for v in df[jc].tolist():
            d = _safe_json_load(v)
            if d is None:
                parsed_rows.append({})
            else:
                parsed_rows.append(_flatten_dict(d, prefix=f"json.{jc}", sep="."))
        extra_frames.append(pd.DataFrame(parsed_rows))

    if extra_frames:
        extras = pd.concat(extra_frames, axis=1)
        extras = extras.loc[:, ~extras.columns.duplicated()]
        df = pd.concat([df.drop(columns=json_feature_cols_present), extras], axis=1)

    # Basic coercion (schema wird danach nochmal sauber “force_numeric” machen)
    df = _coerce_numeric_columns(df)

    # label_up deterministisch
    if "label_return" in df.columns:
        df["label_return"] = pd.to_numeric(df["label_return"], errors="coerce")
        df["label_up"] = (df["label_return"] > 0).astype("int64")

    return df


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    db_cfg = cfg.get("db") or {}
    if not db_cfg.get("enabled"):
        raise RuntimeError("DB must be enabled (backtest/config.yaml -> db.enabled=true).")

    spec = build_export_spec(cfg)
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    conn = pg_connect(db_cfg)
    try:
        snapshot_cols = table_columns(conn, spec.snapshot_table, schema="public")
        if not snapshot_cols:
            raise RuntimeError(f"Snapshot table not found or empty schema: public.{spec.snapshot_table}")

        json_cols_present = [c for c in spec.json_feature_columns if c in snapshot_cols]

        rows = fetch_joined_rows(
            conn,
            run_id=spec.run_id,
            horizon_months=spec.horizon_months,
            snapshot_table=spec.snapshot_table,
            return_table=spec.return_table,
            snapshot_cols=snapshot_cols,
        )

        if not rows:
            raise RuntimeError(
                f"No joined rows found for run_id={spec.run_id} horizon_months={spec.horizon_months}. "
                f"Check bt_snapshot/bt_return coverage."
            )

        df = assemble_dataframe(
            rows=rows,
            snapshot_cols=snapshot_cols,
            json_feature_cols_present=json_cols_present,
        )

    finally:
        conn.close()

    # ------------------------------------------------------------
    # 3.1.2: Feature Schema v1 anwenden + deterministisch exportieren
    # ------------------------------------------------------------
    schema_path = os.getenv("FEATURE_SCHEMA_PATH", str(BASE_DIR / "configs" / "features_schema_v1.yaml"))
    schema = _load_feature_schema(schema_path)
    schema_tag = _schema_version_tag(schema)

    keep_id, keep_label, keep_meta, keep_features = _select_feature_columns(df, schema)

    if not keep_features:
        raise RuntimeError("No feature columns selected by schema. Check allow_prefixes / drops.")

    # Labels müssen vorhanden sein
    for lab in (schema.get("label_cols") or []):
        if lab not in df.columns:
            raise RuntimeError(f"Missing label column required by schema: {lab}")

    # Subset zuerst (deterministisch, stabil)
    df = df[keep_id + keep_meta + keep_label + keep_features].copy()

    # Force numeric nur auf echte Feature-Spalten
    df = _force_numeric(df, keep_features)

    # Missing policy + optional indicators
    df = _apply_missing_policy(df, keep_features, schema)

    # label_up (falls fehlt oder neu berechnen)
    if "label_return" in df.columns:
        df["label_return"] = pd.to_numeric(df["label_return"], errors="coerce")
    if "label_up" not in df.columns:
        df["label_up"] = (df["label_return"] > 0).astype("int64")

    # Write output (versioniert)
    tag = f"{spec.run_id}_h{spec.horizon_months}_{schema_tag}"
    out_parquet = spec.output_dir / f"features_{tag}.parquet"
    out_csv = spec.output_dir / f"features_{tag}.csv"

    df.to_parquet(out_parquet, index=False)

    if spec.export_csv:
        df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] exported rows={len(df)} cols={len(df.columns)} schema={schema_path}")
    print(f"[OK] parquet: {out_parquet}")
    if spec.export_csv:
        print(f"[OK] csv: {out_csv}")


if __name__ == "__main__":
    main()
