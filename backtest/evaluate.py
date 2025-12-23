# evaluate.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalResult:
    """
    Evaluationsergebnis inkl. Regression + Finance Metrics.
    """
    n_rows: int
    rmse: float
    mae: float
    r2: float
    spearman_ic: float
    directional_accuracy: float
    quintile_spread: float
    quintile_mean_returns: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "spearman_ic": self.spearman_ic,
            "directional_accuracy": self.directional_accuracy,
            "quintile_spread": self.quintile_spread,
            "quintile_mean_returns": self.quintile_mean_returns,
        }


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    return float(math.sqrt(mse))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R² = 1 - SSE/SST
    sse = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    sst = float(np.sum((y_true - y_mean) ** 2))
    if sst == 0.0:
        return float("nan")
    return float(1.0 - (sse / sst))


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman Rank IC über alle Zeilen (cross-sectional + time pooled).
    Robust via rank transform.
    """
    # rankdata ohne scipy: pandas rank
    s_true = pd.Series(y_true).rank(method="average")
    s_pred = pd.Series(y_pred).rank(method="average")
    corr = s_true.corr(s_pred, method="pearson")
    return float(corr) if corr is not None else float("nan")


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Anteil korrekter Vorzeichenprognosen: sign(y_pred) == sign(y_true)
    Nullwerte werden als 0 betrachtet.
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float(np.mean(true_sign == pred_sign))


def _quintile_spread(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Quintile Spread: mean(return | top 20% pred) - mean(return | bottom 20% pred)
    Returns außerdem mean returns pro Quintil (Q1..Q5).
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.dropna()
    if len(df) < 10:
        return float("nan"), {}

    # qcut kann bei vielen gleichen Werten Probleme machen; duplicates='drop'
    try:
        df["q"] = pd.qcut(df["y_pred"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    except ValueError:
        # Fallback: rank -> qcut
        df["q"] = pd.qcut(df["y_pred"].rank(method="first"), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    means = df.groupby("q")["y_true"].mean()
    q_means = {str(k): float(v) for k, v in means.to_dict().items()}

    if "Q5" not in q_means or "Q1" not in q_means:
        return float("nan"), q_means

    spread = float(q_means["Q5"] - q_means["Q1"])
    return spread, q_means


def evaluate_predictions(
        df: pd.DataFrame,
        y_true_col: str = "label_return",
        y_pred_col: str = "y_pred",
) -> EvalResult:
    """
    Erwartet DataFrame mit y_true_col und y_pred_col.
    Entfernt NaNs und berechnet Pflichtmetriken.
    """
    if y_true_col not in df.columns:
        raise KeyError(f"Missing y_true column: {y_true_col}")
    if y_pred_col not in df.columns:
        raise KeyError(f"Missing y_pred column: {y_pred_col}")

    dd = df[[y_true_col, y_pred_col]].copy()
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        raise ValueError("No rows left after dropping NaNs/Infs for evaluation.")

    y_true = dd[y_true_col].astype(float).to_numpy()
    y_pred = dd[y_pred_col].astype(float).to_numpy()

    rmse = _rmse(y_true, y_pred)
    mae = _mae(y_true, y_pred)
    r2 = _r2(y_true, y_pred)
    ic = _spearman_ic(y_true, y_pred)
    da = _directional_accuracy(y_true, y_pred)
    spread, q_means = _quintile_spread(y_true, y_pred)

    return EvalResult(
        n_rows=int(len(dd)),
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman_ic=ic,
        directional_accuracy=da,
        quintile_spread=spread,
        quintile_mean_returns=q_means,
    )


def save_eval_artifacts(
        eval_result: EvalResult,
        out_json_path: str | Path,
        out_csv_path: Optional[str | Path] = None,
) -> None:
    out_json = Path(out_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(eval_result.to_dict(), f, ensure_ascii=False, indent=2)

    if out_csv_path:
        out_csv = Path(out_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # "evaluation_table.csv" als Single-Row Tabelle (praktisch für Pipelines)
        flat = eval_result.to_dict().copy()
        # quintiles als separate Spalten
        q = flat.pop("quintile_mean_returns", {})
        for k, v in q.items():
            flat[f"mean_{k}"] = v

        pd.DataFrame([flat]).to_csv(out_csv, index=False)


def main() -> None:
    """
    CLI (optional):
      python -m ml.train.evaluate --predCsv ml/artifacts/predictions_test.csv --outJson ml/artifacts/evaluation_report.json --outCsv ml/artifacts/evaluation_table.csv

    predCsv muss mindestens enthalten:
      - label_return
      - y_pred
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--predCsv", required=True, help="CSV with columns: label_return,y_pred (and optional metadata)")
    ap.add_argument("--outJson", required=True, help="Output JSON path")
    ap.add_argument("--outCsv", default=None, help="Optional output CSV path (single-row metrics table)")
    ap.add_argument("--yTrueCol", default="label_return")
    ap.add_argument("--yPredCol", default="y_pred")
    args = ap.parse_args()

    df = pd.read_csv(args.predCsv)
    res = evaluate_predictions(df, y_true_col=args.yTrueCol, y_pred_col=args.yPredCol)
    save_eval_artifacts(res, out_json_path=args.outJson, out_csv_path=args.outCsv)

    print("[EVAL] n_rows:", res.n_rows)
    print("[EVAL] rmse:", res.rmse)
    print("[EVAL] mae :", res.mae)
    print("[EVAL] r2  :", res.r2)
    print("[EVAL] spearman_ic:", res.spearman_ic)
    print("[EVAL] directional_accuracy:", res.directional_accuracy)
    print("[EVAL] quintile_spread:", res.quintile_spread)
    if res.quintile_mean_returns:
        print("[EVAL] quintile_means:", res.quintile_mean_returns)
    print("[EVAL] written:", args.outJson, ("and " + args.outCsv) if args.outCsv else "")


if __name__ == "__main__":
    main()
