from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


@dataclass
class BucketRow:
    run_id: str
    horizon_months: int
    bucket_scheme: str
    bucket_name: str
    sector: Optional[str]
    n: int
    mean_return: Optional[float]
    median_return: Optional[float]
    hit_rate: Optional[float]
    std_return: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _quantile_edges(values: List[float], k: int) -> List[float]:
    """
    Returns k+1 edges from min..max at quantiles.
    Uses simple percentile index; robust enough for backtest buckets.
    """
    if not values:
        return []
    xs = sorted(values)
    n = len(xs)
    edges = [xs[0]]
    for i in range(1, k):
        # percentile i/k
        idx = int(round((n - 1) * (i / k)))
        edges.append(xs[idx])
    edges.append(xs[-1])
    # ensure non-decreasing edges
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _assign_bucket(score: float, edges: List[float]) -> int:
    """
    Returns bucket index 0..k-1 for edges of length k+1.
    """
    k = len(edges) - 1
    if k <= 0:
        return 0
    # last bucket inclusive
    if score >= edges[-2]:
        return k - 1
    for i in range(k - 1):
        if edges[i] <= score < edges[i + 1]:
            return i
    return k - 1


def compute_bucket_report(
        *,
        rows: List[Dict[str, Any]],
        run_id: str,
        horizon_months: int,
        scheme: str = "quintiles",
        sector: Optional[str] = None,
) -> List[BucketRow]:
    """
    Input rows must include:
      value_score (float)
      return_value (float)
    Optional:
      sector (str)
    """
    # Filter valid rows
    clean: List[Tuple[float, float]] = []
    scores: List[float] = []

    for r in rows:
        s = _safe_float(r.get("value_score"))
        ret = _safe_float(r.get("return_value"))
        if s is None or ret is None:
            continue
        scores.append(s)
        clean.append((s, ret))

    if not clean:
        return []

    if scheme == "deciles":
        k = 10
        labels = [f"D{i+1}" for i in range(k)]
        labels[-1] = "D10 (Top)"
        labels[0] = "D1 (Bottom)"
    else:
        # default quintiles
        k = 5
        labels = [f"Q{i+1}" for i in range(k)]
        labels[-1] = "Q5 (Top)"
        labels[0] = "Q1 (Bottom)"
        scheme = "quintiles"

    edges = _quantile_edges(scores, k=k)

    buckets: List[List[float]] = [[] for _ in range(k)]
    bucket_scores: List[List[float]] = [[] for _ in range(k)]

    for s, ret in clean:
        bi = _assign_bucket(s, edges)
        buckets[bi].append(ret)
        bucket_scores[bi].append(s)

    out: List[BucketRow] = []
    for i in range(k):
        rets = buckets[i]
        scs = bucket_scores[i]
        if not rets:
            out.append(
                BucketRow(
                    run_id=run_id,
                    horizon_months=horizon_months,
                    bucket_scheme=scheme,
                    bucket_name=labels[i],
                    sector=sector,
                    n=0,
                    mean_return=None,
                    median_return=None,
                    hit_rate=None,
                    std_return=None,
                    min_score=None,
                    max_score=None,
                )
            )
            continue

        mean_r = statistics.fmean(rets) if rets else None
        med_r = statistics.median(rets) if rets else None
        hit = sum(1 for x in rets if x > 0) / len(rets) if rets else None
        std = statistics.pstdev(rets) if len(rets) >= 2 else 0.0

        out.append(
            BucketRow(
                run_id=run_id,
                horizon_months=horizon_months,
                bucket_scheme=scheme,
                bucket_name=labels[i],
                sector=sector,
                n=len(rets),
                mean_return=mean_r,
                median_return=med_r,
                hit_rate=hit,
                std_return=std,
                min_score=min(scs) if scs else None,
                max_score=max(scs) if scs else None,
            )
        )

    return out
