from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def flatten_bundle(bundle: Dict[str, Any], *, symbol: str, asof: str, run_id: str) -> Dict[str, Any]:
    """
    Normalisiert ein Bundle-JSON (REST /bundle oder DIRECT run_default_value_bundle)
    in einen flachen Feature-Row, der für Backtesting, Kalibrierung und später ML
    stabil nutzbar ist.

    Erwartet (je nach Quelle):
    - DIRECT bundle: keys wie "graham", "buffett", ... "combined", "piotroski", "beneish", "montier"
    - REST /bundle: identisch (wenn Ihr Endpoint run_default_value_bundle zurückgibt)
    Optional: Engine-Metadaten können vor dem Flattening injiziert werden:
      bundle["_engine_mode"] = "DIRECT" | "REST"
      bundle["_engine_base_url"] = "http://localhost:8001"
    """

    row: Dict[str, Any] = {
        "run_id": run_id,
        "asof": asof,
        "symbol": symbol,
    }

    # ----------------------------
    # Engine-Metadaten (optional)
    # ----------------------------
    row["engine_mode"] = bundle.get("_engine_mode") or ""
    row["engine_base_url"] = bundle.get("_engine_base_url") or ""

    # ----------------------------
    # Fundamentals (optional, wenn im Bundle enthalten)
    # ----------------------------
    fundamentals = bundle.get("fundamentals") or {}
    row["sector"] = fundamentals.get("sector")
    row["currency"] = fundamentals.get("currency")

    # ----------------------------
    # Combined Value block
    # ----------------------------
    # In DIRECT bundle heißt es typischerweise "combined"
    # In /score heißt es typischerweise "value"
    value = bundle.get("combined") or bundle.get("value") or {}
    row["value_score"] = _safe_num(value.get("score"))
    row["value_rating"] = _safe_str(value.get("rating"))
    row["value_method"] = _safe_str(value.get("method"))

    # Aggregation meta (optional)
    agg = value.get("aggregation") or {}
    row["insufficient_data_factor"] = _safe_num(agg.get("insufficient_data_factor"))
    row["agg_sector"] = agg.get("sector")

    # weights_used (optional, audit-trace)
    weights = agg.get("weights_used") or {}
    for k, v in weights.items():
        row["w__" + _slug(_safe_str(k))] = _safe_num(v)

    # components (optional, audit-trace pro Methode)
    comps = value.get("components") or []
    if isinstance(comps, list):
        for c in comps:
            method = _safe_str(c.get("method"))
            if not method:
                continue
            base = "c__" + _slug(method)
            row[base + "__score"] = _safe_num(c.get("score"))
            row[base + "__rating"] = _safe_str(c.get("rating"))

    # ----------------------------
    # Einzelstrategien
    # ----------------------------
    # Wir flattenen nur method/score/rating stabil.
    # Zusätzliche, strategie-spezifische Kennzahlen können später ergänzt werden.
    for strat_key in [
        "graham",
        "buffett",
        "greenblatt",
        "munger",
        "lynch",
        "schloss",
        "davis",
        "templeton",
        "klarman",
    ]:
        obj = bundle.get(strat_key) or {}
        row[f"{strat_key}__method"] = _safe_str(obj.get("method"))
        row[f"{strat_key}__score"] = _safe_num(obj.get("score"))
        row[f"{strat_key}__rating"] = _safe_str(obj.get("rating"))

    # ----------------------------
    # Quality / Forensics (falls als Top-Level keys vorhanden)
    # ----------------------------
    # Bei /score liegen sie oft in "quality_forensics". Bundle liefert meist top-level.
    quality_forensics = bundle.get("quality_forensics") or {}

    piot = bundle.get("piotroski") or quality_forensics.get("piotroski") or {}
    bene = bundle.get("beneish") or quality_forensics.get("beneish") or {}
    mont = bundle.get("montier") or quality_forensics.get("montier") or {}

    for name, obj in [("piotroski", piot), ("beneish", bene), ("montier", mont)]:
        row[f"{name}__method"] = _safe_str(obj.get("method"))
        row[f"{name}__score"] = _safe_num(obj.get("score"))
        row[f"{name}__rating"] = _safe_str(obj.get("rating"))

        metrics = obj.get("metrics") or {}
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                key = f"{name}__m__{_slug(_safe_str(mk))}"
                # numerisch oder string, aber flach
                if isinstance(mv, (int, float)):
                    row[key] = float(mv)
                elif mv is None:
                    row[key] = None
                else:
                    # z.B. "high", "low", "ok", etc.
                    row[key] = _safe_str(mv)

        flags = obj.get("red_flags") or obj.get("flags") or []
        if isinstance(flags, list):
            row[f"{name}__flags_count"] = len(flags)
            # optional: Flags als CSV-String (audit), aber nicht zu groß machen
            if flags:
                row[f"{name}__flags"] = ";".join(_safe_str(x) for x in flags)

    return row


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
