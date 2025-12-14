# strategies.py
"""
================================================================================
Value Services - Strategies Layer (Rule-based Value Investing Engine)
================================================================================

Ziel dieser Datei:
- Implementiert regelbasierte Value-Investing-Strategien (Graham, Buffett, Greenblatt, usw.)
- Liefert pro Strategie ein einheitliches JSON-kompatibles Response-Schema, damit die UI
  ohne Sonderfälle arbeiten kann.
- Zentralisiert alle Grenzwerte (Thresholds) über thresholds.py, um:
    1) Magic Numbers zu eliminieren
    2) Backtesting/Kalibrierung (Phase 2) zu erleichtern
    3) Feature Engineering (Phase 3) zu stabilisieren

Wichtige Prinzipien:
1) Single Source of Truth für Grenzwerte:
   - KEIN Hardcoding von Schwellenwerten in Strategien.
   - Alle Schwellen kommen aus thresholds.get_thresholds(sector).

2) UI-Mapping ohne Sonderfälle:
   - Jede Strategie liefert dieselben Top-Level Keys:
        method, score, rating, thresholds_used, metrics, checks, missing_data
   - checks ist eine Liste einheitlicher Objekte:
        { key, pass, value, threshold, weight, note }
   - missing_data ist eine Liste fehlender Metrik-Keys (strings)

3) Robustheit bei fehlenden Daten:
   - Missing Data verursacht keine Exceptions und keine “Fake Scores”.
   - Bei zu vielen fehlenden Kernwerten: rating = "Insufficient Data"
     und score wird konservativ berechnet (nur auf verfügbaren Checks).

4) Erklärbarkeit:
   - Jede Check-Regel ist transparent: was wurde geprüft, gegen welche Schwelle,
     mit welchem Gewicht und ob bestanden.

Integration / Schnittstelle:
- Diese Funktionen erwarten i.d.R. Dictionaries, die du aus deinem Data Layer befüllst.
  Typische Inputs:
    fundamentals: Kennzahlen (pe, pb, roe, debt_to_equity, etc.)
    growth: Wachstumsdaten (eps_growth_5y, revenue_growth, etc.)
    market: Kurs-/Marktdaten (price, market_cap, etc.)
    sector: Sektorstring (optional; für threshold overrides)

- Die APIs können (wie bisher) je Endpoint eine Strategie berechnen,
  oder du aggregierst mehrere Strategien in combined_value_score().

Hinweis:
- Die konkreten Metric-Keys (z.B. "roe", "debt_to_equity") müssen zu deiner Data-Layer-
  Payload passen. Wenn Key-Namen abweichen, mappe sie im Data Layer oder hier zentral
  in einem Adapter (empfohlen: Data Layer).

================================================================================
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from thresholds import get_thresholds, get_strategy_aggregation_config, get_quality_risk_config



# ------------------------------------------------------------------------------
# Helpers: Rating, Safe access, Check building
# ------------------------------------------------------------------------------

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def rating_from_score(score_0_1: float) -> str:
    """
    Standardisiertes Rating für UI.
    """
    s = clamp01(score_0_1)
    if s >= 0.85:
        return "Strong Buy"
    if s >= 0.70:
        return "Buy"
    if s >= 0.50:
        return "Hold"
    return "Avoid"


def safe_float(value: Any) -> Optional[float]:
    """
    Konvertiert value robust zu float (oder None).
    Akzeptiert int/float/str, ignoriert leere Strings, NaNs werden nicht speziell behandelt.
    """
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except Exception:
        return None

def _pick_two_periods(fin: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Tries to extract (current, previous) period dicts from multiple common shapes:
    - {"current": {...}, "previous": {...}}
    - {"t": {...}, "t_1": {...}}
    - {"latest": {...}, "prior": {...}}
    - {"annualReports": [ {...}, {...}, ... ]}  (AlphaVantage style)
    - {"annual": [ {...}, {...} ]} or {"reports":[...]}
    Returns (cur, prev) or (None, None) if not possible.
    """
    if not isinstance(fin, dict):
        return None, None

    for a, b in [("current", "previous"), ("t", "t_1"), ("latest", "prior")]:
        if isinstance(fin.get(a), dict) and isinstance(fin.get(b), dict):
            return fin[a], fin[b]

    for arr_key in ["annualReports", "annual", "reports"]:
        arr = fin.get(arr_key)
        if isinstance(arr, list) and len(arr) >= 2 and isinstance(arr[0], dict) and isinstance(arr[1], dict):
            # convention: [0] current, [1] previous
            return arr[0], arr[1]

    return None, None


def _get_num(d: Optional[Dict[str, Any]], keys: List[str]) -> Optional[float]:
    """
    Get numeric value from dict by trying multiple candidate keys.
    """
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            v = safe_float(d.get(k))
            if v is not None:
                return v
    return None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b



def build_check(
        *,
        key: str,
        value: Optional[float],
        threshold: Optional[float],
        weight: float,
        comparator: str,
        note: str,
        missing: List[str],
) -> Dict[str, Any]:
    """
    Baut ein einheitliches Check-Objekt für die UI.

    comparator:
      - ">=" : value >= threshold
      - "<=" : value <= threshold
      - ">"  : value > threshold
      - "<"  : value < threshold

    Missing Handling:
      - Wenn value oder threshold None: pass = False und key wird in missing aufgenommen.
    """
    passed = False

    if value is None:
        if key not in missing:
            missing.append(key)
    if threshold is None:
        # threshold fehlt (sollte in 1.1 kaum passieren, aber robust)
        t_key = f"threshold:{key}"
        if t_key not in missing:
            missing.append(t_key)

    if (value is not None) and (threshold is not None):
        if comparator == ">=":
            passed = value >= threshold
        elif comparator == "<=":
            passed = value <= threshold
        elif comparator == ">":
            passed = value > threshold
        elif comparator == "<":
            passed = value < threshold
        else:
            # Unbekannter Comparator -> fail-safe
            passed = False

    return {
        "key": key,
        "pass": bool(passed),
        "value": value,
        "threshold": threshold,
        "weight": float(weight),
        "note": note,
    }


def score_from_checks(checks: List[Dict[str, Any]]) -> float:
    """
    Aggregiert Score aus Checks:
    - gewichtete Summe (bestanden => weight, nicht bestanden => 0)
    - normalisiert durch Summe aller weights
    - falls keine weights: 0.0
    """
    total_w = 0.0
    got_w = 0.0
    for c in checks:
        w = safe_float(c.get("weight")) or 0.0
        total_w += w
        if c.get("pass") is True:
            got_w += w
    if total_w <= 0.0:
        return 0.0
    return clamp01(got_w / total_w)


def make_response(
        *,
        method: str,
        thresholds_used: Dict[str, Any],
        metrics: Dict[str, Any],
        checks: List[Dict[str, Any]],
        missing_data: List[str],
        min_required_passable_checks: int = 2,
) -> Dict[str, Any]:
    """
    Standard-Response für jede Strategie.

    min_required_passable_checks:
      - Mindestanzahl Checks, die überhaupt auswertbar sind (value+threshold vorhanden),
        sonst rating = "Insufficient Data".
    """
    # Zähle auswertbare Checks (value & threshold sind nicht None)
    passable = 0
    for c in checks:
        if (c.get("value") is not None) and (c.get("threshold") is not None):
            passable += 1

    score = score_from_checks(checks)
    if passable < min_required_passable_checks:
        rating = "Insufficient Data"
    else:
        rating = rating_from_score(score)

    return {
        "method": method,
        "score": float(score),
        "rating": rating,
        "thresholds_used": thresholds_used,
        "metrics": metrics,
        "checks": checks,
        "missing_data": sorted(list(set(missing_data))),
    }


# ------------------------------------------------------------------------------
# Strategy: Graham (simplified, rule-based)
# ------------------------------------------------------------------------------

def graham_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Graham (vereinfacht):
    - Niedrige Bewertung (PE, PB)
    - Solide Liquidität (current ratio)
    - Niedrige Verschuldung (debt_to_equity)
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    pe = safe_float(fundamentals.get("pe"))
    pb = safe_float(fundamentals.get("pb"))
    current_ratio = safe_float(fundamentals.get("current_ratio"))
    dte = safe_float(fundamentals.get("debt_to_equity"))

    checks.append(build_check(
        key="pe_max",
        value=pe,
        threshold=safe_float(t.get("pe_max")),
        weight=0.30,
        comparator="<=",
        note="Bewertung: KGV (PE) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="pb_max",
        value=pb,
        threshold=safe_float(t.get("pb_max")),
        weight=0.25,
        comparator="<=",
        note="Bewertung: KBV (PB) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="current_ratio_min",
        value=current_ratio,
        threshold=safe_float(t.get("current_ratio_min")),
        weight=0.20,
        comparator=">=",
        note="Stabilität: Current Ratio über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="debt_to_equity_max",
        value=dte,
        threshold=safe_float(t.get("debt_to_equity_max")),
        weight=0.25,
        comparator="<=",
        note="Verschuldung: Debt/Equity unter Maximalwert.",
        missing=missing,
    ))

    thresholds_used = {
        "pe_max": t.get("pe_max"),
        "pb_max": t.get("pb_max"),
        "current_ratio_min": t.get("current_ratio_min"),
        "debt_to_equity_max": t.get("debt_to_equity_max"),
    }
    metrics = {
        "pe": pe,
        "pb": pb,
        "current_ratio": current_ratio,
        "debt_to_equity": dte,
    }

    return make_response(
        method="Graham (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


# ------------------------------------------------------------------------------
# Strategy: Buffett Quality (simplified)
# ------------------------------------------------------------------------------

def buffett_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Buffett (vereinfacht als Quality-Proxy):
    - ROE >= min_roe
    - Niedrige Verschuldung (debt_to_equity <= max)
    - Optional: Interest Coverage >= min
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    roe = safe_float(fundamentals.get("roe"))
    dte = safe_float(fundamentals.get("debt_to_equity"))
    ic = safe_float(fundamentals.get("interest_coverage"))

    checks.append(build_check(
        key="min_roe",
        value=roe,
        threshold=safe_float(t.get("min_roe")),
        weight=0.45,
        comparator=">=",
        note="Qualität: ROE über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="debt_to_equity_max",
        value=dte,
        threshold=safe_float(t.get("debt_to_equity_max")),
        weight=0.35,
        comparator="<=",
        note="Verschuldung: Debt/Equity unter Maximalwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="interest_coverage_min",
        value=ic,
        threshold=safe_float(t.get("interest_coverage_min")),
        weight=0.20,
        comparator=">=",
        note="Zinslast: Interest Coverage über Mindestwert.",
        missing=missing,
    ))

    thresholds_used = {
        "min_roe": t.get("min_roe"),
        "debt_to_equity_max": t.get("debt_to_equity_max"),
        "interest_coverage_min": t.get("interest_coverage_min"),
    }
    metrics = {
        "roe": roe,
        "debt_to_equity": dte,
        "interest_coverage": ic,
    }

    return make_response(
        method="Buffett-Qualität (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


# ------------------------------------------------------------------------------
# Strategy: Greenblatt Magic Formula (simplified scoring)
# ------------------------------------------------------------------------------

def greenblatt_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Greenblatt (vereinfacht):
    - ROIC >= min_roic
    - Earnings Yield Proxy: 1/PE (oder ebit_yield, falls vorhanden)
    In 1.1 vermeiden wir Magic Numbers; Schwelle für PE/ROIC aus thresholds.
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    roic = safe_float(fundamentals.get("roic"))
    pe = safe_float(fundamentals.get("pe"))
    ebit_yield = safe_float(fundamentals.get("ebit_yield"))  # optionaler direkter Wert

    # Earnings yield proxy
    ey = ebit_yield
    if ey is None and pe is not None and pe > 0:
        ey = 1.0 / pe

    # Schwellen:
    # - ROIC Minimum über threshold
    # - PE Maximum als Proxy für "ausreichenden Earnings Yield"
    checks.append(build_check(
        key="min_roic",
        value=roic,
        threshold=safe_float(t.get("min_roic")),
        weight=0.60,
        comparator=">=",
        note="Qualität: ROIC über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="pe_max",
        value=pe,
        threshold=safe_float(t.get("pe_max")),
        weight=0.40,
        comparator="<=",
        note="Ertragsrendite-Proxy: KGV (PE) unter Schwellwert.",
        missing=missing,
    ))

    thresholds_used = {
        "min_roic": t.get("min_roic"),
        "pe_max": t.get("pe_max"),
    }
    metrics = {
        "roic": roic,
        "pe": pe,
        "earnings_yield_proxy": ey,
        "ebit_yield": ebit_yield,
    }

    return make_response(
        method="Greenblatt Magic Formula (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


# ------------------------------------------------------------------------------
# Strategy: Lynch (simplified PEG)
# ------------------------------------------------------------------------------

def lynch_strategy(
        fundamentals: Dict[str, Any],
        growth: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Peter Lynch (vereinfacht):
    - PEG <= peg_max
    - EPS Growth 5y >= min_eps_growth_5y
    PEG = PE / (EPS Growth %)
    (Hier: EPS Growth als Dezimalzahl, z.B. 0.10 für 10%.)
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    pe = safe_float(fundamentals.get("pe"))
    eps_g5 = safe_float(growth.get("eps_growth_5y"))

    peg = None
    if pe is not None and eps_g5 is not None and eps_g5 > 0:
        peg = pe / (eps_g5 * 100.0)  # growth in % for PEG conventional definition
        # Hinweis: Diese Umrechnung ist bewusst dokumentiert; wenn du Growth bereits in %
        # lieferst, entferne *100.0 im Data Layer oder passe hier an.

    checks.append(build_check(
        key="peg_max",
        value=peg,
        threshold=safe_float(t.get("peg_max")),
        weight=0.60,
        comparator="<=",
        note="Wachstum/Bewertung: PEG unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="min_eps_growth_5y",
        value=eps_g5,
        threshold=safe_float(t.get("min_eps_growth_5y")),
        weight=0.40,
        comparator=">=",
        note="Wachstum: EPS Growth (5y) über Mindestwert.",
        missing=missing,
    ))

    thresholds_used = {
        "peg_max": t.get("peg_max"),
        "min_eps_growth_5y": t.get("min_eps_growth_5y"),
    }
    metrics = {
        "pe": pe,
        "eps_growth_5y": eps_g5,
        "peg": peg,
    }

    return make_response(
        method="Peter Lynch (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


# ------------------------------------------------------------------------------
# Strategy: Klarman (Margin of Safety proxy - simplified)
# ------------------------------------------------------------------------------

def klarman_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Seth Klarman (vereinfacht):
    - Fokus auf Sicherheitsmarge-Proxy über günstige Multiples und Cashflow
    - PE <= pe_max, PFCF <= pfcf_max, FCF Margin >= fcf_margin_min
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    pe = safe_float(fundamentals.get("pe"))
    pfcf = safe_float(fundamentals.get("pfcf"))
    fcf_margin = safe_float(fundamentals.get("fcf_margin"))
    fcf = safe_float(fundamentals.get("free_cash_flow"))

    checks.append(build_check(
        key="pe_max",
        value=pe,
        threshold=safe_float(t.get("pe_max")),
        weight=0.35,
        comparator="<=",
        note="Sicherheitsmarge-Proxy: KGV (PE) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="pfcf_max",
        value=pfcf,
        threshold=safe_float(t.get("pfcf_max")),
        weight=0.35,
        comparator="<=",
        note="Sicherheitsmarge-Proxy: P/FCF unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="fcf_margin_min",
        value=fcf_margin,
        threshold=safe_float(t.get("fcf_margin_min")),
        weight=0.30,
        comparator=">=",
        note="Cashflow-Qualität: FCF-Marge über Mindestwert.",
        missing=missing,
    ))

    # Optionaler Zusatz-Flag: FCF positiv erforderlich
    if bool(t.get("fcf_positive_required", True)):
        # Wir nutzen denselben Standard-Check-Mechanismus mit einem "virtuellen threshold"
        # und Comparator ">" gegen 0.0 – aber 0.0 ist ebenfalls ein Threshold und kommt
        # idealerweise aus config. Für 1.1 halten wir es pragmatisch:
        # (Wenn du absolut keine 0.0 willst: ergänze in thresholds.py z.B. fcf_min = 0.0)
        checks.append(build_check(
            key="fcf_positive_required",
            value=fcf,
            threshold=0.0,
            weight=0.15,
            comparator=">",
            note="Cashflow: Free Cash Flow positiv (wenn erforderlich).",
            missing=missing,
        ))

    thresholds_used = {
        "pe_max": t.get("pe_max"),
        "pfcf_max": t.get("pfcf_max"),
        "fcf_margin_min": t.get("fcf_margin_min"),
        "fcf_positive_required": t.get("fcf_positive_required", True),
    }
    metrics = {
        "pe": pe,
        "pfcf": pfcf,
        "fcf_margin": fcf_margin,
        "free_cash_flow": fcf,
    }

    return make_response(
        method="Klarman (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


# ------------------------------------------------------------------------------
# Placeholder Strategies (Munger/Schloss/Davis/Templeton)
# In 1.1 Fokus: Threshold-Integration + Response-Standard.
# Du kannst diese Regeln später in 1.2 / 2.x weiter ausbauen.
# ------------------------------------------------------------------------------

def munger_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Charlie Munger (vereinfacht):
    - hoher Moat/Quality Proxy: ROE/ROIC, Gross Margin
    - konservative Verschuldung
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    roe = safe_float(fundamentals.get("roe"))
    roic = safe_float(fundamentals.get("roic"))
    gm = safe_float(fundamentals.get("gross_margin"))
    dte = safe_float(fundamentals.get("debt_to_equity"))

    checks.append(build_check(
        key="min_roe",
        value=roe,
        threshold=safe_float(t.get("min_roe")),
        weight=0.30,
        comparator=">=",
        note="Qualität: ROE über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="min_roic",
        value=roic,
        threshold=safe_float(t.get("min_roic")),
        weight=0.30,
        comparator=">=",
        note="Qualität: ROIC über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="min_gross_margin",
        value=gm,
        threshold=safe_float(t.get("min_gross_margin")),
        weight=0.20,
        comparator=">=",
        note="Moat-Proxy: Gross Margin über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="debt_to_equity_max",
        value=dte,
        threshold=safe_float(t.get("debt_to_equity_max")),
        weight=0.20,
        comparator="<=",
        note="Verschuldung: Debt/Equity unter Maximalwert.",
        missing=missing,
    ))

    thresholds_used = {
        "min_roe": t.get("min_roe"),
        "min_roic": t.get("min_roic"),
        "min_gross_margin": t.get("min_gross_margin"),
        "debt_to_equity_max": t.get("debt_to_equity_max"),
    }
    metrics = {
        "roe": roe,
        "roic": roic,
        "gross_margin": gm,
        "debt_to_equity": dte,
    }

    return make_response(
        method="Munger (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


def schloss_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Walter Schloss (vereinfacht):
    - Fokus auf “billig” (PB, PE) und Bilanz-Konservatismus (D/E)
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    pb = safe_float(fundamentals.get("pb"))
    pe = safe_float(fundamentals.get("pe"))
    dte = safe_float(fundamentals.get("debt_to_equity"))

    checks.append(build_check(
        key="pb_max",
        value=pb,
        threshold=safe_float(t.get("pb_max")),
        weight=0.45,
        comparator="<=",
        note="Bewertung: KBV (PB) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="pe_max",
        value=pe,
        threshold=safe_float(t.get("pe_max")),
        weight=0.35,
        comparator="<=",
        note="Bewertung: KGV (PE) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="debt_to_equity_max",
        value=dte,
        threshold=safe_float(t.get("debt_to_equity_max")),
        weight=0.20,
        comparator="<=",
        note="Bilanz: Debt/Equity unter Maximalwert.",
        missing=missing,
    ))

    thresholds_used = {
        "pb_max": t.get("pb_max"),
        "pe_max": t.get("pe_max"),
        "debt_to_equity_max": t.get("debt_to_equity_max"),
    }
    metrics = {
        "pb": pb,
        "pe": pe,
        "debt_to_equity": dte,
    }

    return make_response(
        method="Schloss (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


def davis_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shelby Davis (vereinfacht):
    - Finanzielle Solidität + Growth/Quality Proxy
    - In 1.1 minimal: ROE + D/E + PE
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    roe = safe_float(fundamentals.get("roe"))
    dte = safe_float(fundamentals.get("debt_to_equity"))
    pe = safe_float(fundamentals.get("pe"))

    checks.append(build_check(
        key="min_roe",
        value=roe,
        threshold=safe_float(t.get("min_roe")),
        weight=0.40,
        comparator=">=",
        note="Qualität: ROE über Mindestwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="debt_to_equity_max",
        value=dte,
        threshold=safe_float(t.get("debt_to_equity_max")),
        weight=0.30,
        comparator="<=",
        note="Verschuldung: Debt/Equity unter Maximalwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="pe_max",
        value=pe,
        threshold=safe_float(t.get("pe_max")),
        weight=0.30,
        comparator="<=",
        note="Bewertung: KGV (PE) unter Schwellwert.",
        missing=missing,
    ))

    thresholds_used = {
        "min_roe": t.get("min_roe"),
        "debt_to_equity_max": t.get("debt_to_equity_max"),
        "pe_max": t.get("pe_max"),
    }
    metrics = {
        "roe": roe,
        "debt_to_equity": dte,
        "pe": pe,
    }

    return make_response(
        method="Davis (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


def templeton_strategy(
        fundamentals: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    John Templeton (vereinfacht):
    - Contrarian-Value Proxy: günstige Multiples + solide Bilanz
    - Minimal in 1.1: PE, PS, D/E
    """
    t = get_thresholds(sector)
    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    pe = safe_float(fundamentals.get("pe"))
    ps = safe_float(fundamentals.get("ps"))
    dte = safe_float(fundamentals.get("debt_to_equity"))

    checks.append(build_check(
        key="pe_max",
        value=pe,
        threshold=safe_float(t.get("pe_max")),
        weight=0.40,
        comparator="<=",
        note="Bewertung: KGV (PE) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="ps_max",
        value=ps,
        threshold=safe_float(t.get("ps_max")),
        weight=0.35,
        comparator="<=",
        note="Bewertung: KUV (PS) unter Schwellwert.",
        missing=missing,
    ))
    checks.append(build_check(
        key="debt_to_equity_max",
        value=dte,
        threshold=safe_float(t.get("debt_to_equity_max")),
        weight=0.25,
        comparator="<=",
        note="Bilanz: Debt/Equity unter Maximalwert.",
        missing=missing,
    ))

    thresholds_used = {
        "pe_max": t.get("pe_max"),
        "ps_max": t.get("ps_max"),
        "debt_to_equity_max": t.get("debt_to_equity_max"),
    }
    metrics = {
        "pe": pe,
        "ps": ps,
        "debt_to_equity": dte,
    }

    return make_response(
        method="Templeton (vereinfacht)",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=2,
    )


# ------------------------------------------------------------------------------
# Aggregation: Combined Value Score (clean, short, stable)
# ------------------------------------------------------------------------------

def combined_value_score(
        *strategy_results: Dict[str, Any],
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gewichtete Aggregation (rein konfigurationsbasiert, Phase-2-kalibrierbar).

    - Strategy-Gewichte und Insufficient-Data-Faktor kommen ausschließlich aus thresholds.py
      via get_strategy_aggregation_config(sector).
    - Keine Magic Numbers im Code.
    """
    agg_cfg = get_strategy_aggregation_config(sector)
    weights_cfg: Dict[str, float] = agg_cfg.get("weights", {}) or {}
    insufficient_factor: float = float(agg_cfg.get("insufficient_factor", 1.0))

    components: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    weight_sum = 0.0

    for r in strategy_results:
        if not isinstance(r, dict):
            continue

        method = r.get("method", "Unknown")
        score = safe_float(r.get("score"))
        rating = r.get("rating", "Unknown")

        components.append({"method": method, "score": score, "rating": rating})

        if score is None:
            continue

        base_w = safe_float(weights_cfg.get(method))
        # Wenn keine Konfig vorhanden: default = 1.0, aber auch das ist "konfigurierbar",
        # indem man den Method-Namen in STRATEGY_WEIGHTS_DEFAULT ergänzt.
        if base_w is None:
            base_w = 1.0

        w = float(base_w)
        if rating == "Insufficient Data":
            w = w * float(insufficient_factor)

        weighted_sum += float(score) * w
        weight_sum += w

    if weight_sum <= 0.0:
        final_score = 0.0
        final_rating = "Insufficient Data"
    else:
        final_score = clamp01(weighted_sum / weight_sum)
        final_rating = rating_from_score(final_score)

    return {
        "method": "Combined Value Score",
        "score": float(final_score),
        "rating": final_rating,
        "aggregation": {
            "sector": sector,
            "insufficient_data_factor": insufficient_factor,
            "weights_used": weights_cfg,
        },
        "components": components,
    }

def piotroski_f_score(
        income: Dict[str, Any],
        balance: Dict[str, Any],
        cashflow: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Piotroski F-Score (0..9) – requires current & previous year financials.

    Inputs expected (flexible key mapping):
    - Income: revenue/sales, net_income, gross_profit (or gross_margin), shares_outstanding (optional)
    - Balance: total_assets, long_term_debt (or total_liabilities), current_assets, current_liabilities, shares_outstanding (optional)
    - Cashflow: operating_cash_flow (CFO)

    Output:
    - score normalized 0..1 for aggregation
    - metrics include 'f_score' (0..9)
    """
    cfg = get_quality_risk_config()["piotroski"]
    max_score = int(cfg["max_score"])

    inc_t, inc_t1 = _pick_two_periods(income)
    bal_t, bal_t1 = _pick_two_periods(balance)
    cf_t, cf_t1 = _pick_two_periods(cashflow)

    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    # Key candidates (adaptable without changing scoring logic)
    rev_t = _get_num(inc_t, ["revenue", "totalRevenue", "sales"])
    rev_t1 = _get_num(inc_t1, ["revenue", "totalRevenue", "sales"])

    ni_t = _get_num(inc_t, ["net_income", "netIncome"])
    ni_t1 = _get_num(inc_t1, ["net_income", "netIncome"])

    ta_t = _get_num(bal_t, ["total_assets", "totalAssets"])
    ta_t1 = _get_num(bal_t1, ["total_assets", "totalAssets"])

    cfo_t = _get_num(cf_t, ["operating_cash_flow", "operatingCashflow", "cashflowFromOperations"])
    # CFO t-1 not strictly needed for classic 9 signals, but may be useful later

    ltd_t = _get_num(bal_t, ["long_term_debt", "longTermDebt"])
    ltd_t1 = _get_num(bal_t1, ["long_term_debt", "longTermDebt"])

    ca_t = _get_num(bal_t, ["current_assets", "totalCurrentAssets", "currentAssets"])
    ca_t1 = _get_num(bal_t1, ["current_assets", "totalCurrentAssets", "currentAssets"])

    cl_t = _get_num(bal_t, ["current_liabilities", "totalCurrentLiabilities", "currentLiabilities"])
    cl_t1 = _get_num(bal_t1, ["current_liabilities", "totalCurrentLiabilities", "currentLiabilities"])

    gp_t = _get_num(inc_t, ["gross_profit", "grossProfit"])
    gp_t1 = _get_num(inc_t1, ["gross_profit", "grossProfit"])

    so_t = _get_num(inc_t, ["shares_outstanding", "commonStockSharesOutstanding", "weightedAverageShsOut"])
    if so_t is None:
        so_t = _get_num(bal_t, ["shares_outstanding", "commonStockSharesOutstanding"])
    so_t1 = _get_num(inc_t1, ["shares_outstanding", "commonStockSharesOutstanding", "weightedAverageShsOut"])
    if so_t1 is None:
        so_t1 = _get_num(bal_t1, ["shares_outstanding", "commonStockSharesOutstanding"])

    # Derived ratios
    roa_t = _safe_div(ni_t, ta_t)
    roa_t1 = _safe_div(ni_t1, ta_t1)
    d_roa = None if (roa_t is None or roa_t1 is None) else (roa_t - roa_t1)

    leverage_t = _safe_div(ltd_t, ta_t)
    leverage_t1 = _safe_div(ltd_t1, ta_t1)
    d_leverage = None if (leverage_t is None or leverage_t1 is None) else (leverage_t - leverage_t1)

    cr_t = _safe_div(ca_t, cl_t)
    cr_t1 = _safe_div(ca_t1, cl_t1)
    d_cr = None if (cr_t is None or cr_t1 is None) else (cr_t - cr_t1)

    gm_t = _safe_div(gp_t, rev_t)
    gm_t1 = _safe_div(gp_t1, rev_t1)
    d_gm = None if (gm_t is None or gm_t1 is None) else (gm_t - gm_t1)

    at_t = _safe_div(rev_t, ta_t)
    at_t1 = _safe_div(rev_t1, ta_t1)
    d_at = None if (at_t is None or at_t1 is None) else (at_t - at_t1)

    # Each Piotroski signal is a binary check (pass => +1)
    # To keep UI uniform we model them as checks with threshold=0 and comparator ">" (threshold is "model constant").
    # NOTE: model constants are explicitly tracked in thresholds.py via PIOTROSKI_MAX_SCORE (normalization).
    # For checks we use threshold=0.0 as conceptual boundary (not a tuned threshold).
    def _bin_check(key: str, value: Optional[float], note: str) -> Dict[str, Any]:
        return build_check(
            key=key,
            value=value,
            threshold=0.0,
            weight=1.0,
            comparator=">",
            note=note,
            missing=missing,
        )

    # Profitability
    checks.append(_bin_check("piotroski_roa_positive", roa_t, "ROA > 0"))
    checks.append(_bin_check("piotroski_cfo_positive", cfo_t, "CFO > 0"))
    checks.append(_bin_check("piotroski_delta_roa_positive", d_roa, "ΔROA > 0"))
    # Accruals: CFO > Net Income => quality earnings
    accrual = None if (cfo_t is None or ni_t is None) else (cfo_t - ni_t)
    checks.append(_bin_check("piotroski_accrual_cfo_gt_ni", accrual, "CFO - NI > 0 (low accruals)"))

    # Leverage/Liquidity
    # ΔLeverage < 0 is good => invert sign so ">" check works
    inv_d_leverage = None if d_leverage is None else (-d_leverage)
    checks.append(_bin_check("piotroski_leverage_decrease", inv_d_leverage, "Leverage decreased (Δ < 0)"))
    checks.append(_bin_check("piotroski_current_ratio_increase", d_cr, "ΔCurrent Ratio > 0"))

    # Share issuance: no dilution is good => (so_t1 - so_t) >= 0 (shares not increased)
    # Transform to (so_t1 - so_t) so that positive means good.
    no_dilution = None if (so_t is None or so_t1 is None) else (so_t1 - so_t)
    checks.append(_bin_check("piotroski_no_share_dilution", no_dilution, "No share dilution (shares not increased)"))

    # Operating efficiency
    checks.append(_bin_check("piotroski_gross_margin_increase", d_gm, "ΔGross Margin > 0"))
    checks.append(_bin_check("piotroski_asset_turnover_increase", d_at, "ΔAsset Turnover > 0"))

    # Score = number of passed checks (each weight=1)
    f_score = int(sum(1 for c in checks if c.get("pass") is True))
    score_0_1 = 0.0 if max_score <= 0 else clamp01(f_score / float(max_score))

    thresholds_used = {"piotroski_max_score": max_score}
    metrics = {
        "f_score": f_score,
        "roa": roa_t,
        "cfo": cfo_t,
        "delta_roa": d_roa,
        "accrual_cfo_minus_ni": accrual,
        "delta_leverage": d_leverage,
        "delta_current_ratio": d_cr,
        "shares_outstanding_t": so_t,
        "shares_outstanding_t1": so_t1,
        "delta_gross_margin": d_gm,
        "delta_asset_turnover": d_at,
    }

    # We keep the standard response format; 'score' is normalized 0..1.
    res = make_response(
        method="Piotroski F-Score",
        thresholds_used=thresholds_used,
        metrics=metrics,
        checks=checks,
        missing_data=missing,
        min_required_passable_checks=6,  # Piotroski needs substantial coverage
    )
    # override computed score with normalized piotroski score (since make_response scores by weights)
    res["score"] = float(score_0_1)
    res["rating"] = rating_from_score(res["score"]) if res["rating"] != "Insufficient Data" else "Insufficient Data"
    return res

def beneish_penalty(
        income: Dict[str, Any],
        balance: Dict[str, Any],
        cashflow: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:

    """
    Beneish M-Score (8-variable model) using indices derived from Income/Balance/Cashflow.

    Returns:
    - metrics includes m_score and each index
    - score is 0..1 "goodness" (lower manipulation risk => higher score)
    - checks include threshold decision
    """
    qcfg = get_quality_risk_config()["beneish"]
    coef = qcfg["coefficients"]
    threshold = float(qcfg["threshold"])
    risk_range = float(qcfg["risk_range"])

    inc_t, inc_t1 = _pick_two_periods(income)
    bal_t, bal_t1 = _pick_two_periods(balance)
    cf_t, cf_t1 = _pick_two_periods(cashflow)

    missing: List[str] = []
    checks: List[Dict[str, Any]] = []

    # Core fields (candidate keys)
    sales_t = _get_num(inc_t, ["revenue", "totalRevenue", "sales"])
    sales_t1 = _get_num(inc_t1, ["revenue", "totalRevenue", "sales"])

    ar_t = _get_num(bal_t, ["net_receivables", "netReceivables", "currentNetReceivables", "accountsReceivable"])
    ar_t1 = _get_num(bal_t1, ["net_receivables", "netReceivables", "currentNetReceivables", "accountsReceivable"])

    cogs_t = _get_num(inc_t, ["cost_of_revenue", "costOfRevenue", "cogs"])
    cogs_t1 = _get_num(inc_t1, ["cost_of_revenue", "costOfRevenue", "cogs"])

    ca_t = _get_num(bal_t, ["current_assets", "totalCurrentAssets", "currentAssets"])
    ca_t1 = _get_num(bal_t1, ["current_assets", "totalCurrentAssets", "currentAssets"])

    ppe_t = _get_num(bal_t, ["property_plant_equipment", "propertyPlantEquipment", "propertyPlantEquipmentNet", "netPPE"])
    ppe_t1 = _get_num(bal_t1, ["property_plant_equipment", "propertyPlantEquipment", "propertyPlantEquipmentNet", "netPPE"])

    ta_t = _get_num(bal_t, ["total_assets", "totalAssets"])
    ta_t1 = _get_num(bal_t1, ["total_assets", "totalAssets"])

    dep_t = _get_num(inc_t, ["depreciation", "depreciationAndAmortization"])
    dep_t1 = _get_num(inc_t1, ["depreciation", "depreciationAndAmortization"])

    sga_t = _get_num(inc_t, ["selling_general_administrative", "sellingGeneralAdministrative", "sga"])
    sga_t1 = _get_num(inc_t1, ["selling_general_administrative", "sellingGeneralAdministrative", "sga"])

    td_t = _get_num(bal_t, ["total_debt", "totalDebt", "shortLongTermDebtTotal", "liabilities"])
    td_t1 = _get_num(bal_t1, ["total_debt", "totalDebt", "shortLongTermDebtTotal", "liabilities"])

    ni_t = _get_num(inc_t, ["net_income", "netIncome"])
    cfo_t = _get_num(cf_t, ["operating_cash_flow", "operatingCashflow", "cashflowFromOperations"])

    # Indices
    dsri = None
    gmi = None
    aqi = None
    sgi = None
    depi = None
    sgai = None
    lvgi = None
    tata = None

    # DSRI = (AR/Sales)_t / (AR/Sales)_t1
    dsri = _safe_div(_safe_div(ar_t, sales_t), _safe_div(ar_t1, sales_t1))

    # GMI = (GrossMargin)_t1 / (GrossMargin)_t
    gm_t = None
    gm_t1 = None
    if sales_t is not None and cogs_t is not None:
        gm_t = _safe_div((sales_t - cogs_t), sales_t)
    if sales_t1 is not None and cogs_t1 is not None:
        gm_t1 = _safe_div((sales_t1 - cogs_t1), sales_t1)
    gmi = _safe_div(gm_t1, gm_t)

    # AQI = [1 - (CA + PPE)/TA]_t / [1 - (CA + PPE)/TA]_t1
    aqi_num_t = None
    aqi_num_t1 = None
    if ta_t is not None and ca_t is not None and ppe_t is not None:
        aqi_num_t = 1.0 - _safe_div((ca_t + ppe_t), ta_t)
    if ta_t1 is not None and ca_t1 is not None and ppe_t1 is not None:
        aqi_num_t1 = 1.0 - _safe_div((ca_t1 + ppe_t1), ta_t1)
    aqi = _safe_div(aqi_num_t, aqi_num_t1)

    # SGI = Sales_t / Sales_t1
    sgi = _safe_div(sales_t, sales_t1)

    # DEPI = (DepRate)_t1 / (DepRate)_t, DepRate = Dep/(Dep+PPE)
    dep_rate_t = None
    dep_rate_t1 = None
    if dep_t is not None and ppe_t is not None and (dep_t + ppe_t) != 0:
        dep_rate_t = dep_t / (dep_t + ppe_t)
    if dep_t1 is not None and ppe_t1 is not None and (dep_t1 + ppe_t1) != 0:
        dep_rate_t1 = dep_t1 / (dep_t1 + ppe_t1)
    depi = _safe_div(dep_rate_t1, dep_rate_t)

    # SGAI = (SGA/Sales)_t / (SGA/Sales)_t1
    sgai = _safe_div(_safe_div(sga_t, sales_t), _safe_div(sga_t1, sales_t1))

    # LVGI = (Debt/TA)_t / (Debt/TA)_t1
    lvgi = _safe_div(_safe_div(td_t, ta_t), _safe_div(td_t1, ta_t1))

    # TATA = (NI - CFO) / TA
    tata = _safe_div((None if (ni_t is None or cfo_t is None) else (ni_t - cfo_t)), ta_t)

    # Missing tracking (for indices)
    for k, v in [("dsri", dsri), ("gmi", gmi), ("aqi", aqi), ("sgi", sgi), ("depi", depi), ("sgai", sgai), ("lvgi", lvgi), ("tata", tata)]:
        if v is None:
            missing.append(k)

    # M-Score
    m_score = None
    if all(v is not None for v in [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]):
        m_score = (
                float(coef["intercept"]) +
                float(coef["dsri"]) * dsri +
                float(coef["gmi"]) * gmi +
                float(coef["aqi"]) * aqi +
                float(coef["sgi"]) * sgi +
                float(coef["depi"]) * depi +
                float(coef["sgai"]) * sgai +
                float(coef["lvgi"]) * lvgi +
                float(coef["tata"]) * tata
        )

    # Convert to goodness score (config-based)
    score_0_1 = 0.0
    if m_score is None:
        rating = "Insufficient Data"
    else:
        if m_score <= threshold:
            score_0_1 = 1.0
        else:
            # degrade linearly above threshold
            score_0_1 = clamp01(1.0 - ((m_score - threshold) / risk_range))
        rating = rating_from_score(score_0_1)

    # One check: is m_score <= threshold ?
    checks.append(build_check(
        key="beneish_m_score_threshold",
        value=m_score,
        threshold=threshold,
        weight=1.0,
        comparator="<=",
        note="Beneish: m_score <= threshold gilt als niedrigeres Manipulationsrisiko.",
        missing=missing,
    ))

    thresholds_used = {
        "beneish_threshold": threshold,
        "beneish_risk_range": risk_range,
        "beneish_coefficients": coef,
    }
    metrics = {
        "m_score": m_score,
        "dsri": dsri,
        "gmi": gmi,
        "aqi": aqi,
        "sgi": sgi,
        "depi": depi,
        "sgai": sgai,
        "lvgi": lvgi,
        "tata": tata,
    }

    res = {
        "method": "Beneish M-Score (Penalty)",
        "score": float(score_0_1),
        "rating": rating,
        "thresholds_used": thresholds_used,
        "metrics": metrics,
        "checks": checks,
        "missing_data": sorted(list(set(missing))),
    }
    return res



def montier_penalty(
        income: Dict[str, Any],
        balance: Dict[str, Any],
        cashflow: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Montier C-Score (vereinfachte Red-Flag Logik; transparenter als "Blackbox").

    Implementierte Flags (konfigurierbar):
    1) CFO < Net Income (Earnings quality)
    2) Receivables grow faster than Sales (spread threshold)
    3) Inventory grow faster than Sales (spread threshold)
    4) Gross margin declines (spread threshold)
    5) Leverage increases (spread threshold)
    6) Share dilution (shares increased)

    Outputs:
    - metrics include c_score and red_flags
    - score is 0..1 (fewer flags => higher score)
    """
    qcfg = get_quality_risk_config()["montier"]
    max_flags = int(qcfg["max_flags"])
    th = qcfg["thresholds"]

    inc_t, inc_t1 = _pick_two_periods(income)
    bal_t, bal_t1 = _pick_two_periods(balance)
    cf_t, cf_t1 = _pick_two_periods(cashflow)

    missing: List[str] = []
    checks: List[Dict[str, Any]] = []
    red_flags: List[str] = []

    sales_t = _get_num(inc_t, ["revenue", "totalRevenue", "sales"])
    sales_t1 = _get_num(inc_t1, ["revenue", "totalRevenue", "sales"])

    ni_t = _get_num(inc_t, ["net_income", "netIncome"])
    cfo_t = _get_num(cf_t, ["operating_cash_flow", "operatingCashflow", "cashflowFromOperations"])

    ar_t = _get_num(bal_t, ["net_receivables", "netReceivables", "accountsReceivable"])
    ar_t1 = _get_num(bal_t1, ["net_receivables", "netReceivables", "accountsReceivable"])

    inv_t = _get_num(bal_t, ["inventory", "totalInventory"])
    inv_t1 = _get_num(bal_t1, ["inventory", "totalInventory"])

    cogs_t = _get_num(inc_t, ["cost_of_revenue", "costOfRevenue", "cogs"])
    cogs_t1 = _get_num(inc_t1, ["cost_of_revenue", "costOfRevenue", "cogs"])

    ta_t = _get_num(bal_t, ["total_assets", "totalAssets"])
    ta_t1 = _get_num(bal_t1, ["total_assets", "totalAssets"])

    td_t = _get_num(bal_t, ["total_debt", "totalDebt", "shortLongTermDebtTotal", "liabilities"])
    td_t1 = _get_num(bal_t1, ["total_debt", "totalDebt", "shortLongTermDebtTotal", "liabilities"])

    so_t = _get_num(inc_t, ["shares_outstanding", "commonStockSharesOutstanding", "weightedAverageShsOut"])
    if so_t is None:
        so_t = _get_num(bal_t, ["shares_outstanding", "commonStockSharesOutstanding"])
    so_t1 = _get_num(inc_t1, ["shares_outstanding", "commonStockSharesOutstanding", "weightedAverageShsOut"])
    if so_t1 is None:
        so_t1 = _get_num(bal_t1, ["shares_outstanding", "commonStockSharesOutstanding"])

    def _growth(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0:
            return None
        return (a - b) / b

    sales_g = _growth(sales_t, sales_t1)
    ar_g = _growth(ar_t, ar_t1)
    inv_g = _growth(inv_t, inv_t1)

    # Gross margin trend
    gm_t = None
    gm_t1 = None
    if sales_t is not None and cogs_t is not None:
        gm_t = _safe_div((sales_t - cogs_t), sales_t)
    if sales_t1 is not None and cogs_t1 is not None:
        gm_t1 = _safe_div((sales_t1 - cogs_t1), sales_t1)
    d_gm = None if (gm_t is None or gm_t1 is None) else (gm_t - gm_t1)

    # Leverage trend
    lev_t = _safe_div(td_t, ta_t)
    lev_t1 = _safe_div(td_t1, ta_t1)
    d_lev = None if (lev_t is None or lev_t1 is None) else (lev_t - lev_t1)

    # 1) CFO < NI
    if bool(th.get("cfo_below_ni", True)):
        flag_val = None
        if cfo_t is not None and ni_t is not None:
            flag_val = ni_t - cfo_t  # positive means CFO < NI
        chk = build_check(
            key="montier_cfo_vs_ni",
            value=flag_val,
            threshold=0.0,
            weight=1.0,
            comparator=">",
            note="Red Flag wenn NI - CFO > 0 (CFO < NI).",
            missing=missing,
        )
        checks.append(chk)
        if chk["pass"] is True:
            red_flags.append("CFO < Net Income")

    # 2) Receivables grow faster than sales by spread
    spread_ar = safe_float(th.get("receivables_vs_sales_growth_spread"))
    if spread_ar is not None:
        val = None
        if ar_g is not None and sales_g is not None:
            val = ar_g - sales_g
        chk = build_check(
            key="montier_receivables_vs_sales_growth",
            value=val,
            threshold=spread_ar,
            weight=1.0,
            comparator=">",
            note="Red Flag wenn Receivables-Wachstum deutlich über Sales-Wachstum liegt.",
            missing=missing,
        )
        checks.append(chk)
        if chk["pass"] is True:
            red_flags.append("Receivables growth > Sales growth")

    # 3) Inventory grow faster than sales by spread
    spread_inv = safe_float(th.get("inventory_vs_sales_growth_spread"))
    if spread_inv is not None:
        val = None
        if inv_g is not None and sales_g is not None:
            val = inv_g - sales_g
        chk = build_check(
            key="montier_inventory_vs_sales_growth",
            value=val,
            threshold=spread_inv,
            weight=1.0,
            comparator=">",
            note="Red Flag wenn Inventory-Wachstum deutlich über Sales-Wachstum liegt.",
            missing=missing,
        )
        checks.append(chk)
        if chk["pass"] is True:
            red_flags.append("Inventory growth > Sales growth")

    # 4) Gross margin decline
    gm_decline = safe_float(th.get("gross_margin_decline_spread"))
    if gm_decline is not None:
        # flag if d_gm < -gm_decline => use transformed value (-d_gm) > gm_decline
        val = None if d_gm is None else (-d_gm)
        chk = build_check(
            key="montier_gross_margin_decline",
            value=val,
            threshold=gm_decline,
            weight=1.0,
            comparator=">",
            note="Red Flag wenn Gross Margin fällt (über Schwelle).",
            missing=missing,
        )
        checks.append(chk)
        if chk["pass"] is True:
            red_flags.append("Gross margin declining")

    # 5) Leverage increase
    lev_inc = safe_float(th.get("leverage_increase_spread"))
    if lev_inc is not None:
        chk = build_check(
            key="montier_leverage_increase",
            value=d_lev,
            threshold=lev_inc,
            weight=1.0,
            comparator=">",
            note="Red Flag wenn Leverage zunimmt (über Schwelle).",
            missing=missing,
        )
        checks.append(chk)
        if chk["pass"] is True:
            red_flags.append("Leverage increasing")

    # 6) Share dilution
    if bool(th.get("share_dilution", True)):
        val = None
        if so_t is not None and so_t1 is not None:
            val = so_t - so_t1  # positive means shares increased => flag
        chk = build_check(
            key="montier_share_dilution",
            value=val,
            threshold=0.0,
            weight=1.0,
            comparator=">",
            note="Red Flag wenn Shares Outstanding steigen (Dilution).",
            missing=missing,
        )
        checks.append(chk)
        if chk["pass"] is True:
            red_flags.append("Share dilution")

    c_score = int(len(red_flags))
    score_0_1 = clamp01(1.0 - (c_score / float(max_flags))) if max_flags > 0 else 0.0

    thresholds_used = {
        "montier_max_flags": max_flags,
        "montier_thresholds": dict(th),
    }
    metrics = {
        "c_score": c_score,
        "red_flags": red_flags,
        "sales_growth": sales_g,
        "receivables_growth": ar_g,
        "inventory_growth": inv_g,
        "delta_gross_margin": d_gm,
        "delta_leverage": d_lev,
        "net_income": ni_t,
        "operating_cash_flow": cfo_t,
        "shares_outstanding_t": so_t,
        "shares_outstanding_t1": so_t1,
    }

    rating = rating_from_score(score_0_1) if len(checks) >= 3 else "Insufficient Data"

    return {
        "method": "Montier C-Score (Penalty)",
        "score": float(score_0_1),
        "rating": rating,
        "thresholds_used": thresholds_used,
        "metrics": metrics,
        "checks": checks,
        "missing_data": sorted(list(set(missing))),
    }


# ------------------------------------------------------------------------------
# Optional convenience: run a default “bundle” for one symbol (engine-side)
# ------------------------------------------------------------------------------

def run_default_value_bundle(
        fundamentals: Dict[str, Any],
        growth: Dict[str, Any],
        income: Dict[str, Any],
        balance: Dict[str, Any],
        cashflow: Dict[str, Any],
        *,
        sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Berechnet das vollständige Default-Value-Bundle für ein Symbol.

    Enthaltene Module:
    - Klassische Value-/Quality-Strategien (Graham, Buffett, Greenblatt, Lynch, etc.)
    - Accounting-/Fraud-Checks (Piotroski F-Score, Beneish M-Score, Montier C-Score)
    - Gewichtete, konfigurationsbasierte Aggregation (Phase-2-kalibrierbar)

    Erwartete Inputs:
    - fundamentals : Snapshot-Kennzahlen (PE, PB, ROE, ROIC, Debt/Equity, etc.)
    - growth       : Growth-Kennzahlen (EPS Growth 5y, etc.)
    - income       : Income Statements (mind. current & previous year)
    - balance      : Balance Sheets (mind. current & previous year)
    - cashflow     : Cashflow Statements (mind. current & previous year)
    - sector       : Optionaler Sektor (für Threshold- & Gewicht-Overrides)
    """

    # --- Snapshot-basierte Value / Quality Strategien ---
    g = graham_strategy(fundamentals, sector=sector)
    b = buffett_strategy(fundamentals, sector=sector)
    gr = greenblatt_strategy(fundamentals, sector=sector)
    l = lynch_strategy(fundamentals, growth, sector=sector)
    k = klarman_strategy(fundamentals, sector=sector)
    m = munger_strategy(fundamentals, sector=sector)
    s = schloss_strategy(fundamentals, sector=sector)
    d = davis_strategy(fundamentals, sector=sector)
    t = templeton_strategy(fundamentals, sector=sector)

    # --- Periodenbasierte Quality / Risk / Fraud Module ---
    p = piotroski_f_score(income, balance, cashflow, sector=sector)
    bm = beneish_penalty(income, balance, cashflow, sector=sector)
    mc = montier_penalty(income, balance, cashflow, sector=sector)

    # --- Gewichtete, konfigurationsbasierte Aggregation ---
    combined = combined_value_score(
        g, b, gr, l, k, m, s, d, t, p, bm, mc,
        sector=sector
    )

    return {
        "graham": g,
        "buffett": b,
        "greenblatt": gr,
        "lynch": l,
        "klarman": k,
        "munger": m,
        "schloss": s,
        "davis": d,
        "templeton": t,
        "piotroski": p,
        "beneish": bm,
        "montier": mc,
        "combined": combined,
    }


