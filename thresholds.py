# thresholds.py

"""Zielbild für 1.1: “Thresholds als Konfiguration” (ohne Magic Numbers)
Eine zentrale, versionierbare Konfiguration, die:
in allen Strategien verwendet wird,
optional Sektor/Industrie-Overrides unterstützt,
im Response immer dieselben Keys liefert (UI ohne Sonderfälle).
Empfehlung: thresholds.py (für schnelle Iteration) oder thresholds.json
(wenn man ohne Code deployed). Für den Stack ist thresholds.py pragmatisch."""


from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass(frozen=True)
class Thresholds:
    # Global / General
    min_market_cap: float = 0.0  # optional

    # Value
    pe_max: float = 20.0
    pb_max: float = 3.0
    ps_max: float = 3.0
    pfcf_max: float = 25.0

    # Growth / PEG
    peg_max: float = 2.0
    min_eps_growth_5y: float = 0.03  # 3%

    # Quality
    min_roe: float = 0.12            # 12%
    min_roic: float = 0.10           # 10%
    min_gross_margin: float = 0.30   # 30%

    # Leverage / Solvency
    debt_to_equity_max: float = 1.5
    net_debt_to_ebitda_max: float = 3.0
    interest_coverage_min: float = 4.0
    current_ratio_min: float = 1.2

    # Cashflow
    fcf_margin_min: float = 0.05     # 5%
    fcf_positive_required: bool = True

    # Dividend (optional)
    dividend_yield_full_score: float = 0.05  # 5% => cap

DEFAULT_THRESHOLDS = Thresholds()

# Optional: Overrides by sector or industry (keep minimal in 1.1; expand later)
SECTOR_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Example:
    # "Financial Services": {"debt_to_equity_max": 8.0, "net_debt_to_ebitda_max": 999.0}
}

def get_thresholds(sector: str | None = None) -> Dict[str, Any]:
    base = asdict(DEFAULT_THRESHOLDS)
    if sector and sector in SECTOR_OVERRIDES:
        base.update(SECTOR_OVERRIDES[sector])
    return base
"""
Warum diese Keys:
Sie sind stabil benennbar (für UI/Logs).
Sie decken die typischen Value/Quality/Leverage-Schalter ab.
Sie erlauben später leichtes Kalibrieren (2.x), ohne Code zu ändern.
"""


# Gewichte für Aggregation (Phase-2-kalibrierbar).
# Regeln:
# - Summe muss nicht 1.0 sein; normalisieren wir in combined_value_score.
# - "combined" ist kein Input, nur Output.
STRATEGY_WEIGHTS_DEFAULT = {
    "Graham (vereinfacht)": 1.0,
    "Buffett-Qualität (vereinfacht)": 1.2,
    "Greenblatt Magic Formula (vereinfacht)": 1.1,
    "Peter Lynch (vereinfacht)": 1.0,
    "Klarman (vereinfacht)": 1.1,
    "Munger (vereinfacht)": 1.2,
    "Schloss (vereinfacht)": 0.9,
    "Davis (vereinfacht)": 1.0,
    "Templeton (vereinfacht)": 0.9,

    "Piotroski F-Score": 1.1,
    "Beneish M-Score (Penalty)": 0.9,
    "Montier C-Score (Penalty)": 0.9,

}

# Abwertung, wenn eine Strategie "Insufficient Data" liefert (anstatt Magic Number im Code).
INSUFFICIENT_DATA_WEIGHT_FACTOR_DEFAULT = 0.25

# Optional: sektorabhängige Gewichtung (z.B. Financials anders behandeln).
# Struktur: { "<Sector>": {"weights": {...}, "insufficient_factor": 0.25 } }
STRATEGY_WEIGHT_OVERRIDES_BY_SECTOR = {
    # "Financial Services": {
    #     "weights": {
    #         "Graham (vereinfacht)": 1.0,
    #         "Buffett-Qualität (vereinfacht)": 1.1,
    #         "Greenblatt Magic Formula (vereinfacht)": 1.0,
    #         "Peter Lynch (vereinfacht)": 0.9,
    #         "Klarman (vereinfacht)": 1.0,
    #         "Munger (vereinfacht)": 1.1,
    #         "Schloss (vereinfacht)": 1.0,
    #         "Davis (vereinfacht)": 1.1,
    #         "Templeton (vereinfacht)": 1.0,
    #     },
    #     "insufficient_factor": 0.25,
    # }
}

def get_strategy_aggregation_config(sector: str | None = None) -> dict:
    """
    Liefert Aggregations-Konfiguration rein aus config:
    - weights: Dict[str, float] per Strategy-Method-Name
    - insufficient_factor: float
    """
    weights = dict(STRATEGY_WEIGHTS_DEFAULT)
    insufficient_factor = float(INSUFFICIENT_DATA_WEIGHT_FACTOR_DEFAULT)

    if sector and sector in STRATEGY_WEIGHT_OVERRIDES_BY_SECTOR:
        ov = STRATEGY_WEIGHT_OVERRIDES_BY_SECTOR[sector]
        if isinstance(ov.get("weights"), dict):
            weights.update(ov["weights"])
        if ov.get("insufficient_factor") is not None:
            insufficient_factor = float(ov["insufficient_factor"])

    return {"weights": weights, "insufficient_factor": insufficient_factor}

# ------------------------------------------------------------------------------
# 1.2 Quality & Risk Models (Config only; no Magic Numbers in strategies.py)
# ------------------------------------------------------------------------------

# Beneish M-Score coefficients (1999 8-variable model).
# Keep as config so calibration/experimentation is explicit/versioned.
BENEISH_COEFFICIENTS = {
    "intercept": -4.84,
    "dsri": 0.920,
    "gmi": 0.528,
    "aqi": 0.404,
    "sgi": 0.892,
    "depi": 0.115,
    "sgai": -0.172,
    "lvgi": 4.679,
    "tata": -0.327,
}

# Standard heuristic threshold: higher than this => higher manipulation risk.
BENEISH_THRESHOLD = -2.22

# Map M-Score to a 0..1 "goodness" score for aggregation.
# If m_score <= threshold: score=1. If above threshold: linearly down to 0 with range.
BENEISH_RISK_RANGE = 2.0  # how quickly the score degrades above threshold (calibratable)

# Piotroski F-Score normalization (0..9)
PIOTROSKI_MAX_SCORE = 9

# Montier simplified red-flag model configuration
MONTIER_MAX_FLAGS = 6  # number of implemented rules; must match strategy implementation
MONTIER_THRESHOLDS = {
    # receivables growth vs sales growth (flag if receivables grow faster)
    "receivables_vs_sales_growth_spread": 0.05,  # 5% spread
    # inventory growth vs sales growth
    "inventory_vs_sales_growth_spread": 0.05,
    # leverage increase tolerance
    "leverage_increase_spread": 0.02,
    # gross margin decline threshold
    "gross_margin_decline_spread": 0.01,
    # CFO vs NI: flag if CFO < NI
    "cfo_below_ni": True,
    # share dilution: flag if shares increased
    "share_dilution": True,
}

def get_quality_risk_config() -> dict:
    """
    Central access point for 1.2 configs.
    """
    return {
        "beneish": {
            "coefficients": dict(BENEISH_COEFFICIENTS),
            "threshold": float(BENEISH_THRESHOLD),
            "risk_range": float(BENEISH_RISK_RANGE),
        },
        "piotroski": {
            "max_score": int(PIOTROSKI_MAX_SCORE),
        },
        "montier": {
            "max_flags": int(MONTIER_MAX_FLAGS),
            "thresholds": dict(MONTIER_THRESHOLDS),
        },
    }
