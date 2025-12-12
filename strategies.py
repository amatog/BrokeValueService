from typing import Dict


def graham_valuation(f: Dict) -> Dict:
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0

    intrinsic_value = 15 * eps
    mos = 0.0
    if intrinsic_value != 0:
        mos = (intrinsic_value - price) / intrinsic_value

    mos_score = max(min((mos + 1) / 2, 1.0), 0.0)

    return {
        "strategy": "graham",
        "method": "Graham (vereinfacht)",
        "price": price,
        "eps": eps,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": mos,
        "is_undervalued": mos > 0.3,
        "score": mos_score,
    }


def buffett_quality(f: Dict) -> Dict:
    roe = f.get("roe") or 0.0

    debt_to_equity = f.get("debt_to_equity")
    if debt_to_equity is None:
        debt_to_equity = 0.5  # Fallback, falls Balance Sheet nichts hergibt

    growth = f.get("earnings_growth_5y")
    if growth is None:
        growth = 0.05  # Fallback, falls EARNINGS keine verwertbaren Daten liefern

    roe_score = min(roe / 0.20, 1.0)
    debt_score = 1.0 if debt_to_equity < 0.5 else 0.5
    growth_score = min(growth / 0.10, 1.0)

    quality_score = (roe_score + debt_score + growth_score) / 3

    return {
        "strategy": "buffett",
        "method": "Buffett-Qualität (vereinfacht)",
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "earnings_growth_5y": growth,
        "quality_score": quality_score,
        "is_high_quality": quality_score >= 0.7,
        "score": quality_score,
    }


def greenblatt_magic_formula(f: Dict) -> Dict:
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0
    roe = f.get("roe") or 0.0

    earnings_yield = 0.0
    if price:
        earnings_yield = eps / price

    return_on_capital = roe  # Platzhalter

    ey_score = min(earnings_yield / 0.10, 1.0)
    roc_score = min(return_on_capital / 0.20, 1.0)
    magic_score = (ey_score + roc_score) / 2

    return {
        "strategy": "greenblatt",
        "method": "Greenblatt Magic Formula (vereinfacht)",
        "earnings_yield": earnings_yield,
        "return_on_capital": return_on_capital,
        "magic_score": magic_score,
        "score": magic_score,
    }


def combined_value_score(
        g: Dict, b: Dict, gr: Dict, m: Dict, l: Dict, s: Dict, d: Dict, t: Dict, k: Dict, f: Dict
) -> Dict:
    def clamp01(x):
        try:
            x = float(x)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, x))

    # Einheitliche Inputs (0..1)
    components = {
        "graham": clamp01(g.get("score")),
        "buffett": clamp01(b.get("score")),
        "greenblatt": clamp01(gr.get("score")),
        "munger": clamp01(m.get("score")),
        "lynch": clamp01(l.get("score")),
        "schloss": clamp01(s.get("score")),
        "davis": clamp01(d.get("score")),
        "templeton": clamp01(t.get("score")),
        "klarman": clamp01(k.get("score")),
        "dividend": clamp01((max(float(f.get("dividend_yield") or 0.0), 0.0)) / 0.05)),  # 5% = voller Score
    }

    # Gewichte (Summe = 1.0) – bewusst ausgewogen, ohne Sonderlogik
    weights = {
    "graham": 0.12,
    "klarman": 0.10,
    "greenblatt": 0.14,
    "buffett": 0.10,
    "munger": 0.10,
    "lynch": 0.10,
    "schloss": 0.10,
    "davis": 0.10,
    "templeton": 0.07,
    "dividend": 0.07,
    }

    score = 0.0
    for name, w in weights.items():
        score += w * components.get(name, 0.0)
    score = clamp01(score)

    if score >= 0.80:
        level = "Sehr attraktiv"
    elif score >= 0.65:
        level = "Attraktiv"
    elif score >= 0.50:
        level = "Neutral"
    else:
        level = "Unattraktiv"

    return {
        "value_score": round(score, 4),              # 0..1 (stabil für Backend)
        "value_score_100": round(score * 100, 1),    # 0..100 (bequem für UI)
        "value_level": level,
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": weights,
    }



# ---------------------------------------------------
# Erweiterte Strategien
# ---------------------------------------------------

def munger_quality(f: Dict) -> Dict:
    """
    Charlie Munger – Fokus auf Qualität, hohe Kapitalrendite, wenig Schulden.
    Nutzt vorhandene Kennzahlen aus dem DataLayer.
    """
    roe = f.get("roe") or 0.0            # z.B. 0.18 = 18 %
    debt_to_equity = f.get("debt_to_equity")
    if debt_to_equity is None:
        debt_to_equity = 0.5

    growth = f.get("earnings_growth_5y")
    if growth is None:
        growth = 0.05  # 5 % als Fallback

    # Scores 0..1
    roe_score = min(roe / 0.20, 1.0)              # >= 20 % = voller Score
    debt_score = 1.0 if debt_to_equity <= 0.5 else max(0.0, 1.0 - (debt_to_equity - 0.5))
    growth_score = min(growth / 0.10, 1.0)        # 10 % Wachstum = voller Score

    quality_score = (0.5 * roe_score + 0.3 * growth_score + 0.2 * debt_score)

    return {
        "strategy": "munger",
        "method": "Munger-Qualität (vereinfacht)",
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "earnings_growth_5y": growth,
        "quality_score": quality_score,
        "is_high_quality": quality_score >= 0.7,
        "score": quality_score,
    }


def lynch_growth_value(f: Dict) -> Dict:
    """
    Peter Lynch – Wachstum zu vernünftigem Preis (PEG).
    earnings_growth_5y kommt als Rate (z.B. 0.12 = 12 %).
    """
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0
    growth = f.get("earnings_growth_5y")  # Rate, z.B. 0.15 = 15 %

    pe = 0.0
    if eps > 0:
        pe = price / eps

    peg = None
    growth_pct = None
    if growth is not None and growth > 0:
        growth_pct = growth * 100.0
        if growth_pct > 0:
            peg = pe / growth_pct

    # Bewertung: PEG nahe 1 ist ideal
    peg_score = 0.0
    if peg is not None:
        if peg <= 1.0:
            peg_score = 1.0
        elif peg <= 1.5:
            peg_score = 0.7
        elif peg <= 2.0:
            peg_score = 0.4
        else:
            peg_score = 0.1

    growth_score = 0.0
    if growth is not None:
        if growth >= 0.15:
            growth_score = 1.0
        elif growth >= 0.10:
            growth_score = 0.7
        elif growth >= 0.05:
            growth_score = 0.4
        else:
            growth_score = 0.1

    debt_to_equity = f.get("debt_to_equity")
    if debt_to_equity is None:
        debt_to_equity = 0.5

    if debt_to_equity <= 0.5:
        debt_score = 1.0
    elif debt_to_equity <= 1.0:
        debt_score = 0.6
    else:
        debt_score = 0.2

    lynch_score = 0.5 * peg_score + 0.3 * growth_score + 0.2 * debt_score

    return {
        "strategy": "lynch",
        "method": "Lynch – Growth at a Reasonable Price (vereinfacht)",
        "price": price,
        "eps": eps,
        "pe": pe,
        "earnings_growth_5y": growth,
        "growth_pct": growth_pct,
        "peg": peg,
        "debt_to_equity": debt_to_equity,
        "score": lynch_score,
    }


def schloss_deep_value(f: Dict) -> Dict:
    """
    Walter Schloss – klassischer Deep-Value-Ansatz: niedriges Kurs/Buchwert, wenig Schulden.
    """
    price = f.get("price") or 0.0
    book_value = f.get("book_value") or 0.0

    p_b = None
    if book_value > 0:
        p_b = price / book_value

    debt_to_equity = f.get("debt_to_equity")
    if debt_to_equity is None:
        debt_to_equity = 0.5

    pb_score = 0.0
    if p_b is not None:
        if p_b <= 0.7:
            pb_score = 1.0
        elif p_b <= 1.0:
            pb_score = 0.7
        elif p_b <= 1.5:
            pb_score = 0.4
        else:
            pb_score = 0.1

    if debt_to_equity <= 0.3:
        debt_score = 1.0
    elif debt_to_equity <= 0.6:
        debt_score = 0.6
    else:
        debt_score = 0.2

    schloss_score = 0.7 * pb_score + 0.3 * debt_score

    return {
        "strategy": "schloss",
        "method": "Schloss – Deep Value (vereinfacht)",
        "price": price,
        "book_value": book_value,
        "price_to_book": p_b,
        "debt_to_equity": debt_to_equity,
        "score": schloss_score,
    }


def davis_growth_quality(f: Dict) -> Dict:
    """
    Shelby Davis – Fokus auf vernünftige Bewertung, Wachstum und Qualität.
    """
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0
    roe = f.get("roe") or 0.0
    growth = f.get("earnings_growth_5y")
    if growth is None:
        growth = 0.05

    pe = 0.0
    if eps > 0:
        pe = price / eps

    # P/E im Bereich 8–15 ist ideal
    if 8 <= pe <= 15:
        pe_score = 1.0
    elif 6 <= pe <= 20:
        pe_score = 0.7
    else:
        pe_score = 0.3

    if growth >= 0.12:
        growth_score = 1.0
    elif growth >= 0.08:
        growth_score = 0.7
    elif growth >= 0.04:
        growth_score = 0.4
    else:
        growth_score = 0.1

    if roe >= 0.15:
        roe_score = 1.0
    elif roe >= 0.10:
        roe_score = 0.7
    else:
        roe_score = 0.3

    dividend_yield = f.get("dividend_yield")
    if dividend_yield is None:
        dividend_yield = 0.0

    # moderate Dividende bis ca. 4 % bevorzugt
    if 0 < dividend_yield <= 0.04:
        dividend_score = 1.0
    elif dividend_yield == 0:
        dividend_score = 0.5
    else:
        dividend_score = 0.6  # sehr hohe Dividende kann Risiko anzeigen

    davis_score = (
            0.3 * pe_score +
            0.3 * growth_score +
            0.25 * roe_score +
            0.15 * dividend_score
    )

    return {
        "strategy": "davis",
        "method": "Davis – Wachstum & Qualität (vereinfacht)",
        "price": price,
        "eps": eps,
        "pe": pe,
        "roe": roe,
        "earnings_growth_5y": growth,
        "dividend_yield": dividend_yield,
        "score": davis_score,
    }


def templeton_contrarian_value(f: Dict) -> Dict:
    """
    John Templeton – contrarian/value-orientiert.
    Hier approximieren wir den Markt-P/E mit 15 und schauen, ob das Unternehmen deutlich darunter notiert.
    """
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0
    book_value = f.get("book_value") or 0.0
    debt_to_equity = f.get("debt_to_equity")
    if debt_to_equity is None:
        debt_to_equity = 0.5

    pe = 0.0
    if eps > 0:
        pe = price / eps

    p_b = None
    if book_value > 0:
        p_b = price / book_value

    MARKET_PE_APPROX = 15.0
    relative_pe = None
    if MARKET_PE_APPROX > 0:
        relative_pe = pe / MARKET_PE_APPROX

    pe_score = 0.0
    if relative_pe is not None:
        if relative_pe <= 0.6:
            pe_score = 1.0
        elif relative_pe <= 0.8:
            pe_score = 0.7
        elif relative_pe <= 1.0:
            pe_score = 0.4
        else:
            pe_score = 0.1

    pb_score = 0.0
    if p_b is not None:
        if p_b <= 1.0:
            pb_score = 1.0
        elif p_b <= 1.5:
            pb_score = 0.7
        else:
            pb_score = 0.3

    if debt_to_equity <= 0.5:
        debt_score = 1.0
    elif debt_to_equity <= 1.0:
        debt_score = 0.6
    else:
        debt_score = 0.2

    templeton_score = 0.5 * pe_score + 0.3 * pb_score + 0.2 * debt_score

    return {
        "strategy": "templeton",
        "method": "Templeton – Contrarian Value (vereinfacht)",
        "price": price,
        "eps": eps,
        "pe": pe,
        "relative_pe_vs_market15": relative_pe,
        "book_value": book_value,
        "price_to_book": p_b,
        "debt_to_equity": debt_to_equity,
        "score": templeton_score,
    }


def klarman_margin_of_safety(f: Dict) -> Dict:
    """
    Seth Klarman – starker Fokus auf Margin of Safety.
    Wir verwenden eine vereinfachte Graham-intrinsic-Value-Formel und leiten daraus die Sicherheitsmarge ab.
    """
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0

    intrinsic_value = 15 * eps
    mos = None
    if intrinsic_value > 0:
        mos = (intrinsic_value - price) / intrinsic_value

    # Score anhand Margin of Safety
    mos_score = 0.0
    if mos is not None:
        if mos >= 0.4:
            mos_score = 1.0
        elif mos >= 0.25:
            mos_score = 0.7
        elif mos >= 0.15:
            mos_score = 0.4
        elif mos >= 0.0:
            mos_score = 0.2
        else:
            mos_score = 0.0

    debt_to_equity = f.get("debt_to_equity")
    if debt_to_equity is None:
        debt_to_equity = 0.5

    if debt_to_equity <= 0.5:
        debt_score = 1.0
    elif debt_to_equity <= 1.0:
        debt_score = 0.6
    else:
        debt_score = 0.2

    klarman_score = 0.7 * mos_score + 0.3 * debt_score

    return {
        "strategy": "klarman",
        "method": "Klarman – Margin of Safety (vereinfacht)",
        "price": price,
        "eps": eps,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": mos,
        "debt_to_equity": debt_to_equity,
        "score": klarman_score,
    }

from typing import Optional, List, Any

def piotroski_f_score(cur: Dict, prev: Dict) -> Dict:
    # Wenn Daten fehlen: stabiler Output, UI ohne Sonderfälle
    required = [
        "net_income", "total_assets", "operating_cash_flow",
        "long_term_debt", "current_assets", "current_liabilities",
        "shares_outstanding", "gross_profit", "revenue",
    ]
    available = all(cur.get(k) is not None and prev.get(k) is not None for k in required)

    def safe_div(a, b):
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)

    if not available:
        return {
            "available": False,
            "f_score": None,
            "quality_score": 0.0,
            "passes_gate": True,   # wichtig: kein Fake-REJECT wenn Daten fehlen
            "bucket": "UNKNOWN",
            "signals": {},
            "missing": [k for k in required if cur.get(k) is None or prev.get(k) is None],
        }

    roa_cur = safe_div(cur["net_income"], cur["total_assets"])
    roa_prev = safe_div(prev["net_income"], prev["total_assets"])

    leverage_cur = safe_div(cur["long_term_debt"], cur["total_assets"])
    leverage_prev = safe_div(prev["long_term_debt"], prev["total_assets"])

    cr_cur = safe_div(cur["current_assets"], cur["current_liabilities"])
    cr_prev = safe_div(prev["current_assets"], prev["current_liabilities"])

    gm_cur = safe_div(cur["gross_profit"], cur["revenue"])
    gm_prev = safe_div(prev["gross_profit"], prev["revenue"])

    at_cur = safe_div(cur["revenue"], cur["total_assets"])
    at_prev = safe_div(prev["revenue"], prev["total_assets"])

    signals = {
        # Profitability
        "roa_pos": 1 if (roa_cur is not None and roa_cur > 0) else 0,
        "cfo_pos": 1 if (cur["operating_cash_flow"] > 0) else 0,
        "delta_roa_pos": 1 if (roa_cur is not None and roa_prev is not None and roa_cur > roa_prev) else 0,
        "accruals_cfo_gt_ni": 1 if (cur["operating_cash_flow"] > cur["net_income"]) else 0,
        # Leverage/Liquidity
        "delta_leverage_neg": 1 if (leverage_cur is not None and leverage_prev is not None and leverage_cur < leverage_prev) else 0,
        "delta_current_ratio_pos": 1 if (cr_cur is not None and cr_prev is not None and cr_cur > cr_prev) else 0,
        "no_dilution": 1 if (cur["shares_outstanding"] <= prev["shares_outstanding"]) else 0,
        # Operating efficiency
        "delta_gross_margin_pos": 1 if (gm_cur is not None and gm_prev is not None and gm_cur > gm_prev) else 0,
        "delta_asset_turnover_pos": 1 if (at_cur is not None and at_prev is not None and at_cur > at_prev) else 0,
    }

    f_score = int(sum(signals.values()))
    quality_score = round((f_score / 9.0) * 100.0, 2)

    rejected = f_score <= 3
    if rejected:
        bucket = "REJECT"
    elif f_score >= 7:
        bucket = "PASS"
    else:
        bucket = "WEAK_MEDIUM"

    return {
        "available": True,
        "f_score": f_score,
        "quality_score": quality_score,
        "passes_gate": not rejected,
        "bucket": bucket,
        "signals": signals,
        "missing": [],
    }


def beneish_penalty(m_score: Optional[float]) -> Dict:
    # Profil A: NIE reject – nur Warnung + Penalty
    if m_score is None:
        return {"available": False, "m_score": None, "warning": False, "severity": "NONE", "penalty": 0}

    if m_score <= -1.78:
        return {"available": True, "m_score": m_score, "warning": False, "severity": "NONE", "penalty": 0}
    if m_score <= -1.50:
        return {"available": True, "m_score": m_score, "warning": True, "severity": "MEDIUM", "penalty": 10}
    if m_score <= -1.00:
        return {"available": True, "m_score": m_score, "warning": True, "severity": "HIGH", "penalty": 15}
    return {"available": True, "m_score": m_score, "warning": True, "severity": "HIGH", "penalty": 20}


def montier_penalty(c_score: Optional[int], red_flags: Optional[List[str]] = None) -> Dict:
    red_flags = red_flags or []
    if c_score is None:
        return {"available": False, "c_score": None, "warning": False, "severity": "NONE", "penalty": 0, "red_flags": red_flags}

    c_score = int(c_score)
    if c_score <= 1:
        return {"available": True, "c_score": c_score, "warning": False, "severity": "NONE", "penalty": 0, "red_flags": red_flags}
    if c_score == 2:
        return {"available": True, "c_score": c_score, "warning": True, "severity": "MEDIUM", "penalty": 5, "red_flags": red_flags}
    return {"available": True, "c_score": c_score, "warning": True, "severity": "HIGH", "penalty": 10, "red_flags": red_flags}


def finalize_assessment(value_score_01: float, piotroski: Dict, beneish: Dict, montier: Dict) -> Dict:
    """
    Profil A, 70/30:
      - Gate: Piotroski F <= 3 => REJECT (nur wenn piotroski available)
      - FinalScore = 0.70*Value(0..100) + 0.30*Quality(0..100) - Penalty(0..30)
    """
    # stabile Defaults
    value100 = max(0.0, min(100.0, float(value_score_01) * 100.0))

    # Gate nur anwenden, wenn Piotroski verfügbar ist
    rejected = False
    rejection_reason = None
    if piotroski.get("available") and not piotroski.get("passes_gate", True):
        rejected = True
        rejection_reason = "Piotroski F-Score <= 3 (Value Trap Risk)"

    quality100 = float(piotroski.get("quality_score") or 0.0)  # 0..100
    penalty = int((beneish.get("penalty") or 0) + (montier.get("penalty") or 0))

    if rejected:
        final100 = 0.0
        decision = "REJECT"
    else:
        final100 = (0.70 * value100) + (0.30 * quality100) - float(penalty)
        final100 = max(0.0, min(100.0, final100))

        if final100 >= 80:
            decision = "BUY"
        elif final100 >= 65:
            decision = "WATCHLIST"
        elif final100 >= 50:
            decision = "HOLD"
        else:
            decision = "AVOID"

    # UI Badges ohne Sonderfälle
    badges = []

    # Quality badge
    if not piotroski.get("available"):
        badges.append({"type": "QUALITY", "label": "Quality Check unavailable", "severity": "INFO"})
    else:
        f = piotroski.get("f_score")
        if rejected:
            badges.append({"type": "QUALITY", "label": "Rejected: Value Trap Risk", "severity": "HIGH"})
        elif f is not None and f >= 7:
            badges.append({"type": "QUALITY", "label": "Quality Pass", "severity": "INFO"})
        else:
            badges.append({"type": "QUALITY", "label": "Quality Weak/Medium", "severity": "MEDIUM"})

    if beneish.get("warning"):
        badges.append({"type": "ACCOUNTING", "label": "Accounting Risk", "severity": beneish.get("severity", "MEDIUM")})
    if montier.get("warning"):
        badges.append({"type": "FORENSICS", "label": "Cooking Risk", "severity": montier.get("severity", "MEDIUM")})

    return {
        "decision": decision,
        "final_score_100": round(final100, 1),
        "rejected": rejected,
        "rejection_reason": rejection_reason,
        "penalty_total": penalty,
        "weights": {"value": 0.7, "quality": 0.3},
        "formula": "final=0.70*value(0..100) + 0.30*quality(0..100) - penalty(0..30)",
        "badges": badges,
    }

