from typing import Dict


def graham_valuation(f: Dict) -> Dict:
    price = f.get("price") or 0.0
    eps = f.get("eps") or 0.0

    intrinsic_value = 15 * eps
    mos = 0.0
    if intrinsic_value != 0:
        mos = (intrinsic_value - price) / intrinsic_value

    return {
        "method": "Graham (vereinfacht)",
        "price": price,
        "eps": eps,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": mos,
        "is_undervalued": mos > 0.3,
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
        "method": "Buffett-Qualität (vereinfacht)",
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "earnings_growth_5y": growth,
        "quality_score": quality_score,
        "is_high_quality": quality_score >= 0.7,
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
        "method": "Greenblatt Magic Formula (vereinfacht)",
        "earnings_yield": earnings_yield,
        "return_on_capital": return_on_capital,
        "magic_score": magic_score,
    }


def combined_value_score(
    g: Dict,
    b: Dict,
    gr: Dict,
    m: Dict,
    l: Dict,
    s: Dict,
    d: Dict,
    t: Dict,
    k: Dict,
    f: Dict,
    financials: Dict | None = None,
) -> Dict:
    """
    Gewichtete Aggregation aller Value-Strategien plus Quality/Forensics.

    * Liefert stabile Keys: value, quality_forensics, final_assessment.
    * Breakdown für Debug/UI: jede Komponente mit Gewicht und Beitrag.
    * Piotroski-Gate nur, wenn genügend Daten vorhanden sind.
    """

    def _clamp_01(x: float | None) -> float | None:
        if x is None:
            return None
        return max(0.0, min(1.0, x))

    mos = g.get("margin_of_safety")
    mos_component = None
    if mos is not None:
        mos_component = _clamp_01((mos + 1) / 2)

    dividend = f.get("dividend_yield")
    dividend_component = None
    if dividend is not None:
        dividend_component = _clamp_01(dividend / 0.05)  # 5 % = voller Score

    value_components = {
        "graham_mos": (0.18, mos_component),
        "buffett_quality": (0.1, _clamp_01(b.get("quality_score"))),
        "greenblatt_magic": (0.1, _clamp_01(gr.get("magic_score"))),
        "munger_quality": (0.08, _clamp_01(m.get("quality_score"))),
        "lynch_growth": (0.1, _clamp_01(l.get("score"))),
        "schloss_value": (0.08, _clamp_01(s.get("score"))),
        "davis_growth_quality": (0.08, _clamp_01(d.get("score"))),
        "templeton_value": (0.08, _clamp_01(t.get("score"))),
        "klarman_mos": (0.1, _clamp_01(k.get("score"))),
        "dividend_yield": (0.1, dividend_component),
    }

    total_weight = sum(w for w, score in value_components.values() if score is not None)
    weighted_sum = sum(w * score for w, score in value_components.values() if score is not None)
    value_score = weighted_sum / total_weight if total_weight else 0.0

    if value_score >= 0.8:
        value_label = "Sehr attraktiv"
    elif value_score >= 0.6:
        value_label = "Attraktiv"
    elif value_score >= 0.4:
        value_label = "Neutral"
    else:
        value_label = "Unattraktiv"

    value_block = {
        "score": value_score,
        "label": value_label,
        "breakdown": {
            key: {
                "weight": weight,
                "score": score,
                "contribution": weight * score if score is not None else None,
            }
            for key, (weight, score) in value_components.items()
        },
    }

    quality_block = quality_forensics_assessment(financials)
    final_block = final_assessment(value_block, quality_block)

    return {
        "value": value_block,
        "quality_forensics": quality_block,
        "final_assessment": final_block,
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
        "method": "Munger-Qualität (vereinfacht)",
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "earnings_growth_5y": growth,
        "quality_score": quality_score,
        "is_high_quality": quality_score >= 0.7,
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
        "method": "Klarman – Margin of Safety (vereinfacht)",
        "price": price,
        "eps": eps,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": mos,
        "debt_to_equity": debt_to_equity,
        "score": klarman_score,
    }


# ---------------------------------------------------
# Quality / Forensics & Final Assessment
# ---------------------------------------------------


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator, denominator):
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den in (None, 0):
        return None
    return num / den


def _piotroski_f_score(financials: Dict) -> Dict:
    periods = (financials or {}).get("periods", [])
    if len(periods) < 2:
        return {
            "score": None,
            "available": False,
            "reason": "insufficient_periods",
        }

    current = periods[0]
    previous = periods[1]

    def get_period_metric(period: Dict, section: str, key: str):
        return _to_float(period.get(section, {}).get(key))

    net_income_t = get_period_metric(current, "income", "netIncome")
    net_income_t1 = get_period_metric(previous, "income", "netIncome")
    total_assets_t = get_period_metric(current, "balance", "totalAssets")
    total_assets_t1 = get_period_metric(previous, "balance", "totalAssets")
    ocf_t = get_period_metric(current, "cashflow", "operatingCashflow")
    long_term_debt_t = get_period_metric(current, "balance", "longTermDebt")
    long_term_debt_t1 = get_period_metric(previous, "balance", "longTermDebt")
    curr_assets_t = get_period_metric(current, "balance", "totalCurrentAssets")
    curr_assets_t1 = get_period_metric(previous, "balance", "totalCurrentAssets")
    curr_liab_t = get_period_metric(current, "balance", "totalCurrentLiabilities")
    curr_liab_t1 = get_period_metric(previous, "balance", "totalCurrentLiabilities")
    shares_t = get_period_metric(current, "balance", "commonStockSharesOutstanding")
    shares_t1 = get_period_metric(previous, "balance", "commonStockSharesOutstanding")
    gross_profit_t = get_period_metric(current, "income", "grossProfit")
    gross_profit_t1 = get_period_metric(previous, "income", "grossProfit")
    revenue_t = get_period_metric(current, "income", "totalRevenue")
    revenue_t1 = get_period_metric(previous, "income", "totalRevenue")

    roa_t = _safe_div(net_income_t, total_assets_t)
    roa_t1 = _safe_div(net_income_t1, total_assets_t1)
    cfo_positive = ocf_t is not None and ocf_t > 0
    accrual = None if ocf_t is None or net_income_t is None else ocf_t > net_income_t

    leverage_delta = None
    if long_term_debt_t is not None and long_term_debt_t1 is not None and total_assets_t is not None and total_assets_t1 is not None:
        leverage_delta = _safe_div(long_term_debt_t, total_assets_t) - _safe_div(long_term_debt_t1, total_assets_t1)

    current_ratio_delta = None
    if all(x is not None for x in (curr_assets_t, curr_assets_t1, curr_liab_t, curr_liab_t1)):
        current_ratio_delta = _safe_div(curr_assets_t, curr_liab_t) - _safe_div(curr_assets_t1, curr_liab_t1)

    asset_turnover_t = _safe_div(revenue_t, total_assets_t)
    asset_turnover_t1 = _safe_div(revenue_t1, total_assets_t1)

    gross_margin_t = _safe_div(gross_profit_t, revenue_t)
    gross_margin_t1 = _safe_div(gross_profit_t1, revenue_t1)

    indicators = {
        "roa_positive": roa_t is not None and roa_t > 0,
        "roa_improving": roa_t is not None and roa_t1 is not None and roa_t > roa_t1,
        "cfo_positive": cfo_positive,
        "accruals": accrual,
        "leverage_decrease": leverage_delta is not None and leverage_delta < 0,
        "liquidity_improving": current_ratio_delta is not None and current_ratio_delta > 0,
        "equity_no_dilution": shares_t is not None and shares_t1 is not None and shares_t <= shares_t1,
        "gross_margin_improving": gross_margin_t is not None and gross_margin_t1 is not None and gross_margin_t > gross_margin_t1,
        "asset_turnover_improving": asset_turnover_t is not None and asset_turnover_t1 is not None and asset_turnover_t > asset_turnover_t1,
    }

    available_signals = [val for val in indicators.values() if val is not None]
    if not available_signals:
        return {
            "score": None,
            "available": False,
            "reason": "missing_metrics",
        }

    score = sum(1 for val in available_signals if val)
    score_normalized = score / len(available_signals)

    return {
        "score_raw": score,
        "score": score_normalized,
        "available": True,
        "indicators": indicators,
    }


def _beneish_m_score(financials: Dict) -> Dict:
    periods = (financials or {}).get("periods", [])
    if len(periods) < 2:
        return {"score": None, "available": False, "reason": "insufficient_periods"}

    current = periods[0]
    previous = periods[1]

    def metric(period: Dict, section: str, key: str):
        return _to_float(period.get(section, {}).get(key))

    revenue_t = metric(current, "income", "totalRevenue")
    revenue_t1 = metric(previous, "income", "totalRevenue")
    receivables_t = metric(current, "balance", "netReceivables")
    receivables_t1 = metric(previous, "balance", "netReceivables")
    gross_profit_t = metric(current, "income", "grossProfit")
    gross_profit_t1 = metric(previous, "income", "grossProfit")
    total_assets_t = metric(current, "balance", "totalAssets")
    total_assets_t1 = metric(previous, "balance", "totalAssets")
    depreciation_t = metric(current, "cashflow", "depreciation")
    depreciation_t1 = metric(previous, "cashflow", "depreciation")
    sga_t = metric(current, "income", "sellingGeneralAndAdministrative")
    sga_t1 = metric(previous, "income", "sellingGeneralAndAdministrative")
    total_liab_t = metric(current, "balance", "totalLiabilities")
    total_liab_t1 = metric(previous, "balance", "totalLiabilities")
    cashflow_t = metric(current, "cashflow", "operatingCashflow")
    net_income_t = metric(current, "income", "netIncome")

    # Berechnungen mit möglichst vielen verfügbaren Kennzahlen
    dsri = None
    if receivables_t is not None and revenue_t is not None and receivables_t1 is not None and revenue_t1 is not None and revenue_t1 != 0:
        dsri = _safe_div(receivables_t / revenue_t, receivables_t1 / revenue_t1)

    gmi = None
    if all(x not in (None, 0) for x in (gross_profit_t1, revenue_t1, gross_profit_t, revenue_t)):
        gmi = _safe_div((revenue_t1 - gross_profit_t1) / revenue_t1, (revenue_t - gross_profit_t) / revenue_t)

    aqi = None
    if all(x is not None for x in (total_assets_t, total_assets_t1, revenue_t, revenue_t1)):
        aqi = _safe_div((total_assets_t - gross_profit_t) / total_assets_t if total_assets_t else None,
                        (total_assets_t1 - gross_profit_t1) / total_assets_t1 if total_assets_t1 else None)

    sgi = _safe_div(revenue_t, revenue_t1)

    depi = None
    if depreciation_t is not None and depreciation_t1 is not None and gross_profit_t is not None and gross_profit_t1 not in (None, 0):
        depi = _safe_div(depreciation_t1 / gross_profit_t1, depreciation_t / gross_profit_t)

    sgai = None
    if sga_t is not None and sga_t1 is not None and revenue_t not in (None, 0) and revenue_t1 not in (None, 0):
        sgai = _safe_div(sga_t / revenue_t, sga_t1 / revenue_t1)

    lvgi = None
    if all(x not in (None, 0) for x in (total_liab_t, total_liab_t1, total_assets_t, total_assets_t1)):
        lvgi = _safe_div(total_liab_t / total_assets_t, total_liab_t1 / total_assets_t1)

    tata = None
    if cashflow_t is not None and total_assets_t not in (None, 0) and net_income_t is not None:
        tata = (cashflow_t - net_income_t) / total_assets_t

    factors = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    available = [f for f in factors if f is not None]
    if len(available) < 3:
        return {"score": None, "available": False, "reason": "missing_metrics"}

    # Vereinfachte Gewichtung orientiert an Beneish (keine perfekten Koeffizienten, aber stabile Bewertung)
    m_score = (
        0.92 * (dsri or 0)
        + 0.528 * (gmi or 0)
        + 0.404 * (aqi or 0)
        + 0.892 * (sgi or 0)
        + 0.115 * (depi or 0)
        - 0.172 * (sgai or 0)
        + 4.679 * (tata or 0)
        - 0.327 * (lvgi or 0)
    )

    # Mapping zu 0..1: < -2.22 = sehr gut, > -1 = schwach
    if m_score <= -2.22:
        normalized = 1.0
    elif m_score <= -1.78:
        normalized = 0.7
    elif m_score <= -1.0:
        normalized = 0.4
    else:
        normalized = 0.1

    return {
        "score_raw": m_score,
        "score": normalized,
        "available": True,
        "factors": {
            "dsri": dsri,
            "gmi": gmi,
            "aqi": aqi,
            "sgi": sgi,
            "depi": depi,
            "sgai": sgai,
            "lvgi": lvgi,
            "tata": tata,
        },
    }


def _montier_c_score(financials: Dict) -> Dict:
    periods = (financials or {}).get("periods", [])
    if len(periods) < 2:
        return {"score": None, "available": False, "reason": "insufficient_periods"}

    current = periods[0]
    previous = periods[1]

    def metric(period: Dict, section: str, key: str):
        return _to_float(period.get(section, {}).get(key))

    net_income_t = metric(current, "income", "netIncome")
    cashflow_t = metric(current, "cashflow", "operatingCashflow")
    gross_profit_t = metric(current, "income", "grossProfit")
    gross_profit_t1 = metric(previous, "income", "grossProfit")
    revenue_t = metric(current, "income", "totalRevenue")
    revenue_t1 = metric(previous, "income", "totalRevenue")
    total_assets_t = metric(current, "balance", "totalAssets")
    total_assets_t1 = metric(previous, "balance", "totalAssets")
    inventory_t = metric(current, "balance", "inventory")
    inventory_t1 = metric(previous, "balance", "inventory")
    receivables_t = metric(current, "balance", "netReceivables")
    receivables_t1 = metric(previous, "balance", "netReceivables")

    c_score_flags = {
        "cash_vs_profit": net_income_t is not None and cashflow_t is not None and cashflow_t < net_income_t,
        "gaap_quality": gross_profit_t is not None and gross_profit_t1 is not None and gross_profit_t < gross_profit_t1,
        "margin_deterioration": revenue_t is not None and revenue_t1 not in (None, 0) and (gross_profit_t is not None and gross_profit_t1 is not None) and (gross_profit_t / revenue_t) < (gross_profit_t1 / revenue_t1),
        "asset_turnover_drop": revenue_t is not None and revenue_t1 is not None and total_assets_t not in (None, 0) and total_assets_t1 not in (None, 0) and (revenue_t / total_assets_t) < (revenue_t1 / total_assets_t1),
        "inventory_growth": inventory_t is not None and inventory_t1 is not None and inventory_t > inventory_t1,
        "receivables_growth": receivables_t is not None and receivables_t1 is not None and receivables_t > receivables_t1,
    }

    available = [v for v in c_score_flags.values() if v is not None]
    if not available:
        return {"score": None, "available": False, "reason": "missing_metrics"}

    score_raw = sum(1 for v in available if v)

    if score_raw == 0:
        normalized = 1.0
    elif score_raw <= 2:
        normalized = 0.8
    elif score_raw <= 4:
        normalized = 0.5
    else:
        normalized = 0.2

    return {
        "score_raw": score_raw,
        "score": normalized,
        "available": True,
        "red_flags": c_score_flags,
    }


def quality_forensics_assessment(financials: Dict | None) -> Dict:
    piotroski = _piotroski_f_score(financials)
    beneish = _beneish_m_score(financials)
    montier = _montier_c_score(financials)

    components = {
        "piotroski": (0.6, piotroski.get("score")),
        "beneish": (0.25, beneish.get("score")),
        "montier": (0.15, montier.get("score")),
    }

    total_weight = sum(weight for weight, score in components.values() if score is not None)
    combined = None
    if total_weight:
        combined = sum(weight * score for weight, score in components.values() if score is not None) / total_weight

    label = None
    if combined is not None:
        if combined >= 0.75:
            label = "Stark"
        elif combined >= 0.5:
            label = "Gut"
        elif combined >= 0.35:
            label = "Schwach"
        else:
            label = "Auffällig"

    return {
        "score": combined,
        "label": label,
        "components": {
            name: {
                "weight": weight,
                "score": score,
                "available": (source.get("available") if isinstance(source, dict) else False),
                "raw": source,
            }
            for (name, (weight, score)), source in zip(components.items(), [piotroski, beneish, montier])
        },
    }


def final_assessment(value: Dict, quality: Dict) -> Dict:
    value_score = value.get("score", 0.0)
    quality_score = quality.get("score")

    combined_score = value_score
    quality_weight = 0.3
    if quality_score is not None:
        combined_score = 0.7 * value_score + quality_weight * quality_score

    piotroski_component = quality.get("components", {}).get("piotroski", {})
    piotroski_available = piotroski_component.get("raw", {}).get("available")
    piotroski_score = piotroski_component.get("score")

    gate_pass = True
    gate_reason = None
    if piotroski_available and piotroski_score is not None:
        gate_pass = piotroski_score >= (5 / 9)
        gate_reason = "piotroski_threshold" if not gate_pass else None
    elif not piotroski_available:
        gate_reason = "piotroski_data_missing"

    if combined_score >= 0.8:
        label = "Kaufen"
    elif combined_score >= 0.6:
        label = "Beobachten"
    elif combined_score >= 0.4:
        label = "Neutral"
    else:
        label = "Meiden"

    return {
        "score": combined_score,
        "label": label,
        "piotroski_gate_pass": gate_pass,
        "gate_reason": gate_reason,
    }
