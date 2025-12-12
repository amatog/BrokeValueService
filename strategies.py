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


def combined_value_score(g: Dict, b: Dict, gr: Dict, m: Dict, l: Dict, s: Dict, d: Dict, t: Dict, k: Dict,  f: Dict) -> Dict:
    mos = g.get("margin_of_safety", 0.0)
    mos = max(min(mos, 1.0), -1.0)

    quality = b.get("quality_score", 0.0)
    magic = gr.get("magic_score", 0.0)
    dividend = f.get("dividend_yield") or 0.0

    score = (
            0.4 * (mos + 1) / 2 +
            0.3 * quality +
            0.2 * magic +
            0.1 * min(dividend / 0.05, 1.0)  # 5 % Dividende = voller Score
    )

    if score >= 0.8:
        level = "Sehr attraktiv"
    elif score >= 0.6:
        level = "Attraktiv"
    elif score >= 0.4:
        level = "Neutral"
    else:
        level = "Unattraktiv"

    return {
        "value_score": score,
        "value_level": level,
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
