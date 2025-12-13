import os
import logging
from typing import Dict

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .env laden, falls vorhanden
load_dotenv()


class DataLayer:
    BASE_URL = "https://www.alphavantage.co/query"
    ENV_NAME = "ALPHAVANTAGE_API_KEY"

    def __init__(self) -> None:
        self.api_key = os.getenv(self.ENV_NAME)
        logger.info("[DataLayer] %s = %r", self.ENV_NAME, self.api_key)
        if not self.api_key:
            logger.error("[DataLayer] Kein API-Key gefunden – verwende Dummy-Daten.")
            self.use_dummy = True
        else:
            logger.info("[DataLayer] API-Key gefunden – verwende AlphaVantage.")
            self.use_dummy = False

    # ---------------------------------------------------
    # Öffentliche Methode, die von app.py genutzt wird
    # ---------------------------------------------------
    def get_basic_fundamentals(self, symbol: str) -> Dict:
        symbol = symbol.upper()

        if self.use_dummy:
            data = self._dummy_fundamentals(symbol)
            data["source"] = "dummy"
            return data

        try:
            data = self._alphavantage_fundamentals(symbol)
            data["source"] = "alphavantage"
            return data
        except Exception as e:
            logger.exception("[DataLayer] Fehler bei AlphaVantage, fallback auf Dummy: %s", e)
            data = self._dummy_fundamentals(symbol)
            data["source"] = "dummy-fallback"
            return data

    def get_financials_yoy(self, symbol: str) -> Dict:
        """Liefer t/t-1 Finanzdaten für Piotroski/Beneish/Montier."""
        symbol = symbol.upper()

        if self.use_dummy:
            data = self._dummy_financials(symbol)
            data["source"] = "dummy"
            return data

        try:
            data = self._alphavantage_financials(symbol)
            data["source"] = "alphavantage"
            return data
        except Exception as e:
            logger.exception("[DataLayer] Fehler bei AlphaVantage Finanzdaten, fallback auf Dummy: %s", e)
            data = self._dummy_financials(symbol)
            data["source"] = "dummy-fallback"
            return data

    # ---------------------------------------------------
    # Dummy-Daten
    # ---------------------------------------------------
    def _dummy_fundamentals(self, symbol: str) -> Dict:
        return {
            "symbol": symbol,
            "price": 150.0,
            "eps": 5.0,
            "book_value": 30.0,
            "roe": 0.18,
            "dividend_yield": 0.025,
            "market_cap": 1_000_000_000,
            "sector": "Dummy",
            "industry": "Dummy Industry",
            "currency": "USD",
            "debt_to_equity": 0.5,
            "earnings_growth_5y": 0.05,
        }

    def _dummy_financials(self, symbol: str) -> Dict:
        periods = [
            {
                "fiscalDateEnding": "2023-12-31",
                "balance": {
                    "totalAssets": 5000.0,
                    "totalCurrentAssets": 2000.0,
                    "totalCurrentLiabilities": 900.0,
                    "longTermDebt": 800.0,
                    "totalLiabilities": 2500.0,
                    "commonStockSharesOutstanding": 100.0,
                    "netReceivables": 400.0,
                    "inventory": 300.0,
                },
                "income": {
                    "netIncome": 600.0,
                    "grossProfit": 1500.0,
                    "totalRevenue": 4000.0,
                    "sellingGeneralAndAdministrative": 500.0,
                },
                "cashflow": {
                    "operatingCashflow": 650.0,
                    "depreciation": 120.0,
                },
            },
            {
                "fiscalDateEnding": "2022-12-31",
                "balance": {
                    "totalAssets": 4600.0,
                    "totalCurrentAssets": 1800.0,
                    "totalCurrentLiabilities": 950.0,
                    "longTermDebt": 900.0,
                    "totalLiabilities": 2400.0,
                    "commonStockSharesOutstanding": 100.0,
                    "netReceivables": 380.0,
                    "inventory": 280.0,
                },
                "income": {
                    "netIncome": 520.0,
                    "grossProfit": 1400.0,
                    "totalRevenue": 3600.0,
                    "sellingGeneralAndAdministrative": 480.0,
                },
                "cashflow": {
                    "operatingCashflow": 540.0,
                    "depreciation": 110.0,
                },
            },
        ]

        return {"symbol": symbol, "periods": periods}

    # ---------------------------------------------------
    # Echte Daten von AlphaVantage
    # ---------------------------------------------------
    def _alphavantage_fundamentals(self, symbol: str) -> Dict:
        symbol = symbol.upper()

        overview = self._get({
            "function": "OVERVIEW",
            "symbol": symbol,
        })

        quote = self._get({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        })
        q = quote.get("Global Quote", {})

        balance = self._get({
            "function": "BALANCE_SHEET",
            "symbol": symbol,
        })

        earnings = self._get({
            "function": "EARNINGS",
            "symbol": symbol,
        })

        def to_float(x):
            try:
                if x is None:
                    return None
                s = str(x).strip()
                if not s or s.lower() == "none":
                    return None
                return float(s)
            except ValueError:
                return None

        # ---------- debt_to_equity ----------
        debt_to_equity = None

        # 1) Versuch: direkt aus OVERVIEW (DebtEquityRatio)
        try:
            de_ratio = to_float(overview.get("DebtEquityRatio"))
            if de_ratio is not None:
                debt_to_equity = de_ratio
        except Exception as e:
            logger.warning("[DataLayer] Konnte DebtEquityRatio aus OVERVIEW nicht lesen: %s", e)

        # 2) Fallback: aus BALANCE_SHEET berechnen
        if debt_to_equity is None:
            try:
                annual_reports = balance.get("annualReports") or []
                if annual_reports:
                    last = annual_reports[0]
                    total_liab = to_float(last.get("totalLiabilities"))
                    total_equity = to_float(last.get("totalShareholderEquity"))
                    if total_liab is not None and total_equity not in (None, 0):
                        debt_to_equity = total_liab / total_equity
            except Exception as e:
                logger.warning("[DataLayer] Konnte debt_to_equity nicht berechnen: %s", e)


        # ---------- earnings_growth_5y aus EARNINGS ----------
        earnings_growth_5y = None
        try:
            annual_earnings = earnings.get("annualEarnings") or []
            if len(annual_earnings) >= 2:
                eps_data = []
                for item in annual_earnings:
                    eps_val = to_float(item.get("reportedEPS"))
                    date_str = item.get("fiscalDateEnding")
                    if eps_val is not None and date_str:
                        eps_data.append((date_str, eps_val))

                # neueste → älteste sortieren
                eps_data.sort(key=lambda x: x[0], reverse=True)

                # auf max. 5 Jahre begrenzen
                eps_data = eps_data[:5]

                if len(eps_data) >= 2:
                    eps_new = eps_data[0][1]
                    eps_old = eps_data[-1][1]
                    n_years = max(1, len(eps_data) - 1)
                    if eps_old not in (None, 0):
                        earnings_growth_5y = (eps_new / eps_old) ** (1 / n_years) - 1
        except Exception as e:
            logger.warning("[DataLayer] Konnte earnings_growth_5y nicht berechnen: %s", e)


        return {
            "symbol": symbol,
            "price": to_float(q.get("05. price")),
            "eps": to_float(overview.get("EPS")),
            "book_value": to_float(overview.get("BookValue")),
            "roe": to_float(overview.get("ReturnOnEquityTTM")),
            "dividend_yield": to_float(overview.get("DividendYield")),
            "market_cap": to_float(overview.get("MarketCapitalization")),
            "sector": overview.get("Sector"),
            "industry": overview.get("Industry"),
            "currency": overview.get("Currency"),
            "debt_to_equity": debt_to_equity,
            "earnings_growth_5y": earnings_growth_5y,
        }

    def _alphavantage_financials(self, symbol: str) -> Dict:
        symbol = symbol.upper()

        balance = self._get({"function": "BALANCE_SHEET", "symbol": symbol})
        income = self._get({"function": "INCOME_STATEMENT", "symbol": symbol})
        cashflow = self._get({"function": "CASH_FLOW", "symbol": symbol})

        def to_float(x):
            try:
                if x is None:
                    return None
                s = str(x).strip()
                if not s or s.lower() == "none":
                    return None
                return float(s)
            except ValueError:
                return None

        balance_reports = balance.get("annualReports") or []
        income_reports = income.get("annualReports") or []
        cashflow_reports = cashflow.get("annualReports") or []

        count = min(len(balance_reports), len(income_reports), len(cashflow_reports), 2)
        periods = []
        for idx in range(count):
            bal = balance_reports[idx]
            inc = income_reports[idx]
            cf = cashflow_reports[idx]
            periods.append(
                {
                    "fiscalDateEnding": bal.get("fiscalDateEnding") or inc.get("fiscalDateEnding"),
                    "balance": {
                        "totalAssets": to_float(bal.get("totalAssets")),
                        "totalCurrentAssets": to_float(bal.get("totalCurrentAssets")),
                        "totalCurrentLiabilities": to_float(bal.get("totalCurrentLiabilities")),
                        "longTermDebt": to_float(bal.get("longTermDebt")),
                        "totalLiabilities": to_float(bal.get("totalLiabilities")),
                        "commonStockSharesOutstanding": to_float(bal.get("commonStockSharesOutstanding")),
                        "netReceivables": to_float(bal.get("totalReceivables")) or to_float(bal.get("netReceivables")),
                        "inventory": to_float(bal.get("inventory")),
                    },
                    "income": {
                        "netIncome": to_float(inc.get("netIncome")),
                        "grossProfit": to_float(inc.get("grossProfit")),
                        "totalRevenue": to_float(inc.get("totalRevenue")),
                        "sellingGeneralAndAdministrative": to_float(inc.get("sellingGeneralAndAdministrative")),
                    },
                    "cashflow": {
                        "operatingCashflow": to_float(cf.get("operatingCashflow")),
                        "depreciation": to_float(cf.get("depreciation")),
                    },
                }
            )

        return {"symbol": symbol, "periods": periods}

    # ---------------------------------------------------
    # HTTP-Helfer
    # ---------------------------------------------------
    def _get(self, params: Dict) -> Dict:
        params = {**params, "apikey": self.api_key}
        resp = requests.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "Note" in data or "Information" in data:
            logger.warning("[DataLayer] AlphaVantage Hinweis/Limit: %s", data)
        return data
