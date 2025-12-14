import os
import logging
from typing import Dict, Any

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

    # ---------------------------------------------------
    # Growth-Daten (für Lynch / PEG etc.)
    # ---------------------------------------------------
    def get_growth(self, symbol: str) -> Dict[str, Any]:
        """
        Liefert Growth-Daten in einem stabilen Format.
        Aktuell: eps_growth_5y (dezimal, z.B. 0.08 für 8%)
        """
        symbol = symbol.upper()

        if self.use_dummy:
            return {"eps_growth_5y": 0.08, "source": "dummy"}

        # Wir nutzen hier dieselbe Berechnung wie in _alphavantage_fundamentals(),
        # um doppelte Logik zu vermeiden. Alternativ kann man EARNINGS direkt erneut ziehen.
        try:
            fundamentals = self._alphavantage_fundamentals(symbol)
            return {
                "eps_growth_5y": fundamentals.get("earnings_growth_5y"),
                "source": "alphavantage",
            }
        except Exception as e:
            logger.exception("[DataLayer] Fehler bei get_growth, fallback auf Dummy: %s", e)
            return {"eps_growth_5y": 0.08, "source": "dummy-fallback"}

    # ---------------------------------------------------
    # Income / Balance / Cashflow (für Piotroski/Beneish/Montier)
    # ---------------------------------------------------
    def get_income(self, symbol: str) -> Dict[str, Any]:
        """
        Returns AlphaVantage INCOME_STATEMENT response (or dummy with annualReports[0/1]).
        """
        symbol = symbol.upper()

        if self.use_dummy:
            # Minimal + Beneish/Montier relevant fields
            return {
                "annualReports": [
                    {
                        "fiscalDateEnding": "2024-12-31",
                        "totalRevenue": 40_000_000,
                        "grossProfit": 20_000_000,
                        "netIncome": 5_000_000,
                        "costOfRevenue": 20_000_000,
                        # AV Key varies; we provide one common candidate
                        "sellingGeneralAdministrative": 5_000_000,
                        "depreciationAndAmortization": 1_000_000,
                    },
                    {
                        "fiscalDateEnding": "2023-12-31",
                        "totalRevenue": 38_000_000,
                        "grossProfit": 18_000_000,
                        "netIncome": 4_000_000,
                        "costOfRevenue": 20_000_000,
                        "sellingGeneralAdministrative": 4_800_000,
                        "depreciationAndAmortization": 950_000,
                    },
                ],
                "source": "dummy",
            }

        try:
            data = self._get({"function": "INCOME_STATEMENT", "symbol": symbol})
            data["source"] = "alphavantage"
            return data
        except Exception as e:
            logger.exception("[DataLayer] Fehler bei get_income, fallback auf Dummy: %s", e)
            return self.get_income(symbol="DUMMY")  # safe fallback shape

    def get_balance(self, symbol: str) -> Dict[str, Any]:
        """
        Returns AlphaVantage BALANCE_SHEET response (or dummy with annualReports[0/1]).
        """
        symbol = symbol.upper()

        if self.use_dummy:
            return {
                "annualReports": [
                    {
                        "fiscalDateEnding": "2024-12-31",
                        "totalAssets": 50_000_000,
                        "totalCurrentAssets": 12_000_000,
                        "totalCurrentLiabilities": 6_000_000,
                        "longTermDebt": 10_000_000,
                        "totalDebt": 12_000_000,
                        # Beneish DSRI:
                        "netReceivables": 3_000_000,
                        # Montier inventory:
                        "totalInventory": 2_000_000,
                        # Beneish AQI / DEPI:
                        "propertyPlantEquipment": 15_000_000,
                    },
                    {
                        "fiscalDateEnding": "2023-12-31",
                        "totalAssets": 48_000_000,
                        "totalCurrentAssets": 10_000_000,
                        "totalCurrentLiabilities": 6_500_000,
                        "longTermDebt": 11_000_000,
                        "totalDebt": 13_000_000,
                        "netReceivables": 2_600_000,
                        "totalInventory": 1_800_000,
                        "propertyPlantEquipment": 14_500_000,
                    },
                ],
                "source": "dummy",
            }

        try:
            data = self._get({"function": "BALANCE_SHEET", "symbol": symbol})
            data["source"] = "alphavantage"
            return data
        except Exception as e:
            logger.exception("[DataLayer] Fehler bei get_balance, fallback auf Dummy: %s", e)
            return self.get_balance(symbol="DUMMY")

    def get_cashflow(self, symbol: str) -> Dict[str, Any]:
        """
        Returns AlphaVantage CASH_FLOW response (or dummy with annualReports[0/1]).
        """
        symbol = symbol.upper()

        if self.use_dummy:
            return {
                "annualReports": [
                    {"fiscalDateEnding": "2024-12-31", "operatingCashflow": 6_000_000},
                    {"fiscalDateEnding": "2023-12-31", "operatingCashflow": 4_500_000},
                ],
                "source": "dummy",
            }

        try:
            data = self._get({"function": "CASH_FLOW", "symbol": symbol})
            data["source"] = "alphavantage"
            return data
        except Exception as e:
            logger.exception("[DataLayer] Fehler bei get_cashflow, fallback auf Dummy: %s", e)
            return self.get_cashflow(symbol="DUMMY")

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

    # ---------------------------------------------------
    # Legacy / Helper: YOY financials as merged dicts (still usable)
    # ---------------------------------------------------
    def get_financials_yoy(self, symbol: str):
        """
        Returns: (cur, prev) dicts for Piotroski.
        Keys:
          net_income, total_assets, operating_cash_flow, long_term_debt,
          current_assets, current_liabilities, shares_outstanding,
          gross_profit, revenue
        """
        symbol = symbol.upper()

        if self.use_dummy:
            cur = {
                "net_income": 5_000_000,
                "total_assets": 50_000_000,
                "operating_cash_flow": 6_000_000,
                "long_term_debt": 10_000_000,
                "current_assets": 12_000_000,
                "current_liabilities": 6_000_000,
                "shares_outstanding": 1_000_000,
                "gross_profit": 20_000_000,
                "revenue": 40_000_000,
            }
            prev = {
                "net_income": 4_000_000,
                "total_assets": 48_000_000,
                "operating_cash_flow": 4_500_000,
                "long_term_debt": 11_000_000,
                "current_assets": 10_000_000,
                "current_liabilities": 6_500_000,
                "shares_outstanding": 1_000_000,
                "gross_profit": 18_000_000,
                "revenue": 38_000_000,
            }
            return cur, prev

        # AlphaVantage fetches
        overview = self._get({"function": "OVERVIEW", "symbol": symbol})
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
            except Exception:
                return None

        # Grab annual reports: [0]=latest, [1]=prior
        b = balance.get("annualReports") or []
        i = income.get("annualReports") or []
        c = cashflow.get("annualReports") or []

        # shares outstanding from overview (best available in AV)
        shares = to_float(overview.get("SharesOutstanding"))

        def pick(report_list, idx):
            return report_list[idx] if len(report_list) > idx else {}

        b0, b1 = pick(b, 0), pick(b, 1)
        i0, i1 = pick(i, 0), pick(i, 1)
        c0, c1 = pick(c, 0), pick(c, 1)

        cur = {
            "net_income": to_float(i0.get("netIncome")),
            "total_assets": to_float(b0.get("totalAssets")),
            "operating_cash_flow": to_float(c0.get("operatingCashflow")),
            "long_term_debt": to_float(b0.get("longTermDebt")),
            "current_assets": to_float(b0.get("totalCurrentAssets")),
            "current_liabilities": to_float(b0.get("totalCurrentLiabilities")),
            "shares_outstanding": shares,  # same for both years unless you have year-specific series
            "gross_profit": to_float(i0.get("grossProfit")),
            "revenue": to_float(i0.get("totalRevenue")),
        }

        prev = {
            "net_income": to_float(i1.get("netIncome")),
            "total_assets": to_float(b1.get("totalAssets")),
            "operating_cash_flow": to_float(c1.get("operatingCashflow")),
            "long_term_debt": to_float(b1.get("longTermDebt")),
            "current_assets": to_float(b1.get("totalCurrentAssets")),
            "current_liabilities": to_float(b1.get("totalCurrentLiabilities")),
            "shares_outstanding": shares,
            "gross_profit": to_float(i1.get("grossProfit")),
            "revenue": to_float(i1.get("totalRevenue")),
        }

        return cur, prev
