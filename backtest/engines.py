from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

# DIRECT mode imports (lokale Engine)
from data_layer import DataLayer
from strategies import run_default_value_bundle


class SnapshotEngine:
    def snapshot(self, symbol: str, asof: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class ValueRestEngine(SnapshotEngine):
    base_url: str
    timeout_seconds: int = 25

    def snapshot(self, symbol: str, asof: Optional[str] = None) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + "/bundle"
        params = {"symbol": symbol}
        # future-proof; kann später serverseitig genutzt werden
        if asof:
            params["asof"] = asof

        r = requests.get(url, params=params, timeout=self.timeout_seconds)
        r.raise_for_status()
        return r.json()


@dataclass
class DirectStrategyEngine(SnapshotEngine):
    data_layer: DataLayer

    def snapshot(self, symbol: str, asof: Optional[str] = None) -> Dict[str, Any]:
        # Hinweis: asof wird hier noch nicht genutzt (AlphaVantage liefert "latest").
        # Für echtes historisches Backtesting wird in Phase 2.1.3+ ein Snapshot-Cache/DB ergänzt.
        fundamentals = self.data_layer.get_basic_fundamentals(symbol)
        growth = self.data_layer.get_growth(symbol)
        income = self.data_layer.get_income(symbol)
        balance = self.data_layer.get_balance(symbol)
        cashflow = self.data_layer.get_cashflow(symbol)

        sector = fundamentals.get("sector")
        return run_default_value_bundle(
            fundamentals=fundamentals,
            growth=growth,
            income=income,
            balance=balance,
            cashflow=cashflow,
            sector=sector,
        )
