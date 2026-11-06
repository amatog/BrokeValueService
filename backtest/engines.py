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
        if asof:
            raise ValueError("Historical snapshots require point-in-time fundamentals and are disabled.")
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
        if asof:
            raise ValueError("Historical snapshots require point-in-time fundamentals and are disabled.")
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
