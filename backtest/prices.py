from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


class PriceProvider:
    def price_on_or_after(self, symbol: str, d: date) -> Optional[float]:
        raise NotImplementedError


@dataclass
class CsvPriceProvider(PriceProvider):
    path: str
    date_col: str = "date"
    symbol_col: str = "symbol"
    price_col: str = "close"

    def __post_init__(self) -> None:
        df = pd.read_csv(self.path)
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.date
        df[self.symbol_col] = df[self.symbol_col].astype(str).str.upper().str.strip()
        df = df.dropna(subset=[self.date_col, self.symbol_col, self.price_col])

        # Sortierung für "first price on/after"
        self.df = df.sort_values([self.symbol_col, self.date_col]).reset_index(drop=True)

    def price_on_or_after(self, symbol: str, d: date) -> Optional[float]:
        symbol = symbol.upper().strip()
        sub = self.df[(self.df[self.symbol_col] == symbol) & (self.df[self.date_col] >= d)]
        if sub.empty:
            return None
        v = sub.iloc[0][self.price_col]
        try:
            return float(v)
        except Exception:
            return None
