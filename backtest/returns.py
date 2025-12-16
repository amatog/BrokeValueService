from __future__ import annotations

from datetime import date
from typing import Optional, Dict, Any

from dateutil.relativedelta import relativedelta


def forward_return(
        *,
        provider,
        symbol: str,
        asof: date,
        horizon_months: int,
) -> Optional[Dict[str, Any]]:
    """
    Kalenderbasierter Forward Return:
      p_t  = price_on_or_after(symbol, asof)
      p_th = price_on_or_after(symbol, asof + horizon_months)
      r = (p_th / p_t) - 1
    """
    p_t = provider.price_on_or_after(symbol, asof)
    if p_t is None or p_t <= 0:
        return None

    target = asof + relativedelta(months=horizon_months)
    p_th = provider.price_on_or_after(symbol, target)
    if p_th is None or p_th <= 0:
        return None

    return {
        "price_t": float(p_t),
        "price_t_h": float(p_th),
        "return_value": (float(p_th) / float(p_t)) - 1.0,
        "target_date": target.isoformat(),
    }