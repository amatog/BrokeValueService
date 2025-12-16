from __future__ import annotations
from typing import Any, Dict, List


def get_asof_dates(cfg: Dict[str, Any]) -> List[str]:
    r = (cfg.get("run") or {})
    dates = r.get("asof_dates") or []
    return [str(d).strip() for d in dates if str(d).strip()]
