from __future__ import annotations
from typing import Any, Dict, List


def get_symbols(cfg: Dict[str, Any]) -> List[str]:
    u = (cfg.get("universe") or {})
    symbols = u.get("symbols") or []
    return [str(s).upper().strip() for s in symbols if str(s).strip()]
