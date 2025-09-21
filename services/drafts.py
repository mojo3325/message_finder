import time
import uuid
from typing import Any, Dict, Optional


# Simple in-memory drafts store with TTL eviction


class _DraftsStore:
    def __init__(self, ttl_seconds: int = 60 * 30) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._ttl = int(ttl_seconds)

    def _now(self) -> float:
        return time.time()

    def _cleanup(self) -> None:
        now = self._now()
        expired: list[str] = []
        for draft_id, item in list(self._items.items()):
            created_at_ts = float(item.get("_created_at_ts", 0.0) or 0.0)
            if created_at_ts + self._ttl < now:
                expired.append(draft_id)
        for d in expired:
            try:
                del self._items[d]
            except Exception:
                pass

    def create(self, payload: Dict[str, Any]) -> str:
        self._cleanup()
        did = str(uuid.uuid4())
        item = dict(payload)
        item["draft_id"] = did
        # Store machine timestamp alongside ISO8601 created_at
        item["_created_at_ts"] = self._now()
        if "created_at" not in item:
            item["created_at"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        self._items[did] = item
        return did

    def get(self, draft_id: str) -> Optional[Dict[str, Any]]:
        self._cleanup()
        return self._items.get(str(draft_id))

    def update(self, draft_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._cleanup()
        it = self._items.get(str(draft_id))
        if not it:
            return None
        it.update(dict(fields))
        return it

    def delete(self, draft_id: str) -> None:
        self._cleanup()
        try:
            del self._items[str(draft_id)]
        except Exception:
            pass


drafts_store = _DraftsStore()


