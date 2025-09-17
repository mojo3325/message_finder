import time
from typing import Dict, Tuple


class DedupStore:
    def __init__(self, ttl_seconds: int = 24 * 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: Dict[Tuple[int, int], float] = {}

    def _now(self) -> float:
        return time.time()

    def seen(self, chat_id: int, message_id: int) -> bool:
        key = (chat_id, message_id)
        expiry = self._store.get(key)
        if expiry is None:
            return False
        if expiry < self._now():
            del self._store[key]
            return False
        return True

    def mark(self, chat_id: int, message_id: int) -> None:
        self._store[(chat_id, message_id)] = self._now() + self.ttl_seconds


