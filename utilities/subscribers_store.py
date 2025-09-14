import logging
from typing import Any, Dict, Optional, Set

import httpx


logger = logging.getLogger("supabase_store")


class SupabaseSubscriberStore:
    def __init__(self, supabase_url: str, api_key: str, timeout_s: float = 10.0) -> None:
        base = (supabase_url or "").rstrip("/")
        if not base:
            raise ValueError("SUPABASE_URL is required")
        if not api_key:
            raise ValueError("SUPABASE_ANON_KEY is required")
        self._rest_base = f"{base}/rest/v1"
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client = httpx.Client(timeout=self._timeout_s)

        # Table names/keys
        self._subscribers_table = "subscribers"
        self._kv_table = "kv_store"
        self._offset_key = "bot_last_update_id"
        # Store for classified messages (for later summarization/portraits)
        self._messages_table = "classified_messages"

        # Readiness check to avoid noisy 404s until schema exists
        self._ready = False
        self._ready_logged = False
        self._ready = self._check_ready()

        # Separate readiness for messages table to avoid blocking subscriber flow
        self._ready_messages = False
        self._ready_messages_logged = False
        self._ready_messages = self._check_messages_ready()

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _check_ready(self) -> bool:
        try:
            h = self._headers()
            r1 = self._client.get(f"{self._rest_base}/{self._subscribers_table}?select=user_id&limit=0", headers=h)
            r2 = self._client.get(f"{self._rest_base}/{self._kv_table}?select=k&limit=0", headers=h)
            if r1.status_code == 404 or r2.status_code == 404:
                if not self._ready_logged:
                    logger.warning(
                        "supabase_schema_missing",
                        extra={
                            "extra": {
                                "subscribers_status": r1.status_code,
                                "kv_status": r2.status_code,
                                "hint": "Create tables 'public.subscribers' and 'public.kv_store' (see supabase_schema.sql)",
                            }
                        },
                    )
                    self._ready_logged = True
                return False
            if r1.status_code >= 300 or r2.status_code >= 300:
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_ready_probe_error", extra={"extra": {"error": str(e)}})
            return False

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        self._ready = self._check_ready()
        return self._ready

    def _check_messages_ready(self) -> bool:
        try:
            h = self._headers()
            r = self._client.get(f"{self._rest_base}/{self._messages_table}?select=chat_id&limit=0", headers=h)
            if r.status_code == 404:
                if not self._ready_messages_logged:
                    logger.warning(
                        "supabase_messages_schema_missing",
                        extra={
                            "extra": {
                                "messages_status": r.status_code,
                                "hint": "Create table 'public.classified_messages' with unique (chat_id, message_id)",
                            }
                        },
                    )
                    self._ready_messages_logged = True
                return False
            if r.status_code >= 300:
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_messages_ready_probe_error", extra={"extra": {"error": str(e)}})
            return False

    def _ensure_messages_ready(self) -> bool:
        if self._ready_messages:
            return True
        self._ready_messages = self._check_messages_ready()
        return self._ready_messages

    # ------------------------------
    # Subscribers API
    # ------------------------------
    def add(self, user_id: int) -> bool:
        if not self._ensure_ready():
            return False
        try:
            url = f"{self._rest_base}/{self._subscribers_table}?on_conflict=user_id"
            headers = dict(self._headers())
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
            payload = [{"user_id": int(user_id)}]
            resp = self._client.post(url, headers=headers, json=payload)
            if resp.status_code >= 300:
                logger.warning("supabase_add_failed", extra={"extra": {"status": resp.status_code, "text": resp.text}})
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_add_error", extra={"extra": {"error": str(e)}})
            return False

    def remove(self, user_id: int) -> bool:
        if not self._ensure_ready():
            return False
        try:
            url = f"{self._rest_base}/{self._subscribers_table}?user_id=eq.{int(user_id)}"
            headers = dict(self._headers())
            headers["Prefer"] = "return=minimal"
            resp = self._client.delete(url, headers=headers)
            if resp.status_code >= 300:
                logger.warning("supabase_remove_failed", extra={"extra": {"status": resp.status_code, "text": resp.text}})
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_remove_error", extra={"extra": {"error": str(e)}})
            return False

    def contains(self, user_id: int) -> bool:
        if not self._ensure_ready():
            return False
        try:
            url = f"{self._rest_base}/{self._subscribers_table}?user_id=eq.{int(user_id)}&select=user_id"
            headers = self._headers()
            resp = self._client.get(url, headers=headers)
            if resp.status_code >= 300:
                logger.warning("supabase_contains_failed", extra={"extra": {"status": resp.status_code, "text": resp.text}})
                return False
            data = resp.json()
            return isinstance(data, list) and len(data) > 0
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_contains_error", extra={"extra": {"error": str(e)}})
            return False

    def snapshot(self) -> Set[int]:
        if not self._ensure_ready():
            return set()
        try:
            url = f"{self._rest_base}/{self._subscribers_table}?select=user_id"
            headers = self._headers()
            resp = self._client.get(url, headers=headers)
            if resp.status_code >= 300:
                logger.warning("supabase_snapshot_failed", extra={"extra": {"status": resp.status_code, "text": resp.text}})
                return set()
            rows = resp.json()
            result: Set[int] = set()
            if isinstance(rows, list):
                for r in rows:
                    try:
                        uid = int(r.get("user_id"))
                        result.add(uid)
                    except Exception:
                        continue
            return result
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_snapshot_error", extra={"extra": {"error": str(e)}})
            return set()

    def count(self) -> int:
        return len(self.snapshot())

    # ------------------------------
    # Offset API (bot getUpdates offset)
    # ------------------------------
    def get_offset(self) -> int:
        if not self._ensure_ready():
            return 0
        try:
            url = f"{self._rest_base}/{self._kv_table}?k=eq.{self._offset_key}&select=v"
            headers = self._headers()
            resp = self._client.get(url, headers=headers)
            if resp.status_code >= 300:
                logger.warning("supabase_get_offset_failed", extra={"extra": {"status": resp.status_code, "text": resp.text}})
                return 0
            rows = resp.json()
            if isinstance(rows, list) and rows:
                try:
                    return int(rows[0].get("v") or 0)
                except Exception:
                    return 0
            return 0
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_get_offset_error", extra={"extra": {"error": str(e)}})
            return 0

    def advance_offset(self, update_id: int) -> bool:
        if not self._ensure_ready():
            return False
        try:
            url = f"{self._rest_base}/{self._kv_table}?on_conflict=k"
            headers = dict(self._headers())
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
            payload = [{"k": self._offset_key, "v": str(int(update_id))}]
            resp = self._client.post(url, headers=headers, json=payload)
            if resp.status_code >= 300:
                logger.warning("supabase_set_offset_failed", extra={"extra": {"status": resp.status_code, "text": resp.text}})
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_set_offset_error", extra={"extra": {"error": str(e)}})
            return False

    # ------------------------------
    # Classified messages API
    # ------------------------------
    def save_classified_message(self, record: Dict[str, Any]) -> bool:
        """
        Upsert a classified message for later aggregation/summarization.

        Expects keys like: chat_id, message_id, author_user_id, text, label,
        themes, link, context, date_ts, chat_title, chat_username, chat_type, author_reason.
        """
        if not self._ensure_messages_ready():
            return False

        def _truncate(value: Optional[str], limit: int) -> Optional[str]:
            if value is None:
                return None
            try:
                s = str(value)
            except Exception:
                return None
            if len(s) <= limit:
                return s
            return s[: limit - 1] + "…"

        try:
            # Normalize and limit potentially large fields
            payload: Dict[str, Any] = {
                "chat_id": int(record.get("chat_id")) if record.get("chat_id") is not None else None,
                "message_id": int(record.get("message_id")) if record.get("message_id") is not None else None,
                "author_user_id": int(record.get("author_user_id")) if record.get("author_user_id") is not None else None,
                "author_reason": _truncate(record.get("author_reason"), 64),
                "text": _truncate(record.get("text"), 8000),
                "label": _truncate(record.get("label"), 8),
                "themes": _truncate(record.get("themes"), 4000),
                "link": _truncate(record.get("link"), 1024),
                "context": _truncate(record.get("context"), 8000),
                "date_ts": int(record.get("date_ts")) if record.get("date_ts") is not None else None,
                "chat_title": _truncate(record.get("chat_title"), 256),
                "chat_username": _truncate(record.get("chat_username"), 256),
                "chat_type": _truncate(record.get("chat_type"), 64),
            }

            url = f"{self._rest_base}/{self._messages_table}?on_conflict=chat_id,message_id"
            headers = dict(self._headers())
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
            resp = self._client.post(url, headers=headers, json=[payload])
            if resp.status_code >= 300:
                logger.warning(
                    "supabase_save_message_failed",
                    extra={"extra": {"status": resp.status_code, "text": resp.text}},
                )
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase_save_message_error", extra={"extra": {"error": str(e)}})
            return False


