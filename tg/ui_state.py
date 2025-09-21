from typing import Any, Dict, Optional, List

from core.types import ClassificationResult


class ReplyUIState:
    def __init__(
        self,
        user_id: int,
        original_body_html: str,
        original_text: str,
        context_for_model: Optional[str] = None,
        classification_result: Optional["ClassificationResult"] = None,
        author_user_id: Optional[int] = None,
        source_chat_id: Optional[int] = None,
        author_username: Optional[str] = None,
        author_access_hash: Optional[int] = None,
    ) -> None:
        self.user_id = int(user_id)
        self.original_body_html = original_body_html
        self.original_text = original_text
        self.context_for_model = context_for_model
        self.last_reply_text: Optional[str] = None
        self.classification_result = classification_result
        self.author_user_id = int(author_user_id) if author_user_id is not None else None
        self.source_chat_id = int(source_chat_id) if source_chat_id is not None else None
        # Telegram username of the author (without leading '@'), if available
        self.author_username = str(author_username) if author_username else None
        try:
            self.author_access_hash = int(author_access_hash) if author_access_hash is not None else None
        except Exception:
            self.author_access_hash = None
        self.is_cancelled: bool = False


class ReplyUIStore:
    def __init__(self) -> None:
        self._seq = 0
        self._states: Dict[str, ReplyUIState] = {}

    def create(
        self,
        user_id: int,
        original_body_html: str,
        original_text: str,
        context_for_model: Optional[str] = None,
        classification_result: Optional["ClassificationResult"] = None,
        author_user_id: Optional[int] = None,
        source_chat_id: Optional[int] = None,
        author_username: Optional[str] = None,
        author_access_hash: Optional[int] = None,
    ) -> str:
        self._seq += 1
        sid = f"s{self._seq}"
        self._states[sid] = ReplyUIState(
            user_id=user_id,
            original_body_html=original_body_html,
            original_text=original_text,
            context_for_model=context_for_model,
            classification_result=classification_result,
            author_user_id=author_user_id,
            source_chat_id=source_chat_id,
            author_username=author_username,
            author_access_hash=author_access_hash,
        )
        return sid

    def get(self, sid: str) -> Optional[ReplyUIState]:
        return self._states.get(sid)

    def set_reply(self, sid: str, reply_text: str) -> None:
        st = self._states.get(sid)
        if st:
            st.last_reply_text = reply_text

    def mark_cancelled(self, sid: str) -> None:
        st = self._states.get(sid)
        if st:
            st.is_cancelled = True

    def clear_cancelled(self, sid: str) -> None:
        st = self._states.get(sid)
        if st:
            st.is_cancelled = False

    def is_cancelled(self, sid: str) -> bool:
        st = self._states.get(sid)
        return bool(st and st.is_cancelled)


reply_ui_store = ReplyUIStore()


# ------------------------------
# Account onboarding FSM store
# ------------------------------

class AccountState:
    def __init__(self) -> None:
        self.state: str = "ACCOUNT_IDLE"
        self.phone: Optional[str] = None
        self.phone_code_hash: Optional[str] = None
        self.tmp_client_session: Optional[str] = None  # Telethon StringSession during onboarding
        self.ui_message_id: Optional[int] = None  # Bot message id for onboarding UI edits/deletion
        self.ui_aux_message_ids: List[int] = []  # Extra bot messages to delete (e.g., QR image, keyboards)
        self.user_message_ids: List[int] = []  # User messages to try deleting (may fail in private chats)


class AccountFSM:
    def __init__(self) -> None:
        self._states: Dict[int, AccountState] = {}

    def get(self, user_id: int) -> AccountState:
        st = self._states.get(int(user_id))
        if not st:
            st = AccountState()
            self._states[int(user_id)] = st
        return st

    def clear(self, user_id: int) -> None:
        if int(user_id) in self._states:
            del self._states[int(user_id)]


account_fsm = AccountFSM()


# ------------------------------
# Warp Chat: in-memory index cache
# ------------------------------

# Keyed by "<telegram_user_id>__<chat_id>"
_warp_index_cache: Dict[str, Dict[str, Any]] = {}


def _warp_key(user_id: int, chat_id: int) -> str:
    return f"{int(user_id)}__{int(chat_id)}"


def set_warp_index(user_id: int, chat_id: int, payload: Dict[str, Any]) -> None:
    _warp_index_cache[_warp_key(user_id, chat_id)] = dict(payload)


def get_warp_index(user_id: int, chat_id: int) -> Optional[Dict[str, Any]]:
    return _warp_index_cache.get(_warp_key(user_id, chat_id))


# ------------------------------
# Warp Chat: dialogs cache (per user, short TTL)
# ------------------------------

_DIALOGS_CACHE_TTL_S: int = 60
_DIALOGS_CACHE_VERSION: int = 3
_dialogs_cache: Dict[int, Dict[str, Any]] = {}


def get_cached_dialogs(user_id: int) -> Optional[List[Dict[str, Any]]]:
    try:
        entry = _dialogs_cache.get(int(user_id))
        if not entry:
            return None
        if int(entry.get("version", 0) or 0) != _DIALOGS_CACHE_VERSION:
            try:
                del _dialogs_cache[int(user_id)]
            except Exception:
                pass
            return None
        ts = float(entry.get("ts", 0.0) or 0.0)
        import time as _t
        if (_t.time() - ts) > _DIALOGS_CACHE_TTL_S:
            # Expired
            try:
                del _dialogs_cache[int(user_id)]
            except Exception:
                pass
            return None
        dialogs = entry.get("dialogs")
        if isinstance(dialogs, list):
            return dialogs  # type: ignore[return-value]
        return None
    except Exception:
        return None


def set_cached_dialogs(user_id: int, dialogs: List[Dict[str, Any]]) -> None:
    try:
        import time as _t
        _dialogs_cache[int(user_id)] = {
            "ts": _t.time(),
            "version": _DIALOGS_CACHE_VERSION,
            "dialogs": list(dialogs),
        }
    except Exception:
        pass


def clear_cached_dialogs(user_id: int) -> None:
    try:
        _dialogs_cache.pop(int(user_id), None)
    except Exception:
        pass


# ------------------------------
# Warp Chat: primary UI message tracking
# ------------------------------

_warp_ui_message_ids: Dict[int, int] = {}


def get_warp_ui_message_id(user_id: int) -> Optional[int]:
    try:
        mid = _warp_ui_message_ids.get(int(user_id))
        return int(mid) if mid is not None else None
    except Exception:
        return None


def set_warp_ui_message_id(user_id: int, message_id: int) -> None:
    try:
        _warp_ui_message_ids[int(user_id)] = int(message_id)
    except Exception:
        pass


def clear_warp_ui_message_id(user_id: int) -> None:
    try:
        _warp_ui_message_ids.pop(int(user_id), None)
    except Exception:
        pass
