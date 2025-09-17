from typing import Any, Dict, Optional

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
    ) -> None:
        self.user_id = int(user_id)
        self.original_body_html = original_body_html
        self.original_text = original_text
        self.context_for_model = context_for_model
        self.last_reply_text: Optional[str] = None
        self.classification_result = classification_result
        self.author_user_id = int(author_user_id) if author_user_id is not None else None
        self.source_chat_id = int(source_chat_id) if source_chat_id is not None else None
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


