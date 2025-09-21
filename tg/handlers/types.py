from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(slots=True)
class CallbackContext:
    callback_id: str
    user_id: Optional[int]
    chat_id: Optional[int]
    message_id: Optional[int]
    data: str
    action: str
    sid: str
    raw: Dict[str, Any]
    message: Dict[str, Any]


@dataclass(slots=True)
class MessageContext:
    user_id: int
    chat_id: int
    message_id: int
    text: str
    entities: list[Dict[str, Any]]
    raw: Dict[str, Any]
    start_payload: Optional[str]
    contact: Optional[Dict[str, Any]]

    @property
    def lower_text(self) -> str:
        return self.text.lower()
