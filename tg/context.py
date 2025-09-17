import html
from typing import Any, List, Optional, Tuple
import re

from telethon import events, TelegramClient
from telethon.tl.types import Channel, PeerUser, PeerChannel, PeerChat

from logging_config import logger


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def sanitize_telegram_html(raw_html: str) -> str:
    """Normalize LLM output to Telegram HTML subset.

    - Replace <br>, <br/>, <br /> with newlines
    - Strip unsupported tags, keep only <b>, <i>, <u>
    - Remove attributes from allowed tags
    """
    if not raw_html:
        return ""

    text = raw_html

    # Normalize common line-break tags to newlines
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    # Treat common block containers as line breaks
    text = re.sub(r"<\s*/?\s*(?:div|p)\s*>", "\n", text, flags=re.IGNORECASE)

    # Strip attributes from allowed tags
    def _strip_attrs(match: re.Match[str]) -> str:
        # match groups: 1 = optional '/', 2 = tag name
        slash = match.group(1) or ""
        tag = (match.group(2) or "").lower()
        return f"<{slash}{tag}>"

    text = re.sub(r"<\s*(/?)\s*(b|i|u)(?:\s+[^>]*?)?>", _strip_attrs, text, flags=re.IGNORECASE)

    # Remove any other tags not in the allowed list
    text = re.sub(r"<\s*/?\s*(?!b\b|i\b|u\b)[a-z0-9]+(?:\s+[^>]*?)?>", "", text, flags=re.IGNORECASE)

    # Remove any accidental self-closing allowed tags like <b/> that Telegram rejects
    text = re.sub(r"<\s*(b|i|u)\s*/\s*>", "", text, flags=re.IGNORECASE)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def build_internal_chat_c_id(chat_id: int) -> Optional[str]:
    raw = str(abs(chat_id))
    if raw.startswith("100"):
        return raw[3:]
    return raw


async def build_message_link(event: events.NewMessage.Event) -> Optional[str]:
    try:
        chat = await event.get_chat()
        username = getattr(chat, "username", None)
        if username:
            return f"https://t.me/{username}/{event.id}"
        if isinstance(chat, Channel) or event.is_channel:
            internal = build_internal_chat_c_id(event.chat_id)  # type: ignore[arg-type]
            return f"https://t.me/c/{internal}/{event.id}" if internal else None
    except Exception as e:  # noqa: BLE001
        logger.warning("link_build_failed", extra={"extra": {"error": str(e)}})
    return None


async def collect_reply_context(event: events.NewMessage.Event, depth_limit: int = 6) -> Tuple[Optional[str], Optional[str]]:
    try:
        chain: List[Any] = []
        current = event.message
        steps = 0
        while current is not None and steps < depth_limit:
            try:
                parent = await current.get_reply_message()  # type: ignore[attr-defined]
            except Exception:
                parent = None
            if not parent:
                break
            chain.append(parent)
            current = parent
            steps += 1

        if not chain:
            return None, None

        chain = list(reversed(chain))

        def _msg_text(msg: Any) -> Optional[str]:
            try:
                raw = getattr(msg, "message", None)
                return (raw or "").strip() or None
            except Exception:
                return None

        parts_plain: List[str] = []
        parts_html: List[str] = ["<b>Контекст</b>", "<b>━━━━━━━━━━━━━━━━━━━━</b>"]

        for idx, msg in enumerate(chain):
            text = _msg_text(msg)
            if not text:
                continue
            if idx == 0:
                parts_plain.append(f"Post: {text}")
            else:
                parts_plain.append(f"Reply{idx}: {text}")
            safe = escape_html(text)
            prefix = "Пост" if idx == 0 else f"Ответ {idx}"
            parts_html.append(f"• <i>{prefix}</i>:\n{safe}")

        if not parts_plain:
            return None, None

        context_plain = "\n".join(parts_plain)
        context_html = "\n".join(parts_html)
        return context_plain, context_html
    except Exception as e:  # noqa: BLE001
        logger.warning("collect_reply_context_failed", extra={"extra": {"error": str(e)}})
        return None, None


async def resolve_author_user_id(event: events.NewMessage.Event):
    try:
        from_id = getattr(event.message, "from_id", None)
        if isinstance(from_id, PeerUser):
            return int(from_id.user_id), None
        if isinstance(from_id, PeerChannel):
            return None, "author_is_channel"
        if isinstance(from_id, PeerChat):
            sid = int(getattr(event, "sender_id", 0) or 0)
            return (sid if sid else None), (None if sid else "no_user_id")
        sid = int(getattr(event, "sender_id", 0) or 0)
        if sid:
            return sid, None
        sender = await event.get_sender()
        uid = int(getattr(sender, "id", 0))
        return (uid if uid else None), (None if uid else "no_user_id")
    except Exception as e:  # noqa: BLE001
        logger.warning("resolve_author_failed", extra={"extra": {"error": str(e)}})
        return None, "resolve_error"


async def fetch_author_texts_from_history(
    client: TelegramClient,
    chat_id: int,
    author_user_id: int,
    *,
    limit_msgs: int = 1000,
    max_collect: int = 200,
) -> List[str]:
    """Fetch up to max_collect non-empty text messages from a specific author in a chat history.

    Messages are returned oldest-first for better readability/processing downstream.
    """
    if client is None:
        return []
    if not chat_id or not author_user_id:
        return []

    texts: List[str] = []
    try:
        # Telethon yields newest-first; we'll reverse later
        async for msg in client.iter_messages(entity=chat_id, limit=limit_msgs, from_user=author_user_id):
            try:
                raw = getattr(msg, "message", None)
            except Exception:  # noqa: BLE001
                raw = None
            if not raw:
                continue
            text = str(raw).strip()
            if not text:
                continue
            texts.append(text)
            if len(texts) >= max_collect:
                break
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "fetch_author_history_failed",
            extra={"extra": {"chat_id": int(chat_id), "author_user_id": int(author_user_id), "error": str(e)}},
        )
        return []

    if not texts:
        return []
    # Oldest-first
    texts.reverse()
    return texts


async def fetch_author_messages_with_meta_from_history(
    client: TelegramClient,
    chat_id: int,
    author_user_id: int,
    *,
    limit_msgs: int = 1000,
    max_collect: int = 200,
) -> List[dict]:
    """Fetch up to max_collect messages with metadata for a specific author.

    Returns a list of dicts with keys: text (str), date_ts (int, UTC epoch seconds),
    hour_utc (int 0..23), weekday (int 0..6, Monday=0). Oldest-first.
    """
    if client is None:
        return []
    if not chat_id or not author_user_id:
        return []

    items: List[dict] = []
    try:
        async for msg in client.iter_messages(entity=chat_id, limit=limit_msgs, from_user=author_user_id):
            try:
                raw = getattr(msg, "message", None)
            except Exception:  # noqa: BLE001
                raw = None
            if not raw:
                continue
            text = str(raw).strip()
            if not text:
                continue
            try:
                dt = getattr(msg, "date", None)
            except Exception:  # noqa: BLE001
                dt = None
            if dt is None:
                # Skip messages without date to keep stats simple
                continue
            try:
                ts = int(dt.timestamp())
                hour = int(getattr(dt, "hour", 0) or 0)
                weekday = int(getattr(dt, "weekday", lambda: 0)()) if hasattr(dt, "weekday") else 0
            except Exception:  # noqa: BLE001
                ts, hour, weekday = 0, 0, 0
            items.append({
                "text": text,
                "date_ts": ts,
                "hour_utc": hour,
                "weekday": weekday,
            })
            if len(items) >= max_collect:
                break
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "fetch_author_history_with_meta_failed",
            extra={"extra": {"chat_id": int(chat_id), "author_user_id": int(author_user_id), "error": str(e)}},
        )
        return []

    if not items:
        return []
    items.reverse()
    return items

