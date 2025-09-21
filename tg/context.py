import html
from typing import Any, List, Optional, Tuple
import re

from telethon import events, TelegramClient
from telethon.tl.types import Channel, InputPeerUser, PeerUser, PeerChannel, PeerChat

from logging_config import logger


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


_ALLOWED_HTML_TAGS: tuple[str, ...] = ("b", "i", "u")
_PLACEHOLDER_TEMPLATE = "__TG_TAG_{idx}__"


def sanitize_telegram_html(raw_html: str) -> str:
    """Normalize LLM output to Telegram HTML subset.

    The sanitizer keeps only <b>, <i>, <u> tags, removes attributes, closes
    unbalanced tags, and escapes every other HTML construct so Telegram can
    parse the result safely.
    """

    if not raw_html:
        return ""

    text = raw_html

    # Normalise newlines first so later escaping keeps structure readable.
    text = text.replace("\r", "")
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?\s*(?:div|p)\s*>", "\n", text, flags=re.IGNORECASE)

    placeholders: list[tuple[str, bool, str]] = []

    def _extract_allowed(match: re.Match[str]) -> str:
        slash = bool(match.group(1))
        tag = (match.group(2) or "").lower()
        idx = len(placeholders)
        placeholder = _PLACEHOLDER_TEMPLATE.format(idx=idx)
        placeholders.append((placeholder, slash, tag))
        return placeholder

    # Extract allowed tags (with attributes stripped) and remember ordering.
    text = re.sub(
        r"<\s*(/?)\s*(b|i|u)(?:\s+[^>]*?)?>",
        _extract_allowed,
        text,
        flags=re.IGNORECASE,
    )

    # Drop any other tags entirely.
    text = re.sub(r"<[^>]+>", "", text)

    # Escape the remainder so raw angle brackets do not break Telegram parsing.
    safe = html.escape(text, quote=False)

    # Collapse excessive blank lines after escaping to keep layout compact.
    safe = re.sub(r"\n{3,}", "\n\n", safe)

    # Decide which placeholders can be safely re-inserted while keeping the
    # HTML balanced. Unmatched tags are dropped.
    open_stack: list[tuple[str, str]] = []
    replacements: dict[str, str] = {}

    for placeholder, is_closing, tag in placeholders:
        if tag not in _ALLOWED_HTML_TAGS:
            replacements[placeholder] = ""
            continue
        if not is_closing:
            open_stack.append((tag, placeholder))
            replacements[placeholder] = f"<{tag}>"
            continue

        # Closing tag: find the nearest matching open tag.
        match_idx = None
        for idx in range(len(open_stack) - 1, -1, -1):
            open_tag, _ = open_stack[idx]
            if open_tag == tag:
                match_idx = idx
                break

        if match_idx is None:
            replacements[placeholder] = ""
            continue

        # Drop any still-open tags stacked above the match – their markup would be unbalanced.
        for extra_tag, extra_placeholder in open_stack[match_idx + 1 :]:
            replacements[extra_placeholder] = ""

        # Pop the matched tag and anything above it.
        open_stack = open_stack[:match_idx]
        replacements[placeholder] = f"</{tag}>"

    # Any unmatched opening tags are removed to avoid dangling markup.
    for _, placeholder in open_stack:
        replacements[placeholder] = ""

    for placeholder, replacement in replacements.items():
        safe = safe.replace(placeholder, replacement)

    # Remove any placeholders that were never assigned (should not happen but safe-guard).
    safe = re.sub(r"__TG_TAG_\d+__", "", safe)

    return safe.strip()

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
    author_access_hash: Optional[int] = None,
) -> List[str]:
    """Fetch up to max_collect non-empty text messages from a specific author in a chat history.

    Messages are returned oldest-first for better readability/processing downstream.
    """
    if client is None:
        return []
    if not chat_id or not author_user_id:
        return []

    texts: List[str] = []
    from_user_param: Any = int(author_user_id)
    if author_access_hash is not None:
        try:
            from_user_param = InputPeerUser(int(author_user_id), int(author_access_hash))
            await client.get_input_entity(from_user_param)
        except Exception:
            from_user_param = int(author_user_id)
    try:
        # Telethon yields newest-first; we'll reverse later
        async for msg in client.iter_messages(
            entity=chat_id,
            limit=limit_msgs,
            from_user=from_user_param,
        ):
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
    author_access_hash: Optional[int] = None,
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
    from_user_param: Any = int(author_user_id)
    if author_access_hash is not None:
        try:
            from_user_param = InputPeerUser(int(author_user_id), int(author_access_hash))
            await client.get_input_entity(from_user_param)
        except Exception:
            from_user_param = int(author_user_id)
    try:
        async for msg in client.iter_messages(
            entity=chat_id,
            limit=limit_msgs,
            from_user=from_user_param,
        ):
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
