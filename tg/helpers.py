import asyncio
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

from logging_config import logger

from tg import bot_api, ui
from tg.context import escape_html
from tg.ui_state import reply_ui_store


LinkChatIdentifier = Union[int, str]


def shorten(text: str, limit: int = 60) -> str:
    """Collapse whitespace and trim text to the given limit with an ellipsis."""
    collapsed = " ".join((text or "").strip().split())
    if len(collapsed) > limit:
        # Reserve one character for the ellipsis.
        return collapsed[: limit - 1].rstrip() + "…"
    return collapsed


_SPINNER_FRAMES: tuple[str, ...] = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_LOADING_INTERVAL_S: float = 0.8


def get_loading_interval(default: float = _LOADING_INTERVAL_S) -> float:
    return max(0.1, float(default))


def _normalize_headers(header: Union[str, Sequence[str]]) -> tuple[str, ...]:
    if isinstance(header, (list, tuple)):
        parts = tuple(str(h).strip() for h in header if str(h).strip())
    else:
        value = str(header).strip()
        parts = (value,) if value else tuple()
    return parts or ("Загрузка…",)


def build_loading_frame(step: int, header: str) -> str:
    spinner = _SPINNER_FRAMES[step % len(_SPINNER_FRAMES)]
    width = 10
    filled = (step % width) + 1
    bar = "█" * filled + "░" * (width - filled)
    return "\n".join(
        [
            f"<b>{header} {spinner}</b>",
            "<b>━━━━━━━━━━━━━━━━━━━━</b>",
            f"<code>{bar}</code>",
        ]
    )


async def start_loading_animation(
    chat_id: int,
    message_id: int,
    sid: str,
    header: Union[str, Sequence[str]],
    *,
    reply_markup: Optional[Dict[str, Any]] = None,
    interval: Optional[float] = None,
    freeze_on_last: bool = False,
) -> Tuple[asyncio.Event, asyncio.Task[None]]:
    """Animate loading state by editing the target message until stopped."""
    stop_event: asyncio.Event = asyncio.Event()
    headers = _normalize_headers(header)
    markup = reply_markup if reply_markup is not None else ui.build_reply_keyboard(sid, None)
    effective_interval = get_loading_interval(interval if interval is not None else _LOADING_INTERVAL_S)

    async def _loop() -> None:
        step = 0
        try:
            while not stop_event.is_set() and not reply_ui_store.is_cancelled(sid):
                if headers:
                    if freeze_on_last:
                        header_idx = min(step, len(headers) - 1)
                    else:
                        header_idx = step % len(headers)
                    current_header = headers[header_idx]
                else:
                    current_header = "Загрузка…"
                body = build_loading_frame(step, current_header)
                try:
                    await bot_api.bot_edit_message_text(
                        chat_id,
                        message_id,
                        body,
                        reply_markup=markup,
                    )
                except Exception:
                    # Ignore animation edit errors.
                    pass
                step += 1
                await asyncio.sleep(effective_interval)
        except Exception as exc:
            try:
                logger.debug("loading_animation_failed", extra={"extra": {"error": str(exc)}})
            except Exception:
                pass

    task = asyncio.create_task(_loop())
    return stop_event, task


def extract_link(message: dict) -> Optional[str]:
    """Return the first Telegram link found in message entities or text."""
    try:
        entities = message.get("entities") or []
        for entity in entities:
            if entity.get("type") == "text_link" and entity.get("url"):
                return str(entity.get("url"))
    except Exception:
        pass

    text = str(message.get("text") or "").strip()
    if not text:
        return None

    for token in text.split():
        lowered = token.lower()
        if lowered.startswith("http://t.me/") or lowered.startswith("https://t.me/"):
            return token
        if lowered.startswith("t.me/"):
            return f"https://{token}"
    return None


def parse_link_to_ids(link: str) -> Optional[Tuple[LinkChatIdentifier, int]]:
    """Parse a Telegram link into chat identifier (id or username) and message id."""
    if not link:
        return None

    link = link.rstrip("/")
    try:
        if "/c/" in link:
            _, right = link.split("/c/", 1)
            internal, message = right.split("/", 1)
            chat_id = int(f"-100{int(internal)}")
            return chat_id, int(message)
        segments = link.split("/")
        if len(segments) < 2:
            return None
        username = segments[-2]
        message = segments[-1]
        if not username or not message:
            return None
        return username, int(message)
    except Exception:
        return None


async def restore_reply_state_from_callback_message(
    callback_message: dict,
    *,
    get_telethon_client: Callable[[], Any],
    requester_user_id: Optional[int],
    fallback_user_id: Optional[int],
) -> Optional[Tuple[str, Any]]:
    """Rebuild reply UI state when callbacks arrive after process restarts."""

    link = extract_link(callback_message)
    if not link:
        return None

    client_ref = get_telethon_client()
    if client_ref is None:
        return None

    ids = parse_link_to_ids(link)
    if ids is None:
        return None

    try:
        if isinstance(ids[0], str):
            entity = await client_ref.get_entity(ids[0])  # type: ignore[arg-type]
            source_chat_id = int(getattr(entity, "id", 0) or 0)
        else:
            source_chat_id = int(ids[0])  # type: ignore[index]
        source_msg_id = int(ids[1])  # type: ignore[index]
    except Exception:
        return None

    try:
        src_msg = await client_ref.get_messages(entity=source_chat_id, ids=source_msg_id)
    except Exception:
        return None
    if not src_msg:
        return None

    try:
        raw_text = (getattr(src_msg, "message", None) or "").strip()
    except Exception:
        raw_text = ""
    if not raw_text:
        return None

    try:
        sender = await src_msg.get_sender()
    except Exception:
        sender = None
    try:
        chat_entity = await client_ref.get_entity(source_chat_id)
    except Exception:
        chat_entity = None

    async def _collect_context() -> Tuple[Optional[str], Optional[str]]:
        try:
            chain = []
            current = src_msg
            steps = 0
            while current is not None and steps < 6:
                try:
                    parent = await current.get_reply_message()
                except Exception:
                    parent = None
                if not parent:
                    break
                chain.append(parent)
                current = parent
                steps += 1
            if not chain:
                return None, None
            chain.reverse()

            parts_plain = []
            parts_html = ["<b>Контекст</b>", "<b>━━━━━━━━━━━━━━━━━━━━</b>"]
            for idx, msg in enumerate(chain):
                try:
                    text = (getattr(msg, "message", None) or "").strip()
                except Exception:
                    text = ""
                if not text:
                    continue
                prefix_plain = "Post" if idx == 0 else f"Reply{idx}"
                parts_plain.append(f"{prefix_plain}: {text}")
                safe = escape_html(text)
                prefix = "Пост" if idx == 0 else f"Ответ {idx}"
                parts_html.append(f"• <i>{prefix}</i>:\n{safe}")
            if not parts_plain:
                return None, None
            return "\n".join(parts_plain), "\n".join(parts_html)
        except Exception:
            return None, None

    context_plain, context_html = await _collect_context()

    def _build_body_html() -> str:
        full_name = " ".join(
            filter(None, [getattr(sender, "first_name", None), getattr(sender, "last_name", None)])
        ) or "отсутствует"
        username_val = (
            f"@{getattr(sender, 'username', '')}"
            if getattr(sender, "username", None)
            else "отсутствует"
        )
        chat_title_val = (
            getattr(chat_entity, "title", None)
            or getattr(chat_entity, "username", None)
            or (chat_entity.__class__.__name__ if chat_entity is not None else "")
        )
        underlined = f"<u>{escape_html(raw_text)}</u>"
        header = "<b>🔎 Обнаружен \"возможный диалог\"</b>"
        divider = "<b>━━━━━━━━━━━━━━━━━━━━</b>"
        parts_local = [
            header,
            divider,
            f"• <b>Имя</b>: {escape_html(full_name)}",
            f"• <b>Никнейм</b>: <i>{escape_html(username_val)}</i>",
            f"• <b>Чат</b>: <i>{escape_html(str(chat_title_val))}</i>",
            divider,
            f"<b>Сообщение</b>:\n{underlined}",
        ]
        if context_html:
            parts_local.extend([divider, context_html])
        parts_local.append(
            f"• <b>Ссылка</b>: <a href=\"{escape_html(link)}\">перейти</a>"
        )
        return "\n".join(parts_local)

    try:
        author_username = str(getattr(sender, "username", None) or "") or None
    except Exception:
        author_username = None

    try:
        author_access_hash = (
            int(getattr(sender, "access_hash", 0)) if sender is not None and getattr(sender, "access_hash", None) is not None else None
        )
    except Exception:
        author_access_hash = None

    owner_id = requester_user_id if requester_user_id is not None else fallback_user_id
    if owner_id is None:
        return None

    try:
        author_uid = int(getattr(sender, "id", 0) or 0) if sender is not None else None
    except Exception:
        author_uid = None

    new_sid = reply_ui_store.create(
        user_id=int(owner_id),
        original_body_html=_build_body_html(),
        original_text=raw_text,
        context_for_model=context_plain,
        classification_result=None,
        author_user_id=author_uid,
        source_chat_id=int(source_chat_id),
        author_username=author_username,
        author_access_hash=author_access_hash,
    )
    return new_sid, reply_ui_store.get(new_sid)


async def safe_delete_messages(
    chat_id: int,
    message_ids: Iterable[int],
    *,
    delay: float = 0.0,
    attempts: int = 3,
    retry_delay: float = 1.0,
) -> None:
    """Delete messages sequentially with optional retries, ignoring errors."""
    ids = [int(mid) for mid in message_ids if mid is not None]
    if not ids:
        return

    if delay > 0:
        await asyncio.sleep(delay)

    remaining = ids
    for attempt in range(attempts):
        next_round: list[int] = []
        for mid in remaining:
            try:
                await bot_api.bot_delete_message(int(chat_id), int(mid))
            except Exception:
                next_round.append(mid)
        if not next_round:
            return
        remaining = next_round
        if retry_delay > 0 and attempt < attempts - 1:
            await asyncio.sleep(retry_delay)

    # Final failure is silently ignored to match best-effort semantics.
