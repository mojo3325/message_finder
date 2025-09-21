from __future__ import annotations

import asyncio
import unicodedata
from datetime import timezone
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from telethon.tl.types import InputPeerUser

from logging_config import logger

from config import WARP_CONTEXT_LIMIT, WARP_LIST_PAGE_SIZE, WARP_MINIATURE_LAST, WARP_BOT_USERNAME
from const import CMD_ACCOUNT, ACCOUNT_STATUS_NONE, ACCOUNT_PROMPT_START
from services.drafts import drafts_store
from services.feedback import save_feedback
from services.user_sessions import create_client_from_session
import services.replier as replier_service

from tg import bot_api, ui
from utilities.accounts_store import get_user_account
from tg.context import escape_html
from tg.handlers.types import CallbackContext, MessageContext
from tg.helpers import (
    build_loading_frame,
    restore_reply_state_from_callback_message,
    shorten,
    start_loading_animation,
)
from tg.ui_state import (
    reply_ui_store,
    get_cached_dialogs,
    set_cached_dialogs,
    clear_cached_dialogs,
    get_warp_index,
    set_warp_index,
    get_warp_ui_message_id,
    set_warp_ui_message_id,
    clear_warp_ui_message_id,
)


_WARP_PREVIEW_SUPPRESS_PREFIXES: tuple[str, ...] = (
    "индексация чатов",
    "индексирую чат",
    "генерация ответа",
    "генерирую",
    "думаю над ответом",
    "анализирую контекст",
)

_WARP_SPINNER_GLYPHS = "⣾⣽⣻⢿⡿⣟⣯⣷"

_WARP_DIALOG_ENTRY_VERSION: int = 3
_WARP_DIALOG_FETCH_LIMIT: Optional[int] = 500

_WARP_EXCLUDED_USER_IDS: tuple[int, ...] = (777000,)


def _format_media_duration(raw_duration: Any) -> Optional[str]:
    try:
        seconds_float = float(raw_duration)
    except (TypeError, ValueError):
        return None
    if seconds_float < 0:
        return None
    total_seconds = int(round(seconds_float))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _extract_document_filename(document: Any) -> Optional[str]:
    attributes = getattr(document, "attributes", None)
    if not isinstance(attributes, (list, tuple)):
        return None
    for attr in attributes:
        filename = getattr(attr, "file_name", None)
        if isinstance(filename, str) and filename.strip():
            return filename.strip()
    return None


def _collect_media_descriptions(message: Any) -> list[tuple[str, str]]:
    descriptions: list[tuple[str, str]] = []

    photo = getattr(message, "photo", None)
    if photo is not None:
        descriptions.append(("🖼 Фото", "[Фото]"))

    voice = getattr(message, "voice", None)
    if voice is not None:
        duration = _format_media_duration(getattr(voice, "duration", None))
        if duration is None:
            attrs = getattr(voice, "attributes", None)
            if isinstance(attrs, (list, tuple)):
                for attr in attrs:
                    duration = _format_media_duration(getattr(attr, "duration", None))
                    if duration:
                        break
        label = "🎙 Голосовое сообщение"
        context_label = "[Голосовое сообщение]"
        if duration:
            label = f"{label} {duration}"
            context_label = f"[Голосовое сообщение {duration}]"
        descriptions.append((label, context_label))

    video = getattr(message, "video", None)
    video_note = getattr(message, "video_note", None)
    video_like = video or video_note
    if video_like is not None:
        duration = _format_media_duration(getattr(video_like, "duration", None))
        if duration is None:
            attrs = getattr(video_like, "attributes", None)
            if isinstance(attrs, (list, tuple)):
                for attr in attrs:
                    duration = _format_media_duration(getattr(attr, "duration", None))
                    if duration:
                        break
        label = "🎬 Видео"
        context_label = "[Видео]"
        if duration:
            label = f"{label} {duration}"
            context_label = f"[Видео {duration}]"
        descriptions.append((label, context_label))

    document = getattr(message, "document", None)
    if document is not None:
        mime_type_raw = getattr(document, "mime_type", None)
        mime_type = str(mime_type_raw or "")
        filename = _extract_document_filename(document)
        is_voice_doc = voice is not None
        is_video_doc = video_like is not None
        is_sticker = bool(getattr(message, "sticker", None))
        is_gif = bool(getattr(message, "gif", None))

        if mime_type.startswith("image/") and photo is None:
            label = "🖼 Фото"
            context_label = "[Фото]"
            if filename:
                label = f"{label} ({filename})"
                context_label = f"[Фото: {filename}]"
            descriptions.append((label, context_label))
        elif not (is_voice_doc or is_video_doc or is_sticker or is_gif):
            label = "📄 Документ"
            context_label = "[Документ]"
            detail = filename or (mime_type if mime_type else "")
            if detail:
                label = f"{label} ({detail})"
                context_label = f"[Документ: {detail}]"
            descriptions.append((label, context_label))

    return descriptions


def _build_message_preview_lines(message: Any) -> tuple[list[str], Optional[str]]:
    raw_text = getattr(message, "message", None)
    text_clean = str(raw_text or "").strip()
    preview_lines: list[str] = []
    if text_clean:
        preview_lines.append(text_clean)

    media_descriptions = _collect_media_descriptions(message)
    media_preview_lines = [desc[0] for desc in media_descriptions if desc[0]]
    preview_lines.extend(media_preview_lines)

    media_context_parts = [desc[1] for desc in media_descriptions if desc[1]]
    context_line: Optional[str]
    if text_clean:
        context_line = text_clean
        if media_context_parts:
            context_line = f"{context_line} {' '.join(media_context_parts)}"
    elif media_context_parts:
        context_line = " ".join(media_context_parts)
    else:
        context_line = None

    return preview_lines, context_line


def _build_dialog_preview_text(message: Any) -> Optional[str]:
    """Format a short preview for dialog listings, including media placeholders."""

    if message is None:
        return None

    preview_lines, context_line = _build_message_preview_lines(message)
    compact_lines = [str(line).strip() for line in preview_lines if str(line).strip()]
    preview_source = " · ".join(compact_lines)
    if not preview_source and context_line:
        preview_source = str(context_line).strip()
    if not preview_source:
        return None

    candidate = shorten(preview_source)
    return _clean_preview(candidate)


def _normalize_for_match(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    trimmed = text.strip()
    if not trimmed:
        return ""
    decomposed = unicodedata.normalize("NFKD", trimmed)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = without_marks.casefold()
    return lowered


def _normalize_username(raw: Any) -> Optional[str]:
    if isinstance(raw, str) and raw.strip():
        normalized = _normalize_for_match(raw)
        if not normalized:
            return None
        return normalized.lstrip("@")
    return None


def _extract_username(entity: Any) -> Optional[str]:
    username = _normalize_username(getattr(entity, "username", None))
    if username:
        return username
    usernames = getattr(entity, "usernames", None)
    if isinstance(usernames, (list, tuple)):
        for item in usernames:
            candidate = _normalize_username(getattr(item, "username", None))
            if candidate:
                return candidate
    return None


def _looks_like_bot_username(raw: Any) -> bool:
    username = _normalize_username(raw)
    if not username:
        return False
    warp_username = _normalize_username(WARP_BOT_USERNAME)
    if warp_username:
        normalized = username.replace("_", "")
        warp_normalized = warp_username.replace("_", "")
        if username == warp_username or normalized == warp_normalized:
            return True
    username_compact = username.replace("_", "")
    if username_compact in {"telegram", "telegrampremium"}:
        return True
    if username_compact.endswith("bot") or username_compact.endswith("support"):
        return True
    if "bot" in username_compact or "support" in username_compact:
        return True
    return False


def _looks_like_bot_display_name(raw: Any) -> bool:
    if raw is None:
        return False
    text = str(raw)
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    normalized = (
        normalized
        .replace("-", " ")
        .replace("—", " ")
        .replace("|", " ")
        .replace("_", " ")
    )
    tokens = [tok for tok in normalized.split() if tok]
    compact = normalized.replace(" ", "")
    bot_markers = {"bot", "бот", "support", "поддержка", "бота", "ботов"}
    for token in tokens:
        if token in bot_markers:
            return True
        if token.endswith("bot") or token.endswith("бот"):
            return True
        if "bot" in token or "бот" in token:
            return True
        if token.endswith("support"):
            return True
    warp_username = _normalize_username(WARP_BOT_USERNAME)
    if warp_username:
        warp_compact = warp_username.replace("_", "")
        if warp_compact and warp_compact in compact:
            return True
        warp_tokens = [tok for tok in warp_compact.split() if tok]
        if warp_tokens and all(tok in compact for tok in warp_tokens):
            return True
    if "warpchat" in compact or "warp" in compact and "chat" in compact:
        return True
    if "telegram" in compact or "telegrampremium" in compact:
        return True
    if "grey" in compact and "scheme" in compact:
        return True
    if "bot" in compact or "support" in compact:
        return True
    return False


def _clean_preview(preview: Optional[str]) -> Optional[str]:
    if not preview:
        return None
    cleaned = preview
    for glyph in _WARP_SPINNER_GLYPHS:
        cleaned = cleaned.replace(glyph, "")
    cleaned = cleaned.replace("…", "...")
    cleaned = cleaned.replace("▌", "").replace("█", "").replace("░", "")
    normalized = cleaned.strip().lower()
    for prefix in _WARP_PREVIEW_SUPPRESS_PREFIXES:
        if normalized.startswith(prefix):
            return None
    return preview


def _is_human_user_entity(entity: Any) -> bool:
    """Return True for private dialogs with real users only."""
    if entity is None:
        return False
    flags = (
        getattr(entity, "bot", False),
        getattr(entity, "is_bot", False),
        getattr(entity, "is_self", False),
        getattr(entity, "deleted", False),
        getattr(entity, "is_deleted", False),
        getattr(entity, "fake", False),
        getattr(entity, "is_fake", False),
        getattr(entity, "scam", False),
        getattr(entity, "is_scam", False),
        getattr(entity, "support", False),
        getattr(entity, "is_support", False),
    )
    if any(bool(flag) for flag in flags):
        return False
    try:
        entity_id = int(getattr(entity, "id", 0) or 0)
    except Exception:
        entity_id = 0
    if entity_id in _WARP_EXCLUDED_USER_IDS:
        # Telegram service notifications and other known non-human peers
        return False
    username = _extract_username(entity)
    if _looks_like_bot_username(username):
        return False
    display_parts = []
    for attr in ("first_name", "last_name"):
        try:
            value = getattr(entity, attr, None)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            display_parts.append(value.strip())
    display_name = " ".join(display_parts)
    if _looks_like_bot_display_name(display_name):
        return False
    return True


def _parse_access_hash(raw: Any) -> Optional[int]:
    try:
        if raw is None:
            return None
        value = int(raw)
        return value
    except (TypeError, ValueError):
        return None


def _lookup_dialog_entry(dialogs: Sequence[Dict[str, Any]], chat_id: int) -> Optional[Dict[str, Any]]:
    for item in dialogs:
        try:
            if int(item.get("chat_id")) == int(chat_id):
                return item
        except Exception:
            continue
    return None


def _build_input_peer_from_entry(entry: Optional[Dict[str, Any]]) -> Optional[InputPeerUser]:
    if not entry:
        return None
    access_hash = _parse_access_hash(entry.get("access_hash"))
    if access_hash is None:
        return None
    try:
        chat_id = int(entry.get("chat_id"))
    except Exception:
        return None
    return InputPeerUser(chat_id, access_hash)

def _is_valid_dialog_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        version = int(entry.get("version", 0) or 0)
    except Exception:
        return False
    if version != _WARP_DIALOG_ENTRY_VERSION:
        return False
    if _parse_access_hash(entry.get("access_hash")) is None:
        return False
    try:
        chat_id = int(entry.get("chat_id"))
    except Exception:
        return False
    if chat_id in _WARP_EXCLUDED_USER_IDS:
        return False
    if _looks_like_bot_username(entry.get("username")):
        return False
    if _looks_like_bot_display_name(entry.get("title")):
        return False
    if not bool(entry.get("is_human")):
        return False
    return True


def _filter_dialog_entries(dialogs: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [item for item in dialogs if _is_valid_dialog_entry(item)]


def _load_cached_dialogs(user_id: int) -> list[dict]:
    dialogs = get_cached_dialogs(int(user_id)) or []
    if not dialogs:
        return []
    filtered = _filter_dialog_entries(dialogs)
    if not filtered:
        try:
            clear_cached_dialogs(int(user_id))
        except Exception:
            pass
        return []
    if len(filtered) != len(dialogs):
        try:
            set_cached_dialogs(int(user_id), filtered)
        except Exception:
            pass
    return filtered


async def _refresh_dialogs_cache(
    user_id: int,
    *,
    client: Optional[Any] = None,
    limit: Optional[int] = _WARP_DIALOG_FETCH_LIMIT,
) -> list[dict]:
    acc = get_user_account(int(user_id))
    if not acc:
        try:
            clear_cached_dialogs(int(user_id))
        except Exception:
            pass
        return []

    managed_client = False
    active_client = client
    if active_client is None:
        active_client = create_client_from_session(acc.string_session)
        await active_client.connect()
        managed_client = True

    try:
        dialogs_raw = await _fetch_private_dialogs_with_preview(active_client, limit=limit)
        dialogs = _filter_dialog_entries(dialogs_raw)
        if dialogs:
            try:
                set_cached_dialogs(int(user_id), dialogs)
            except Exception:
                pass
        else:
            try:
                clear_cached_dialogs(int(user_id))
            except Exception:
                pass
        return dialogs
    finally:
        if managed_client:
            try:
                await active_client.disconnect()
            except Exception:
                pass


async def _ensure_dialogs_loaded(
    user_id: int,
    *,
    client: Optional[Any] = None,
    limit: Optional[int] = _WARP_DIALOG_FETCH_LIMIT,
) -> list[dict]:
    cached = _load_cached_dialogs(user_id)
    if cached:
        return cached
    return await _refresh_dialogs_cache(user_id, client=client, limit=limit)


async def _get_or_fetch_dialogs(
    user_id: int,
    *,
    client: Optional[Any] = None,
    limit: Optional[int] = _WARP_DIALOG_FETCH_LIMIT,
    force_refresh: bool = False,
) -> list[dict]:
    """Return cached dialogs or fetch and refresh the cache when needed."""

    cached_dialogs = list(_load_cached_dialogs(user_id))

    if force_refresh:
        try:
            refreshed = await _refresh_dialogs_cache(
                user_id,
                client=client,
                limit=limit,
            )
        except Exception:
            refreshed = []
        if refreshed:
            return _filter_dialog_entries(refreshed)
        return cached_dialogs

    if cached_dialogs:
        return cached_dialogs

    try:
        refreshed = await _refresh_dialogs_cache(
            user_id,
            client=client,
            limit=limit,
        )
    except Exception:
        refreshed = []

    if refreshed:
        return _filter_dialog_entries(refreshed)

    return cached_dialogs


async def _fetch_private_dialogs_with_preview(
    client: Any,
    limit: Optional[int] = _WARP_DIALOG_FETCH_LIMIT,
) -> list[dict]:
    dialogs: list[dict] = []
    async for d in client.iter_dialogs(limit=limit):
        try:
            if getattr(d, "is_user", False):
                entity = d.entity
                if not _is_human_user_entity(entity):
                    continue
                uid = int(getattr(entity, "id", 0) or 0)
                if uid <= 0:
                    continue
                access_hash = _parse_access_hash(getattr(entity, "access_hash", None))
                if access_hash is None:
                    continue
                fn = (getattr(entity, "first_name", None) or "").strip()
                ln = (getattr(entity, "last_name", None) or "").strip()
                username = _extract_username(entity)
                title = (f"{fn} {ln}".strip()) or (f"@{username}" if username else str(uid))

                last_msg = getattr(d, "message", None)
                preview = None
                time_str = None
                try:
                    preview = _build_dialog_preview_text(last_msg)
                except Exception:
                    preview = None
                try:
                    dt = getattr(last_msg, "date", None)
                    if dt is not None:
                        if getattr(dt, "tzinfo", None) is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        time_str = dt.strftime("%H:%M")
                except Exception:
                    time_str = None

                dialogs.append({
                    "chat_id": uid,
                    "access_hash": access_hash,
                    "title": title,
                    "preview": preview,
                    "time": time_str,
                    "username": username,
                    "is_human": True,
                    "version": _WARP_DIALOG_ENTRY_VERSION,
                })
        except Exception:
            continue
    return dialogs


async def _ensure_dialog_entry_for_user(
    user_id: int,
    chat_id: int,
    *,
    client: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    dialogs = get_cached_dialogs(int(user_id)) or []
    entry = _lookup_dialog_entry(dialogs, chat_id)
    if entry and _is_valid_dialog_entry(entry):
        return entry

    async def _refresh_with_client(active_client: Any, *, limit: Optional[int]) -> list[dict]:
        fresh_dialogs = await _fetch_private_dialogs_with_preview(active_client, limit=limit)
        filtered_dialogs = _filter_dialog_entries(fresh_dialogs)
        if filtered_dialogs:
            try:
                set_cached_dialogs(int(user_id), filtered_dialogs)
            except Exception:
                pass
        else:
            try:
                clear_cached_dialogs(int(user_id))
            except Exception:
                pass
        return filtered_dialogs

    async def _try_lookup(active_client: Any) -> Optional[Dict[str, Any]]:
        refreshed = await _refresh_with_client(active_client, limit=_WARP_DIALOG_FETCH_LIMIT)
        entry_local = _lookup_dialog_entry(refreshed, chat_id)
        if entry_local and _is_valid_dialog_entry(entry_local):
            return entry_local
        refreshed_full = await _refresh_with_client(active_client, limit=None)
        entry_full = _lookup_dialog_entry(refreshed_full, chat_id)
        if entry_full and _is_valid_dialog_entry(entry_full):
            return entry_full
        return None

    try:
        if client is not None:
            return await _try_lookup(client)

        acc = get_user_account(int(user_id))
        if not acc:
            return None
        temp_client = create_client_from_session(acc.string_session)
        await temp_client.connect()
        try:
            return await _try_lookup(temp_client)
        finally:
            try:
                await temp_client.disconnect()
            except Exception:
                pass
    except Exception:
        return None


_WARP_SPINNER_FRAMES: tuple[str, ...] = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_WARP_LOADING_SEQUENCE: tuple[str, ...] = (
    "Генерирую…",
    "Думаю над ответом…",
    "Анализирую контекст…",
)
_WARP_INDEXING_CHATS_HEADER = "Индексирую чаты…"
_WARP_INDEXING_CHAT_HEADER = "Индексирую чат…"
_WARP_GEN_LOADING_INTERVAL = 1.6


def _normalize_loading_headers(header_spec: Union[str, Sequence[str]]) -> tuple[str, ...]:
    if isinstance(header_spec, (list, tuple)):
        parts = tuple(str(h).strip() for h in header_spec if str(h).strip())
    else:
        value = str(header_spec).strip()
        parts = (value,) if value else tuple()
    return parts or ("Загрузка…",)


def _build_warp_loading_frame(header: str, step: int) -> str:
    spinner = _WARP_SPINNER_FRAMES[step % len(_WARP_SPINNER_FRAMES)]
    width = 10
    pos = step % width
    bar = "█" * (pos + 1) + "░" * (width - pos - 1)
    return "\n".join(
        [
            f"<b>{header} {spinner}</b>",
            "<b>━━━━━━━━━━━━━━━━━━━━</b>",
            f"<code>{bar}</code>",
        ]
    )


def _make_loading_placeholder(
    header_spec: Union[str, Sequence[str]],
    *,
    for_spinner: bool,
) -> tuple[str, tuple[str, ...]]:
    headers = _normalize_loading_headers(header_spec)
    if for_spinner:
        return _build_warp_loading_frame(headers[0], 0), headers
    # Static placeholder (no spinner animation requested)
    return headers[0], headers


def _start_warp_loading_animation(
    chat_id: int,
    message_id: int,
    header: Union[str, Sequence[str]],
    *,
    reply_markup: Optional[Dict[str, Any]] = None,
    interval: float = 0.8,
    freeze_on_last: bool = False,
) -> Tuple[asyncio.Event, asyncio.Task[None]]:
    stop_event: asyncio.Event = asyncio.Event()
    headers = _normalize_loading_headers(header)
    effective_interval = max(0.1, float(interval))

    async def _loop() -> None:
        step = 0
        try:
            while not stop_event.is_set():
                if freeze_on_last and headers:
                    header_idx = min(step, len(headers) - 1)
                    current_header = headers[header_idx]
                else:
                    current_header = headers[step % len(headers)]
                body = _build_warp_loading_frame(current_header, step)
                try:
                    await bot_api.bot_edit_message_text(
                        chat_id,
                        message_id,
                        body,
                        reply_markup=reply_markup,
                    )
                except Exception:
                    pass
                step += 1
                await asyncio.sleep(effective_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            try:
                logger.debug("warp_spinner_failed", extra={"extra": {"chat_id": chat_id}})
            except Exception:
                pass

    task = asyncio.create_task(_loop())
    return stop_event, task


async def _send_or_edit_loading_message(
    user_id: int,
    placeholder_spec: Union[str, Sequence[str]] = _WARP_LOADING_SEQUENCE,
    *,
    with_spinner: bool = True,
    reply_markup: Optional[Dict[str, Any]] = None,
    spinner_interval: float = 0.8,
    freeze_on_last: bool = False,
) -> tuple[Optional[int], Optional[asyncio.Event], Optional[asyncio.Task[None]]]:
    placeholder, headers = _make_loading_placeholder(placeholder_spec, for_spinner=with_spinner)
    message_id = get_warp_ui_message_id(user_id)
    stop_spinner: Optional[asyncio.Event] = None
    spinner_task: Optional[asyncio.Task[None]] = None

    if message_id is not None:
        edit_ok = await bot_api.bot_edit_message_text(user_id, message_id, placeholder, reply_markup=reply_markup)
        if not edit_ok:
            message_id = None

    if message_id is None:
        message_id = await bot_api.bot_send_html_message_with_id(user_id, placeholder, reply_markup=reply_markup)
        if message_id is None:
            await bot_api.bot_send_html_message(user_id, placeholder, reply_markup=reply_markup)
            clear_warp_ui_message_id(user_id)
            return None, None, None
        set_warp_ui_message_id(user_id, message_id)
    else:
        set_warp_ui_message_id(user_id, message_id)

    if with_spinner:
        stop_spinner, spinner_task = _start_warp_loading_animation(
            user_id,
            message_id,
            headers,
            reply_markup=reply_markup,
            interval=spinner_interval,
            freeze_on_last=freeze_on_last,
        )
    return message_id, stop_spinner, spinner_task


async def _deliver_warp_message(
    user_id: int,
    message_id: Optional[int],
    body: str,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    if message_id is not None:
        edit_ok = await bot_api.bot_edit_message_text(user_id, message_id, body, reply_markup=reply_markup)
        if edit_ok:
            set_warp_ui_message_id(user_id, message_id)
            return message_id
        message_id = None

    new_message_id = await bot_api.bot_send_html_message_with_id(user_id, body, reply_markup=reply_markup)
    if new_message_id is not None:
        set_warp_ui_message_id(user_id, new_message_id)
        return new_message_id

    await bot_api.bot_send_html_message(user_id, body, reply_markup=reply_markup)
    clear_warp_ui_message_id(user_id)
    return None


async def _collect_warp_dialog_state(
    cache_user_id: int,
    acc: Any,
    chat_id: int,
    *,
    cached_ctx: Optional[str],
    cached_ids: list[int],
) -> Tuple[str, list[int], list[dict], str, Optional[str]]:
    need_context = not cached_ctx or not cached_ids
    limit_ctx = int(WARP_CONTEXT_LIMIT) if need_context else 0
    limit_fetch = max(int(WARP_MINIATURE_LAST), limit_ctx)
    if limit_fetch <= 0:
        limit_fetch = int(WARP_MINIATURE_LAST)
    preview_buffer: list[dict] = []
    context_entries: list[str] = []
    context_ids: list[int] = []
    header_time: Optional[str] = None
    title = str(chat_id)
    client_idx: Optional[Any] = None
    owner_uid = int(getattr(acc, "telegram_user_id", 0) or 0)
    t0 = __import__("time").time()
    duration_ms = 0
    try:
        client_idx = create_client_from_session(acc.string_session)
        await client_idx.connect()
        dialog_entry = await _ensure_dialog_entry_for_user(cache_user_id, chat_id, client=client_idx)
        peer_target = _build_input_peer_from_entry(dialog_entry)
        fetch_target: Any = peer_target if peer_target is not None else chat_id
        if dialog_entry and dialog_entry.get("title"):
            title = str(dialog_entry.get("title") or str(chat_id))
        else:
            title = str(chat_id)
        try:
            ent = await client_idx.get_entity(fetch_target)
            fn = (getattr(ent, "first_name", None) or "").strip()
            ln = (getattr(ent, "last_name", None) or "").strip()
            uname = getattr(ent, "username", None)
            resolved_title = (f"{fn} {ln}".strip()) or (f"@{uname}" if uname else str(chat_id))
            if resolved_title:
                title = resolved_title
        except Exception:
            # Fall back to cached title if available
            if dialog_entry and dialog_entry.get("title"):
                title = str(dialog_entry.get("title") or title)
            else:
                title = str(chat_id)

        if need_context:
            try:
                logger.info(
                    "index_start",
                    extra={"extra": {"user_id": cache_user_id, "chat_id": chat_id}},
                )
            except Exception:
                pass

        async for mm in client_idx.iter_messages(entity=fetch_target, limit=max(1, limit_fetch)):
            try:
                dt = getattr(mm, "date", None)
                if header_time is None and dt is not None:
                    if getattr(dt, "tzinfo", None) is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    header_time = dt.strftime("%H:%M")
            except Exception:
                pass

            preview_lines, context_text = _build_message_preview_lines(mm)

            if preview_lines and len(preview_buffer) < int(WARP_MINIATURE_LAST):
                is_out = bool(getattr(mm, "out", False))
                author = "Вы" if is_out else title
                preview_buffer.append(
                    {
                        "direction": "out" if is_out else "in",
                        "author": author,
                        "text": "\n".join(preview_lines),
                    }
                )

            if need_context and context_text:
                try:
                    snd = await mm.get_sender()
                except Exception:
                    snd = None
                try:
                    auid = int(getattr(snd, "id", 0) or 0)
                except Exception:
                    auid = 0
                try:
                    auser = getattr(snd, "username", None)
                except Exception:
                    auser = None
                role = "OWNER" if bool(getattr(mm, "out", False)) or (owner_uid and auid and int(auid) == owner_uid) else "USER"
                ident = (
                    f"@{auser}"
                    if auser
                    else (f"id:{auid}" if auid else (f"id:{owner_uid}" if role == "OWNER" and owner_uid else "id:?"))
                )
                context_entries.append(f"{role} {ident}: {context_text}")
                try:
                    context_ids.append(int(getattr(mm, "id", 0) or 0))
                except Exception:
                    context_ids.append(0)

            mini_ready = len(preview_buffer) >= int(WARP_MINIATURE_LAST)
            ctx_ready = (not need_context) or (len(context_ids) >= int(WARP_CONTEXT_LIMIT))
            if mini_ready and ctx_ready:
                break

        if not preview_buffer:
            # Ensure at least stub entry to avoid empty box
            preview_buffer.append({"direction": "in", "author": title, "text": "Нет сообщений"})

        preview_buffer = list(reversed(preview_buffer))

        transcript = cached_ctx or ""
        message_ids = list(cached_ids)
        if need_context and context_entries and context_ids:
            context_entries = list(reversed(context_entries))
            context_ids = [int(x) for x in reversed(context_ids) if isinstance(x, (int, float)) and int(x) > 0]
            transcript = "\n".join(context_entries)
            message_ids = context_ids
            try:
                from datetime import datetime as _dt
                set_warp_index(
                    int(cache_user_id),
                    int(chat_id),
                    {
                        "indexed_at": _dt.utcnow().isoformat() + "Z",
                        "limit": int(WARP_CONTEXT_LIMIT),
                        "message_ids": list(message_ids),
                        "sample": (transcript[:160] if transcript else ""),
                        "transcript": transcript,
                        "miniature_last": int(WARP_MINIATURE_LAST),
                    },
                )
            except Exception:
                pass
            duration_ms = int(max(0.0, (__import__("time").time() - t0) * 1000))
            try:
                logger.info(
                    "index_done",
                    extra={
                        "extra": {
                            "user_id": cache_user_id,
                            "chat_id": chat_id,
                            "messages": len(message_ids),
                            "duration_ms": duration_ms,
                        }
                    },
                )
            except Exception:
                pass
        elif not need_context:
            try:
                logger.info(
                    "index_done",
                    extra={
                        "extra": {
                            "user_id": cache_user_id,
                            "chat_id": chat_id,
                            "messages": len(message_ids),
                            "cached": True,
                        }
                    },
                )
            except Exception:
                pass

        return transcript or "", message_ids, preview_buffer, title, header_time
    except Exception as exc:  # noqa: BLE001
        try:
            logger.error(
                "index_failed",
                extra={"extra": {"user_id": cache_user_id, "chat_id": chat_id, "error": str(exc)}},
            )
        except Exception:
            pass
        raise
    finally:
        if client_idx is not None:
            try:
                await client_idx.disconnect()
            except Exception:
                pass


async def _handle_warp_generation(
    *,
    sid: str,
    message_chat_id: int,
    message_id: int,
    warp_chat_id: Optional[int],
    draft_for_regen: Optional[Dict[str, Any]],
    from_user_id: int,
) -> None:
    target_chat_id: Optional[int] = None
    cache_user_id: Optional[int] = None
    if draft_for_regen is not None:
        try:
            target_chat_id = int(draft_for_regen.get("chat_id")) if draft_for_regen.get("chat_id") is not None else None
        except Exception:
            target_chat_id = None
        try:
            cache_user_id = int(draft_for_regen.get("telegram_user_id")) if draft_for_regen.get("telegram_user_id") is not None else int(from_user_id)
        except Exception:
            cache_user_id = int(from_user_id)
    else:
        target_chat_id = int(warp_chat_id) if warp_chat_id is not None else None
        cache_user_id = int(from_user_id)

    if target_chat_id is None or cache_user_id is None:
        await bot_api.bot_edit_message_text(
            message_chat_id,
            message_id,
            "Не удалось определить чат для генерации",
        )
        return

    acc = get_user_account(cache_user_id)
    if not acc:
        body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
        await bot_api.bot_edit_message_text(message_chat_id, message_id, body, reply_markup=kb)
        return

    cached = get_warp_index(cache_user_id, target_chat_id) or {}
    cached_ctx = cached.get("transcript")
    cached_ids_raw = cached.get("message_ids") or []
    cached_ids = [int(x) for x in cached_ids_raw if isinstance(x, (int, float, str))]

    existing_reply = None
    draft_id: Optional[str] = None
    if draft_for_regen is not None:
        draft_id = str(draft_for_regen.get("draft_id") or sid)
        try:
            existing_reply = str(draft_for_regen.get("text", "") or "").strip() or None
        except Exception:
            existing_reply = None
        ctx_ids_draft = draft_for_regen.get("context_message_ids") or []
        if not cached_ids and ctx_ids_draft:
            cached_ids = [int(x) for x in ctx_ids_draft if isinstance(x, (int, float, str))]

    transcript, message_ids, preview_messages, title, header_time = await _collect_warp_dialog_state(
        cache_user_id=cache_user_id,
        acc=acc,
        chat_id=target_chat_id,
        cached_ctx=str(cached_ctx) if cached_ctx else None,
        cached_ids=cached_ids,
    )

    reply_to_id = None

    loading_body, loading_kb = ui.build_warp_miniature(
        title,
        header_time,
        preview_messages,
        target_chat_id,
        loading=True,
        draft_id=draft_id,
        generated_reply=existing_reply,
    )
    await bot_api.bot_edit_message_text(
        message_chat_id,
        message_id,
        loading_body,
        reply_markup=loading_kb,
    )

    stop_spinner: Optional[asyncio.Event] = None
    spinner_task: Optional[asyncio.Task[None]] = None
    final_body: Optional[str] = None
    final_kb: Optional[Dict[str, Any]] = None

    try:
        try:
            stop_spinner, spinner_task = _start_warp_loading_animation(
                message_chat_id,
                message_id,
                _WARP_LOADING_SEQUENCE,
                reply_markup=loading_kb,
                interval=_WARP_GEN_LOADING_INTERVAL,
                freeze_on_last=True,
            )
        except Exception:
            stop_spinner = None
            spinner_task = None

        t0 = __import__("time").time()
        try:
            logger.info(
                "gen_start",
                extra={
                    "extra": {
                        "mode": "warp",
                        "sid": sid,
                        "warp_chat_id": target_chat_id,
                    }
                },
            )
        except Exception:
            pass

        reply_text_raw = await replier_service.generate_reply("", context=transcript)
        reply_text = str(reply_text_raw or "").strip()
        try:
            duration_ms = int(max(0.0, (__import__("time").time() - t0) * 1000))
            logger.info(
                "gen_done",
                extra={
                    "extra": {
                        "mode": "warp",
                        "sid": sid,
                        "warp_chat_id": target_chat_id,
                        "reply_len": len(reply_text),
                        "duration_ms": duration_ms,
                    }
                },
            )
        except Exception:
            pass

        if draft_id is None:
            payload_init: Dict[str, Any] = {
                "telegram_user_id": cache_user_id,
                "chat_id": target_chat_id,
                "reply_to_message_id": None,
                "text": reply_text,
                "html": None,
                "author_user_id": None,
                "source_chat_id": target_chat_id,
                "context_message_ids": list(message_ids),
            }
            draft_id = drafts_store.create(payload_init)
        else:
            drafts_store.update(
                str(draft_id),
                {
                    "text": reply_text,
                    "context_message_ids": list(message_ids),
                    "reply_to_message_id": None,
                },
            )

        final_body, final_kb = ui.build_warp_miniature(
            title,
            header_time,
            preview_messages,
            target_chat_id,
            generated_reply=reply_text,
            draft_id=str(draft_id),
        )
    finally:
        if stop_spinner is not None and not stop_spinner.is_set():
            stop_spinner.set()
            await asyncio.sleep(0)
        if spinner_task is not None:
            if not spinner_task.done():
                spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass

    if final_body is not None and final_kb is not None:
        await bot_api.bot_edit_message_text(
            message_chat_id,
            message_id,
            final_body,
            reply_markup=final_kb,
        )
        drafts_store.update(str(draft_id), {"html": final_body})
    else:
        raise RuntimeError("warp_generation_no_result")


async def handle_callback(
    ctx: CallbackContext,
    *,
    get_telethon_client: Callable[[], Any],
) -> bool:
    cb_id = ctx.callback_id
    action = ctx.action
    sid = ctx.sid
    msg_chat_id = ctx.chat_id
    msg_id = ctx.message_id
    from_user_id = ctx.user_id

    if cb_id is None or msg_chat_id is None or msg_id is None:
        return False

    msg = ctx.message
    st = reply_ui_store.get(sid) if sid else None

    if action == "noop":
        reason = str(sid or "")
        if reason == "gen":
            msg = "Подождите, идёт генерация…"
        elif reason == "send":
            msg = "Ответ ещё не готов"
        elif reason == "back":
            msg = "Подождите завершения операции"
        elif reason == "list":
            msg = "Загрузка…"
        else:
            msg = "Действие недоступно"
        await bot_api.bot_answer_callback_query(cb_id, text=msg, show_alert=False)
        return True

    if action == "send_disabled":
        await bot_api.bot_answer_callback_query(cb_id, text="Сначала сгенерируйте ответ", show_alert=True)
        return True

    if action == "acc_link":
        await bot_api.bot_answer_callback_query(cb_id, text="Привязка аккаунта", show_alert=False)
        try:
            kb = ui.build_account_start_keyboard()
        except Exception:
            kb = None
        await bot_api.bot_edit_message_text(
            msg_chat_id,
            msg_id,
            ACCOUNT_PROMPT_START,
            reply_markup=kb,
        )
        return True

    if action == "list":
        await bot_api.bot_answer_callback_query(cb_id, text="Загружаю…", show_alert=False)
        try:
            page = int(sid) if sid else 1
            if page <= 0:
                page = 1
        except (TypeError, ValueError):
            page = 1
        uid_effective = int(from_user_id) if from_user_id is not None else int(msg_chat_id)
        acc = get_user_account(uid_effective)
        if not acc:
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
            await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
            return True

        dialogs = _load_cached_dialogs(uid_effective)
        stop_spinner: Optional[asyncio.Event] = None
        spinner_task: Optional[asyncio.Task[None]] = None
        if not dialogs:
            try:
                stop_spinner, spinner_task = _start_warp_loading_animation(
                    int(msg_chat_id),
                    int(msg_id),
                    _WARP_INDEXING_CHATS_HEADER,
                )
            except Exception:
                stop_spinner = None
                spinner_task = None

            try:
                dialogs = await _get_or_fetch_dialogs(uid_effective, force_refresh=True)
            finally:
                if stop_spinner is not None:
                    stop_spinner.set()
                    await asyncio.sleep(0)
                if spinner_task is not None:
                    if not spinner_task.done():
                        spinner_task.cancel()
                    try:
                        await spinner_task
                    except asyncio.CancelledError:
                        pass
        dialogs = _filter_dialog_entries(dialogs)

        page_size = max(1, int(WARP_LIST_PAGE_SIZE))
        total_pages = max(1, (len(dialogs) + page_size - 1) // page_size)
        page = min(page, total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = dialogs[start:end]
        body, kb = ui.build_warp_chats_list("Мои приватные чаты", page_items, page, total_pages)
        await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
        try:
            logger.info(
                "list_open",
                extra={
                    "extra": {
                        "user_id": uid_effective,
                        "total": len(dialogs),
                        "source": "callback",
                        "page": page,
                    }
                },
            )
        except Exception:
            pass
        return True

    if action == "open":
        await bot_api.bot_answer_callback_query(cb_id, text="Открываю…", show_alert=False)
        try:
            target_chat_id = int(sid)
        except (TypeError, ValueError):
            await bot_api.bot_answer_callback_query(cb_id, text="Некорректный чат", show_alert=True)
            return True
        cache_user_id = int(from_user_id) if from_user_id is not None else int(msg_chat_id)
        acc = get_user_account(cache_user_id)
        if not acc:
            await bot_api.bot_answer_callback_query(cb_id, text="Аккаунт не подключён", show_alert=True)
            return True
        client = None
        stop_spinner: Optional[asyncio.Event] = None
        spinner_task: Optional[asyncio.Task[None]] = None
        try:
            stop_spinner, spinner_task = _start_warp_loading_animation(
                int(msg_chat_id),
                int(msg_id),
                _WARP_INDEXING_CHAT_HEADER,
            )
            client = create_client_from_session(acc.string_session)
            await client.connect()
            dialog_entry = await _ensure_dialog_entry_for_user(cache_user_id, target_chat_id, client=client)
            if dialog_entry is None:
                raise RuntimeError("dialog_not_available")
            peer_target = _build_input_peer_from_entry(dialog_entry)
            fetch_target: Any = peer_target if peer_target is not None else target_chat_id
            # Resolve entity and display title first (needed for authorship labels)
            try:
                ent = await client.get_entity(fetch_target)
                fn = (getattr(ent, "first_name", None) or "").strip()
                ln = (getattr(ent, "last_name", None) or "").strip()
                uname = getattr(ent, "username", None)
                title = (f"{fn} {ln}".strip()) or (f"@{uname}" if uname else str(target_chat_id))
            except Exception:
                raw_title = dialog_entry.get("title") if dialog_entry else None
                title = str(raw_title or target_chat_id)

            # Collect last messages with direction and header time (from newest)
            messages: list[dict] = []
            header_time: Optional[str] = None
            async for msg in client.iter_messages(entity=fetch_target, limit=max(1, int(WARP_MINIATURE_LAST))):
                if header_time is None:
                    try:
                        dt = getattr(msg, "date", None)
                        if dt is not None:
                            if getattr(dt, "tzinfo", None) is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            header_time = dt.strftime("%H:%M")
                    except Exception:
                        header_time = None
                preview_lines, _ = _build_message_preview_lines(msg)
                if not preview_lines:
                    continue
                is_out = bool(getattr(msg, "out", False))
                author = "Вы" if is_out else title
                messages.append({
                    "direction": "out" if is_out else "in",
                    "author": author,
                    "text": "\n".join(preview_lines),
                })
            messages.reverse()
            if stop_spinner is not None:
                stop_spinner.set()
                await asyncio.sleep(0)
            body, kb = ui.build_warp_miniature(title, header_time, messages, target_chat_id)
            await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
            try:
                logger.info(
                    "open_chat",
                    extra={
                        "extra": {
                            "user_id": int(from_user_id) if from_user_id is not None else int(msg_chat_id),
                            "chat_id": target_chat_id,
                        }
                    },
                )
            except Exception:
                pass
        except Exception as e:
            logger.error("warp_open_failed", extra={"extra": {"chat_id": target_chat_id, "error": str(e)}})
            await bot_api.bot_answer_callback_query(cb_id, text="Ошибка открытия", show_alert=True)
        finally:
            try:
                if client:
                    await client.disconnect()
            except Exception:
                pass
            if stop_spinner is not None and not stop_spinner.is_set():
                stop_spinner.set()
                await asyncio.sleep(0)
            if spinner_task is not None:
                if not spinner_task.done():
                    spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass
        return True

    if (not st) and sid:
        restored = await restore_reply_state_from_callback_message(
            msg,
            get_telethon_client=get_telethon_client,
            requester_user_id=int(from_user_id) if from_user_id is not None else None,
            fallback_user_id=int(msg_chat_id),
        )
        if restored is not None:
            sid, st = restored[0], restored[1]


    # Account onboarding callbacks (start message buttons)

    if action == "send":
        await bot_api.bot_answer_callback_query(cb_id, text="Отправляю…", show_alert=False)
        # Warp draft send if present
        draft = drafts_store.get(sid)
        if draft is not None:
            reply_text = str(draft.get("text", "") or "").strip()
            if not reply_text:
                await bot_api.bot_answer_callback_query(cb_id, text="Сначала сгенерируйте ответ", show_alert=True)
                return True
            acc = get_user_account(int(from_user_id) if from_user_id is not None else int(msg_chat_id))
            if acc is None:
                await bot_api.bot_answer_callback_query(cb_id, text="Аккаунт не подключён", show_alert=True)
                try:
                    from tg.ui import build_account_start_keyboard
                    await bot_api.bot_edit_message_text(
                        msg_chat_id,
                        msg_id,
                        "\n".join([
                            "Аккаунт не подключён.",
                            "Чтобы отправлять сообщения от вашего имени, привяжите аккаунт:",
                        ]),
                        reply_markup=build_account_start_keyboard(),
                    )
                except Exception:
                    pass
                try:
                    logger.info(
                        "send_skipped_no_session",
                        extra={
                            "extra": {
                                "user_id": int(from_user_id) if from_user_id is not None else int(msg_chat_id),
                                "chat_id": int(draft.get("chat_id") or 0),
                            }
                        },
                    )
                except Exception:
                    pass
                return True
            try:
                client = create_client_from_session(acc.string_session)
                await client.connect()
                entity = int(draft.get("chat_id"))
                try:
                    await client.send_message(entity, reply_text)
                finally:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                await bot_api.bot_answer_callback_query(cb_id, text="Отправлено", show_alert=False)
                try:
                    logger.info(
                        "send_ok",
                        extra={
                            "extra": {
                                "user_id": int(from_user_id) if from_user_id is not None else int(msg_chat_id),
                                "chat_id": int(entity),
                                "reply_len": int(len(reply_text)),
                            }
                        },
                    )
                except Exception:
                    pass
                try:
                    body_ok = "\n".join([
                        "<b>✉️ Сообщение отправлено</b>",
                        "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                        escape_html(reply_text),
                    ])
                    success_markup = {"inline_keyboard": [[{"text": "⬅ Назад", "callback_data": "back:chats"}]]}
                    await bot_api.bot_edit_message_text(
                        msg_chat_id,
                        msg_id,
                        body_ok,
                        reply_markup=success_markup,
                    )
                except Exception:
                    pass
                finally:
                    try:
                        drafts_store.delete(sid)
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                logger.error("send_via_account_failed", extra={"extra": {"user": from_user_id, "error": str(e)}})
                await bot_api.bot_answer_callback_query(cb_id, text="Не удалось отправить", show_alert=True)
            return True

        # Legacy ReplyUIState send
        if not st:
            await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
            return True
        reply_text = getattr(st, "last_reply_text", None)
        if not reply_text:
            await bot_api.bot_answer_callback_query(cb_id, text="Сначала сгенерируйте ответ", show_alert=True)
            return True
        acc = get_user_account(int(from_user_id) if from_user_id is not None else int(msg_chat_id))
        if acc is None:
            await bot_api.bot_answer_callback_query(cb_id, text="Аккаунт не подключён", show_alert=True)
            try:
                from tg.ui import build_account_start_keyboard
                await bot_api.bot_edit_message_text(
                    msg_chat_id,
                    msg_id,
                    "\n".join([
                        "Аккаунт не подключён.",
                        "Чтобы отправлять сообщения от вашего имени, привяжите аккаунт:",
                    ]),
                    reply_markup=build_account_start_keyboard(),
                )
            except Exception:
                pass
            return True
        try:
            client = create_client_from_session(acc.string_session)
            await client.connect()
            target_user_id = getattr(st, "author_user_id", None)
            target_username = getattr(st, "author_username", None)
            target_access_hash = getattr(st, "author_access_hash", None)

            entity: Any = None
            if target_user_id is not None:
                if target_access_hash is not None:
                    try:
                        entity = InputPeerUser(int(target_user_id), int(target_access_hash))
                    except Exception:
                        entity = None
                if entity is None:
                    try:
                        entity = await client.get_input_entity(int(target_user_id))
                    except Exception:
                        entity = None
            if entity is None and target_username:
                try:
                    entity = await client.get_input_entity(target_username)
                except Exception:
                    entity = None
            if entity is None:
                raise RuntimeError("no_target")
            try:
                await client.send_message(entity, reply_text)
            finally:
                try:
                    await client.disconnect()
                except Exception:
                    pass
            await bot_api.bot_answer_callback_query(cb_id, text="Отправлено", show_alert=False)
            try:
                logger.info(
                    "send_ok",
                    extra={
                        "extra": {
                            "user_id": int(from_user_id) if from_user_id is not None else int(msg_chat_id),
                            "chat_id": (
                                int(entity) if isinstance(entity, int)
                                else (int(getattr(entity, "user_id", 0)) if getattr(entity, "user_id", None) is not None else None)
                            ),
                            "reply_len": int(len(reply_text)),
                        }
                    },
                )
            except Exception:
                pass
            try:
                body_ok = "\n".join([
                    "<b>✉️ Сообщение отправлено</b>",
                    "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                    escape_html(reply_text),
                ])
                await bot_api.bot_edit_message_text(
                    msg_chat_id, msg_id, body_ok, reply_markup=ui.build_reply_keyboard(sid, None)
                )
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            logger.error("send_via_account_failed", extra={"extra": {"user": from_user_id, "error": str(e)}})
            await bot_api.bot_answer_callback_query(cb_id, text="Не удалось отправить", show_alert=True)
        return True

    if action == "gen" or action == "regen":
        await bot_api.bot_answer_callback_query(cb_id, text="Генерирую ответ…", show_alert=False)
        warp_chat_id: Optional[int] = None
        if not st and action == "gen":
            try:
                warp_chat_id = int(sid)
            except Exception:
                warp_chat_id = None
        draft_for_regen = drafts_store.get(sid) if (not st and action == "regen") else None
        if not st and warp_chat_id is None and draft_for_regen is None:
            await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
            return True
        reply_ui_store.clear_cancelled(str(sid))

        if warp_chat_id is not None or draft_for_regen is not None:
            try:
                await _handle_warp_generation(
                    sid=str(sid),
                    message_chat_id=int(msg_chat_id),
                    message_id=int(msg_id),
                    warp_chat_id=warp_chat_id,
                    draft_for_regen=draft_for_regen,
                    from_user_id=int(from_user_id) if from_user_id is not None else int(msg_chat_id),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "warp_gen_failed",
                    extra={"extra": {"sid": sid, "error": str(exc)}},
                )
                error_markup = {"inline_keyboard": [[{"text": "⬅ Назад", "callback_data": "back:chats"}]]}
                await bot_api.bot_edit_message_text(
                    msg_chat_id,
                    msg_id,
                    "Не удалось сгенерировать ответ",
                    reply_markup=error_markup,
                )
            return True

        if st is None:
            await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
            return True

        loading_headers: tuple[str, ...] = _WARP_LOADING_SEQUENCE
        first_header = loading_headers[0] if loading_headers else "Генерирую…"
        reply_keyboard = ui.build_reply_keyboard(sid, None)
        loading_body = build_loading_frame(0, first_header)
        try:
            await bot_api.bot_edit_message_text(
                msg_chat_id,
                msg_id,
                loading_body,
                reply_markup=reply_keyboard,
            )
        except Exception:
            pass

        stop_anim: Optional[asyncio.Event] = None
        spinner_task: Optional[asyncio.Task[None]] = None
        try:
            stop_anim, spinner_task = await start_loading_animation(
                int(msg_chat_id),
                int(msg_id),
                sid,
                header=loading_headers,
                reply_markup=reply_keyboard,
                interval=_WARP_GEN_LOADING_INTERVAL,
                freeze_on_last=True,
            )
        except Exception:
            stop_anim = None
            spinner_task = None

        _orig_text = getattr(st, "original_text", "")
        _ctx = getattr(st, "context_for_model", None)

        async def _stop_spinner() -> None:
            nonlocal stop_anim, spinner_task
            event_local = stop_anim
            task_local = spinner_task
            if event_local is not None and not event_local.is_set():
                try:
                    event_local.set()
                except Exception:
                    pass
            if task_local is not None and not task_local.done():
                try:
                    await task_local
                except asyncio.CancelledError:
                    pass
            stop_anim = None
            spinner_task = None

        async def _legacy_generate() -> None:
            try:
                logger.info(
                    "gen_start",
                    extra={"extra": {"mode": "legacy", "sid": sid}},
                )
                t0 = __import__("time").time()
                reply_text_raw = await replier_service.generate_reply(_orig_text, context=_ctx)
                reply_text = str(reply_text_raw or "").strip()
                if reply_ui_store.is_cancelled(sid):
                    return
                reply_ui_store.set_reply(sid, reply_text)
                safe = escape_html(reply_text)
                body = "\n".join([
                    "<b>✍️ Предложенный ответ</b>",
                    "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                    safe,
                ])
                reply_markup = ui.build_reply_keyboard(sid, reply_text)
                await _stop_spinner()
                await bot_api.bot_edit_message_text(
                    msg_chat_id,
                    msg_id,
                    body,
                    reply_markup=reply_markup,
                )
                duration_ms = int(max(0.0, (__import__("time").time() - t0) * 1000))
                logger.info(
                    "gen_done",
                    extra={
                        "extra": {
                            "mode": "legacy",
                            "sid": sid,
                            "reply_len": len(reply_text),
                            "duration_ms": duration_ms,
                        }
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "gen_bg_error",
                    extra={"extra": {"sid": sid, "error": str(exc)}},
                )
                await _stop_spinner()
                await bot_api.bot_edit_message_text(
                    msg_chat_id,
                    msg_id,
                    "Не удалось сгенерировать ответ",
                )
            finally:
                await _stop_spinner()

        asyncio.create_task(_legacy_generate())
        return True




    if action == "dislike":
        await bot_api.bot_answer_callback_query(cb_id, text="Спасибо за отзыв!", show_alert=False)
        if not st:
            logger.warning(
                "feedback_handler_missing_state", extra={"extra": {"sid": sid, "action": action}}
            )
            return True

        feedback_data = {
            "message": st.original_text,
            "output": {"classification": "0"},
            "label": 0,
        }

        try:
            await asyncio.to_thread(save_feedback, feedback_data)
            logger.info("feedback_saved", extra={"extra": {"sid": sid, "action": action}})
        except Exception as e:
            logger.error("save_feedback_failed", extra={"extra": {"error": str(e)}})

        # Try to delete the bot message tied to this callback
        try:
            await bot_api.bot_delete_message(msg_chat_id, msg_id)
        except Exception as e:
            logger.error("delete_message_failed", extra={"extra": {"chat_id": msg_chat_id, "message_id": msg_id, "error": str(e)}})

        return True

    if action == "back":
        await bot_api.bot_answer_callback_query(cb_id, text="Возвращаю…", show_alert=False)
        # Warp Chat: back to chats list
        if sid == "chats":
            uid_effective = int(from_user_id) if from_user_id is not None else int(msg_chat_id)
            acc = get_user_account(uid_effective)
            if not acc:
                body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
                await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
                return True
            dialogs_cached = _load_cached_dialogs(uid_effective)
            dialogs = list(dialogs_cached)
            stop_spinner: Optional[asyncio.Event] = None
            spinner_task: Optional[asyncio.Task[None]] = None
            try:
                stop_spinner, spinner_task = _start_warp_loading_animation(
                    int(msg_chat_id),
                    int(msg_id),
                    _WARP_INDEXING_CHATS_HEADER,
                )
            except Exception:
                stop_spinner = None
                spinner_task = None

            try:
                dialogs = await _get_or_fetch_dialogs(uid_effective, force_refresh=True)
                if not dialogs and dialogs_cached:
                    dialogs = dialogs_cached
            finally:
                if stop_spinner is not None:
                    stop_spinner.set()
                    await asyncio.sleep(0)
                if spinner_task is not None:
                    if not spinner_task.done():
                        spinner_task.cancel()
                    try:
                        await spinner_task
                    except asyncio.CancelledError:
                        pass
            dialogs = _filter_dialog_entries(dialogs)
            page_size = max(1, int(WARP_LIST_PAGE_SIZE))
            total_pages = max(1, (len(dialogs) + page_size - 1) // page_size)
            page = 1
            page_items = dialogs[:page_size]
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", page_items, page, total_pages)
            await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
            try:
                logger.info(
                    "list_open",
                    extra={
                        "extra": {
                            "user_id": uid_effective,
                            "total": len(dialogs),
                            "source": "back",
                            "page": page,
                        }
                    },
                )
            except Exception:
                pass
            return True
        draft = drafts_store.get(sid)
        if draft is not None:
            try:
                u_id = int(draft.get("telegram_user_id"))
                ch_id = int(draft.get("chat_id"))
            except (TypeError, ValueError):
                u_id = int(from_user_id) if from_user_id is not None else int(msg_chat_id)
                ch_id = draft.get("chat_id") or 0
            # Return to chats list instead of placeholder
            acc_local = get_user_account(u_id)
            if not acc_local:
                body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
                await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
                return True
            dialogs_cached = _load_cached_dialogs(u_id)
            dialogs = list(dialogs_cached)
            stop_spinner: Optional[asyncio.Event] = None
            spinner_task: Optional[asyncio.Task[None]] = None
            try:
                stop_spinner, spinner_task = _start_warp_loading_animation(
                    int(msg_chat_id),
                    int(msg_id),
                    _WARP_INDEXING_CHATS_HEADER,
                )
            except Exception:
                stop_spinner = None
                spinner_task = None

            try:
                dialogs = await _get_or_fetch_dialogs(u_id, force_refresh=True)
                if not dialogs and dialogs_cached:
                    dialogs = dialogs_cached
            finally:
                if stop_spinner is not None:
                    stop_spinner.set()
                    await asyncio.sleep(0)
                if spinner_task is not None:
                    if not spinner_task.done():
                        spinner_task.cancel()
                    try:
                        await spinner_task
                    except asyncio.CancelledError:
                        pass
            dialogs = _filter_dialog_entries(dialogs)
            page = 1
            page_size = max(1, int(WARP_LIST_PAGE_SIZE))
            total_pages = max(1, (len(dialogs) + page_size - 1) // page_size)
            page_items = dialogs[:page_size]
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", page_items, page, total_pages)
            await bot_api.bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=kb)
            try:
                logger.info(
                    "list_open",
                    extra={
                        "extra": {
                            "user_id": u_id,
                            "total": len(dialogs),
                            "source": "back_draft",
                            "page": page,
                            "chat_id": ch_id,
                        }
                    },
                )
            except Exception:
                pass
            return True
        if not st:
            await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
            return True
        reply_ui_store.mark_cancelled(sid)
        orig_keyboard = ui.build_feedback_keyboard(sid)
        ok = await bot_api.bot_edit_message_text(
            msg_chat_id, msg_id, st.original_body_html, reply_markup=orig_keyboard
        )
        if not ok:
            await bot_api.bot_send_html_message(
                msg_chat_id, st.original_body_html, reply_markup=orig_keyboard
            )
        return True
    try:
        logger.debug(
            "warp_unknown_callback",
            extra={"extra": {"action": action, "sid": sid}},
        )
    except Exception:
        pass
    return False


async def handle_start_payload(
    ctx: MessageContext,
    *,
    get_telethon_client: Callable[[], Any],
) -> bool:
    user_id = ctx.user_id
    text_s = ctx.text
    payload: Optional[str] = None
    parts = text_s.split(maxsplit=1)
    if len(parts) == 2:
        payload = parts[1].strip()
    if not payload:
        return False

    async def _delete_source_message() -> None:
        try:
            await bot_api.bot_delete_message(ctx.chat_id, ctx.message_id)
        except Exception:
            pass

    if payload.startswith("open_"):
        try:
            target_chat_id = int(payload.replace("open_", "", 1))
        except (TypeError, ValueError):
            await bot_api.bot_send_html_message(user_id, "Некорректный идентификатор чата")
            await _delete_source_message()
            return True
        acc = get_user_account(user_id)
        if not acc:
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
            await bot_api.bot_send_html_message(user_id, body, reply_markup=kb)
            await _delete_source_message()
            return True
        message_id, stop_spinner, spinner_task = await _send_or_edit_loading_message(
            user_id,
            _WARP_INDEXING_CHAT_HEADER,
        )
        client = None
        try:
            client = create_client_from_session(acc.string_session)
            await client.connect()
            dialog_entry = await _ensure_dialog_entry_for_user(user_id, target_chat_id, client=client)
            if dialog_entry is None:
                raise RuntimeError("dialog_not_available")
            peer_target = _build_input_peer_from_entry(dialog_entry)
            fetch_target: Any = peer_target if peer_target is not None else target_chat_id
            try:
                ent = await client.get_entity(fetch_target)
                fn = (getattr(ent, "first_name", None) or "").strip()
                ln = (getattr(ent, "last_name", None) or "").strip()
                uname = getattr(ent, "username", None)
                title = (f"{fn} {ln}".strip()) or (f"@{uname}" if uname else str(target_chat_id))
            except Exception:
                raw_title = dialog_entry.get("title") if dialog_entry else None
                title = str(raw_title or target_chat_id)

            messages: list[dict] = []
            header_time: Optional[str] = None
            async for msg in client.iter_messages(entity=fetch_target, limit=max(1, int(WARP_MINIATURE_LAST))):
                if header_time is None:
                    try:
                        dt = getattr(msg, "date", None)
                        if dt is not None:
                            if getattr(dt, "tzinfo", None) is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            header_time = dt.strftime("%H:%M")
                    except Exception:
                        header_time = None
                try:
                    raw = getattr(msg, "message", None)
                except Exception:
                    raw = None
                if not raw:
                    continue
                txt = str(raw).strip()
                if not txt:
                    continue
                is_out = bool(getattr(msg, "out", False))
                author = "Вы" if is_out else title
                messages.append({
                    "direction": "out" if is_out else "in",
                    "author": author,
                    "text": txt,
                })
            messages.reverse()
            if stop_spinner is not None:
                stop_spinner.set()
                await asyncio.sleep(0)
            body, kb = ui.build_warp_miniature(title, header_time, messages, target_chat_id)
            message_id = await _deliver_warp_message(user_id, message_id, body, reply_markup=kb)
            try:
                logger.info(
                    "open_chat",
                    extra={"extra": {"user_id": user_id, "chat_id": target_chat_id, "source": "start_payload"}},
                )
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            logger.error("start_open_failed", extra={"extra": {"user_id": user_id, "error": str(e)}})
            await bot_api.bot_send_html_message(user_id, "Ошибка открытия чата")
        finally:
            try:
                if client:
                    await client.disconnect()
            except Exception:
                pass
            if stop_spinner is not None and not stop_spinner.is_set():
                stop_spinner.set()
                await asyncio.sleep(0)
            if spinner_task is not None:
                if not spinner_task.done():
                    spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass
            await _delete_source_message()
        return True

    # /start list_<page>
    if payload.startswith("list_"):
        try:
            page = int(payload.replace("list_", "", 1))
        except (TypeError, ValueError):
            page = 1
        acc = get_user_account(user_id)
        if not acc:
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
            await bot_api.bot_send_html_message(user_id, body, reply_markup=kb)
            await _delete_source_message()
            return True
        message_id, stop_spinner, spinner_task = await _send_or_edit_loading_message(
            user_id,
            _WARP_INDEXING_CHATS_HEADER,
        )
        dialogs_cached = _load_cached_dialogs(user_id)
        dialogs = list(dialogs_cached)
        if not dialogs:
            dialogs = await _get_or_fetch_dialogs(user_id, force_refresh=True)
        if stop_spinner is not None:
            stop_spinner.set()
            await asyncio.sleep(0)
        dialogs = _filter_dialog_entries(dialogs)
        page_size = max(1, int(WARP_LIST_PAGE_SIZE))
        total_pages = max(1, (len(dialogs) + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = dialogs[start:end]
        body, kb = ui.build_warp_chats_list("Мои приватные чаты", page_items, page, total_pages)
        await _deliver_warp_message(user_id, message_id, body, reply_markup=kb)
        if spinner_task is not None:
            if not spinner_task.done():
                spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass
        await _delete_source_message()
        try:
            logger.info(
                "list_open",
                extra={"extra": {"user_id": user_id, "total": len(dialogs), "source": "start_payload", "page": page}},
            )
        except Exception:
            pass
        return True

    return False


async def handle_command(
    ctx: MessageContext,
    *,
    get_telethon_client: Callable[[], Any],
) -> bool:
    user_id = ctx.user_id
    text_raw = ctx.text.strip()
    lower = text_raw.lower()

    if lower.startswith("open:"):
        suffix = text_raw.split(":", 1)[1].strip() if ":" in text_raw else ""
        text_raw = f"/open_{suffix}" if suffix else "/open_"
        lower = text_raw.lower()
    elif lower.startswith("list:"):
        suffix = text_raw.split(":", 1)[1].strip() if ":" in text_raw else ""
        text_raw = f"/list_{suffix}" if suffix else "/list_"
        lower = text_raw.lower()

    if lower == CMD_ACCOUNT:
        acc = get_user_account(user_id)
        if not acc:
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
            text_parts = [ACCOUNT_STATUS_NONE, "", body]
            await bot_api.bot_send_html_message(user_id, "\n".join(part for part in text_parts if part), reply_markup=kb)
            return True
        message_id, stop_spinner, spinner_task = await _send_or_edit_loading_message(
            user_id,
            _WARP_INDEXING_CHATS_HEADER,
        )
        try:
            dialogs = await _get_or_fetch_dialogs(user_id, force_refresh=True)
        finally:
            if stop_spinner is not None:
                stop_spinner.set()
                await asyncio.sleep(0)
            if spinner_task is not None:
                if not spinner_task.done():
                    spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass

        dialogs = _filter_dialog_entries(dialogs)
        page_size = max(1, int(WARP_LIST_PAGE_SIZE))
        total = len(dialogs)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = 1
        page_items = dialogs[:page_size]
        body, kb = ui.build_warp_chats_list("Мои приватные чаты", page_items, page, total_pages)
        await _deliver_warp_message(user_id, message_id, body, reply_markup=kb)
        try:
            logger.info(
                "list_open",
                extra={"extra": {"user_id": user_id, "total": total, "source": "command"}},
            )
        except Exception:
            pass
        return True

    if lower.startswith("/open_"):
        try:
            target_chat_id = int(lower.replace("/open_", "", 1))
        except (TypeError, ValueError):
            await bot_api.bot_send_html_message(user_id, "Некорректный идентификатор чата")
            return True
        acc = get_user_account(user_id)
        if not acc:
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
            await bot_api.bot_send_html_message(user_id, body, reply_markup=kb)
            return True
        message_id, stop_spinner, spinner_task = await _send_or_edit_loading_message(
            user_id,
            _WARP_INDEXING_CHAT_HEADER,
        )
        client = None
        try:
            client = create_client_from_session(acc.string_session)
            await client.connect()
            dialog_entry = await _ensure_dialog_entry_for_user(user_id, target_chat_id, client=client)
            if dialog_entry is None:
                raise RuntimeError("dialog_not_available")
            peer_target = _build_input_peer_from_entry(dialog_entry)
            fetch_target: Any = peer_target if peer_target is not None else target_chat_id
            try:
                ent = await client.get_entity(fetch_target)
                fn = (getattr(ent, "first_name", None) or "").strip()
                ln = (getattr(ent, "last_name", None) or "").strip()
                uname = getattr(ent, "username", None)
                title = (f"{fn} {ln}".strip()) or (f"@{uname}" if uname else str(target_chat_id))
            except Exception:
                raw_title = dialog_entry.get("title") if dialog_entry else None
                title = str(raw_title or target_chat_id)
            messages: list[dict] = []
            header_time: Optional[str] = None
            async for msg in client.iter_messages(entity=fetch_target, limit=max(1, int(WARP_MINIATURE_LAST))):
                if header_time is None:
                    try:
                        dt = getattr(msg, "date", None)
                        if dt is not None:
                            if getattr(dt, "tzinfo", None) is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            header_time = dt.strftime("%H:%M")
                    except Exception:
                        header_time = None
                try:
                    raw = getattr(msg, "message", None)
                except Exception:
                    raw = None
                if not raw:
                    continue
                txt = str(raw).strip()
                if not txt:
                    continue
                is_out = bool(getattr(msg, "out", False))
                author = "Вы" if is_out else title
                messages.append({
                    "direction": "out" if is_out else "in",
                    "author": author,
                    "text": txt,
                })
            messages.reverse()
            if stop_spinner is not None:
                stop_spinner.set()
                await asyncio.sleep(0)
            body, kb = ui.build_warp_miniature(title, header_time, messages, target_chat_id)
            message_id = await _deliver_warp_message(user_id, message_id, body, reply_markup=kb)
            try:
                logger.info(
                    "open_chat",
                    extra={"extra": {"user_id": user_id, "chat_id": target_chat_id, "source": "command"}},
                )
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            logger.error("cmd_open_failed", extra={"extra": {"user_id": user_id, "error": str(e)}})
            await bot_api.bot_send_html_message(user_id, "Ошибка открытия чата")
        finally:
            try:
                if client:
                    await client.disconnect()
            except Exception:
                pass
            if stop_spinner is not None and not stop_spinner.is_set():
                stop_spinner.set()
                await asyncio.sleep(0)
            if spinner_task is not None:
                if not spinner_task.done():
                    spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass
        return True

    if lower.startswith("/list_"):
        try:
            page = int(lower.replace("/list_", "", 1))
        except (TypeError, ValueError):
            page = 1
        acc = get_user_account(user_id)
        if not acc:
            body, kb = ui.build_warp_chats_list("Мои приватные чаты", [], 1, 1, session_available=False)
            await bot_api.bot_send_html_message(user_id, body, reply_markup=kb)
            return True
        message_id, stop_spinner, spinner_task = await _send_or_edit_loading_message(
            user_id,
            _WARP_INDEXING_CHATS_HEADER,
        )
        dialogs_cached = _load_cached_dialogs(user_id)
        dialogs = list(dialogs_cached)
        fetch_error: Optional[Exception] = None
        try:
            if not dialogs:
                dialogs = await _get_or_fetch_dialogs(user_id, force_refresh=True)
        except Exception as exc:  # noqa: BLE001
            fetch_error = exc
            dialogs = list(dialogs_cached)
        finally:
            if stop_spinner is not None and not stop_spinner.is_set():
                stop_spinner.set()
                await asyncio.sleep(0)
            if spinner_task is not None:
                if not spinner_task.done():
                    spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass

        if fetch_error is not None and not dialogs:
            logger.error(
                "cmd_list_failed",
                extra={"extra": {"user_id": user_id, "error": str(fetch_error)}},
            )
            await bot_api.bot_send_html_message(user_id, "Ошибка загрузки списка")
            return True

        dialogs = _filter_dialog_entries(dialogs)
        page_size = max(1, int(WARP_LIST_PAGE_SIZE))
        total_pages = max(1, (len(dialogs) + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = dialogs[start:end]
        body, kb = ui.build_warp_chats_list("Мои приватные чаты", page_items, page, total_pages)
        await _deliver_warp_message(user_id, message_id, body, reply_markup=kb)
        try:
            logger.info(
                "list_open",
                extra={"extra": {"user_id": user_id, "total": len(dialogs), "source": "command_list", "page": page}},
            )
        except Exception:
            pass
        return True
        # No linked account → start linking flow
        st = account_fsm.get(user_id)
        st.state = "ACCOUNT_IDLE"
        st.phone = None
        st.phone_code_hash = None
        st.tmp_client_session = None
        body = ACCOUNT_PROMPT_START
        kb = ui.build_account_start_keyboard()
        mid = await bot_api.bot_send_html_message_with_id(user_id, body, reply_markup=kb)
        st.ui_message_id = int(mid) if mid is not None else None
        # try remember the user's command message (if possible) for later deletion
        try:
            msg_id_user = int((message or {}).get("message_id"))
            st.user_message_ids.append(msg_id_user)
        except Exception:
            pass
        return True

    return False
