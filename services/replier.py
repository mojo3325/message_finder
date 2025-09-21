from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from config import GEMINI_RATE_RPD, GEMINI_REPLY_MODEL
from const import DIALOGUE_PROMPT
from core.rate_limiter import estimate_prompt_tokens, gemini_rate_limiter
from logging_config import logger
from services.clients import get_gemini_client
from services.errors import (
    is_gemini_quota_exhausted_error,
    is_gemini_transient_or_rate_error,
)


StructuredContextEntry = Dict[str, Any]
ReplyContext = Union[str, Dict[str, Any], Sequence[StructuredContextEntry]]


def normalize_short_reply(raw: str) -> str:
    """Normalize model output by trimming whitespace and redundant quotes."""
    if not raw:
        return ""

    text = (raw or "").strip()

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    text = re.sub(r"[\t \f\v]+", " ", text).strip()

    return text


def _parse_reply_context(context: Optional[ReplyContext]) -> Tuple[str, List[StructuredContextEntry]]:
    context_text = ""
    entries: List[StructuredContextEntry] = []
    if context is None:
        return context_text, entries
    if isinstance(context, str):
        return context.strip(), entries
    if isinstance(context, dict):
        context_text = str(context.get("text") or "").strip()
        raw_entries = context.get("entries")
        if isinstance(raw_entries, (list, tuple)):
            for entry in raw_entries:
                if isinstance(entry, dict):
                    entries.append(entry)
        return context_text, entries
    if isinstance(context, (list, tuple)):
        for entry in context:
            if isinstance(entry, dict):
                entries.append(entry)
    return context_text, entries


def _format_context_entry(entry: StructuredContextEntry) -> str:
    role_raw = entry.get("role")
    role = str(role_raw).strip() if isinstance(role_raw, str) else str(role_raw or "USER").strip() or "USER"

    identifier_raw = entry.get("identifier")
    if isinstance(identifier_raw, str):
        identifier = identifier_raw.strip()
    elif identifier_raw is None:
        identifier = ""
    else:
        identifier = str(identifier_raw).strip()

    text_raw = entry.get("text")
    message_text = str(text_raw).strip() if isinstance(text_raw, str) else ""

    prefix = role
    if identifier:
        prefix = f"{role} {identifier}"

    return f"{prefix}: {message_text}" if message_text else f"{prefix}:"


def _should_add_segment(candidate: Optional[str], existing_segments: Sequence[str]) -> bool:
    if not isinstance(candidate, str):
        return False
    text = candidate.strip()
    if not text:
        return False
    for existing in existing_segments:
        if not isinstance(existing, str):
            continue
        existing_text = existing.strip()
        if not existing_text:
            continue
        if text == existing_text:
            return False
        if text in existing_text or existing_text in text:
            return False
    return True


def _describe_media_for_log(media: Dict[str, Any]) -> str:
    label_source = media.get("display") or media.get("context") or media.get("type") or "media"
    label = str(label_source).strip() if isinstance(label_source, str) else str(label_source)

    extras: List[str] = []
    for key in ("format", "duration", "subtype"):
        value = media.get(key)
        if isinstance(value, str) and value.strip():
            extras.append(f"{key}={value.strip()}")

    if extras:
        return f"[MEDIA {label} ({', '.join(extras)})]"
    return f"[MEDIA {label}]"


def _infer_format_from_mime(mime: Optional[str]) -> Optional[str]:
    if not isinstance(mime, str):
        return None
    parts = mime.split("/", 1)
    if len(parts) != 2:
        return None
    subtype = parts[1].strip().lower()
    if not subtype:
        return None
    if subtype == "jpeg":
        return "jpg"
    if "+" in subtype:
        subtype = subtype.split("+", 1)[0]
    return subtype or None


def _normalize_audio_format_hint(raw: Optional[str]) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if not value:
        return None

    aliases = {
        "mpeg": "mp3",
        "mpga": "mp3",
        "mp2": "mp3",
        "mp1": "mp3",
        "mpega": "mp3",
        "wave": "wav",
        "x-wav": "wav",
        "x-pn-wav": "wav",
        "oga": "ogg",
        "opus": "ogg",
    }
    return aliases.get(value, value)


def _media_to_content_parts(media: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    data_raw = media.get("data")
    if not isinstance(data_raw, str) or not data_raw.strip():
        return parts
    data = data_raw.strip()

    media_type = str(media.get("type") or "").lower()

    format_hint_raw = media.get("format")
    format_hint = str(format_hint_raw).strip().lower() if isinstance(format_hint_raw, str) else ""

    mime_raw = media.get("mime_type")
    mime_type = str(mime_raw).strip() if isinstance(mime_raw, str) else ""

    if not format_hint:
        inferred = _infer_format_from_mime(mime_type)
        if inferred:
            format_hint = inferred

    if media_type in {"photo", "image"}:
        mime_to_use = mime_type or "image/jpeg"
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_to_use};base64,{data}"}})
    elif media_type in {"voice", "audio"}:
        fmt_candidate = format_hint or _infer_format_from_mime(mime_type)
        fmt = _normalize_audio_format_hint(fmt_candidate)
        if fmt in {"mp3", "wav"}:
            parts.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
        else:
            logger.warning(
                "Skipping audio attachment with unsupported format",
                extra={"format": fmt_candidate or "unknown", "media_type": media_type},
            )
    elif media_type == "video":
        fmt = format_hint or "mp4"
        parts.append({"type": "input_video", "input_video": {"data": data, "format": fmt}})

    return parts


def _estimate_context_tokens(context_text: str, entries: Sequence[StructuredContextEntry]) -> int:
    text_for_estimation = context_text.strip()
    if not text_for_estimation and entries:
        rendered = [
            _format_context_entry(entry)
            for entry in entries
            if isinstance(entry, dict)
        ]
        text_for_estimation = "\n".join(line for line in rendered if line)
    if not text_for_estimation:
        return 0
    return max(1, len(text_for_estimation) // 4)


def generate_reply_sync(
    message_text: str,
    context: Optional[ReplyContext] = None,
    *,
    is_from_owner: bool = False,
) -> str:
    safe_text = (message_text or "").strip()
    context_text, context_entries = _parse_reply_context(context)
    has_context = bool(context_text) or bool(context_entries)

    if not safe_text and not has_context:
        return "Можешь уточнить, что именно хочешь сделать? Я помогу."

    attempts = 0
    while True:
        try:
            instructions = "\n".join(
                [
                    "ROLES:",
                    "OWNER — владелец аккаунта (я)",
                    "USER — собеседник/другой участник чата",
                    "Если говорящий не указан явно — считай это USER",
                ]
            )

            user_content: List[Dict[str, Any]] = [{"type": "text", "text": instructions}]
            log_lines: List[str] = [instructions]

            if context_entries:
                user_content.append({"type": "text", "text": "CHAT_CONTEXT (chronological):"})
                log_lines.append("CHAT_CONTEXT (chronological):")
                for entry in context_entries:
                    if not isinstance(entry, dict):
                        continue
                    entry_segments: List[str] = []
                    entry_line = _format_context_entry(entry)
                    if entry_line:
                        user_content.append({"type": "text", "text": entry_line})
                        log_lines.append(entry_line)
                        entry_segments.append(entry_line)
                    media_list = entry.get("media")
                    if isinstance(media_list, (list, tuple)):
                        for media in media_list:
                            if not isinstance(media, dict):
                                continue
                            display_candidate = media.get("display") or media.get("context")
                            if isinstance(display_candidate, str) and display_candidate.strip():
                                media_text = display_candidate.strip()
                            else:
                                media_text = _describe_media_for_log(media)
                            if _should_add_segment(media_text, entry_segments):
                                user_content.append({"type": "text", "text": media_text})
                                log_lines.append(media_text)
                                entry_segments.append(media_text)
                            analysis = media.get("analysis")
                            if isinstance(analysis, dict):
                                label = analysis.get("label")
                                text_val = analysis.get("text")
                                if isinstance(label, str) and label.strip() and isinstance(text_val, str) and text_val.strip():
                                    snippet = f"{label.strip()}: {text_val.strip()}"
                                    if _should_add_segment(snippet, entry_segments):
                                        user_content.append({"type": "text", "text": snippet})
                                        log_lines.append(snippet)
                                        entry_segments.append(snippet)
                            for part in _media_to_content_parts(media):
                                user_content.append(part)
            elif context_text:
                context_segment = f"CHAT_CONTEXT (chronological):\n{context_text}"
                user_content.append({"type": "text", "text": context_segment})
                log_lines.append(context_segment)

            sender_label = "OWNER" if is_from_owner else "USER"
            if safe_text:
                current_segment = f"CURRENT_MESSAGE_FROM: {sender_label}\n{safe_text}"
            else:
                current_segment = f"CURRENT_MESSAGE_FROM: {sender_label}"
            user_content.append({"type": "text", "text": current_segment})
            log_lines.append(current_segment)

            client = get_gemini_client()

            est = estimate_prompt_tokens(safe_text)
            est += _estimate_context_tokens(context_text, context_entries)
            est += 64
            try:
                gemini_rate_limiter.acquire_sync(est, rpd=GEMINI_RATE_RPD)
            except Exception:
                time.sleep(0.2)

            log_prompt = "\n".join(log_lines)
            logger.info(
                f"replier prompt: {DIALOGUE_PROMPT}\n{log_prompt}",
                extra={"plain": True},
            )

            cc = client.chat.completions.create(
                model=GEMINI_REPLY_MODEL,
                reasoning_effort="high",
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": DIALOGUE_PROMPT}]},
                    {"role": "user", "content": user_content},
                ],
            )

            message_content = cc.choices[0].message.content
            if isinstance(message_content, list):
                raw_parts = [
                    part.get("text", "")
                    for part in message_content
                    if isinstance(part, dict)
                ]
                raw = "\n".join(part for part in raw_parts if part).strip()
            else:
                raw = (message_content or "").strip()

            text = normalize_short_reply(raw)
            logger.info(f"replier output:\n{text}", extra={"plain": True})
            return text or "Давай уточним цель в одном предложении."
        except Exception as e:
            attempts += 1
            logger.exception("reply generation failed: %s", e)
            if is_gemini_quota_exhausted_error(e):
                return "Секунду, пожалуйста. Кажется, сеть шалит — попробуем ещё раз?"
            if is_gemini_transient_or_rate_error(e):
                if attempts >= 3:
                    return "Секунду, пожалуйста. Кажется, сеть шалит — попробуем ещё раз?"
                time.sleep(min(2 ** attempts, 10))
                continue
            if attempts > 3:
                return "Секунду, пожалуйста. Кажется, сеть шалит — попробуем ещё раз?"
            time.sleep(min(2 ** attempts, 10))


async def generate_reply(
    message_text: str,
    context: Optional[ReplyContext] = None,
    *,
    is_from_owner: bool = False,
) -> str:
    import asyncio

    context_text, context_entries = _parse_reply_context(context)

    estimated = estimate_prompt_tokens(message_text)
    estimated += _estimate_context_tokens(context_text, context_entries)
    estimated += 48

    await gemini_rate_limiter.acquire(estimated)
    return await asyncio.to_thread(
        generate_reply_sync,
        message_text,
        context,
        is_from_owner=is_from_owner,
    )

