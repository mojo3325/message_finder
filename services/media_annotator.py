from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from config import GEMINI_MODEL, GEMINI_RATE_RPD
from core.rate_limiter import estimate_prompt_tokens, gemini_rate_limiter
from logging_config import logger
from services.clients import get_gemini_client


def _collect_text_from_content(content: Any) -> str:
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text_val = part.get("text")
            if isinstance(text_val, str) and text_val.strip():
                pieces.append(text_val.strip())
        return "\n".join(pieces).strip()
    if isinstance(content, str):
        return content.strip()
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr.strip()
    return ""


def _call_media_completion(messages: Iterable[Dict[str, Any]], *, estimated_tokens: int = 512) -> Optional[str]:
    try:
        gemini_rate_limiter.acquire_sync(max(estimated_tokens, 64), rpd=GEMINI_RATE_RPD)
    except Exception:
        # Best effort; fall through on limiter issues
        pass

    client = get_gemini_client()
    try:
        completion = client.chat.completions.create(
            model=GEMINI_MODEL,
            temperature=1,
            messages=list(messages)
        )
    except Exception as exc:  # pragma: no cover - network failures
        logger.exception("media annotation request failed: %s", exc)
        return None

    try:
        choice = completion.choices[0]
        content = choice.message.content
    except Exception:  # pragma: no cover - unexpected SDK shape
        return None

    text = _collect_text_from_content(content)
    return text or None


def transcribe_audio_base64(data_base64: str, fmt: str) -> Optional[str]:
    if not isinstance(data_base64, str) or not data_base64.strip():
        return None
    if not isinstance(fmt, str) or not fmt.strip():
        return None

    system_prompt = (
        "Ты точный расшифровщик голосовых сообщений из Telegram. "
        "Передавай текст в исходном языке без пояснений."
    )
    user_prompt = (
        "Расшифруй это голосовое сообщение и ответь только расшифровкой, без кавычек и комментариев."
    )

    estimated = estimate_prompt_tokens(user_prompt) + 128

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "input_audio", "input_audio": {"data": data_base64.strip(), "format": fmt.strip().lower()}},
            ],
        },
    ]

    return _call_media_completion(messages, estimated_tokens=estimated)


def describe_image_base64(data_base64: str, mime_type: str) -> Optional[str]:
    if not isinstance(data_base64, str) or not data_base64.strip():
        return None
    if not isinstance(mime_type, str) or not mime_type.strip():
        return None

    system_prompt = "Ты кратко описываешь изображения из переписки. Пиши по-русски и по делу."
    user_prompt = "Опиши изображение в одном-двух предложениях, отметь важные детали или текст."
    estimated = estimate_prompt_tokens(user_prompt) + 96

    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type.strip()};base64,{data_base64.strip()}"},
    }

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}, image_part]},
    ]

    return _call_media_completion(messages, estimated_tokens=estimated)


def summarize_video_base64(data_base64: str, fmt: str) -> Optional[str]:
    if not isinstance(data_base64, str) or not data_base64.strip():
        return None
    if not isinstance(fmt, str) or not fmt.strip():
        return None

    system_prompt = (
        "Ты помогаешь кратко пересказывать видеосообщения из чатов."
        " Опиши ключевые действия и речь на русском."
    )
    user_prompt = "Передай основную суть этого короткого видео в одном-двух предложениях."
    estimated = estimate_prompt_tokens(user_prompt) + 160

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "input_video", "input_video": {"data": data_base64.strip(), "format": fmt.strip().lower()}},
            ],
        },
    ]

    return _call_media_completion(messages, estimated_tokens=estimated)
