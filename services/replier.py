import re
import time
from typing import List, Optional

from const import DIALOGUE_PROMPT
from core.rate_limiter import estimate_prompt_tokens
from logging_config import logger
from config import GEMINI_RATE_RPD, GEMINI_REPLY_MODEL
from services.clients import get_gemini_client
from services.errors import is_gemini_quota_exhausted_error, is_gemini_transient_or_rate_error
from core.rate_limiter import gemini_rate_limiter


def normalize_short_reply(raw: str) -> str:
    # Keep model output intact; only trim whitespace and surrounding quotes.
    if not raw:
        return ""

    text = (raw or "").strip()

    # Drop surrounding single/double quotes if the whole reply is quoted
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Collapse excessive internal spaces but preserve punctuation and sentence boundaries
    text = re.sub(r"[\t \f\v]+", " ", text).strip()

    return text


def generate_reply_sync(message_text: str, context: Optional[str] = None, *, is_from_owner: bool = False) -> str:
    safe_text = (message_text or "").strip()
    has_context = bool(context and context.strip())
    if not safe_text and not has_context:
        return "Можешь уточнить, что именно хочешь сделать? Я помогу."

    attempts = 0
    while True:
        try:
            user_content_parts: List[str] = []
            # Explicit role legend to disambiguate who wrote what
            user_content_parts.append(
                "\n".join([
                    "ROLES:",
                    "OWNER — владелец аккаунта (я)",
                    "USER — собеседник/другой участник чата",
                    "Если говорящий не указан явно — считай это USER",
                ])
            )
            if has_context:
                user_content_parts.append(f"CHAT_CONTEXT (chronological):\n{context.strip()}")
            if safe_text:
                sender_label = "OWNER" if is_from_owner else "USER"
                user_content_parts.append(f"CURRENT_MESSAGE_FROM: {sender_label}\n{safe_text}")
            else:
                # Context-only mode: request a short helpful reply suggestion based on context
                user_content_parts.append("")

            user_content = "\n\n".join(user_content_parts)
            client = get_gemini_client()
            est = estimate_prompt_tokens(safe_text)
            if context:
                est += max(1, len(context) // 4)
            est += 64
            try:
                gemini_rate_limiter.acquire_sync(est, rpd=GEMINI_RATE_RPD)
            except Exception:
                time.sleep(0.2)

            logger.info(f"replier prompt: {DIALOGUE_PROMPT}\n{user_content}", extra={"plain": True})


            cc = client.chat.completions.create(
                model=GEMINI_REPLY_MODEL,
                reasoning_effort="high",
                messages=[
                    {"role": "system", "content": DIALOGUE_PROMPT},
                    {"role": "user", "content": user_content}
                ]
            )

            raw = (cc.choices[0].message.content or "").strip()
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


async def generate_reply(message_text: str, context: Optional[str] = None, *, is_from_owner: bool = False) -> str:
    import asyncio

    estimated = estimate_prompt_tokens(message_text)
    if context:
        estimated += max(1, len(context) // 4)
    estimated += 48
    from core.rate_limiter import gemini_rate_limiter as _gemini_rate_limiter

    await _gemini_rate_limiter.acquire(estimated)
    return await asyncio.to_thread(generate_reply_sync, message_text, context, is_from_owner=is_from_owner)


