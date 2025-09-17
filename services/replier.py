import re
import time
from typing import List, Optional

from const import DIALOGUE_PROMPT
from core.rate_limiter import estimate_prompt_tokens
from logging_config import logger
from config import GROQ_RATE_RPD, GROQ_REPLY_MODEL
from services.clients import get_groq_client
from services.errors import is_gemini_quota_exhausted_error, is_gemini_transient_or_rate_error
from core.rate_limiter import groq_rate_limiter


def normalize_short_reply(raw: str) -> str:
    text = re.sub(r"\s+", " ", (raw or "").strip()).strip('"\' ')
    if not text:
        return text
    m = re.search(r"[.!?…]", text)
    if m:
        return text[: m.start() + 1].strip()
    return (text + ".") if not text.endswith((".", "!", "?", "…")) else text


def generate_reply_sync(message_text: str, context: Optional[str] = None) -> str:
    safe_text = (message_text or "").strip()
    if not safe_text:
        return "Можешь уточнить, что именно хочешь сделать? Я помогу."

    attempts = 0
    while True:
        try:
            user_content_parts: List[str] = []
            if context and context.strip():
                user_content_parts.append(f"CONTEXT (may be truncated):\n{context.strip()}")
            user_content_parts.append(f"CURRENT_MESSAGE:\n{safe_text}")
            user_content = "\n\n".join(user_content_parts)
            logger.info(f"replier input:\n{user_content}", extra={"plain": True})
            client = get_groq_client()
            est = estimate_prompt_tokens(safe_text)
            if context:
                est += max(1, len(context) // 4)
            est += 64
            try:
                groq_rate_limiter.acquire_sync(est, rpd=GROQ_RATE_RPD)
            except Exception:
                time.sleep(0.2)

            cc = client.chat.completions.create(
                model=GROQ_REPLY_MODEL,
                messages=[
                    {"role": "system", "content": DIALOGUE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.85,
                top_p=0.95,
                max_tokens=4000,
                frequency_penalty=0.1,
                presence_penalty=0.3,
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


async def generate_reply(message_text: str, context: Optional[str] = None) -> str:
    import asyncio

    estimated = estimate_prompt_tokens(message_text)
    if context:
        estimated += max(1, len(context) // 4)
    estimated += 48
    from core.rate_limiter import rate_limiter

    await rate_limiter.acquire(estimated)
    return await asyncio.to_thread(generate_reply_sync, message_text, context)


