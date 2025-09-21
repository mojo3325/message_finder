from typing import List
import time

from logging_config import logger
from services.clients import get_groq_client
from services.errors import (
    is_gemini_quota_exhausted_error,
    is_gemini_transient_or_rate_error,
    is_groq_tpd_limit_error,
)
from core.rate_limiter import estimate_prompt_tokens, groq_rate_limiter
from config import GROQ_RATE_RPD, GROQ_REPLY_MODEL, PORTRAIT_FALLBACK_MODEL


def _build_system_prompt() -> str:
    return (
        "Ты аналитик цифровой коммуникации. На основе реальных сообщений составь ёмкий, "
        "но насыщенный портрет человека. Твоя цель — уникальные изюминки: узнаваемые привычки письма, темы, установки, триггеры. "
        "Избегай гаданий и общих мест; опирайся только на наблюдаемые факты.\n\n"
        "Требования к ответу (строго соблюдай):\n"
        "- Без таблиц, Markdown, кода и JSON.\n"
        "- Формат Telegram HTML: используй только <b>, <i>, <u> и переносы строк.\n"
        "- Заголовки разделов выделяй <b>ЖИРНЫМ</b>, пункты короче одной строки.\n"
        "- Только вывод; без преамбул, дисклеймеров и повторов входных сообщений.\n"
        "- Если данных мало — коротко укажи это и дай лишь самое очевидное.\n"
        "- Используй переданные метаданные (распределение по часам/дням, длины и т.п.) для вывода паттернов и аккуратного прогноза поведения.\n\n"
        "Структура (следуй и не добавляй лишних разделов):\n"
        "<b>КТО ЭТО В ОБЩЕНИИ</b> — 2–3 фразы о роли/позиционировании, уверенности, образе.\n"
        "<b>СТИЛЬ ПИСЬМА</b> — длина сообщений; эмодзи/стикеры; пунктуация; регистр; обращения; частые слова/конструкции.\n"
        "<b>ТЕМЫ И ИНТЕРЕСЫ</b> — 5–8 пунктов от более частого к менее частому (• …).\n"
        "<b>КОМПЕТЕНТНОСТИ</b> — 3–5 заметных областей опыта/экспертизы (если явны).\n"
        "<b>ЦЕННОСТИ И УСТАНОВКИ</b> — 3–5 наблюдаемых выводов без психологизации.\n"
        "<b>ПОВЕДЕНИЕ В ЧАТЕ</b> — инициативность, просьбы/советы, юмор, конфликтность, дедлайны.\n"
        "<b>ТРИГГЕРЫ / АНТИТРИГГЕРЫ</b> — что цепляет/раздражает по текстам.\n"
        "<b>ПРЕДИКТОР ПОВЕДЕНИЯ</b> — вероятные часы активности (локально), дни недели, ритм ответа; осторожная гипотеза по часовому поясу/региону (если уверенность достаточна); ожидаемые темы ближайших сообщений; риски эскалации.\n"
        "<b>ЦИТАТЫ/ПРИМЕРЫ</b> — 2–4 короткие характерные фразы в кавычках.\n"
    )

def _build_user_messages(messages: List[str]) -> str:
    examples_joined = "\n\n".join(f"— {m.strip()}" for m in messages if m and m.strip())
    return f"Сообщения пользователя:\n{examples_joined}"


def _build_analysis_block(analysis: str | None) -> str:
    if not analysis:
        return ""
    return f"\n\nНаблюдения и метаданные:\n{analysis.strip()}"


def generate_portrait_sync(author_texts: List[str], *, analysis: str | None = None) -> str:
    safe_msgs = [m.strip() for m in author_texts if isinstance(m, str) and m.strip()]
    if len(safe_msgs) < 2:
        return "К сожалению сообщений не достаточно для составления портрета человека"

    joined_len = sum(len(m) for m in safe_msgs)
    if joined_len > 20000:
        acc: List[str] = []
        total = 0
        for m in reversed(safe_msgs):
            total += len(m)
            acc.append(m)
            if total > 18000:
                break
        safe_msgs = list(reversed(acc))

    system_prompt = _build_system_prompt()
    user_payload = _build_user_messages(safe_msgs) + _build_analysis_block(analysis)

    attempts = 0
    while True:
        try:
            client = get_groq_client()
            est = estimate_prompt_tokens(system_prompt + "\n\n" + user_payload)
            groq_rate_limiter.acquire_sync(est, rpd=GROQ_RATE_RPD)
            try:
                cc = client.chat.completions.create(
                    model=GROQ_REPLY_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=0.85,
                    top_p=0.95,
                    max_tokens=4000,
                    frequency_penalty=0.1,
                    presence_penalty=0.3,
                    extra_body={"reasoning_effort": "high"},
                )
            except Exception as groq_err:
                if is_groq_tpd_limit_error(groq_err):
                    logger.warning(
                        "Groq TPD hit for %s; switching to fallback %s",
                        GROQ_REPLY_MODEL,
                        PORTRAIT_FALLBACK_MODEL,
                    )
                    cc = client.chat.completions.create(
                        model=PORTRAIT_FALLBACK_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_payload},
                        ],
                        temperature=0.85,
                        top_p=0.95,
                        max_tokens=4000,
                        frequency_penalty=0.1,
                        presence_penalty=0.3,
                    )
                else:
                    raise
            text = (cc.choices[0].message.content or "").strip()
            return text or "К сожалению сообщений не достаточно для составления портрета человека"
        except Exception as e:
            attempts += 1
            logger.exception("portrait generation failed: %s", e)
            if is_gemini_quota_exhausted_error(e) or attempts > 3:
                return "К сожалению произошла ошибка при генерации портрета. Попробуйте позже."
            if is_gemini_transient_or_rate_error(e):
                time.sleep(min(2 ** attempts, 10))
                continue
            return "К сожалению произошла ошибка при генерации портрета. Попробуйте позже."


async def generate_portrait(author_texts: List[str], *, analysis: str | None = None) -> str:
    import asyncio

    return await asyncio.to_thread(generate_portrait_sync, author_texts, analysis=analysis)


