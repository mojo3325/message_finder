import re
import re
import time
from typing import Optional

from config import (
    CEREBRAS_MODEL,
    CEREBRAS_EXTRA_API,
    GEMINI_MODEL,
    GEMINI_RATE_RPD,
)
from const import CLASSIFIER_PROMPT
from core.rate_limiter import estimate_prompt_tokens, gemini_rate_limiter
from core.types import ClassificationResult
from logging_config import logger
from services.clients import get_cerebras_client, get_gemini_client, get_lmstudio_client
from services.errors import (
    is_cerebras_tpd_limit_error,
    is_gemini_quota_exhausted_error,
    is_gemini_transient_or_rate_error,
)
from services.feedback import load_feedback_examples

_cerebras_use_extra_key: bool = False
_gemini_fallback_active: bool = False
_local_fallback_active: bool = False
_lm_model_resolved_name: Optional[str] = None


def _parse_classifier_label(raw: str) -> ClassificationResult:
    if not raw:
        return "0"
    cleaned = str(raw).strip().strip('`').strip()
    # Fast path: single character 0/1
    if cleaned in {"0", "1"}:
        return cleaned  # type: ignore[return-value]
    # Try to extract first 0/1 anywhere
    m = re.search(r"[01]", cleaned)
    if m:
        return m.group(0)  # type: ignore[return-value]
    return "0"


def classify_with_openai_sync(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    if not message_text or not message_text.strip():
        return "0"

    global _cerebras_use_extra_key, _gemini_fallback_active, _local_fallback_active, _lm_model_resolved_name

    feedback_examples = load_feedback_examples()
    if feedback_examples:
        feedback_examples_str = "\n\n".join(feedback_examples)
        parts = CLASSIFIER_PROMPT.split("Examples:", 1)
        dynamic_prompt = f"{parts[0]}Examples:\n\n{feedback_examples_str}\n\n{parts[1]}" if len(parts) == 2 else CLASSIFIER_PROMPT
    else:
        dynamic_prompt = CLASSIFIER_PROMPT

    attempts = 0
    while True:
        try:
            if _local_fallback_active:
                client = get_lmstudio_client()
                if not _lm_model_resolved_name:
                    try:
                        models = client.models.list()
                        ids = [m.id for m in getattr(models, "data", [])]
                        desired = "openai/gpt-oss-20b"
                        if desired in ids:
                            _lm_model_resolved_name = desired
                        else:
                            candidates = [mid for mid in ids if "gpt" in mid.lower() and "oss" in mid.lower()]
                            _lm_model_resolved_name = candidates[0] if candidates else (ids[0] if ids else desired)
                        logger.info("lmstudio_model_selected", extra={"extra": {"selected": _lm_model_resolved_name}})
                    except Exception:
                        _lm_model_resolved_name = "openai/gpt-oss-20b"
                model_name = _lm_model_resolved_name
            elif _gemini_fallback_active:
                client = get_gemini_client()
                model_name = GEMINI_MODEL
                est = estimate_prompt_tokens(message_text)
                if context:
                    est += max(1, len(context) // 4)
                est += 256
                try:
                    gemini_rate_limiter.acquire_sync(est, rpd=GEMINI_RATE_RPD)
                except Exception:
                    time.sleep(0.2)
            else:
                client = get_cerebras_client(CEREBRAS_EXTRA_API if _cerebras_use_extra_key and CEREBRAS_EXTRA_API else None)
                model_name = CEREBRAS_MODEL

            user_parts = [f"CONTEXT (may be truncated):\n{context.strip()}"] if context and context.strip() else []
            user_parts.append(f"CURRENT_MESSAGE:\n{message_text.strip()}")
            user_content = "\n\n".join(user_parts)

            if _local_fallback_active:
                kwargs = {
                    "messages": [
                        {"role": "system", "content": dynamic_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.65,
                    "top_p": 1.0,
                    "max_tokens": 10000,
                    "extra_body": {"reasoning_effort": "medium"},
                }
            elif _gemini_fallback_active:
                 kwargs = {
                    "messages": [
                        {"role": "system", "content": dynamic_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.65,
                    "top_p": 1.0,
                    "max_tokens": 10000,
                    "extra_body": {"reasoning_effort": "low"},
                }
            else:
                kwargs = {
                    "messages": [
                        {"role": "system", "content": dynamic_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.65,
                    "top_p": 1.0,
                    "max_tokens": 10000,
                    "extra_body": {"reasoning_effort": "low"},
                }

            cc = client.chat.completions.create(model=model_name, **kwargs)
            return _parse_classifier_label(cc.choices[0].message.content or "")

        except Exception as e:
            attempts += 1
            is_tpd = is_cerebras_tpd_limit_error(e)

            if not _local_fallback_active:
                if _gemini_fallback_active:
                    if is_gemini_quota_exhausted_error(e):
                        _local_fallback_active = True
                        logger.warning("gemini_quota_switch_to_lmstudio", extra={"extra": {"error": str(e)}})
                        continue
                    if is_gemini_transient_or_rate_error(e):
                        sleep_s = min(2 ** attempts, 10)
                        if attempts >= 3:
                            _local_fallback_active = True
                            logger.warning("gemini_transient_switch_to_lmstudio", extra={"extra": {"error": str(e)}})
                            continue
                        logger.warning("gemini_transient_retry", extra={"extra": {"error": str(e), "sleep": sleep_s}})
                        time.sleep(sleep_s)
                        continue
                    _local_fallback_active = True
                    logger.warning("gemini_fail_switch_to_lmstudio", extra={"extra": {"error": str(e)}})
                    continue
                elif is_tpd:
                    if not _cerebras_use_extra_key and CEREBRAS_EXTRA_API:
                        _cerebras_use_extra_key = True
                        logger.warning("cerebras_tpd_switch_key", extra={"extra": {"error": str(e)}})
                        continue
                    else:
                        _gemini_fallback_active = True
                        logger.warning("cerebras_tpd_switch_to_gemini", extra={"extra": {"error": str(e)}})
                        continue

            if attempts > 3:
                logger.error("openai_classify_failed_all_fallbacks", extra={"extra": {"error": str(e)}})
                return "0"

            time.sleep(min(2 ** attempts, 10))


async def classify_with_openai(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    import asyncio

    return await asyncio.to_thread(classify_with_openai_sync, message_text, context)


