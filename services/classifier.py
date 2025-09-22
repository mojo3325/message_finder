import os
import random
import re
import time
from collections import deque
from collections.abc import Iterable, Mapping
from threading import Lock, Semaphore
from typing import Any, Optional

from const import CLASSIFIER_PROMPT
from core.types import ClassificationResult
from core.rate_limiter import estimate_prompt_tokens
from logging_config import logger
from config import (
    CEREBRAS_MODEL,
    CEREBRAS_EXTRA_API,
    MISTRAL_API_KEY,
    MISTRAL_BACKOFF_BASE_MS,
    MISTRAL_BACKOFF_MAX_MS,
    MISTRAL_MAX_CONCURRENCY,
    MISTRAL_MAX_RETRIES,
    MISTRAL_MAX_RPS,
    MISTRAL_MODEL,
    MISTRAL_TPM_SOFT,
    MISTRAL_TPMO_SOFT,
)
from services.clients import (
    get_cerebras_client,
    get_lmstudio_client,
    get_mistral_client,
)
from services.errors import (
    is_cerebras_tpd_limit_error,
)
from services.feedback import load_feedback_examples


_cerebras_use_extra_key: bool = False
_mistral_fallback_active: bool = False
_local_fallback_active: bool = False
_lm_model_resolved_name: Optional[str] = None
_mistral_disabled_logged: bool = False

_mistral_semaphore = Semaphore(MISTRAL_MAX_CONCURRENCY)
_mistral_rps_lock = Lock()
_mistral_usage_lock = Lock()
_mistral_serial_lock = Lock()
_mistral_recent_requests: deque[float] = deque()
_mistral_minute_usage: deque[tuple[float, int]] = deque()
_mistral_month_usage: deque[tuple[float, int]] = deque()
_mistral_force_single_thread: bool = MISTRAL_MAX_CONCURRENCY <= 1
_mistral_downshift_active: bool = False
_mistral_extra_delay_s: float = 0.0
_MISTRAL_MONTH_WINDOW_S = 30 * 24 * 60 * 60


BatchEntry = str | tuple[Any, Optional[Any]] | Mapping[str, Any]


def _can_use_mistral() -> bool:
    api_key = _current_mistral_api_key()
    available = bool(api_key)
    global _mistral_disabled_logged
    if available and _mistral_disabled_logged:
        _mistral_disabled_logged = False
    return available


def _current_mistral_api_key() -> str:
    if MISTRAL_API_KEY:
        return MISTRAL_API_KEY
    return os.getenv("MISTRAL_API_KEY", "").strip()


def _current_mistral_model() -> str:
    return os.getenv("MISTRAL_MODEL", MISTRAL_MODEL)


def _acquire_mistral_slot():
    _mistral_semaphore.acquire()
    acquired_serial = False
    while True:
        now = time.monotonic()
        with _mistral_rps_lock:
            while _mistral_recent_requests and now - _mistral_recent_requests[0] >= 1.0:
                _mistral_recent_requests.popleft()
            if len(_mistral_recent_requests) < MISTRAL_MAX_RPS:
                _mistral_recent_requests.append(now)
                break
            sleep_for = max(0.0, 1.0 - (now - _mistral_recent_requests[0]))
        time.sleep(max(sleep_for, 0.05))
    if _mistral_force_single_thread:
        _mistral_serial_lock.acquire()
        acquired_serial = True
    delay = 0.0
    with _mistral_usage_lock:
        delay = _mistral_extra_delay_s
    if delay > 0:
        time.sleep(delay)

    def _release() -> None:
        if acquired_serial:
            _mistral_serial_lock.release()
        _mistral_semaphore.release()

    return _release


def _mistral_backoff_delay(attempt: int) -> float:
    attempt = max(1, attempt)
    base = MISTRAL_BACKOFF_BASE_MS / 1000.0
    maximum = MISTRAL_BACKOFF_MAX_MS / 1000.0
    delay = base * (2 ** (attempt - 1))
    jitter = random.uniform(0, base)
    return min(maximum, delay + jitter)


def _extract_status_code(e: Exception) -> Optional[int]:
    status = getattr(e, "status_code", None)
    if status is None:
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_mistral_retryable_error(e: Exception) -> tuple[bool, Optional[int], str]:
    status = _extract_status_code(e)
    try:
        message = str(e).lower()
    except Exception:
        message = ""
    if status == 429 or "429" in message or "too many requests" in message or "rate limit" in message:
        return True, status, "rate"
    transient_markers = [
        "timeout",
        "timed out",
        "overload",
        "overloaded",
        "temporarily unavailable",
        "temporarily_unavailable",
        "unavailable",
        "gateway",
        "bad gateway",
        "connection reset",
        "connection aborted",
        "server error",
    ]
    if status in {408} or (status is not None and 500 <= status < 600):
        return True, status, "transient"
    if any(marker in message for marker in transient_markers):
        return True, status, "transient"
    if status is not None and 400 <= status < 500:
        return False, status, "client"
    return True, status, "unknown"


def _record_mistral_usage(usage: object) -> None:
    if not usage:
        return
    try:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        return
    total_tokens = max(0, prompt_tokens + completion_tokens)
    if total_tokens <= 0:
        return
    now = time.time()
    minute_tokens = 0
    month_tokens = 0
    triggered_downshift = False
    restored = False
    global _mistral_force_single_thread, _mistral_downshift_active, _mistral_extra_delay_s
    with _mistral_usage_lock:
        _mistral_minute_usage.append((now, total_tokens))
        _mistral_month_usage.append((now, total_tokens))
        minute_cutoff = now - 60.0
        month_cutoff = now - _MISTRAL_MONTH_WINDOW_S
        while _mistral_minute_usage and _mistral_minute_usage[0][0] < minute_cutoff:
            _mistral_minute_usage.popleft()
        while _mistral_month_usage and _mistral_month_usage[0][0] < month_cutoff:
            _mistral_month_usage.popleft()
        minute_tokens = sum(tokens for _, tokens in _mistral_minute_usage)
        month_tokens = sum(tokens for _, tokens in _mistral_month_usage)
        minute_threshold = MISTRAL_TPM_SOFT * 0.8 if MISTRAL_TPM_SOFT else None
        month_threshold = MISTRAL_TPMO_SOFT * 0.8 if MISTRAL_TPMO_SOFT else None
        downshift_needed = False
        if minute_threshold is not None and minute_tokens >= minute_threshold:
            downshift_needed = True
        if month_threshold is not None and month_tokens >= month_threshold:
            downshift_needed = True
        if downshift_needed and not _mistral_downshift_active:
            _mistral_downshift_active = True
            triggered_downshift = True
            if MISTRAL_MAX_CONCURRENCY > 1:
                _mistral_force_single_thread = True
            if _mistral_extra_delay_s < 0.5:
                _mistral_extra_delay_s = 0.5
        elif not downshift_needed and _mistral_downshift_active:
            _mistral_downshift_active = False
            restored = True
            if MISTRAL_MAX_CONCURRENCY > 1:
                _mistral_force_single_thread = False
            _mistral_extra_delay_s = 0.0
    if triggered_downshift:
        logger.warning(
            "mistral_quota_downshift",
            extra={"extra": {"minute_tokens": minute_tokens, "month_tokens": month_tokens}},
        )
    elif restored:
        logger.info(
            "mistral_quota_recovered",
            extra={"extra": {"minute_tokens": minute_tokens, "month_tokens": month_tokens}},
        )


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


def _coerce_batch_entry(entry: BatchEntry) -> tuple[str, Optional[str]]:
    if isinstance(entry, tuple):
        text_value = entry[0] if len(entry) > 0 else ""
        context_value = entry[1] if len(entry) > 1 else None
        text = str(text_value) if text_value is not None else ""
        context = str(context_value) if context_value is not None else None
        return text, context

    if isinstance(entry, Mapping):
        text_value: Any = entry.get("text")
        if text_value is None:
            text_value = entry.get("message")
        if text_value is None:
            text_value = entry.get("content")
        if text_value is None:
            text_value = entry.get("body")

        context_value: Any = entry.get("context")
        if context_value is None:
            context_value = entry.get("ctx")

        text = str(text_value) if text_value is not None else ""
        context = str(context_value) if context_value is not None else None
        return text, context

    if entry is None:
        return "", None

    return str(entry), None


def classify_with_openai_sync(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    if not message_text or not message_text.strip():
        return "0"

    global _cerebras_use_extra_key, _local_fallback_active, _lm_model_resolved_name, _mistral_fallback_active, _mistral_disabled_logged

    feedback_examples = load_feedback_examples()
    if feedback_examples:
        feedback_examples_str = "\n\n".join(feedback_examples)
        parts = CLASSIFIER_PROMPT.split("Examples:", 1)
        dynamic_prompt = f"{parts[0]}Examples:\n\n{feedback_examples_str}\n\n{parts[1]}" if len(parts) == 2 else CLASSIFIER_PROMPT
    else:
        dynamic_prompt = CLASSIFIER_PROMPT

    attempts = 0
    mistral_attempts = 0
    while True:
        release_mistral = None
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
            elif _mistral_fallback_active:
                if not _can_use_mistral():
                    if not _mistral_disabled_logged:
                        logger.warning("mistral_api_key_missing", extra={"extra": {"configured": False}})
                        _mistral_disabled_logged = True
                    _mistral_fallback_active = False
                    _local_fallback_active = True
                    continue
                client = get_mistral_client()
                model_name = _current_mistral_model()
                release_mistral = _acquire_mistral_slot()
            else:
                client = get_cerebras_client(CEREBRAS_EXTRA_API if _cerebras_use_extra_key and CEREBRAS_EXTRA_API else None)
                model_name = CEREBRAS_MODEL

            user_parts = [f"CONTEXT (may be truncated):\n{context.strip()}"] if context and context.strip() else []
            user_parts.append(f"CURRENT_MESSAGE:\n{message_text.strip()}")
            user_content = "\n\n".join(user_parts)

            kwargs = {
                "messages": [
                    {"role": "system", "content": dynamic_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.65,
                "top_p": 1.0,
                "max_tokens": 10000,
            }
            if _local_fallback_active:
                kwargs["extra_body"] = {"reasoning_effort": "medium"}
            elif _mistral_fallback_active:
                pass
            else:
                kwargs["extra_body"] = {"reasoning_effort": "low"}

            cc = client.chat.completions.create(model=model_name, **kwargs)
            if _mistral_fallback_active:
                _record_mistral_usage(getattr(cc, "usage", None))
            if release_mistral:
                release_mistral()
                release_mistral = None
            return _parse_classifier_label(cc.choices[0].message.content or "")

        except Exception as e:
            if release_mistral:
                release_mistral()
                release_mistral = None
            attempts += 1
            if _mistral_fallback_active and not _local_fallback_active:
                retryable, status_code, category = _is_mistral_retryable_error(e)
                if retryable:
                    mistral_attempts += 1
                    if mistral_attempts <= MISTRAL_MAX_RETRIES:
                        delay = _mistral_backoff_delay(mistral_attempts)
                        logger.warning(
                            "mistral_retry",
                            extra={
                                "extra": {
                                    "attempt": mistral_attempts,
                                    "delay_s": round(delay, 3),
                                    "status": status_code,
                                    "category": category,
                                    "error": str(e),
                                }
                            },
                        )
                        time.sleep(delay)
                        continue
                    _local_fallback_active = True
                    logger.warning(
                        "mistral_retry_exhausted_switch_to_lmstudio",
                        extra={
                            "extra": {
                                "attempts": mistral_attempts,
                                "status": status_code,
                                "category": category,
                                "error": str(e),
                            }
                        },
                    )
                    continue
                _local_fallback_active = True
                logger.warning(
                    "mistral_error_switch_to_lmstudio",
                    extra={"extra": {"status": status_code, "category": category, "error": str(e)}},
                )
                continue

            is_tpd = is_cerebras_tpd_limit_error(e)

            if not _local_fallback_active:
                if is_tpd:
                    if not _cerebras_use_extra_key and CEREBRAS_EXTRA_API:
                        _cerebras_use_extra_key = True
                        logger.warning("cerebras_tpd_switch_key", extra={"extra": {"error": str(e)}})
                        continue
                    else:
                        if _can_use_mistral():
                            _mistral_fallback_active = True
                            mistral_attempts = 0
                            logger.warning("cerebras_tpd_switch_to_mistral", extra={"extra": {"error": str(e)}})
                            continue
                        else:
                            _local_fallback_active = True
                            logger.warning(
                                "cerebras_tpd_switch_to_lmstudio",
                                extra={
                                    "extra": {
                                        "error": str(e),
                                        "mistral_available": False,
                                        "mistral_api_key_present": bool(_current_mistral_api_key()),
                                    }
                                },
                            )
                            continue

            if attempts > 3:
                logger.error("openai_classify_failed_all_fallbacks", extra={"extra": {"error": str(e)}})
                return "0"

            time.sleep(min(2 ** attempts, 10))


async def classify_with_openai(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    import asyncio

    return await asyncio.to_thread(classify_with_openai_sync, message_text, context)


def classify_batch_with_openai_sync(
    message_batch: Iterable[BatchEntry] | None,
) -> list[ClassificationResult]:
    if message_batch is None:
        return []

    results: list[ClassificationResult] = []
    for entry in message_batch:
        text, context = _coerce_batch_entry(entry)
        results.append(classify_with_openai_sync(text, context))
    return results


async def classify_batch_with_openai(
    message_batch: Iterable[BatchEntry] | None,
) -> list[ClassificationResult]:
    import asyncio

    batch_list = list(message_batch) if message_batch is not None else []
    return await asyncio.to_thread(classify_batch_with_openai_sync, batch_list)


