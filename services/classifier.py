import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock, Semaphore
from typing import Any, Iterable, Mapping, Optional

from const import CLASSIFIER_PROMPT
from core.rate_limiter import estimate_prompt_tokens, gemini_rate_limiter
from core.types import ClassificationResult
from logging_config import logger
from config import (
    CEREBRAS_EXTRA_API,
    CEREBRAS_MODEL,
    CLASSIFIER_CONTEXT_CHAR_LIMIT,
    CLASSIFIER_MAX_OUTPUT_TOKENS,
    CLASSIFIER_MESSAGE_CHAR_LIMIT,
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
    get_gemini_client,
    get_lmstudio_client,
    get_mistral_client,
)
from services.errors import is_cerebras_tpd_limit_error
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

_REMOTE_BATCH_CHUNK_SIZE = 4
_LOG_ITEM_PREVIEW_LIMIT = 160


BatchEntry = str | tuple[Any, Optional[Any]] | Mapping[str, Any]


@dataclass
class _BatchWorkItem:
    request_id: str
    index: int
    text: str
    context: Optional[str]


def _format_log_preview(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= _LOG_ITEM_PREVIEW_LIMIT:
        return cleaned
    return f"{cleaned[: _LOG_ITEM_PREVIEW_LIMIT - 1]}…"


def _summarize_batch_items_for_log(items: list["_BatchWorkItem"]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in items:
        text_preview = _format_log_preview(item.text)
        summary: dict[str, Any] = {
            "id": item.request_id,
            "index": item.index,
            "text_preview": text_preview if text_preview is not None else "",
            "text_len": len(item.text),
        }
        if item.context:
            context_preview = _format_log_preview(item.context)
            summary["context_len"] = len(item.context)
            if context_preview is not None:
                summary["context_preview"] = context_preview
        summaries.append(summary)
    return summaries


def _try_parse_json(raw: str) -> Optional[Any]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def _log_classifier_batch(
    provider: str,
    items: list["_BatchWorkItem"],
    payload_json: str,
    raw_text: str,
    result: dict[str, ClassificationResult],
    *,
    level: int = logging.INFO,
    note: Optional[str] = None,
) -> None:
    request_items = _summarize_batch_items_for_log(items)
    log_extra: dict[str, Any] = {
        "provider": provider,
        "batch_size": len(items),
        "request_items": request_items,
        "parsed": result,
    }
    payload_obj = _try_parse_json(payload_json)
    if isinstance(payload_obj, Mapping) and isinstance(payload_obj.get("items"), list):
        sanitized_items: list[dict[str, Any]] = []
        for item in request_items:
            sanitized_entry: dict[str, Any] = {
                "id": item["id"],
                "text": item.get("text_preview", ""),
            }
            if "context_preview" in item:
                sanitized_entry["context"] = item["context_preview"]
            sanitized_items.append(sanitized_entry)
        log_extra["request_payload"] = {"items": sanitized_items}
    elif payload_obj is not None:
        log_extra["request_payload"] = payload_obj
    response_obj = _try_parse_json(raw_text)
    if response_obj is not None:
        log_extra["response_json"] = response_obj
    elif raw_text:
        log_extra["response_raw"] = raw_text
    if note:
        log_extra["note"] = note
    logger.log(level, "classifier_batch_processed", extra={"extra": log_extra})


def _default_batch_result(items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    return {item.request_id: "0" for item in items}


def _finalize_batch_result(
    provider: str,
    payload_json: str,
    raw_text: str,
    items: list[_BatchWorkItem],
    parsed: dict[str, ClassificationResult],
) -> dict[str, ClassificationResult]:
    if parsed:
        _log_classifier_batch(provider, items, payload_json, raw_text, parsed)
        return parsed

    default_result = _default_batch_result(items)
    _log_classifier_batch(
        provider,
        items,
        payload_json,
        raw_text,
        default_result,
        level=logging.WARNING,
        note="empty_or_unparsed_response",
    )
    return default_result


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


def _parse_classifier_label(raw: object) -> ClassificationResult:
    if raw is None:
        return "0"

    if isinstance(raw, (int, float)):
        value = int(raw)
        if value in (0, 1):
            return str(value)  # type: ignore[return-value]
        return "0"

    if isinstance(raw, Mapping):
        if "label" in raw:
            return _parse_classifier_label(raw["label"])
        if "value" in raw:
            return _parse_classifier_label(raw["value"])
        return "0"

    if isinstance(raw, (list, tuple)):
        for item in raw:
            candidate = _parse_classifier_label(item)
            if candidate != "0":
                return candidate
        return "0"

    cleaned = str(raw).strip().strip("`").strip()
    if not cleaned:
        return "0"

    try:
        decoded = json.loads(cleaned)
    except Exception:
        decoded = None
    if decoded is not None:
        return _parse_classifier_label(decoded)

    if cleaned in {"0", "1"}:
        return cleaned  # type: ignore[return-value]

    match = re.search(r"[01]", cleaned)
    if match:
        return match.group(0)  # type: ignore[return-value]

    return "0"


def _coerce_batch_entry(entry: BatchEntry) -> tuple[str, Optional[str], Optional[str]]:
    explicit_id: Optional[str] = None
    if isinstance(entry, tuple):
        text_value = entry[0] if len(entry) > 0 else ""
        context_value = entry[1] if len(entry) > 1 else None
        text = str(text_value) if text_value is not None else ""
        context = str(context_value) if context_value is not None else None
        return text, context, explicit_id

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

        raw_id = entry.get("id")
        if raw_id is not None:
            explicit_id = str(raw_id)

        text = str(text_value) if text_value is not None else ""
        context = str(context_value) if context_value is not None else None
        return text, context, explicit_id

    if entry is None:
        return "", None, None

    return str(entry), None, None


def _truncate_message(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= CLASSIFIER_MESSAGE_CHAR_LIMIT:
        return stripped
    return stripped[:CLASSIFIER_MESSAGE_CHAR_LIMIT]


def _truncate_context(context: Optional[str]) -> Optional[str]:
    if not context:
        return None
    stripped = context.strip()
    if len(stripped) <= CLASSIFIER_CONTEXT_CHAR_LIMIT:
        return stripped
    return stripped[-CLASSIFIER_CONTEXT_CHAR_LIMIT:]


def _prepare_inputs(message_text: str, context: Optional[str]) -> tuple[str, Optional[str]]:
    text = _truncate_message(message_text)
    ctx = _truncate_context(context)
    return text, ctx


_COMMAND_RE = re.compile(r"^[\s]*[!/].+")


def _maybe_short_circuit(text: str, context: Optional[str]) -> Optional[ClassificationResult]:
    if not text.strip():
        return "0"

    normalized = text.strip()
    if _COMMAND_RE.match(normalized):
        return "0"

    if not re.search(r"[A-Za-zА-Яа-я]", normalized):
        return "0"

    return None


def estimate_classifier_tokens(message_text: str, context: Optional[str]) -> int:
    text, ctx = _prepare_inputs(message_text, context)
    estimated = estimate_prompt_tokens(text)
    if ctx:
        estimated += max(1, len(ctx) // 4)
    return estimated


def _extract_choice_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
    except Exception:
        choices = None
    if not choices:
        return ""

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "output_text":
                text_value = part.get("text")
            else:
                text_value = getattr(part, "text", None)
            if text_value:
                return str(text_value)

    text_attr = getattr(message, "text", None)
    if text_attr:
        return str(text_attr)

    return ""


def _parse_nested_value(value: Any, items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    if isinstance(value, str):
        return _parse_batch_response(value, items)
    if isinstance(value, (Mapping, list)):
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            return {}
        return _parse_batch_response(serialized, items)
    return {}


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    retry_after: Optional[str] = None
    if headers is not None:
        try:
            retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        except Exception:  # noqa: BLE001
            retry_after = None
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        try:
            return float(int(retry_after))
        except Exception:  # noqa: BLE001
            return None


def _increase_mistral_delay(retry_after: Optional[float]) -> None:
    global _mistral_extra_delay_s, _mistral_force_single_thread
    increment = retry_after if retry_after is not None else (MISTRAL_BACKOFF_BASE_MS / 1000.0)
    maximum = MISTRAL_BACKOFF_MAX_MS / 1000.0
    with _mistral_usage_lock:
        current = _mistral_extra_delay_s
        if current <= 0:
            new_delay = increment
        else:
            new_delay = min(maximum, current * 1.5 + increment)
        _mistral_extra_delay_s = min(maximum, max(new_delay, increment))
        if MISTRAL_MAX_CONCURRENCY > 1:
            _mistral_force_single_thread = True


def _relax_mistral_delay() -> None:
    global _mistral_extra_delay_s, _mistral_force_single_thread
    with _mistral_usage_lock:
        if _mistral_extra_delay_s <= 0:
            return
        _mistral_extra_delay_s = max(0.0, _mistral_extra_delay_s * 0.5 - 0.1)
        if _mistral_extra_delay_s <= 0 and not _mistral_downshift_active and MISTRAL_MAX_CONCURRENCY > 1:
            _mistral_force_single_thread = False


def _build_classifier_prompt() -> str:
    feedback_examples = load_feedback_examples()
    if not feedback_examples:
        return CLASSIFIER_PROMPT
    parts = CLASSIFIER_PROMPT.split("Examples:", 1)
    if len(parts) != 2:
        return CLASSIFIER_PROMPT
    feedback_section = "\n\n".join(feedback_examples)
    return f"{parts[0]}Examples:\n\n{feedback_section}\n\n{parts[1]}"


def _prepare_batch_items(entries: list[BatchEntry]) -> tuple[list[_BatchWorkItem], dict[int, ClassificationResult]]:
    work_items: list[_BatchWorkItem] = []
    immediate_results: dict[int, ClassificationResult] = {}

    for index, entry in enumerate(entries):
        text_raw, context_raw, explicit_id = _coerce_batch_entry(entry)
        text, ctx = _prepare_inputs(text_raw, context_raw)
        shortcut = _maybe_short_circuit(text, ctx)
        request_id = explicit_id or f"item_{index + 1}"
        if shortcut is not None:
            immediate_results[index] = shortcut
            continue
        work_items.append(_BatchWorkItem(request_id=request_id, index=index, text=text, context=ctx))

    return work_items, immediate_results


def _build_payload_json(items: list[_BatchWorkItem]) -> str:
    payload_items: list[dict[str, str]] = []
    for item in items:
        entry: dict[str, str] = {"id": item.request_id, "text": item.text}
        if item.context:
            entry["context"] = item.context
        payload_items.append(entry)
    return json.dumps({"items": payload_items}, ensure_ascii=False)


def _parse_batch_response(raw_text: str, items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    if not raw_text:
        return {}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, str):
        return _parse_batch_response(data, items)

    if isinstance(data, Mapping):
        entries = data.get("items")
        if isinstance(entries, list):
            result: dict[str, ClassificationResult] = {}
            fallback_by_index: list[tuple[int, ClassificationResult]] = []
            for idx, entry in enumerate(entries):
                if isinstance(entry, Mapping):
                    entry_id_raw = entry.get("id")
                    entry_id = str(entry_id_raw).strip() if entry_id_raw is not None else ""
                    label = _parse_classifier_label(entry.get("label"))
                    if entry_id:
                        result[entry_id] = label
                        continue
                    fallback_by_index.append((idx, label))
                    continue

                # Handle providers returning bare values instead of objects.
                label = _parse_classifier_label(entry)
                fallback_by_index.append((idx, label))

            for idx, label in fallback_by_index:
                if 0 <= idx < len(items):
                    request_id = items[idx].request_id
                    if request_id not in result:
                        result[request_id] = label

            if result:
                return result

        for value in data.values():
            nested = _parse_nested_value(value, items)
            if nested:
                return nested

    if isinstance(data, list):
        # Sometimes providers may return list of numbers without ids; rely on order
        digits = [str(int(val)) for val in data if isinstance(val, (int, float)) and int(val) in (0, 1)]
        if digits:
            padded = digits + ["0"] * max(0, len(items) - len(digits))
            return {item.request_id: padded[idx] for idx, item in enumerate(items)}

        for entry in data:
            nested = _parse_nested_value(entry, items)
            if nested:
                return nested

    stripped_full = raw_text.strip()
    for line in raw_text.splitlines():
        candidate = line.strip().strip("`").strip()
        if not candidate or candidate == stripped_full:
            continue
        nested = _parse_batch_response(candidate, items)
        if nested:
            return nested

    label_matches: list[ClassificationResult] = []
    for match in re.finditer(r'"label"\s*:\s*("?[01]"?|true|false)', raw_text, flags=re.IGNORECASE):
        value_raw = match.group(1).strip().strip('"').lower()
        if value_raw in {"0", "1"}:
            label_matches.append(value_raw)
        elif value_raw == "true":
            label_matches.append("1")
        elif value_raw == "false":
            label_matches.append("0")
    if label_matches:
        padded = label_matches + ["0"] * max(0, len(items) - len(label_matches))
        return {item.request_id: padded[idx] for idx, item in enumerate(items)}

    digits = re.findall(r"[01]", raw_text)
    if digits and items:
        padded = digits + ["0"] * max(0, len(items) - len(digits))
        return {item.request_id: _parse_classifier_label(padded[idx]) for idx, item in enumerate(items)}

    if len(items) == 1:
        label = _parse_classifier_label(raw_text)
        return {items[0].request_id: label}

    return {}


def _call_cerebras_batch(prompt: str, payload_json: str, items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    client = get_cerebras_client(CEREBRAS_EXTRA_API if _cerebras_use_extra_key and CEREBRAS_EXTRA_API else None)
    response = client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload_json},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max(CLASSIFIER_MAX_OUTPUT_TOKENS, CLASSIFIER_MAX_OUTPUT_TOKENS * len(items) + 8),
        extra_body={"reasoning_effort": "high"},
    )
    raw_text = _extract_choice_text(response)
    parsed = _parse_batch_response(raw_text, items)
    return _finalize_batch_result("cerebras", payload_json, raw_text, items, parsed)


def _resolve_lmstudio_model() -> str:
    global _lm_model_resolved_name
    if _lm_model_resolved_name:
        return _lm_model_resolved_name
    client = get_lmstudio_client()
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
    except Exception:  # noqa: BLE001
        _lm_model_resolved_name = "openai/gpt-oss-20b"
    return _lm_model_resolved_name


def _call_local_batch(prompt: str, payload_json: str, items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    client = get_lmstudio_client()
    model_name = _resolve_lmstudio_model()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload_json},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max(CLASSIFIER_MAX_OUTPUT_TOKENS, CLASSIFIER_MAX_OUTPUT_TOKENS * len(items) + 8),
        extra_body={"reasoning_effort": "medium"},
    )
    raw_text = _extract_choice_text(response)
    parsed = _parse_batch_response(raw_text, items)
    return _finalize_batch_result("lmstudio", payload_json, raw_text, items, parsed)


def _call_mistral_batch(prompt: str, payload_json: str, items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    if not _can_use_mistral():
        raise RuntimeError("mistral_api_key_missing")
    release = _acquire_mistral_slot()
    try:
        client = get_mistral_client()
        response = client.chat.completions.create(
            model=_current_mistral_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload_json},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max(CLASSIFIER_MAX_OUTPUT_TOKENS, CLASSIFIER_MAX_OUTPUT_TOKENS * len(items) + 8),
            response_format={"type": "json_object"},
        )
    finally:
        release()
    _record_mistral_usage(getattr(response, "usage", None))
    raw_text = _extract_choice_text(response)
    parsed = _parse_batch_response(raw_text, items)
    _relax_mistral_delay()
    return _finalize_batch_result("mistral", payload_json, raw_text, items, parsed)


def _classify_remote_chunk(items: list[_BatchWorkItem], prompt: str) -> dict[str, ClassificationResult]:
    if not items:
        return {}

    global _local_fallback_active, _cerebras_use_extra_key, _mistral_fallback_active
    if not _local_fallback_active:
        if _can_use_mistral():
            _mistral_fallback_active = True
        else:
            _mistral_fallback_active = False
    payload_json = _build_payload_json(items)
    attempts = 0
    mistral_attempts = 0
    mistral_failed_this_cycle = False

    while True:
        try:
            if _local_fallback_active:
                provider = "lmstudio"
                return _call_local_batch(prompt, payload_json, items)
            if _mistral_fallback_active and not mistral_failed_this_cycle:
                provider = "mistral"
                result = _call_mistral_batch(prompt, payload_json, items)
                _mistral_fallback_active = True
                return result
            provider = "cerebras"
            return _call_cerebras_batch(prompt, payload_json, items)
        except Exception as exc:  # noqa: BLE001
            attempts += 1
            if provider == "mistral":
                retryable, status_code, category = _is_mistral_retryable_error(exc)
                if retryable and not _local_fallback_active:
                    mistral_attempts += 1
                    retry_after = _extract_retry_after_seconds(exc) if status_code == 429 else None
                    if status_code == 429:
                        _increase_mistral_delay(retry_after)
                    if mistral_attempts <= MISTRAL_MAX_RETRIES:
                        delay = _mistral_backoff_delay(mistral_attempts)
                        if retry_after is not None:
                            delay = max(delay, retry_after)
                        logger.warning(
                            "mistral_retry",
                            extra={
                                "extra": {
                                    "attempt": mistral_attempts,
                                    "delay_s": round(delay, 3),
                                    "status": status_code,
                                    "category": category,
                                    "retry_after": retry_after,
                                    "error": str(exc),
                                }
                            },
                        )
                        time.sleep(delay)
                        continue
                    logger.warning(
                        "mistral_retry_exhausted_switch_to_cerebras",
                        extra={
                            "extra": {
                                "attempts": mistral_attempts,
                                "status": status_code,
                                "category": category,
                                "error": str(exc),
                            }
                        },
                    )
                else:
                    logger.warning(
                        "mistral_error_switch_to_cerebras",
                        extra={"extra": {"status": status_code, "category": category, "error": str(exc)}},
                    )
                mistral_failed_this_cycle = True
                _mistral_fallback_active = False
                continue

            if provider == "cerebras" and not _local_fallback_active:
                is_tpd = is_cerebras_tpd_limit_error(exc)
                if is_tpd:
                    if not _cerebras_use_extra_key and CEREBRAS_EXTRA_API:
                        _cerebras_use_extra_key = True
                        logger.warning("cerebras_tpd_switch_key", extra={"extra": {"error": str(exc)}})
                        continue
                    if _can_use_mistral():
                        _mistral_fallback_active = True
                        mistral_attempts = 0
                        mistral_failed_this_cycle = False
                        logger.warning("cerebras_tpd_switch_to_mistral", extra={"extra": {"error": str(exc)}})
                        continue
                    _local_fallback_active = True
                    logger.warning(
                        "cerebras_tpd_switch_to_lmstudio",
                        extra={
                            "extra": {
                                "error": str(exc),
                                "mistral_available": False,
                                "mistral_api_key_present": bool(_current_mistral_api_key()),
                            }
                        },
                    )
                    continue

            if attempts > 3:
                logger.error("openai_classify_failed_all_fallbacks", extra={"extra": {"error": str(exc)}})
                return {}

            time.sleep(min(2 ** attempts, 10))


def _iter_remote_chunks(items: list[_BatchWorkItem], chunk_size: int) -> Iterable[list[_BatchWorkItem]]:
    chunk_size = max(1, chunk_size)
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _classify_remote_batch(items: list[_BatchWorkItem], prompt: str) -> dict[str, ClassificationResult]:
    if not items:
        return {}

    combined: dict[str, ClassificationResult] = {}
    for chunk in _iter_remote_chunks(items, _REMOTE_BATCH_CHUNK_SIZE):
        chunk_result = _classify_remote_chunk(chunk, prompt)
        if chunk_result:
            combined.update(chunk_result)

    return combined


def classify_batch_with_openai_sync(
    message_batch: Iterable[BatchEntry] | None,
) -> list[ClassificationResult]:
    if not message_batch:
        return []

    entries = list(message_batch)
    if not entries:
        return []

    prompt = _build_classifier_prompt()
    work_items, immediate_results = _prepare_batch_items(entries)
    remote_results = _classify_remote_batch(work_items, prompt) if work_items else {}

    results: list[ClassificationResult] = ["0"] * len(entries)
    for index, label in immediate_results.items():
        if 0 <= index < len(results):
            results[index] = label

    for item in work_items:
        label = remote_results.get(item.request_id, "0")
        results[item.index] = label

    return results


def classify_with_openai_sync(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    results = classify_batch_with_openai_sync([(message_text, context)])
    return results[0] if results else "0"


async def classify_with_openai(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    import asyncio

    return await asyncio.to_thread(classify_with_openai_sync, message_text, context)


async def classify_batch_with_openai(
    message_batch: Iterable[BatchEntry] | None,
) -> list[ClassificationResult]:
    import asyncio

    batch_list = list(message_batch) if message_batch is not None else []
    return await asyncio.to_thread(classify_batch_with_openai_sync, batch_list)
