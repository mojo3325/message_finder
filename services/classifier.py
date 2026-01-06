import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Mapping, Optional, Sequence

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
    ENABLE_GEMINI,
    GEMINI_MODEL,
    GEMINI_RATE_RPD,
    GEMINI_RATE_RPM,
    GEMINI_RATE_TPM,
    GEMINI_RPD_GUARD,
    GEMINI_RPM_GUARD,
    GEMINI_TPM_GUARD,
    PROVIDER_ORDER,
)
from services.clients import (
    get_cerebras_client,
    get_gemini_client,
    get_lmstudio_client,
)
from services.errors import is_cerebras_tpd_limit_error
from services.feedback import load_feedback_examples


_cerebras_use_extra_key: bool = False
_lm_model_resolved_name: Optional[str] = None

_REMOTE_BATCH_CHUNK_SIZE = 4
_LOG_ITEM_PREVIEW_LIMIT = 160
_GEMINI_MAX_RETRIES = 2


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
    latency_ms: Optional[float] = None,
    fallback_reason: Optional[str] = None,
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
    if latency_ms is not None:
        log_extra["latency_ms"] = round(latency_ms, 3)
    if fallback_reason:
        log_extra["fallback_reason"] = fallback_reason
    logger.log(level, "classifier_batch_processed", extra={"extra": log_extra})


def _default_batch_result(items: list[_BatchWorkItem]) -> dict[str, ClassificationResult]:
    return {item.request_id: "0" for item in items}


def _finalize_batch_result(
    provider: str,
    payload_json: str,
    raw_text: str,
    items: list[_BatchWorkItem],
    parsed: dict[str, ClassificationResult],
    *,
    latency_ms: Optional[float] = None,
    fallback_reason: Optional[str] = None,
) -> dict[str, ClassificationResult]:
    if parsed:
        _log_classifier_batch(
            provider,
            items,
            payload_json,
            raw_text,
            parsed,
            latency_ms=latency_ms,
            fallback_reason=fallback_reason,
        )
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
        latency_ms=latency_ms,
        fallback_reason=fallback_reason,
    )
    return default_result


@dataclass
class _RequestStamp:
    timestamp: float


@dataclass
class _TokenStamp:
    timestamp: float
    tokens: int


@dataclass
class _GeminiReservation:
    manager: "GeminiQuotaManager"
    minute_entries: list[_RequestStamp]
    day_entries: list[_RequestStamp]
    token_entry: Optional[_TokenStamp]
    estimated_tokens: int
    committed: bool = False
    released: bool = False

    def commit(self, prompt_tokens: int, completion_tokens: int) -> None:
        if self.released or self.committed:
            return
        self.manager.commit(self, prompt_tokens, completion_tokens)
        self.committed = True

    def release(self) -> None:
        if self.released or self.committed:
            return
        self.manager.release(self)
        self.released = True


class GeminiQuotaManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._minute_requests: deque[_RequestStamp] = deque()
        self._minute_tokens: deque[_TokenStamp] = deque()
        self._day_requests: deque[_RequestStamp] = deque()
        self._minute_token_total = 0

    def _cleanup(self, now: Optional[float] = None) -> None:
        timestamp = now or time.time()
        minute_cutoff = timestamp - 60.0
        day_cutoff = timestamp - 86400.0
        while self._minute_requests and self._minute_requests[0].timestamp <= minute_cutoff:
            self._minute_requests.popleft()
        while self._minute_tokens and self._minute_tokens[0].timestamp <= minute_cutoff:
            stamp = self._minute_tokens.popleft()
            self._minute_token_total -= stamp.tokens
        while self._day_requests and self._day_requests[0].timestamp <= day_cutoff:
            self._day_requests.popleft()

    def _guard_limit(self, limit: int, guard: float) -> Optional[int]:
        if limit <= 0:
            return None
        threshold = int(limit * guard) if guard > 0 else 0
        if threshold <= 0 or threshold > limit:
            return limit
        return threshold

    def reserve(
        self,
        estimated_prompt_tokens: int,
        estimated_output_tokens: int,
        *,
        request_count: int = 1,
    ) -> tuple[Optional[_GeminiReservation], Optional[str]]:
        with self._lock:
            self._cleanup()
            request_count = max(1, request_count)
            estimated_prompt_tokens = max(0, estimated_prompt_tokens)
            estimated_output_tokens = max(0, estimated_output_tokens)
            total_tokens = estimated_prompt_tokens + estimated_output_tokens

            projected_rpm = len(self._minute_requests) + request_count
            projected_tpm = self._minute_token_total + total_tokens
            projected_rpd = len(self._day_requests) + request_count

            rpm_guard = self._guard_limit(GEMINI_RATE_RPM, GEMINI_RPM_GUARD)
            if rpm_guard is not None and projected_rpm > rpm_guard:
                return None, "rpm_guard"

            tpm_guard = self._guard_limit(GEMINI_RATE_TPM, GEMINI_TPM_GUARD)
            if tpm_guard is not None and projected_tpm > tpm_guard:
                return None, "tpm_guard"

            rpd_guard = self._guard_limit(GEMINI_RATE_RPD, GEMINI_RPD_GUARD)
            if rpd_guard is not None and projected_rpd > rpd_guard:
                return None, "rpd_guard"

            now = time.time()
            minute_entries = [
                _RequestStamp(timestamp=now) for _ in range(request_count)
            ]
            for entry in minute_entries:
                self._minute_requests.append(entry)
            day_entries = [_RequestStamp(timestamp=now) for _ in range(request_count)]
            for entry in day_entries:
                self._day_requests.append(entry)
            token_entry = None
            if total_tokens > 0:
                token_entry = _TokenStamp(timestamp=now, tokens=total_tokens)
                self._minute_tokens.append(token_entry)
                self._minute_token_total += total_tokens

            reservation = _GeminiReservation(
                manager=self,
                minute_entries=minute_entries,
                day_entries=day_entries,
                token_entry=token_entry,
                estimated_tokens=total_tokens,
            )
            return reservation, None

    def release(self, reservation: _GeminiReservation) -> None:
        with self._lock:
            for entry in reservation.minute_entries:
                try:
                    self._minute_requests.remove(entry)
                except ValueError:
                    continue
            for entry in reservation.day_entries:
                try:
                    self._day_requests.remove(entry)
                except ValueError:
                    continue
            if reservation.token_entry is not None:
                try:
                    self._minute_tokens.remove(reservation.token_entry)
                    self._minute_token_total -= reservation.token_entry.tokens
                except ValueError:
                    pass
            reservation.minute_entries.clear()
            reservation.day_entries.clear()
            reservation.token_entry = None
            reservation.estimated_tokens = 0

    def commit(self, reservation: _GeminiReservation, prompt_tokens: int, completion_tokens: int) -> None:
        total = max(0, (prompt_tokens or 0) + (completion_tokens or 0))
        with self._lock:
            if reservation.token_entry is not None:
                diff = total - reservation.estimated_tokens
                reservation.token_entry.tokens += diff
                self._minute_token_total += diff
            reservation.minute_entries.clear()
            reservation.day_entries.clear()
            reservation.token_entry = None
            reservation.estimated_tokens = total


_gemini_quota_manager = GeminiQuotaManager()

_ALLOWED_PROVIDERS = ("gemini", "cerebras", "local")


def _build_provider_chain() -> tuple[str, ...]:
    configured: list[str] = []
    for provider in PROVIDER_ORDER:
        normalized = provider.strip().lower()
        if normalized not in _ALLOWED_PROVIDERS:
            continue
        if normalized == "gemini" and not ENABLE_GEMINI:
            continue
        if normalized not in configured:
            configured.append(normalized)
    for fallback in _ALLOWED_PROVIDERS:
        if fallback == "gemini" and not ENABLE_GEMINI:
            continue
        if fallback not in configured:
            configured.append(fallback)
    return tuple(configured)


_PROVIDER_CHAIN = _build_provider_chain()


def _extract_status_code(e: Exception) -> Optional[int]:
    status = getattr(e, "status_code", None)
    if status is None:
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _describe_gemini_error(exc: Exception) -> tuple[str, bool]:
    status = _extract_status_code(exc)
    message = str(exc).lower() if exc is not None else ""
    if status == 429 or "429" in message or "rate limit" in message or "quota" in message:
        return "gemini_rate_limit", False
    if "timeout" in message or "timed out" in message:
        return "gemini_timeout", True
    if status is not None and 500 <= status < 600:
        return f"gemini_http_{status}", True
    if status in {408}:
        return "gemini_timeout", True
    if "unavailable" in message or "bad gateway" in message or "gateway" in message:
        return "gemini_service_unavailable", True
    return "gemini_error", True


def _describe_cerebras_error(exc: Exception) -> str:
    status = _extract_status_code(exc)
    if is_cerebras_tpd_limit_error(exc):
        return "cerebras_tpd"
    if status == 429:
        return "cerebras_rate_limit"
    if status is not None and 500 <= status < 600:
        return f"cerebras_http_{status}"
    message = str(exc).lower() if exc is not None else ""
    if "timeout" in message or "timed out" in message:
        return "cerebras_timeout"
    return "cerebras_error"


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


def _estimate_batch_prompt_tokens(prompt: str, payload_json: str) -> int:
    return estimate_prompt_tokens(prompt) + estimate_prompt_tokens(payload_json)


def _estimate_batch_output_tokens(items: Sequence[_BatchWorkItem]) -> int:
    if not items:
        return CLASSIFIER_MAX_OUTPUT_TOKENS
    return max(CLASSIFIER_MAX_OUTPUT_TOKENS, CLASSIFIER_MAX_OUTPUT_TOKENS * len(items) + 8)


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


def _call_gemini_batch(
    prompt: str,
    payload_json: str,
    items: list[_BatchWorkItem],
    reservation: _GeminiReservation,
    estimated_prompt_tokens: int,
    estimated_output_tokens: int,
) -> dict[str, ClassificationResult]:
    total_estimated = max(1, estimated_prompt_tokens + estimated_output_tokens)
    gemini_rate_limiter.acquire_sync(total_estimated, rpd=GEMINI_RATE_RPD)
    client = get_gemini_client()
    max_tokens = _estimate_batch_output_tokens(items)
    start = time.perf_counter()
    try:
     response = client.chat.completions.create(
    model="gemma-3-27b-it",
    messages=[
        {
            "role": "user",
            "content": f"{prompt}\n\n# Входные данные\n{payload_json}"
        },
    ],
    temperature=0.0,
    top_p=1.0,
    max_tokens=max_tokens,
    n=1,
)

    except Exception:
        reservation.release()
        raise
    latency_ms = (time.perf_counter() - start) * 1000.0
    usage = getattr(response, "usage", None)
    prompt_tokens = estimated_prompt_tokens
    completion_tokens = estimated_output_tokens
    if usage is not None:
        try:
            prompt_tokens = int(getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens)
        except (TypeError, ValueError):
            prompt_tokens = estimated_prompt_tokens
        try:
            completion_tokens = int(
                getattr(usage, "completion_tokens", completion_tokens) or completion_tokens
            )
        except (TypeError, ValueError):
            completion_tokens = estimated_output_tokens
    reservation.commit(prompt_tokens, completion_tokens)
    raw_text = _extract_choice_text(response)
    parsed = _parse_batch_response(raw_text, items)
    return _finalize_batch_result(
        "gemini",
        payload_json,
        raw_text,
        items,
        parsed,
        latency_ms=latency_ms,
    )


def _call_cerebras_batch(
    prompt: str,
    payload_json: str,
    items: list[_BatchWorkItem],
    *,
    fallback_reason: Optional[str] = None,
) -> dict[str, ClassificationResult]:
    client = get_cerebras_client(CEREBRAS_EXTRA_API if _cerebras_use_extra_key and CEREBRAS_EXTRA_API else None)
    max_tokens = _estimate_batch_output_tokens(items)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload_json},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        extra_body={"reasoning_effort": "high"},
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    raw_text = _extract_choice_text(response)
    parsed = _parse_batch_response(raw_text, items)
    return _finalize_batch_result(
        "cerebras",
        payload_json,
        raw_text,
        items,
        parsed,
        latency_ms=latency_ms,
        fallback_reason=fallback_reason,
    )


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


def _call_local_batch(
    prompt: str,
    payload_json: str,
    items: list[_BatchWorkItem],
    *,
    fallback_reason: Optional[str] = None,
) -> dict[str, ClassificationResult]:
    client = get_lmstudio_client()
    model_name = _resolve_lmstudio_model()
    max_tokens = _estimate_batch_output_tokens(items)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload_json},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        extra_body={"reasoning_effort": "medium"},
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    raw_text = _extract_choice_text(response)
    parsed = _parse_batch_response(raw_text, items)
    return _finalize_batch_result(
        "lmstudio",
        payload_json,
        raw_text,
        items,
        parsed,
        latency_ms=latency_ms,
        fallback_reason=fallback_reason,
    )


def _classify_remote_chunk(items: list[_BatchWorkItem], prompt: str) -> dict[str, ClassificationResult]:
    if not items:
        return {}

    global _cerebras_use_extra_key

    payload_json = _build_payload_json(items)
    estimated_prompt_tokens = _estimate_batch_prompt_tokens(prompt, payload_json)
    estimated_output_tokens = _estimate_batch_output_tokens(items)
    fallback_reason: Optional[str] = None

    for provider in _PROVIDER_CHAIN:
        if provider == "gemini":
            reservation, guard_reason = _gemini_quota_manager.reserve(
                estimated_prompt_tokens, estimated_output_tokens
            )
            if reservation is None:
                fallback_reason = f"gemini_{guard_reason or 'guard'}"
                logger.info(
                    "gemini_guard_redirect",
                    extra={
                        "extra": {
                            "reason": guard_reason or "guard",
                            "batch_size": len(items),
                            "estimated_prompt_tokens": estimated_prompt_tokens,
                            "estimated_output_tokens": estimated_output_tokens,
                        }
                    },
                )
                continue

            attempt = 0
            while True:
                try:
                    return _call_gemini_batch(
                        prompt,
                        payload_json,
                        items,
                        reservation,
                        estimated_prompt_tokens,
                        estimated_output_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    reservation.release()
                    reason, retryable = _describe_gemini_error(exc)
                    fallback_reason = reason
                    logger.warning(
                        "gemini_error_fallback",
                        extra={
                            "extra": {
                                "attempt": attempt + 1,
                                "retryable": retryable,
                                "reason": reason,
                                "error": str(exc),
                            }
                        },
                    )
                    if retryable and attempt < _GEMINI_MAX_RETRIES:
                        attempt += 1
                        new_reservation, guard_reason = _gemini_quota_manager.reserve(
                            estimated_prompt_tokens, estimated_output_tokens
                        )
                        if new_reservation is None:
                            fallback_reason = f"gemini_{guard_reason or 'guard'}"
                            logger.info(
                                "gemini_guard_after_retry",
                                extra={
                                    "extra": {
                                        "reason": guard_reason or "guard",
                                        "attempt": attempt,
                                    }
                                },
                            )
                            break
                        reservation = new_reservation
                        continue
                    break

        elif provider == "cerebras":
            while True:
                try:
                    return _call_cerebras_batch(
                        prompt,
                        payload_json,
                        items,
                        fallback_reason=fallback_reason,
                    )
                except Exception as exc:  # noqa: BLE001
                    reason = _describe_cerebras_error(exc)
                    fallback_reason = reason
                    logger.warning(
                        "cerebras_error_fallback",
                        extra={"extra": {"reason": reason, "error": str(exc)}},
                    )
                    if is_cerebras_tpd_limit_error(exc) and CEREBRAS_EXTRA_API and not _cerebras_use_extra_key:
                        _cerebras_use_extra_key = True
                        logger.warning(
                            "cerebras_switch_extra_key",
                            extra={"extra": {"error": str(exc)}},
                        )
                        continue
                    break

        elif provider == "local":
            try:
                return _call_local_batch(
                    prompt,
                    payload_json,
                    items,
                    fallback_reason=fallback_reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "local_classifier_error",
                    extra={"extra": {"error": str(exc), "batch_size": len(items)}},
                )
                return {}

    logger.error(
        "classifier_providers_exhausted",
        extra={"extra": {"batch_size": len(items), "fallback_reason": fallback_reason}},
    )
    return {}


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
