import asyncio
import html
import json
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple, List, TypedDict
import re
import filelock

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import Channel, PeerUser, PeerChannel, PeerChat

import httpx

from openai import OpenAI
from const import DIALOGUE_PROMPT, CLASSIFIER_PROMPT


# ------------------------------
# Configuration
# ------------------------------
LOG_LEVEL = "INFO"
REQUEST_TIMEOUT_S = 20.0
RETRY_MAX_ATTEMPTS = 3

# Detect Docker runtime to properly reach host services (e.g., LM Studio)
def _is_running_in_docker() -> bool:
    try:
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    except Exception:
        return False

# Hardcoded credentials per user request (TEST USE ONLY)
TELEGRAM_API_ID = "28738574"
TELEGRAM_API_HASH = "343d6e8a20de4a2f3cc265eafdd64f71"
TELEGRAM_STRING_SESSION = "1AZWarzkBuzIOLASsWrjMkxeeZ5PaJoMtJSZSajB2lJEXQivilsJJHIPX6JQgSFfIVfi0dTf-LbaBHk_8N_kUyXWljBgsAPJVOL6qtTX1fgJAxTTNfkTZQq049Ad9PrlLwfU4AbNXgyYAfXV_tLobDQLALoTssGcqxXulW6b556iDc0xf7msg-QO8OIVzLI28ASxtXdbfTMrBOQ9gp3xaV5oZyp3XNCic9vtRYqPWxkCRBnlM4m8RNwaUZo86rYldDCiugbzRNZwrfqq9VYZtKQ2fTO5FFUKnMZADaBAePy7fsAKpee1IXraaBjMQeRUR6DM8iXAdqkV-ajEkyUPNmMS3IMomhOE="
TELEGRAM_SESSION_PATH = "session"

# LM Studio (local OpenAI-compatible) configuration (Docker-aware default)
LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL",
    "http://host.docker.internal:1234" if _is_running_in_docker() else "http://127.0.0.1:1234",
)
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "not-needed")

# Cerebras (OpenAI-compatible) configuration for classification
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_API_KEY = "csk-5njr2er53p5dwh6ymrfj4jjf9tctx9rpr3xtw4mc6te6dd38"
# Extra API key to use when primary Cerebras key hits TPD
CEREBRAS_EXTRA_API = "csk-6p5w4vjhnk42vnvmwjncn85cvwrp3dj84n5fj26m5n56phex"

# Gemini API (third fallback)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = "AIzaSyD9Ig7Isn6THvmClllKTUucoMWPuEpdziU"
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Bot API for notifications
TELEGRAM_BOT_TOKEN = "8255044221:AAH_MKTuXbuWoLn0OlJRr6amaXMbdQ3jUlg"

GROQ_API_KEY = "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"

# Groq OpenAI-compatible settings for reply generation
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_REPLY_MODEL = "moonshotai/kimi-k2-instruct-0905"

# Optional: limit processing to a single chat during tests (disabled by default)
_TEST_ONLY_CHAT_ID_RAW = os.getenv("TEST_ONLY_CHAT_ID", "").strip()
TEST_ONLY_CHAT_ID: Optional[int] = int(_TEST_ONLY_CHAT_ID_RAW) if _TEST_ONLY_CHAT_ID_RAW else None

# Rate limits (Cerebras quotas)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
RATE_LIMIT_RPH = int(os.getenv("RATE_LIMIT_RPH", "900"))
RATE_LIMIT_TPM = int(os.getenv("RATE_LIMIT_TPM", "60000"))

# Supabase configuration (subscribers storage)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wswatojrgeknsfqujjvi.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indzd2F0b2pyZ2VrbnNmcXVqanZpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2Mjc3MTgsImV4cCI6MjA3MzIwMzcxOH0.zj6HBxOIkoJsZ3qByQfipeoR4S-EbAMH7tJ__SLajtI")


# ------------------------------
# Logging
# ------------------------------
class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        # Allow plain, human-readable lines when requested
        if getattr(record, "plain", False):
            return record.getMessage()
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": int(time.time() * 1000),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logging.basicConfig(level=LOG_LEVEL, handlers=[handler])
logger = logging.getLogger("message_finder")

# Minimize noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.WARNING)


# ------------------------------
# Dedup Store (TTL-based)
# ------------------------------
class DedupStore:
    def __init__(self, ttl_seconds: int = 24 * 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: Dict[Tuple[int, int], float] = {}

    def _now(self) -> float:
        return time.time()

    def seen(self, chat_id: int, message_id: int) -> bool:
        key = (chat_id, message_id)
        expiry = self._store.get(key)
        if expiry is None:
            return False
        if expiry < self._now():
            del self._store[key]
            return False
        return True

    def mark(self, chat_id: int, message_id: int) -> None:
        self._store[(chat_id, message_id)] = self._now() + self.ttl_seconds


dedup_store = DedupStore()


# ------------------------------
# Subscriber Store (Bot API /start)
# ------------------------------
from utilities.subscribers_store import SupabaseSubscriberStore

subscriber_store = SupabaseSubscriberStore(SUPABASE_URL, SUPABASE_ANON_KEY)


# ------------------------------
# UI State for Reply Generation
# ------------------------------
class ReplyUIState:
    def __init__(self, user_id: int, original_body_html: str, original_text: str, context_for_model: Optional[str] = None, classification_result: Optional["ClassificationResult"] = None) -> None:
        self.user_id = int(user_id)
        self.original_body_html = original_body_html
        self.original_text = original_text
        self.context_for_model = context_for_model
        self.last_reply_text: Optional[str] = None
        self.classification_result = classification_result


class ReplyUIStore:
    def __init__(self) -> None:
        self._seq = 0
        self._states: Dict[str, ReplyUIState] = {}

    def create(self, user_id: int, original_body_html: str, original_text: str, context_for_model: Optional[str] = None, classification_result: Optional["ClassificationResult"] = None) -> str:
        self._seq += 1
        sid = f"s{self._seq}"
        self._states[sid] = ReplyUIState(user_id=user_id, original_body_html=original_body_html, original_text=original_text, context_for_model=context_for_model, classification_result=classification_result)
        return sid

    def get(self, sid: str) -> Optional[ReplyUIState]:
        return self._states.get(sid)

    def set_reply(self, sid: str, reply_text: str) -> None:
        st = self._states.get(sid)
        if st:
            st.last_reply_text = reply_text


reply_ui_store = ReplyUIStore()


# ------------------------------
# OpenAI SDK (LM Studio) Client
# ------------------------------

_oa_client: Optional[OpenAI] = None
_oa_client_key: Optional[str] = None
_groq_client: Optional[OpenAI] = None
_gemini_client: Optional[OpenAI] = None
_http_client: Optional[httpx.AsyncClient] = None
COPY_TEXT_ALLOWED: bool = True

# Switch to extra API key after TPD; then to Gemini, then to local LM Studio
_cerebras_use_extra_key: bool = False
_gemini_fallback_active: bool = False
_lm_client: Optional[OpenAI] = None
# Switch to local LM Studio after Cerebras/Gemini fallbacks also hit TPD
_local_fallback_active: bool = False
_lm_model_resolved_name: Optional[str] = None


def _normalize_base_url(raw: str) -> str:
    base = (raw or "http://127.0.0.1:1234").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def get_openai_client() -> OpenAI:
    global _oa_client, _oa_client_key, _cerebras_use_extra_key
    desired_key = (CEREBRAS_EXTRA_API if _cerebras_use_extra_key and CEREBRAS_EXTRA_API else CEREBRAS_API_KEY)
    if _oa_client is not None and _oa_client_key == desired_key:
        return _oa_client
    # Recreate client if key changed or not initialized
    _oa_client = OpenAI(
        base_url=_normalize_base_url(CEREBRAS_BASE_URL),
        api_key=desired_key,
        timeout=REQUEST_TIMEOUT_S,
    )
    _oa_client_key = desired_key
    return _oa_client


def get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    _groq_client = OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        timeout=REQUEST_TIMEOUT_S,
    )
    return _groq_client


def get_gemini_client() -> OpenAI:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    _gemini_client = OpenAI(
        base_url=_normalize_base_url(GEMINI_BASE_URL),
        api_key=GEMINI_API_KEY,
        timeout=REQUEST_TIMEOUT_S,
    )
    return _gemini_client


def get_lmstudio_client() -> OpenAI:
    global _lm_client
    if _lm_client is not None:
        return _lm_client
    _lm_client = OpenAI(
        base_url=_normalize_base_url(LMSTUDIO_BASE_URL),
        api_key=LMSTUDIO_API_KEY,
        timeout=REQUEST_TIMEOUT_S,
        # Reduce SDK-side retry latency for local server
        max_retries=1,
    )
    return _lm_client


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is not None:
        return _http_client
    _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    return _http_client


# Path for feedback data
FEEDBACK_FILE_PATH = "data/feedback.json"
LOCK_FILE_PATH = "data/feedback.json.lock"

def save_feedback(feedback_data: dict):
    """Appends a feedback entry to the feedback JSON file in a thread-safe way."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE_PATH), exist_ok=True)
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        examples = []
        if os.path.exists(FEEDBACK_FILE_PATH):
            try:
                with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        examples = json.loads(content)
                    else:
                        examples = []
            except (json.JSONDecodeError, FileNotFoundError):
                examples = []

        examples.append(feedback_data)

        with open(FEEDBACK_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)


class ClassificationResult(TypedDict):
    classification: str
    themes: str


def _extract_label_from_chat_completion(cc) -> str:
    try:
        raw = (cc.choices[0].message.content or "").strip()
    except Exception:
        raw = ""
    if raw in {"0", "1"}:
        return raw
    m = re.search(r"[01]", raw)
    if m:
        return m.group(0)
    return "0"


def _extract_label_from_responses(resp) -> str:
    try:
        raw = (getattr(resp, "output_text", None) or "").strip()
    except Exception:
        raw = ""
    if raw in {"0", "1"}:
        return raw
    m = re.search(r"[01]", raw)
    if m:
        return m.group(0)
    return "0"


def _is_cerebras_tpd_limit_error(e: Exception) -> bool:
    try:
        msg = str(e).lower()
    except Exception:
        return False
    # Match various representations of daily quota exhaustion and HTTP 429 limit hits
    return (
        "token_quota_exceeded" in msg
        or "too_many_tokens_error" in msg
        or "tokens per day" in msg
        or "limit hit" in msg
        or "http 429" in msg
        or "rate limit" in msg
    )


def _parse_classifier_json(raw: str) -> ClassificationResult:
    # Try direct JSON first
    try:
        obj = json.loads(raw)
    except Exception:
        # Strip code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            try:
                first_newline = cleaned.find("\n")
                cleaned = cleaned[first_newline + 1 :]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            except Exception:
                pass
        # Try to extract first {...}
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(cleaned[start : end + 1])
            else:
                obj = {}
        except Exception:
            obj = {}

    cls = str(obj.get("classification", "")).strip() if isinstance(obj, dict) else ""
    themes = str(obj.get("themes", "")).strip() if isinstance(obj, dict) else ""
    if cls not in {"0", "1"}:
        # Fallback to simple extraction
        m = re.search(r"[01]", raw)
        cls = m.group(0) if m else "0"
        themes = ""
    if cls == "0":
        themes = ""
    return {"classification": cls, "themes": themes}


def load_feedback_examples() -> list[str]:
    """Loads and formats feedback examples from the JSON file."""
    if not os.path.exists(FEEDBACK_FILE_PATH):
        return []

    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        try:
            with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    return []
                examples_data = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    formatted_examples = []
    for ex in examples_data:
        message = ex.get("message", "")
        output = ex.get("output", {})
        label = ex.get("label")

        if not message or not isinstance(output, dict) or label is None:
            continue

        # Correct the classification based on the user's feedback label
        if label == 0:
            original_classification = output.get("classification")
            if original_classification == "1":
                corrected_classification = "0"
                corrected_themes = ""
            elif original_classification == "0":
                corrected_classification = "1"
                corrected_themes = "" # Cannot generate new themes, so leave empty.
            else:
                continue # Skip invalid original classification

            corrected_output = {
                "classification": corrected_classification,
                "themes": corrected_themes
            }
            output_str = json.dumps(corrected_output, ensure_ascii=False)
        else: # label == 1, use the original classification
            output_str = json.dumps(output, ensure_ascii=False)

        # Use json.dumps for the message to handle quotes inside the string properly
        formatted_examples.append(f'Input: {json.dumps(message, ensure_ascii=False)}\nOutput: {output_str}')

    return formatted_examples


def classify_with_openai_sync(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    if not message_text or not message_text.strip():
        return {"classification": "0", "themes": ""}

    global _cerebras_use_extra_key, _gemini_fallback_active, _local_fallback_active, _lm_model_resolved_name

    # Dynamically build the classifier prompt
    feedback_examples = load_feedback_examples()
    if feedback_examples:
        feedback_examples_str = "\n\n".join(feedback_examples)
        parts = CLASSIFIER_PROMPT.split("Examples:", 1)
        dynamic_classifier_prompt = f"{parts[0]}Examples:\n\n{feedback_examples_str}\n\n{parts[1]}" if len(parts) == 2 else CLASSIFIER_PROMPT
    else:
        dynamic_classifier_prompt = CLASSIFIER_PROMPT

    attempts = 0
    while True:
        try:
            # Choose provider/model with multi-tier fallback
            if _local_fallback_active:
                client = get_lmstudio_client()
                if not _lm_model_resolved_name:
                    try:
                        models = client.models.list()
                        ids = [m.id for m in getattr(models, "data", [])]
                        desired = LMSTUDIO_MODEL
                        if desired in ids: _lm_model_resolved_name = desired
                        else:
                            candidates = [mid for mid in ids if "gpt" in mid.lower() and "oss" in mid.lower()]
                            _lm_model_resolved_name = candidates[0] if candidates else (ids[0] if ids else desired)
                        logger.info("lmstudio_model_selected", extra={"extra": {"selected": _lm_model_resolved_name}})
                    except Exception: _lm_model_resolved_name = LMSTUDIO_MODEL
                model_name = _lm_model_resolved_name
            elif _gemini_fallback_active:
                client = get_gemini_client()
                model_name = GEMINI_MODEL
            else:
                client = get_openai_client()
                model_name = CEREBRAS_MODEL

            user_parts = [f"CONTEXT (may be truncated):\n{context.strip()}"] if context and context.strip() else []
            user_parts.append(f"CURRENT_MESSAGE:\n{message_text.strip()}")
            user_content = "\n\n".join(user_parts)

            # Define kwargs based on provider
            if _local_fallback_active:
                kwargs = {"messages": [{"role": "system", "content": dynamic_classifier_prompt}, {"role": "user", "content": user_content}]}
            elif _gemini_fallback_active:
                kwargs = {"messages": [{"role": "system", "content": dynamic_classifier_prompt}, {"role": "user", "content": user_content}], "extra_body": {"reasoning_effort": "low"}}
            else:  # Cerebras
                kwargs = {"messages": [{"role": "system", "content": dynamic_classifier_prompt}, {"role": "user", "content": user_content}], "temperature": 1, "top_p": 1.0, "max_tokens": 4000, "extra_body": {"reasoning_effort": "low"}}

            cc = client.chat.completions.create(model=model_name, **kwargs)
            return _parse_classifier_json(cc.choices[0].message.content or "")

        except Exception as e:
            attempts += 1
            is_tpd = _is_cerebras_tpd_limit_error(e)

            # Fallback Logic
            if not _local_fallback_active:
                if _gemini_fallback_active: # If Gemini fails for any reason, switch to local
                    _local_fallback_active = True
                    logger.warning("gemini_fail_switch_to_lmstudio", extra={"extra": {"error": str(e)}})
                    continue
                elif is_tpd: # If Cerebras fails with TPD
                    if not _cerebras_use_extra_key and CEREBRAS_EXTRA_API:
                        _cerebras_use_extra_key = True
                        logger.warning("cerebras_tpd_switch_key", extra={"extra": {"error": str(e)}})
                        global _oa_client
                        _oa_client = None  # Force re-init
                        continue
                    else: # Both Cerebras keys failed, switch to Gemini
                        _gemini_fallback_active = True
                        logger.warning("cerebras_tpd_switch_to_gemini", extra={"extra": {"error": str(e)}})
                        continue

            if attempts > RETRY_MAX_ATTEMPTS:
                logger.error("openai_classify_failed_all_fallbacks", extra={"extra": {"error": str(e)}})
                return {"classification": "0", "themes": ""}

            time.sleep(min(2 ** attempts, 10))


async def classify_with_openai(message_text: str, context: Optional[str] = None) -> ClassificationResult:
    return await asyncio.to_thread(classify_with_openai_sync, message_text, context)


# ------------------------------
# Reply Generation (LM Studio via OpenAI SDK)
# ------------------------------

def generate_reply_sync(message_text: str, context: Optional[str] = None) -> str:
    safe_text = (message_text or "").strip()
    if not safe_text:
        return "Можешь уточнить, что именно хочешь сделать? Я помогу."

    # Use Groq provider for reply generation per user request
    client = get_groq_client()
    attempts = 0
    while True:
        try:
            user_content_parts: List[str] = []
            if context and context.strip():
                user_content_parts.append(f"CONTEXT (may be truncated):\n{context.strip()}")
            user_content_parts.append(f"CURRENT_MESSAGE:\n{safe_text}")
            # Compose final user input for the model (log to verify context inclusion)
            user_content = "\n\n".join(user_content_parts)
            logger.info(f"replier input:\n{user_content}", extra={"plain": True})
            # Use Chat Completions (Kimi K2 does not support reasoning)
            cc = client.chat.completions.create(
                model=GROQ_REPLY_MODEL,
                messages=[
                    {"role": "system", "content": DIALOGUE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.6,
                top_p=1.0,
                max_tokens=48,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            raw = (cc.choices[0].message.content or "").strip()
            # Normalize to a single short sentence without fluff
            text = normalize_short_reply(raw)
            logger.info(f"replier output:\n{text}", extra={"plain": True})
            return text or "Давай уточним цель в одном предложении."
        except Exception as e:  # noqa: BLE001
            attempts += 1
            if attempts > RETRY_MAX_ATTEMPTS:
                logger.error("openai_reply_failed", extra={"extra": {"error": str(e)}})
                return "Секунду, пожалуйста. Кажется, сеть шалит — попробуем ещё раз?"
            time.sleep(min(2 ** attempts, 10))


async def generate_reply(message_text: str, context: Optional[str] = None) -> str:
    # Use same rate limiter since we consume local model quota
    estimated = estimate_prompt_tokens(message_text)
    if context:
        estimated += max(1, len(context) // 4)
    estimated += 48
    await rate_limiter.acquire(estimated)
    return await asyncio.to_thread(generate_reply_sync, message_text, context)


def _is_copy_text_safe(text: str) -> bool:
    # Telegram copy_text has a strict byte limit (<=256 bytes)
    try:
        return len(text.encode("utf-8")) <= 256
    except Exception:
        return False


def sanitize_copy_text(text: str) -> Optional[str]:
    if not text:
        return None
    # Collapse whitespace and strip
    collapsed = " ".join(text.strip().split())
    # Enforce byte limit
    try:
        data = collapsed.encode("utf-8")
    except Exception:
        return None
    if len(data) <= 256:
        return collapsed
    # Trim to 256 bytes and ensure valid UTF-8
    trimmed = data[:256]
    safe = trimmed.decode("utf-8", errors="ignore").rstrip()
    return safe or None


def build_reply_keyboard(sid: str, reply_text: Optional[str] = None) -> Dict[str, Any]:
    rows: list[list[Dict[str, Any]]] = [[
        {"text": "⬅ Назад", "callback_data": f"back:{sid}"},
        {"text": "🔁 Перегенерировать", "callback_data": f"regen:{sid}"},
    ]]
    if reply_text is not None and COPY_TEXT_ALLOWED:
        safe = sanitize_copy_text(reply_text)
        if safe and _is_copy_text_safe(safe):
            rows.append([
                {"text": "📋 Скопировать", "copy_text": {"text": safe}},
            ])
    return {"inline_keyboard": rows}


def normalize_short_reply(raw: str) -> str:
    # Collapse whitespace and strip simple quotes
    text = re.sub(r"\s+", " ", (raw or "").strip()).strip('\"\' ')
    if not text:
        return text
    # If multiple sentences present, keep the first complete sentence
    m = re.search(r"[.!?…]", text)
    if m:
        return text[: m.start() + 1].strip()
    # Otherwise, return as-is; add a period if it seems unfinished
    return (text + ".") if not text.endswith((".", "!", "?", "…")) else text


def build_message_preview(text: str, limit: int = 160) -> str:
    normalized = " ".join((text or "").strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


# ------------------------------
# Telegram utilities
# ------------------------------
def build_internal_chat_c_id(chat_id: int) -> Optional[str]:
    raw = str(abs(chat_id))
    # For supergroups/channels chat_id often starts with 100... strip leading '100'
    if raw.startswith("100"):
        return raw[3:]
    return raw


async def build_message_link(event: events.NewMessage.Event) -> Optional[str]:
    try:
        chat = await event.get_chat()
        username = getattr(chat, "username", None)
        if username:
            return f"https://t.me/{username}/{event.id}"

        # Private groups/channels without username
        if isinstance(chat, Channel) or event.is_channel:
            internal = build_internal_chat_c_id(event.chat_id)  # type: ignore[arg-type]
            return f"https://t.me/c/{internal}/{event.id}" if internal else None
    except Exception as e:  # noqa: BLE001
        logger.warning("link_build_failed", extra={"extra": {"error": str(e)}})
    return None

# Collect the reply chain context (plain text for LLM and HTML for notification)
async def collect_reply_context(event: events.NewMessage.Event, depth_limit: int = 6) -> Tuple[Optional[str], Optional[str]]:
    try:
        # Walk up the reply chain
        chain: List[Any] = []
        current = event.message
        steps = 0
        while current is not None and steps < depth_limit:
            try:
                parent = await current.get_reply_message()  # type: ignore[attr-defined]
            except Exception:
                parent = None
            if not parent:
                break
            chain.append(parent)
            current = parent
            steps += 1

        if not chain:
            return None, None

        # Oldest first
        chain = list(reversed(chain))

        def _msg_text(msg: Any) -> Optional[str]:
            try:
                raw = getattr(msg, "message", None)
                return (raw or "").strip() or None
            except Exception:
                return None

        parts_plain: List[str] = []
        parts_html: List[str] = ["<b>Контекст</b>", "<b>━━━━━━━━━━━━━━━━━━━━</b>"]

        for idx, msg in enumerate(chain):
            text = _msg_text(msg)
            if not text:
                continue
            # Plain transcript-like formatting
            if idx == 0:
                parts_plain.append(f"Post: {text}")
            else:
                parts_plain.append(f"Reply{idx}: {text}")

            # HTML, escape and emulate quote
            safe = escape_html(text)
            prefix = "Пост" if idx == 0 else f"Ответ {idx}"
            parts_html.append(f"• <i>{prefix}</i>:\n{safe}")

        if not parts_plain:
            return None, None

        context_plain = "\n".join(parts_plain)
        context_html = "\n".join(parts_html)
        return context_plain, context_html
    except Exception as e:  # noqa: BLE001
        logger.warning("collect_reply_context_failed", extra={"extra": {"error": str(e)}})
        return None, None
# Extract sender user_id robustly across groups/channels. Returns (user_id, reason)
async def resolve_author_user_id(event: events.NewMessage.Event) -> Tuple[Optional[int], Optional[str]]:
    try:
        from_id = getattr(event.message, "from_id", None)
        if isinstance(from_id, PeerUser):
            return int(from_id.user_id), None
        if isinstance(from_id, PeerChannel):
            return None, "author_is_channel"
        if isinstance(from_id, PeerChat):
            # Legacy small groups; fall back to sender_id
            sid = int(getattr(event, "sender_id", 0) or 0)
            return (sid if sid else None), (None if sid else "no_user_id")

        # Fallback: try event.sender_id
        sid = int(getattr(event, "sender_id", 0) or 0)
        if sid:
            return sid, None

        # Last resort: resolve entity
        sender = await event.get_sender()
        uid = int(getattr(sender, "id", 0))
        return (uid if uid else None), (None if uid else "no_user_id")
    except Exception as e:  # noqa: BLE001
        logger.warning("resolve_author_failed", extra={"extra": {"error": str(e)}})
        return None, "resolve_error"

def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


async def notifier_send(
    client: TelegramClient,
    user_id: int,
    user: Optional[Any],
    chat: Any,
    text: str,
    link: Optional[str],
    context_html: Optional[str] = None,
    context_plain: Optional[str] = None,
    themes: Optional[str] = None,
    classification_result: Optional[ClassificationResult] = None,
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("bot_token_missing", extra={"extra": {"msg": "TELEGRAM_BOT_TOKEN is not set"}})
        return

    full_name = " ".join(filter(None, [getattr(user, "first_name", None), getattr(user, "last_name", None)])) or "отсутствует"
    username = f"@{getattr(user, 'username', '')}" if getattr(user, "username", None) else "отсутствует"
    chat_title = getattr(chat, "title", None) or getattr(chat, "username", None) or chat.__class__.__name__

    underlined = f"<u>{escape_html(text)}</u>"
    header = "<b>🔎 Обнаружен \"возможный диалог\"</b>"
    divider = "<b>━━━━━━━━━━━━━━━━━━━━</b>"
    parts = [
        header,
        divider,
        f"• <b>Имя</b>: {escape_html(full_name)}",
        f"• <b>Никнейм</b>: <i>{escape_html(username)}</i>",
        f"• <b>Чат</b>: <i>{escape_html(str(chat_title))}</i>",
        divider,
        f"<b>Сообщение</b>:\n{underlined}",
    ]
    if context_html:
        parts.extend([divider, context_html])
    if themes and themes.strip():
        parts.extend([
            divider,
            "<b>Подсказки Диалога</b>",
            "<b>━━━━━━━━━━━━━━━━━━━━</b>",
            escape_html(themes.strip()),
        ])
    if link:
        parts.append(f"• <b>Ссылка</b>: <a href=\"{escape_html(link)}\">перейти</a>")
    body = "\n".join(parts)

    # Create UI state and initial inline keyboard with Generate button
    sid = reply_ui_store.create(
        user_id=int(user_id),
        original_body_html=body,
        original_text=text,
        context_for_model=context_plain,
        classification_result=classification_result,
    )
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "👍", "callback_data": f"like:{sid}"},
                {"text": "👎", "callback_data": f"dislike:{sid}"},
                {"text": "✨", "callback_data": f"gen:{sid}"}
            ]
        ]
    }

    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": int(user_id),
        "text": body,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": keyboard,
    }

    async def _post():
        resp = await http_client.post(url, json=payload)
        # Validate Telegram Bot API response; raise to trigger retry on failure
        if resp.status_code >= 400:
            raise RuntimeError(f"bot_send_http_{resp.status_code}")
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"bot_send_invalid_json:{e}")
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_send_api_error:{desc}")
        return resp

    await send_with_retries(_post)


async def send_with_retries(coro_factory) -> bool:
    attempt = 0
    delay = 0.5
    while True:
        try:
            result = coro_factory()
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "button_copy_text_invalid" in msg or "copy_text_invalid" in msg:
                logger.warning("copy_text_disabled", extra={"extra": {"reason": msg}})
                # Turn off copy_text for subsequent keyboards
                global COPY_TEXT_ALLOWED
                COPY_TEXT_ALLOWED = False
                return False
            attempt += 1
            if attempt > RETRY_MAX_ATTEMPTS:
                logger.error("notify_failed", extra={"extra": {"error": str(e)}})
                return False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)


# ------------------------------
# Telegram Bot API helpers (edit/answer)
# ------------------------------
async def bot_edit_message_text(chat_id: int, message_id: int, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"

    # Telegram hard limit
    def _truncate_html(text: str) -> str:
        return text if len(text) <= 4096 else text[:4080] + "…"

    payload: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "text": _truncate_html(html_text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    async def _post():
        resp = await http_client.post(url, json=payload)
        # Try to parse Telegram error details even on HTTP 4xx
        if resp.status_code >= 400:
            try:
                data = resp.json()
                desc = data.get("description", f"HTTP_{resp.status_code}")
            except Exception:  # noqa: BLE001
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            # Treat benign case as success to avoid infinite retries
            if "message is not modified" in desc.lower():
                return resp
            # If copy_text is invalid, disable feature globally and surface error
            if "button_copy_text_invalid" in desc.lower():
                global COPY_TEXT_ALLOWED
                COPY_TEXT_ALLOWED = False
                raise RuntimeError("copy_text_invalid")
            raise RuntimeError(f"bot_edit_error:{desc}")

        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"bot_edit_invalid_json:{e}")

        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            if "message is not modified" in desc.lower():
                return resp
            raise RuntimeError(f"bot_edit_api_error:{desc}")
        return resp

    return await send_with_retries(_post)


async def bot_answer_callback_query(callback_query_id: str, text: Optional[str] = None, show_alert: bool = False) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    if show_alert:
        payload["show_alert"] = True

    async def _post():
        resp = await http_client.post(url, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"bot_answer_http_{resp.status_code}")
        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_answer_api_error:{desc}")
        return resp

    return await send_with_retries(_post)


async def bot_send_html_message(chat_id: int, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    def _truncate_html(text: str) -> str:
        return text if len(text) <= 4096 else text[:4080] + "…"

    payload: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "text": _truncate_html(html_text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    async def _post():
        resp = await http_client.post(url, json=payload)
        if resp.status_code >= 400:
            try:
                data = resp.json()
                desc = data.get("description", f"HTTP_{resp.status_code}")
            except Exception:  # noqa: BLE001
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            if "button_copy_text_invalid" in desc.lower():
                global COPY_TEXT_ALLOWED
                COPY_TEXT_ALLOWED = False
                raise RuntimeError("copy_text_invalid")
            raise RuntimeError(f"bot_send_error:{desc}")
        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_send_api_error:{desc}")
        return resp

    return await send_with_retries(_post)


# ------------------------------
# Rate Limiter (RPM/RPH/TPM approx)
# ------------------------------
class RateLimiter:
    def __init__(self, rpm: int, rph: int, tpm: int) -> None:
        self.rpm = rpm
        self.rph = rph
        self.tpm = tpm
        self._minute_events: Deque[float] = deque()
        self._hour_events: Deque[float] = deque()
        self._minute_tokens: Deque[Tuple[float, int]] = deque()

    def _cleanup(self) -> None:
        now = time.time()
        one_minute_ago = now - 60.0
        one_hour_ago = now - 3600.0
        while self._minute_events and self._minute_events[0] < one_minute_ago:
            self._minute_events.popleft()
        while self._hour_events and self._hour_events[0] < one_hour_ago:
            self._hour_events.popleft()
        while self._minute_tokens and self._minute_tokens[0][0] < one_minute_ago:
            self._minute_tokens.popleft()

    def _minute_tokens_used(self) -> int:
        return sum(tok for _, tok in self._minute_tokens)

    async def acquire(self, estimated_tokens: int) -> None:
        while True:
            self._cleanup()
            if len(self._minute_events) < self.rpm and len(self._hour_events) < self.rph and (self._minute_tokens_used() + estimated_tokens) <= self.tpm:
                now = time.time()
                self._minute_events.append(now)
                self._hour_events.append(now)
                self._minute_tokens.append((now, estimated_tokens))
                return
            # Sleep until the next window likely frees
            await asyncio.sleep(0.2)


rate_limiter = RateLimiter(RATE_LIMIT_RPM, RATE_LIMIT_RPH, RATE_LIMIT_TPM)


def estimate_prompt_tokens(message_text: str) -> int:
    # Rough heuristic: 1 token ~ 4 chars; add prompt/system overhead
    overhead = 64
    return overhead + max(1, len(message_text) // 4)


# ------------------------------
# Bot updates poller (/start registry)
# ------------------------------
async def bot_updates_poller() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("bot_token_missing", extra={"extra": {"msg": "TELEGRAM_BOT_TOKEN is not set; poller disabled"}})
        return
    http_client = get_http_client()
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    timeout_s = 50
    # Ensure polling mode (disable webhook if previously set)
    try:
        await http_client.get(f"{base}/deleteWebhook", params={"drop_pending_updates": False})
    except Exception as e:  # noqa: BLE001
        logger.warning("delete_webhook_failed", extra={"extra": {"error": str(e)}})
    while True:
        try:
            _offset_val = subscriber_store.get_offset()
            offset = (_offset_val + 1) if _offset_val else None
            params: Dict[str, Any] = {"timeout": timeout_s}
            if offset is not None:
                params["offset"] = offset
            resp = await http_client.get(
                f"{base}/getUpdates",
                params=params,
                timeout=timeout_s + 10,
            )
            data = resp.json()
            if not data.get("ok", False):
                await asyncio.sleep(1.0)
                continue
            for upd in data.get("result", []) or []:
                upd_id = int(upd.get("update_id", 0))
                try:
                    subscriber_store.advance_offset(upd_id)
                except Exception:
                    pass
                # 1) Handle callback_query (inline keyboard actions)
                callback = upd.get("callback_query")
                if callback:
                    try:
                        cb_id = callback.get("id")
                        from_user = callback.get("from", {})
                        from_user_id = int(from_user.get("id")) if from_user and from_user.get("id") is not None else None
                        msg = callback.get("message") or {}
                        msg_chat = msg.get("chat") or {}
                        msg_chat_id = int(msg_chat.get("id")) if msg_chat and msg_chat.get("id") is not None else None
                        msg_id = int(msg.get("message_id")) if msg and msg.get("message_id") is not None else None
                        data_s = callback.get("data") or ""

                        if not data_s or not cb_id or msg_chat_id is None or msg_id is None:
                            continue

                        # Parse action
                        if ":" in data_s:
                            action, sid = data_s.split(":", 1)
                        else:
                            action, sid = data_s, ""

                        st = reply_ui_store.get(sid) if sid else None
                        if st and from_user_id and st.user_id != int(from_user_id):
                            # Prevent others from using the UI intended for a particular user
                            await bot_answer_callback_query(cb_id, text="Эта кнопка не для вас", show_alert=False)
                            continue

                        if action == "gen" or action == "regen":
                            await bot_answer_callback_query(cb_id, text="Генерирую ответ…", show_alert=False)
                            if not st:
                                await bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
                                continue
                            reply_text = await generate_reply(st.original_text, context=st.context_for_model)
                            reply_ui_store.set_reply(sid, reply_text)

                            safe = escape_html(reply_text)
                            body = "\n".join([
                                "<b>✍️ Предложенный ответ</b>",
                                "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                                safe,
                            ])
                            reply_markup: Dict[str, Any] = build_reply_keyboard(sid, reply_text)

                            ok = await bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=reply_markup)
                            if not ok:
                                # Fallback without copy button if Bot API rejects it
                                fallback_markup = build_reply_keyboard(sid, None)
                                ok2 = await bot_edit_message_text(msg_chat_id, msg_id, body, reply_markup=fallback_markup)
                                if not ok2:
                                    # If editing still fails (e.g., message can't be edited), send a new message
                                    await bot_send_html_message(msg_chat_id, body, reply_markup=fallback_markup)
                            continue

                        if action == "like" or action == "dislike":
                            await bot_answer_callback_query(cb_id, text="Спасибо за отзыв!", show_alert=False)
                            if not st or not st.classification_result:
                                logger.warning("feedback_handler_missing_state", extra={"extra": {"sid": sid, "action": action}})
                                continue

                            feedback_data = {
                                "message": st.original_text,
                                "output": st.classification_result,
                                "label": 1 if action == "like" else 0
                            }

                            try:
                                # Using asyncio.to_thread to avoid blocking the event loop with file I/O
                                await asyncio.to_thread(save_feedback, feedback_data)
                                logger.info("feedback_saved", extra={"extra": {"sid": sid, "action": action}})
                            except Exception as e:
                                logger.error("save_feedback_failed", extra={"extra": {"error": str(e)}})

                            continue

                        if action == "back":
                            await bot_answer_callback_query(cb_id, text="Возвращаю…", show_alert=False)
                            if not st:
                                await bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
                                continue
                            orig_keyboard = {
                                "inline_keyboard": [
                                      [
                {"text": "👎", "callback_data": f"dislike:{sid}"},
                {"text": "✨", "callback_data": f"gen:{sid}"}
                                          ]
                                ]
                            }
                            ok = await bot_edit_message_text(msg_chat_id, msg_id, st.original_body_html, reply_markup=orig_keyboard)
                            if not ok:
                                # If editing fails, send a new message with the original body and keyboard
                                await bot_send_html_message(msg_chat_id, st.original_body_html, reply_markup=orig_keyboard)
                            continue

                        # Unknown action
                        await bot_answer_callback_query(cb_id, text="Неизвестное действие", show_alert=False)
                        continue
                    except Exception as e:  # noqa: BLE001
                        logger.error("callback_handle_error", extra={"extra": {"error": f"{type(e).__name__}: {e}"}})
                        continue

                # 2) Handle normal private messages for /start or opt-in
                message = upd.get("message") or upd.get("edited_message")
                if not message:
                    continue
                chat = message.get("chat", {})
                text = message.get("text")
                chat_type = chat.get("type")
                if chat_type == "private":
                    user_id = int(chat.get("id"))
                    # Respect explicit opt-out
                    if isinstance(text, str) and text.strip().lower().startswith("/stop"):
                        if subscriber_store.remove(user_id):
                            logger.info("subscriber_removed", extra={"extra": {"user_id": user_id}})
                    else:
                        # Any interaction with the bot counts as opt-in
                        added_before = subscriber_store.contains(user_id)
                        added_ok = subscriber_store.add(user_id)
                        if added_ok and (not added_before):
                            logger.info("subscriber_added", extra={"extra": {"user_id": user_id}})
        except Exception as e:  # noqa: BLE001
            logger.error("bot_poller_error", extra={"extra": {"error": f"{type(e).__name__}: {e}"}})
            await asyncio.sleep(1.0)


# ------------------------------
# Worker and Listener
# ------------------------------
MessageItem = Tuple[int, int, str]


async def worker(queue: "asyncio.Queue[Tuple[events.NewMessage.Event, str]]") -> None:
    while True:
        event, text = await queue.get()
        try:
            # Idempotency
            if dedup_store.seen(event.chat_id, event.id):  # type: ignore[arg-type]
                queue.task_done()
                continue

            # Collect reply context once; reuse for classification and notification
            context_plain, context_html = await collect_reply_context(event)

            # Rate limit before calling Cerebras (include context in token estimate)
            estimated_tokens = estimate_prompt_tokens(text)
            if context_plain:
                estimated_tokens += max(1, len(context_plain) // 4)
            await rate_limiter.acquire(estimated_tokens)

            t0 = time.time()
            clf_result = await classify_with_openai(text, context=context_plain)
            latency_ms = int((time.time() - t0) * 1000)

            label = clf_result.get("classification", "0")
            themes = clf_result.get("themes", "")

            logger.info(f"message: {text}\nlabel: {label}\nthemes: {themes}", extra={"plain": True})

            # Persist only selected messages (label == "1") to Supabase
            if label == "1":
                try:
                    author_user_id, author_reason = await resolve_author_user_id(event)
                    link_all = await build_message_link(event)
                    chat_obj = await event.get_chat()
                    chat_title = getattr(chat_obj, "title", None) or getattr(chat_obj, "username", None)
                    chat_username = getattr(chat_obj, "username", None)
                    chat_type = chat_obj.__class__.__name__ if chat_obj is not None else None
                    date_ts = None
                    try:
                        date_ts = int(getattr(event.message, "date", None).timestamp())  # type: ignore[union-attr]
                    except Exception:
                        date_ts = None

                    record = {
                        "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                        "message_id": int(event.id),
                        "author_user_id": int(author_user_id) if author_user_id is not None else None,
                        "author_reason": author_reason,
                        "text": text,
                        "label": label,
                        "themes": themes,
                        "link": link_all,
                        "context": context_plain,
                        "date_ts": date_ts,
                        "chat_title": chat_title,
                        "chat_username": chat_username,
                        "chat_type": chat_type,
                    }
                    # Offload blocking HTTP call to thread pool
                    await asyncio.to_thread(subscriber_store.save_classified_message, record)
                except Exception as e:  # noqa: BLE001
                    logger.warning("persist_message_failed", extra={"extra": {"error": str(e)}})

            if label == "1":
                # Reuse prepared metadata if available
                link = link_all if 'link_all' in locals() else await build_message_link(event)
                chat = chat_obj if 'chat_obj' in locals() else await event.get_chat()
                # Keep original author entity for display in the notification
                try:
                    from_user = await event.get_sender()
                except Exception:
                    from_user = None

                # Context already collected above; use as-is

                subscribers = subscriber_store.snapshot()
                if not subscribers:
                    logger.info(
                        "no_subscribers",
                        extra={
                            "extra": {
                                "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                "message_id": int(event.id),
                            }
                        },
                    )
                else:
                    sent = 0
                    total = len(subscribers)
                    for recipient_id in subscribers:
                        try:
                            await notifier_send(
                                event.client,
                                int(recipient_id),
                                from_user,
                                chat,
                                text,
                                link,
                                context_html=context_html,
                                context_plain=context_plain,
                                themes=themes,
                                classification_result=clf_result,
                            )
                            sent += 1
                        except Exception as e:  # noqa: BLE001
                            logger.error(
                                "notify_error",
                                extra={
                                    "extra": {
                                        "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                        "message_id": int(event.id),
                                        "recipient": int(recipient_id),
                                        "error": str(e),
                                    }
                                },
                            )
                    # delivery summary suppressed per user request

            dedup_store.mark(event.chat_id, event.id)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            logger.error(
                "worker_error",
                extra={
                    "extra": {
                        "chat_id": int(getattr(event, "chat_id", 0)),
                        "message_id": int(getattr(event, "id", 0)),
                        "error": str(e),
                    }
                },
            )
        finally:
            queue.task_done()


async def main() -> None:
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")
    # Validate LM Studio configuration
    if not LMSTUDIO_BASE_URL:
        raise RuntimeError("LMSTUDIO_BASE_URL is required")
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("bot_token_missing", extra={"extra": {"msg": "TELEGRAM_BOT_TOKEN is not set; notifications disabled"}})

    if TELEGRAM_STRING_SESSION:
        client = TelegramClient(StringSession(TELEGRAM_STRING_SESSION), api_id=int(TELEGRAM_API_ID), api_hash=TELEGRAM_API_HASH)
    else:
        client = TelegramClient(session=TELEGRAM_SESSION_PATH, api_id=int(TELEGRAM_API_ID), api_hash=TELEGRAM_API_HASH)

    queue: "asyncio.Queue[Tuple[events.NewMessage.Event, str]]" = asyncio.Queue(maxsize=1000)

    @client.on(events.NewMessage(chats=None))
    async def handler(event: events.NewMessage.Event) -> None:  # type: ignore[override]
        try:
            if event.is_private:
                return
            # Test-only chat filter
            if TEST_ONLY_CHAT_ID is not None and int(event.chat_id) != TEST_ONLY_CHAT_ID:  # type: ignore[arg-type]
                return
            if not event.message or not getattr(event.message, "message", None):
                return
            text = event.message.message
            if not text or not text.strip():
                return
            # Only enqueue if not seen
            if dedup_store.seen(event.chat_id, event.id):  # type: ignore[arg-type]
                return
            await queue.put((event, text))
        except Exception as e:  # noqa: BLE001
            logger.error("handler_error", extra={"extra": {"error": str(e)}})

    await client.start()

    # Start workers
    worker_tasks = [asyncio.create_task(worker(queue)) for _ in range(2)]
    bot_poller_task = asyncio.create_task(bot_updates_poller())

    logger.info(
        "started",
        extra={
            "extra": {
                "msg": "listener running",
                "test_only_chat_id": TEST_ONLY_CHAT_ID,
                "subscriber_count": subscriber_store.count(),
                "subscriber_offset": subscriber_store.get_offset(),
                "subscriber_backend": "supabase",
            }
        },
    )
    try:
        await client.run_until_disconnected()
    finally:
        for t in worker_tasks:
            t.cancel()
        bot_poller_task.cancel()
        await asyncio.gather(*worker_tasks, bot_poller_task, return_exceptions=True)
        # Close HTTP client
        global _http_client
        if _http_client is not None:
            await _http_client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass