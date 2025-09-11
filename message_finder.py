import asyncio
import html
import json
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple, List
import re

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import Channel, PeerUser, PeerChannel, PeerChat

import httpx

from cerebras.cloud.sdk import Cerebras
from const import DIALOGUE_PROMPT, CLASSIFIER_PROMPT


# ------------------------------
# Configuration
# ------------------------------
LOG_LEVEL = "INFO"
REQUEST_TIMEOUT_S = 20.0
RETRY_MAX_ATTEMPTS = 3

# Hardcoded credentials per user request (TEST USE ONLY)
TELEGRAM_API_ID = "28738574"
TELEGRAM_API_HASH = "343d6e8a20de4a2f3cc265eafdd64f71"
TELEGRAM_STRING_SESSION = "1AZWarzkBuzIOLASsWrjMkxeeZ5PaJoMtJSZSajB2lJEXQivilsJJHIPX6JQgSFfIVfi0dTf-LbaBHk_8N_kUyXWljBgsAPJVOL6qtTX1fgJAxTTNfkTZQq049Ad9PrlLwfU4AbNXgyYAfXV_tLobDQLALoTssGcqxXulW6b556iDc0xf7msg-QO8OIVzLI28ASxtXdbfTMrBOQ9gp3xaV5oZyp3XNCic9vtRYqPWxkCRBnlM4m8RNwaUZo86rYldDCiugbzRNZwrfqq9VYZtKQ2fTO5FFUKnMZADaBAePy7fsAKpee1IXraaBjMQeRUR6DM8iXAdqkV-ajEkyUPNmMS3IMomhOE="
TELEGRAM_SESSION_PATH = "session"

CEREBRAS_API_KEY = "csk-5njr2er53p5dwh6ymrfj4jjf9tctx9rpr3xtw4mc6te6dd38"

# Bot API for notifications
TELEGRAM_BOT_TOKEN = "8255044221:AAH_MKTuXbuWoLn0OlJRr6amaXMbdQ3jUlg"

# Optional: limit processing to a single chat during tests (disabled by default)
_TEST_ONLY_CHAT_ID_RAW = os.getenv("TEST_ONLY_CHAT_ID", "").strip()
TEST_ONLY_CHAT_ID: Optional[int] = int(_TEST_ONLY_CHAT_ID_RAW) if _TEST_ONLY_CHAT_ID_RAW else None

# Rate limits (Cerebras quotas)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
RATE_LIMIT_RPH = int(os.getenv("RATE_LIMIT_RPH", "900"))
RATE_LIMIT_TPM = int(os.getenv("RATE_LIMIT_TPM", "60000"))

# Persistent subscribers store path (mounted volume recommended)
SUBSCRIBER_STORE_PATH = os.getenv("SUBSCRIBER_STORE_PATH", "/data/subscribers.json")


# ------------------------------
# Logging
# ------------------------------
class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
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
class SubscriberStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._subscribers: Set[int] = set()
        self._last_update_id: int = 0
        self._load()

    def _load(self) -> None:
        try:
            if not self._path:
                return
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                subs = data.get("subscribers", []) or []
                self._subscribers = {int(x) for x in subs}
                self._last_update_id = int(data.get("last_update_id", 0) or 0)
        except Exception as e:  # noqa: BLE001
            logger.warning("subscriber_store_load_failed", extra={"extra": {"error": str(e), "path": self._path}})

    def _save(self) -> None:
        try:
            if not self._path:
                return
            dirpath = os.path.dirname(self._path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            tmp_path = f"{self._path}.tmp"
            payload = {
                "subscribers": sorted(self._subscribers),
                "last_update_id": self._last_update_id,
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except Exception as e:  # noqa: BLE001
            logger.warning("subscriber_store_save_failed", extra={"extra": {"error": str(e), "path": self._path}})

    def add(self, user_id: int) -> None:
        before = len(self._subscribers)
        self._subscribers.add(user_id)
        if len(self._subscribers) != before:
            self._save()

    def remove(self, user_id: int) -> None:
        if user_id in self._subscribers:
            self._subscribers.remove(user_id)
            self._save()

    def contains(self, user_id: int) -> bool:
        return user_id in self._subscribers

    def snapshot(self) -> Set[int]:
        return set(self._subscribers)

    def get_offset(self) -> int:
        return self._last_update_id

    def advance_offset(self, update_id: int) -> None:
        if update_id > self._last_update_id:
            self._last_update_id = update_id
            self._save()

    def count(self) -> int:
        return len(self._subscribers)


subscriber_store = SubscriberStore(SUBSCRIBER_STORE_PATH)


# ------------------------------
# UI State for Reply Generation
# ------------------------------
class ReplyUIState:
    def __init__(self, user_id: int, original_body_html: str, original_text: str, context_for_model: Optional[str] = None) -> None:
        self.user_id = int(user_id)
        self.original_body_html = original_body_html
        self.original_text = original_text
        self.context_for_model = context_for_model
        self.last_reply_text: Optional[str] = None


class ReplyUIStore:
    def __init__(self) -> None:
        self._seq = 0
        self._states: Dict[str, ReplyUIState] = {}

    def create(self, user_id: int, original_body_html: str, original_text: str, context_for_model: Optional[str] = None) -> str:
        self._seq += 1
        sid = f"s{self._seq}"
        self._states[sid] = ReplyUIState(user_id=user_id, original_body_html=original_body_html, original_text=original_text, context_for_model=context_for_model)
        return sid

    def get(self, sid: str) -> Optional[ReplyUIState]:
        return self._states.get(sid)

    def set_reply(self, sid: str, reply_text: str) -> None:
        st = self._states.get(sid)
        if st:
            st.last_reply_text = reply_text


reply_ui_store = ReplyUIStore()


# ------------------------------
# Cerebras Classifier
# ------------------------------

_cb_client: Optional[Cerebras] = None
_http_client: Optional[httpx.AsyncClient] = None
COPY_TEXT_ALLOWED: bool = True


def get_cerebras_client() -> Cerebras:
    global _cb_client
    if _cb_client is not None:
        return _cb_client

    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is required")

    # Pass API key explicitly and set retries/timeouts per docs.
    _cb_client = Cerebras(api_key=CEREBRAS_API_KEY, max_retries=RETRY_MAX_ATTEMPTS, timeout=REQUEST_TIMEOUT_S)
    return _cb_client


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is not None:
        return _http_client
    _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    return _http_client


def classify_with_cerebras_sync(message_text: str) -> str:
    if not message_text or not message_text.strip():
        return "0"

    client = get_cerebras_client()

    attempts = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": message_text.strip()},
                ],
                temperature=0.0,
                max_tokens=1,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw not in {"0", "1"}:
                # Invalid content per strict contract → retry limited times
                raise ValueError(f"invalid_model_output:{raw}")
            return raw
        except Exception as e:  # noqa: BLE001
            attempts += 1
            # Best-effort 429 handling
            should_retry = attempts <= RETRY_MAX_ATTEMPTS
            if not should_retry:
                logger.error("cerebras_call_failed", extra={"extra": {"error": str(e)}})
                return "0"
            time.sleep(min(2 ** attempts, 10))


async def classify_with_cerebras(message_text: str) -> str:
    return await asyncio.to_thread(classify_with_cerebras_sync, message_text)


# ------------------------------
# Cerebras Replier
# ------------------------------

def generate_reply_sync(message_text: str, context: Optional[str] = None) -> str:
    safe_text = (message_text or "").strip()
    if not safe_text:
        return "Можешь уточнить, что именно хочешь сделать? Я помогу."

    client = get_cerebras_client()
    attempts = 0
    while True:
        try:
            user_content_parts: List[str] = []
            if context and context.strip():
                user_content_parts.append(f"CONTEXT (may be truncated):\n{context.strip()}")
            user_content_parts.append(f"CURRENT_MESSAGE:\n{safe_text}")
            response = client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[
                    {"role": "system", "content": DIALOGUE_PROMPT},
                    {"role": "user", "content": "\n\n".join(user_content_parts)},
                ],
                temperature=0.7,
                max_tokens=48,
            )
            raw = (response.choices[0].message.content or "").strip()
            # Normalize to a single short sentence without fluff
            text = normalize_short_reply(raw)
            return text or "Давай уточним цель в одном предложении."
        except Exception as e:  # noqa: BLE001
            attempts += 1
            if attempts > RETRY_MAX_ATTEMPTS:
                logger.error("cerebras_reply_failed", extra={"extra": {"error": str(e)}})
                return "Секунду, пожалуйста. Кажется, сеть шалит — попробуем ещё раз?"
            time.sleep(min(2 ** attempts, 10))


async def generate_reply(message_text: str, context: Optional[str] = None) -> str:
    # Use same rate limiter since we consume Cerebras quota
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
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("bot_token_missing", extra={"extra": {"msg": "TELEGRAM_BOT_TOKEN is not set"}})
        return

    full_name = " ".join(filter(None, [getattr(user, "first_name", None), getattr(user, "last_name", None)])) or "отсутствует"
    username = f"@{getattr(user, 'username', '')}" if getattr(user, "username", None) else "отсутствует"
    chat_title = getattr(chat, "title", None) or getattr(chat, "username", None) or chat.__class__.__name__

    underlined = f"<u>{escape_html(text)}</u>"
    header = "<b>🔎 Обнаружен \"вопрос\"</b>"
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
    if link:
        parts.append(f"• <b>Ссылка</b>: <a href=\"{escape_html(link)}\">перейти</a>")
    body = "\n".join(parts)

    # Create UI state and initial inline keyboard with Generate button
    sid = reply_ui_store.create(
        user_id=int(user_id),
        original_body_html=body,
        original_text=text,
        context_for_model=context_plain,
    )
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "⚡ Сгенерировать ответ", "callback_data": f"gen:{sid}"}
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
            offset = subscriber_store.get_offset() + 1 if subscriber_store.get_offset() else None
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
                subscriber_store.advance_offset(upd_id)
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

                        if action == "back":
                            await bot_answer_callback_query(cb_id, text="Возвращаю…", show_alert=False)
                            if not st:
                                await bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
                                continue
                            orig_keyboard = {
                                "inline_keyboard": [[{"text": "⚡ Сгенерировать ответ", "callback_data": f"gen:{sid}"}]]
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
                        subscriber_store.remove(user_id)
                        logger.info("subscriber_removed", extra={"extra": {"user_id": user_id}})
                    else:
                        # Any interaction with the bot counts as opt-in
                        added_before = subscriber_store.contains(user_id)
                        subscriber_store.add(user_id)
                        if not added_before:
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

            # Rate limit before calling Cerebras
            await rate_limiter.acquire(estimate_prompt_tokens(text))

            t0 = time.time()
            label = await classify_with_cerebras(text)
            latency_ms = int((time.time() - t0) * 1000)

            logger.info(
                "classified",
                extra={
                    "extra": {
                        "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                        "message_id": int(event.id),
                        "latency_ms": latency_ms,
                        "label": label,
                    }
                },
            )

            if label == "1":
                link = await build_message_link(event)
                chat = await event.get_chat()
                # Keep original author entity for display in the notification
                try:
                    from_user = await event.get_sender()
                except Exception:
                    from_user = None

                # Collect reply context if this message is a reply to another message/post
                context_plain, context_html = await collect_reply_context(event)

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
                            )
                            sent += 1
                            logger.info(
                                "notified_subscriber",
                                extra={
                                    "extra": {
                                        "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                        "message_id": int(event.id),
                                        "recipient": int(recipient_id),
                                    }
                                },
                            )
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
                    logger.info(
                        "broadcast_done",
                        extra={
                            "extra": {
                                "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                "message_id": int(event.id),
                                "sent": sent,
                                "total_subscribers": total,
                            }
                        },
                    )

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
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is required")
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
                "subscriber_store_path": SUBSCRIBER_STORE_PATH,
                "subscriber_count": subscriber_store.count(),
                "subscriber_offset": subscriber_store.get_offset(),
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