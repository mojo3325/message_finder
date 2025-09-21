from typing import Any, Dict, Optional

from services.clients import get_http_client
from config import MESSAGE_FUCKERR_TOKEN
import os


def _get_bot_token() -> str:
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    if tok and tok.strip():
        return tok
    return MESSAGE_FUCKERR_TOKEN


def get_bot_token() -> str:
    """Return the configured Telegram bot token."""
    return _get_bot_token()


async def bot_edit_message_text(chat_id: int, message_id: int, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/editMessageText"

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
        if resp.status_code >= 400:
            try:
                data = resp.json()
                desc = data.get("description", f"HTTP_{resp.status_code}")
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            if "message is not modified" in desc.lower():
                return resp
            if "button_copy_text_invalid" in desc.lower():
                raise RuntimeError("copy_text_invalid")
            raise RuntimeError(f"bot_edit_error:{desc}")

        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            if "message is not modified" in desc.lower():
                return resp
            raise RuntimeError(f"bot_edit_api_error:{desc}")
        return resp

    from utils.retry import send_with_retries
    return await send_with_retries(_post)


async def bot_edit_message_reply_markup(chat_id: int, message_id: int, reply_markup: Dict[str, Any]) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/editMessageReplyMarkup"

    payload: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "reply_markup": reply_markup,
    }

    async def _post():
        resp = await http_client.post(url, json=payload)
        if resp.status_code >= 400:
            try:
                data = resp.json()
                desc = data.get("description", f"HTTP_{resp.status_code}")
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            raise RuntimeError(f"bot_edit_markup_error:{desc}")

        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_edit_markup_api_error:{desc}")
        return resp

    from utils.retry import send_with_retries
    return await send_with_retries(_post)


async def bot_answer_callback_query(callback_query_id: str, text: Optional[str] = None, show_alert: bool = False) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/answerCallbackQuery"
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

    from utils.retry import send_with_retries
    return await send_with_retries(_post)


async def bot_send_html_message(chat_id: int, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/sendMessage"

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
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            if "button_copy_text_invalid" in desc.lower():
                raise RuntimeError("copy_text_invalid")
            raise RuntimeError(f"bot_send_error:{desc}")
        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_send_api_error:{desc}")
        return resp

    from utils.retry import send_with_retries
    return await send_with_retries(_post)


async def bot_send_html_message_with_id(chat_id: int, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> Optional[int]:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/sendMessage"

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
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            if "button_copy_text_invalid" in desc.lower():
                raise RuntimeError("copy_text_invalid")
            raise RuntimeError(f"bot_send_error:{desc}")
        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_send_api_error:{desc}")
        return data

    from utils.retry import send_with_retries
    try:
        data = await send_with_retries(_post)
    except Exception:
        return None
    try:
        result = data.get("result") or {}
        return int(result.get("message_id")) if result.get("message_id") is not None else None
    except Exception:
        return None

async def bot_delete_message(chat_id: int, message_id: int) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/deleteMessage"

    payload: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "message_id": int(message_id),
    }

    async def _post():
        resp = await http_client.post(url, json=payload)
        if resp.status_code >= 400:
            try:
                data = resp.json()
                desc = data.get("description", f"HTTP_{resp.status_code}")
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            raise RuntimeError(f"bot_delete_error:{desc}")
        data = resp.json()
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            raise RuntimeError(f"bot_delete_api_error:{desc}")
        return resp

    from utils.retry import send_with_retries
    return await send_with_retries(_post)


# ------------------------------
# Media helpers
# ------------------------------

async def bot_send_photo(chat_id: int, photo_bytes: bytes, caption_html: Optional[str] = None) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/sendPhoto"

    data: Dict[str, Any] = {"chat_id": int(chat_id)}
    if caption_html:
        # Telegram caption limit ~1024
        cap = caption_html if len(caption_html) <= 1000 else caption_html[:990] + "…"
        data["caption"] = cap
        data["parse_mode"] = "HTML"

    files = {"photo": ("qr.png", photo_bytes, "image/png")}

    async def _post():
        resp = await http_client.post(url, data=data, files=files)
        if resp.status_code >= 400:
            try:
                jd = resp.json()
                desc = jd.get("description", f"HTTP_{resp.status_code}")
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            raise RuntimeError(f"bot_send_photo_error:{desc}")
        jd = resp.json()
        if not jd.get("ok", False):
            desc = jd.get("description", "unknown_error")
            raise RuntimeError(f"bot_send_photo_api_error:{desc}")
        return resp

    from utils.retry import send_with_retries
    return await send_with_retries(_post)


async def bot_send_photo_with_id(chat_id: int, photo_bytes: bytes, caption_html: Optional[str] = None) -> Optional[int]:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{_get_bot_token()}/sendPhoto"

    data: Dict[str, Any] = {"chat_id": int(chat_id)}
    if caption_html:
        cap = caption_html if len(caption_html) <= 1000 else caption_html[:990] + "…"
        data["caption"] = cap
        data["parse_mode"] = "HTML"

    files = {"photo": ("qr.png", photo_bytes, "image/png")}

    async def _post():
        resp = await http_client.post(url, data=data, files=files)
        if resp.status_code >= 400:
            try:
                jd = resp.json()
                desc = jd.get("description", f"HTTP_{resp.status_code}")
            except Exception:
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            raise RuntimeError(f"bot_send_photo_error:{desc}")
        return resp.json()

    from utils.retry import send_with_retries
    try:
        jd = await send_with_retries(_post)
    except Exception:
        return None
    try:
        result = jd.get("result") or {}
        return int(result.get("message_id")) if result.get("message_id") is not None else None
    except Exception:
        return None

