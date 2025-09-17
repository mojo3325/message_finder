from typing import Any, Dict, Optional

from services.clients import get_http_client
from config import TELEGRAM_BOT_TOKEN


async def bot_edit_message_text(chat_id: int, message_id: int, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"

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

    from utils.retry import send_with_retries
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


async def bot_delete_message(chat_id: int, message_id: int) -> bool:
    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage"

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


