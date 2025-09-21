from typing import Any, Optional

from logging_config import logger
from config import MESSAGE_FUCKERR_TOKEN
from services.clients import get_http_client
from utils.retry import send_with_retries
from tg.context import escape_html
from tg.ui_state import reply_ui_store
from tg import ui
from core.types import ClassificationResult


async def notifier_send(
    user_id: int,
    user: Optional[Any],
    chat: Any,
    text: str,
    link: Optional[str],
    *,
    context_html: Optional[str] = None,
    context_plain: Optional[str] = None,
    classification_result: Optional[ClassificationResult] = None,
    subscriber_store: Any,
    bot_token: Optional[str] = None,
) -> None:
    token = bot_token or MESSAGE_FUCKERR_TOKEN
    if not token:
        logger.warning("bot_token_missing", extra={"extra": {"msg": "MESSAGE_FUCKERR_TOKEN is not set"}})
        return

    full_name = " ".join(
        filter(None, [getattr(user, "first_name", None), getattr(user, "last_name", None)])
    ) or "отсутствует"
    username = f"@{getattr(user, 'username', '')}" if getattr(user, "username", None) else "отсутствует"
    raw_username = getattr(user, "username", None)
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
    if link:
        parts.append(f"• <b>Ссылка</b>: <a href=\"{escape_html(link)}\">перейти</a>")
    body = "\n".join(parts)

    try:
        author_user_id = int(getattr(user, "id", 0)) if user is not None else None
    except Exception:
        author_user_id = None

    try:
        author_access_hash = (
            int(getattr(user, "access_hash", 0)) if user is not None and getattr(user, "access_hash", None) is not None else None
        )
    except Exception:
        author_access_hash = None

    try:
        source_chat_id = int(getattr(chat, "id", 0) or 0)
    except Exception:
        source_chat_id = None

    sid = reply_ui_store.create(
        user_id=int(user_id),
        original_body_html=body,
        original_text=text,
        context_for_model=context_plain,
        classification_result=classification_result,
        author_user_id=author_user_id,
        source_chat_id=source_chat_id,
        author_username=str(raw_username) if raw_username else None,
        author_access_hash=author_access_hash,
    )
    keyboard = ui.build_feedback_keyboard(sid)

    http_client = get_http_client()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": int(user_id),
        "text": body,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": keyboard,
    }

    async def _post():
        resp = await http_client.post(url, json=payload)
        if resp.status_code >= 400:
            try:
                data = resp.json()
                desc = data.get("description", f"HTTP_{resp.status_code}")
            except Exception:  # noqa: BLE001
                desc = f"HTTP_{resp.status_code}:{resp.text[:200]}"
            if "button_copy_text_invalid" in desc.lower():
                raise RuntimeError("copy_text_invalid")

            if resp.status_code == 403:
                try:
                    lower = desc.lower()
                except Exception:
                    lower = "forbidden"
                removable_markers = [
                    "bot was blocked",
                    "user is deactivated",
                    "chat not found",
                    "can't initiate",
                    "forbidden",
                ]
                if any(marker in lower for marker in removable_markers):
                    try:
                        removed = subscriber_store.remove(int(user_id))
                        if removed:
                            logger.info(
                                "subscriber_removed_on_403",
                                extra={"extra": {"user_id": int(user_id), "reason": desc}},
                            )
                    except Exception:  # noqa: BLE001
                        pass

            raise RuntimeError(f"bot_send_error:{desc}")
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"bot_send_invalid_json:{e}")
        if not data.get("ok", False):
            desc = data.get("description", "unknown_error")
            if isinstance(desc, str) and "button_copy_text_invalid" in desc.lower():
                raise RuntimeError("copy_text_invalid")
            raise RuntimeError(f"bot_send_api_error:{desc}")
        return resp

    await send_with_retries(_post)
