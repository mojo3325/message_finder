from typing import Any, Dict, List, Optional
from urllib.parse import quote

from config import WARP_BOT_USERNAME
from tg.ui_state import reply_ui_store
from tg.context import escape_html


COPY_TEXT_ALLOWED: bool = True


def _is_copy_text_safe(text: str) -> bool:
    try:
        return len(text.encode("utf-8")) <= 256
    except Exception:
        return False


def sanitize_copy_text(text: str) -> Optional[str]:
    if not text:
        return None
    collapsed = " ".join(text.strip().split())
    try:
        data = collapsed.encode("utf-8")
    except Exception:
        return None
    if len(data) <= 256:
        return collapsed
    trimmed = data[:256]
    safe = trimmed.decode("utf-8", errors="ignore").rstrip()
    return safe or None


def build_reply_keyboard(sid: str, reply_text: Optional[str] = None) -> Dict[str, Any]:
    rows: List[List[Dict[str, Any]]] = [[
        {"text": "⬅ Назад", "callback_data": f"back:{sid}"},
        {"text": "🔁 Перегенерировать", "callback_data": f"regen:{sid}"},
        {"text": "👨🏻‍🦰", "callback_data": f"portrait:{sid}"},
    ]]
    if reply_text is not None and COPY_TEXT_ALLOWED:
        safe = sanitize_copy_text(reply_text)
        if safe and _is_copy_text_safe(safe):
            st = reply_ui_store.get(sid)
            author_username = getattr(st, "author_username", None) if st else None
            # Determine if user has a linked personal account
            has_linked_account = False
            try:
                user_id = getattr(st, "user_id", None) if st else None
                if user_id is not None:
                    from utilities.accounts_store import get_user_account  # lazy import to avoid cycles
                    has_linked_account = get_user_account(int(user_id)) is not None
            except Exception:
                has_linked_account = False

            # If the user has a linked account and we know the target, offer callback-based send
            if has_linked_account and st and (getattr(st, "author_user_id", None) is not None or author_username):
                rows.append([{ "text": "📨 Отправить", "callback_data": f"send:{sid}" }])
            else:
                if author_username:
                    # Use percent-encoding (spaces as %20, not '+')
                    url = f"https://t.me/{author_username}?text={quote(safe, safe='')}"
                    rows.append([{ "text": "📨 Отправить", "url": url }])
                else:
                    rows.append([{ "text": "📋 Скопировать", "copy_text": {"text": safe} }])
    return {"inline_keyboard": rows}


def _warp_bot_username() -> str:
    return WARP_BOT_USERNAME.lstrip("@") or "warp_chat_bot"


def build_warp_chats_list(
    title: str,
    items: List[Dict[str, Any]],
    page: int,
    total_pages: int,
    *,
    session_available: bool = True,
) -> tuple[str, Dict[str, Any]]:
    safe_title = escape_html(title).strip() or "Мои приватные чаты"
    lines: List[str] = [f"📒 <b>{safe_title}</b>"]
    lines.append(f"Страница {max(page, 1)}/{max(total_pages, 1)}")
    if not session_available:
        lines.append("")
        lines.append("Подключите аккаунт, чтобы просматривать приватные диалоги.")
    lines.append("")

    has_items = False
    bot_username = _warp_bot_username()
    if session_available:
        for it in items:
            try:
                cid = int(it.get("chat_id"))
            except Exception:
                continue
            raw_name = str(it.get("title") or cid).strip()
            preview = str(it.get("preview") or "").strip()
            time_str = str(it.get("time") or "").strip()

            deep_link = f"https://t.me/{bot_username}?start=open_{cid}"
            lines.append(
                f"• {escape_html(raw_name)} — <a href=\"{deep_link}\">Открыть чат</a>"
            )
            if preview:
                lines.append(f"  ⤷ {escape_html(preview)}")
            if time_str:
                lines.append(f"  🕓 {escape_html(time_str)}")
            lines.append("")
            has_items = True
    if session_available and not has_items:
        lines.append("Пока нет приватных диалогов.")

    body = "\n".join(line for line in lines if line is not None).rstrip()

    keyboard_rows: List[List[Dict[str, Any]]] = []
    if session_available and total_pages > 1:
        nav_row: List[Dict[str, Any]] = []
        if page > 1:
            nav_row.append({"text": "⬅ Prev", "callback_data": f"list:{page-1}"})
        nav_row.append({"text": f"{page}/{total_pages}", "callback_data": "noop:list"})
        if page < total_pages:
            nav_row.append({"text": "Next ➡", "callback_data": f"list:{page+1}"})
        keyboard_rows.append(nav_row)

    if not session_available:
        keyboard_rows.append([
            {"text": "Привязать аккаунт", "callback_data": "acc_link"}
        ])

    return body, {"inline_keyboard": keyboard_rows}


def build_warp_miniature(
    title: str,
    time_str: Optional[str],
    messages: List[Dict[str, Any]],
    chat_id: int,
    *,
    loading: bool = False,
    generated_reply: Optional[str] = None,
    draft_id: Optional[str] = None,
    session_available: bool = True,
) -> tuple[str, Dict[str, Any]]:
    if loading:
        generate_text = "Перегенерировать" if draft_id else "Сгенерировать"
        kb_rows: List[List[Dict[str, Any]]] = []
        if session_available:
            kb_rows.append([
                {"text": f"{generate_text}…", "callback_data": "noop:gen"},
                {"text": "Назад", "callback_data": "noop:back"},
                {"text": "Отправить", "callback_data": "noop:send"},
            ])
        else:
            kb_rows.append([
                {"text": "Назад", "callback_data": "back:chats"},
                {"text": "Привязать аккаунт", "callback_data": "acc_link"},
            ])
        placeholder_lines = [
            "<b>Генерирую… ⣾</b>",
            "<b>━━━━━━━━━━━━━━━━━━━━</b>",
            "<code>█░░░░░░░░░</code>",
        ]
        return "\n".join(placeholder_lines), {"inline_keyboard": kb_rows}

    safe_title = escape_html(title)
    safe_time = escape_html(time_str) if time_str else ""

    header = f"📩 Чат с {safe_title}"
    if safe_time:
        header = f"{header} · {safe_time}"

    top = "┌─ " + header + " " + "─" * 33
    bottom = "└" + "─" * (len(top) - 1)

    body_lines: List[str] = [top, ""]
    for m in messages:
        direction = str(m.get("direction") or "").lower()
        author = escape_html(str(m.get("author") or ""))
        text = escape_html(str(m.get("text") or ""))
        if not text:
            continue
        if direction == "out":
            body_lines.append(f"→ <b>{author}:</b>")
        else:
            body_lines.append(f"← {author}:")
        body_lines.append("   " + text)
        body_lines.append("")

    if generated_reply:
        body_lines.append(f"→ <b>Сгенерированный ответ:</b>")
        body_lines.append(f"   <i><u>{escape_html(generated_reply)}</u></i>")
        body_lines.append("")

    if body_lines and body_lines[-1] == "":
        body_lines.pop()
    body_lines.append("")
    body_lines.append(bottom)

    kb_rows: List[List[Dict[str, Any]]] = []
    if session_available:
        generate_text = "Перегенерировать" if draft_id else "Сгенерировать"
        kb_rows.append([
            {"text": generate_text, "callback_data": (f"regen:{draft_id}" if draft_id else f"gen:{int(chat_id)}")},
            {"text": "Назад", "callback_data": "back:chats"},
            {
                "text": "Отправить",
                "callback_data": (f"send:{draft_id}" if draft_id else "send_disabled"),
            },
        ])
    else:
        kb_rows.append([
            {"text": "Назад", "callback_data": "back:chats"},
            {"text": "Привязать аккаунт", "callback_data": "acc_link"},
        ])

    return "\n".join(body_lines), {"inline_keyboard": kb_rows}


# ------------------------------
# Account onboarding UI helpers
# ------------------------------

def build_account_start_keyboard() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "📱 Телефон", "callback_data": "acc_phone"},
                {"text": "🔳 QR", "callback_data": "acc_qr"},
            ]
        ]
    }


def build_account_confirm_keyboard(url: Optional[str]) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [{"text": "Подтвердить вход", "url": url or "tg://settings"}]
        ]
    }


def build_request_contact_keyboard() -> Dict[str, Any]:
    # ReplyKeyboardMarkup (not inline) to request user's phone number
    # It will be shown once and then hidden after use
    return {
        "keyboard": [[{"text": "📲 Поделиться номером", "request_contact": True}]],
        "resize_keyboard": True,
        "one_time_keyboard": True,
        "input_field_placeholder": "Нажмите, чтобы отправить номер или введите его",
    }

def hide_reply_keyboard() -> Dict[str, Any]:
    return {"remove_keyboard": True}

def build_feedback_keyboard(sid: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "👎", "callback_data": f"dislike:{sid}"},
                {"text": "✨", "callback_data": f"gen:{sid}"},
                {"text": "👨🏻‍🦰", "callback_data": f"portrait:{sid}"},
            ]
        ]
    }

