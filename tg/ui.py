from typing import Any, Dict, List, Optional


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
            rows.append([{ "text": "📋 Скопировать", "copy_text": {"text": safe} }])
    return {"inline_keyboard": rows}


