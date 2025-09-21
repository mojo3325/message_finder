import json
import os
from typing import Any, Dict, List, Optional

import filelock


CHATS_FILE_PATH = "data/user_chats.json"
LOCK_FILE_PATH = "data/user_chats.json.lock"


def _ensure_files() -> None:
    base = os.path.dirname(CHATS_FILE_PATH)
    if base and not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
    if not os.path.exists(CHATS_FILE_PATH):
        with open(CHATS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load_all() -> Dict[str, Any]:
    _ensure_files()
    try:
        with open(CHATS_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                return {}
            return json.loads(content)
    except Exception:
        return {}


def _save_all(doc: Dict[str, Any]) -> None:
    tmp = CHATS_FILE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHATS_FILE_PATH)


def add_chat(
    telegram_user_id: int,
    chat_id: int,
    title: Optional[str],
    chat_type: str,
    is_active: bool = True,
) -> bool:
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        doc = _load_all()
        key = str(int(telegram_user_id))
        rows: List[Dict[str, Any]] = list(doc.get(key, []))

        # dedup by chat_id
        found = False
        for i, it in enumerate(rows):
            try:
                if int(it.get("chat_id")) == int(chat_id):
                    rows[i] = {
                        **it,
                        "title": title if title is not None else it.get("title"),
                        "type": chat_type or it.get("type"),
                        "is_active": bool(is_active),
                    }
                    found = True
                    break
            except Exception:
                continue

        if not found:
            now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
            rows.append(
                {
                    "chat_id": int(chat_id),
                    "title": title,
                    "type": chat_type,
                    "added_at": now,
                    "is_active": bool(is_active),
                }
            )

        doc[key] = rows
        _save_all(doc)
        return not found


def get_user_chats(telegram_user_id: int) -> List[Dict[str, Any]]:
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        doc = _load_all()
        return list(doc.get(str(int(telegram_user_id)), []))


def chat_exists(telegram_user_id: int, chat_id: int) -> bool:
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        doc = _load_all()
        for it in doc.get(str(int(telegram_user_id)), []):
            try:
                if int(it.get("chat_id")) == int(chat_id):
                    return True
            except Exception:
                continue
        return False


def remove_chat(telegram_user_id: int, chat_id: int) -> bool:
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        doc = _load_all()
        key = str(int(telegram_user_id))
        rows = [it for it in doc.get(key, []) if int(it.get("chat_id", -1)) != int(chat_id)]
        if len(rows) == len(doc.get(key, [])):
            return False
        doc[key] = rows
        _save_all(doc)
        return True


