import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from logging_config import logger
from config import ACCOUNTS_FILE


def _mask_phone_e164(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    s = phone.strip()
    if not s.startswith("+") or len(s) < 6:
        return "********"
    return s[:4] + "*" * max(4, len(s) - 6)


@dataclass
class Account:
    telegram_user_id: int
    phone_e164_masked: Optional[str]
    string_session: str
    status: str
    created_at: str
    updated_at: str
    meta: Dict[str, Any]


def _now_iso() -> str:
    try:
        return __import__("datetime").datetime.utcnow().isoformat() + "Z"
    except Exception:
        return str(int(time.time()))


def _ensure_file(path: str) -> None:
    base = os.path.dirname(path)
    if base and not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"users": []}, f)


def _load_all() -> Dict[str, Any]:
    _ensure_file(ACCOUNTS_FILE)
    try:
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {"users": []}
    except Exception as e:  # noqa: BLE001
        logger.warning("accounts_load_failed", extra={"extra": {"error": str(e)}})
        return {"users": []}


def _save_all(doc: Dict[str, Any]) -> None:
    try:
        tmp = ACCOUNTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ACCOUNTS_FILE)
    except Exception as e:  # noqa: BLE001
        logger.error("accounts_save_failed", extra={"extra": {"error": str(e)}})


def get_user_account(telegram_user_id: int) -> Optional[Account]:
    doc = _load_all()
    try:
        for it in doc.get("users", []) or []:
            if int(it.get("telegram_user_id", 0)) == int(telegram_user_id):
                return Account(
                    telegram_user_id=int(it.get("telegram_user_id")),
                    phone_e164_masked=it.get("phone_e164_masked"),
                    string_session=it.get("string_session", ""),
                    status=str(it.get("status", "active")),
                    created_at=str(it.get("created_at") or _now_iso()),
                    updated_at=str(it.get("updated_at") or _now_iso()),
                    meta=dict(it.get("meta") or {}),
                )
    except Exception:
        return None
    return None


def save_or_update_account(
    telegram_user_id: int,
    string_session_plain: str,
    phone_e164: Optional[str] = None,
    method: str = "phone",
) -> None:
    doc = _load_all()
    users = list(doc.get("users", []))
    now = _now_iso()
    payload = {
        "telegram_user_id": int(telegram_user_id),
        "phone_e164_masked": _mask_phone_e164(phone_e164),
        "string_session": string_session_plain,
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "meta": {"method": method, "last_ip": None},
    }
    found = False
    for idx, it in enumerate(users):
        if int(it.get("telegram_user_id", 0)) == int(telegram_user_id):
            users[idx] = {**it, **payload, "created_at": it.get("created_at", now)}
            found = True
            break
    if not found:
        users.append(payload)
    _save_all({"users": users})


def delete_account(telegram_user_id: int) -> bool:
    doc = _load_all()
    users = [it for it in doc.get("users", []) if int(it.get("telegram_user_id", 0)) != int(telegram_user_id)]
    if len(users) == len(doc.get("users", [])):
        return False
    _save_all({"users": users})
    return True


