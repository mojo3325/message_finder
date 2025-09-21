from typing import Optional, Tuple

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import SessionPasswordNeededError, PhoneCodeInvalidError, PhoneCodeExpiredError, FloodWaitError

from config import TELEGRAM_API_ID, TELEGRAM_API_HASH


def create_ephemeral_client() -> TelegramClient:
    return TelegramClient(StringSession(), api_id=int(TELEGRAM_API_ID), api_hash=TELEGRAM_API_HASH)


async def send_login_code(client: TelegramClient, phone: str) -> str:
    """Request a login code using Telethon defaults, like in test_teleton.py.

    This lets Telegram decide the best delivery (in-app/SMS/call) automatically.
    """
    sent = await client.send_code_request(phone)
    return getattr(sent, "phone_code_hash", "")


async def sign_in_with_code(client: TelegramClient, phone: str, code: str, phone_code_hash: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    try:
        await client.sign_in(phone=phone, code=code, phone_code_hash=phone_code_hash)
        return client.session.save(), None
    except SessionPasswordNeededError:
        return None, "need_2fa"
    except PhoneCodeInvalidError:
        return None, "code_invalid"
    except PhoneCodeExpiredError:
        return None, "code_expired"


async def sign_in_with_password(client: TelegramClient, password: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        await client.sign_in(password=password)
        return client.session.save(), None
    except Exception as e:  # noqa: BLE001
        # Treat any failure as generic
        return None, str(e)


def client_from_session(session_str: str) -> TelegramClient:
    return TelegramClient(StringSession(session_str), api_id=int(TELEGRAM_API_ID), api_hash=TELEGRAM_API_HASH)


def create_client_from_session(session_str: str) -> TelegramClient:
    """Backward-compatible alias for existing callers expecting the old name."""
    return client_from_session(session_str)

