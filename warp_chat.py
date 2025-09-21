import asyncio
import os
from typing import Optional

# Ensure token is set BEFORE importing modules that build Bot API URLs
from config import WARP_CHAT_BOT_TOKEN
if not os.getenv("TELEGRAM_BOT_TOKEN") and WARP_CHAT_BOT_TOKEN:
    os.environ["TELEGRAM_BOT_TOKEN"] = WARP_CHAT_BOT_TOKEN

from logging_config import logger
from utilities.subscribers_store import SupabaseSubscriberStore
from tg.bot_poller import bot_updates_poller
from services.clients import close_http_client
from config import SUPABASE_URL, SUPABASE_ANON_KEY


def get_telethon_client() -> Optional[object]:
    # Not needed for basic /account and /unlink flows in initial setup
    return None


async def main() -> None:
    # Ensure correct bot token is used for this process
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = WARP_CHAT_BOT_TOKEN
        except Exception:
            pass
    logger.info(
        "warp_chat_starting",
        extra={
            "extra": {
                "tele_token_set": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
                "offset_key": "warp_chat_last_update_id",
                "supabase_url": SUPABASE_URL,
            }
        },
    )
    store = SupabaseSubscriberStore(SUPABASE_URL, SUPABASE_ANON_KEY, offset_key="warp_chat_last_update_id")
    try:
        await bot_updates_poller(store, get_telethon_client)
    finally:
        try:
            await close_http_client()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("warp_chat_stopped")

