import asyncio
from typing import Optional, Tuple

from telethon import TelegramClient, events
from telethon.sessions import StringSession


from config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_STRING_SESSION,
    TELEGRAM_SESSION_PATH,
    LMSTUDIO_BASE_URL,
    MESSAGE_FUCKERR_TOKEN,
    TEST_ONLY_CHAT_ID,
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
)


# ------------------------------
# Logging
# ------------------------------
from logging_config import logger


from core.dedup import DedupStore
dedup_store = DedupStore()


# ------------------------------
# Subscriber Store (Bot API /start)
# ------------------------------
from utilities.subscribers_store import SupabaseSubscriberStore

subscriber_store = SupabaseSubscriberStore(SUPABASE_URL, SUPABASE_ANON_KEY)


from tg.worker import worker
from tg.bot_poller import bot_updates_poller

# Keep a reference to the Telethon client for cross-task operations (e.g., history fetch in bot poller)
_telethon_client: Optional[TelegramClient] = None


def get_telethon_client() -> Optional[TelegramClient]:
    return _telethon_client


# ------------------------------
# Client utilities moved to services.clients
# ------------------------------


from services.feedback import save_feedback
from core.types import ClassificationResult


async def main() -> None:
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")
    # Validate LM Studio configuration
    if not LMSTUDIO_BASE_URL:
        raise RuntimeError("LMSTUDIO_BASE_URL is required")
    if not MESSAGE_FUCKERR_TOKEN:
        logger.warning("bot_token_missing", extra={"extra": {"msg": "MESSAGE_FUCKERR_TOKEN is not set; notifications disabled"}})

    if TELEGRAM_STRING_SESSION:
        client = TelegramClient(StringSession(TELEGRAM_STRING_SESSION), api_id=int(TELEGRAM_API_ID), api_hash=TELEGRAM_API_HASH)
    else:
        client = TelegramClient(session=TELEGRAM_SESSION_PATH, api_id=int(TELEGRAM_API_ID), api_hash=TELEGRAM_API_HASH)

    # Expose client reference for fallback history fetch
    global _telethon_client
    _telethon_client = client

    queue: "asyncio.Queue[Tuple[events.NewMessage.Event, str]]" = asyncio.Queue(maxsize=1000)

    @client.on(events.NewMessage(chats=None))
    async def handler(event: events.NewMessage.Event) -> None:  # type: ignore[override]
        try:
            if event.is_private:
                return
            # Only process group/supergroup messages; skip broadcast channels
            if not event.is_group:
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
            # Mark immediately to avoid duplicate enqueues from rapid successive updates (e.g., edits)
            dedup_store.mark(event.chat_id, event.id)  # type: ignore[arg-type]
            await queue.put((event, text))
        except Exception as e:  # noqa: BLE001
            logger.error("handler_error", extra={"extra": {"error": str(e)}})

    await client.start()

    # Start workers
    worker_tasks = [asyncio.create_task(worker(queue, dedup_store, subscriber_store)) for _ in range(2)]
    bot_poller_task = asyncio.create_task(bot_updates_poller(subscriber_store, get_telethon_client))

    logger.info(
        "started",
        extra={
            "extra": {
                "msg": "listener running",
                "test_only_chat_id": TEST_ONLY_CHAT_ID,
                "subscriber_count": subscriber_store.count(),
                "subscriber_offset": subscriber_store.get_offset(),
                "subscriber_backend": "supabase",
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
        # Close shared HTTP client via services.clients
        from services.clients import close_http_client
        await close_http_client()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass