import asyncio
from typing import Callable, Awaitable

from logging_config import logger


async def send_with_retries(coro_factory: Callable[[], Awaitable]):
    from tg.ui import COPY_TEXT_ALLOWED  # runtime import to avoid cycles
    attempt = 0
    delay = 0.5
    global_copy_allowed_name = "tg.ui.COPY_TEXT_ALLOWED"
    while True:
        try:
            result = coro_factory()
            if asyncio.iscoroutine(result):
                # Return the actual value produced by the awaited coroutine
                return await result
            # If the factory produced a non-coroutine value, return it as-is
            return result
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "copy_text_invalid" in msg:
                logger.warning("copy_text_disabled", extra={"extra": {"reason": msg, "flag": global_copy_allowed_name}})
                try:
                    import tg.ui as ui

                    ui.COPY_TEXT_ALLOWED = False
                except Exception:
                    pass
                return False
            attempt += 1
            if attempt > 3:
                logger.error("notify_failed", extra={"extra": {"error": str(e)}})
                return False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)


