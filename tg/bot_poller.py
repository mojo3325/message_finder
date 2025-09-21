import asyncio
from typing import Any, Dict, Callable

import httpx

from config import WARP_CHAT_BOT_TOKEN
from const import CMD_ACCOUNT
from logging_config import logger
from services.clients import get_http_client
from tg import bot_api
from tg.ui_state import reply_ui_store
from tg.handlers import account_flow, warp, portrait
from tg.handlers.types import CallbackContext, MessageContext
from utilities.accounts_store import get_user_account

async def bot_updates_poller(
    subscriber_store: Any,
    get_telethon_client: Callable[[], Any],
) -> None:
    bot_token = bot_api.get_bot_token()
    if not bot_token:
        logger.warning(
            "bot_token_missing",
            extra={"extra": {"msg": "TELEGRAM_BOT_TOKEN is not set; poller disabled"}},
        )
        return
    is_warp_chat_mode = bool(bot_token == WARP_CHAT_BOT_TOKEN)

    http_client = get_http_client()
    base = f"https://api.telegram.org/bot{bot_token}"
    timeout_s = 50
    try:
        await http_client.get(f"{base}/deleteWebhook", params={"drop_pending_updates": False})
    except httpx.HTTPError as exc:
        logger.warning(
            "delete_webhook_failed",
            extra={"extra": {"error": f"{type(exc).__name__}: {exc}"}},
        )

    # Helpers: dialogs fetching and caching
    while True:
        try:
            _offset_val = subscriber_store.get_offset()
            offset = (_offset_val + 1) if _offset_val else None
            params: Dict[str, Any] = {"timeout": timeout_s}
            if offset is not None:
                params["offset"] = offset
            try:
                resp = await http_client.get(
                    f"{base}/getUpdates",
                    params=params,
                    timeout=timeout_s + 10,
                )
            except httpx.HTTPError as exc:
                logger.error(
                    "bot_get_updates_http_error",
                    extra={"extra": {"error": f"{type(exc).__name__}: {exc}"}},
                )
                await asyncio.sleep(1.0)
                continue

            try:
                data = resp.json()
            except ValueError as exc:
                logger.error(
                    "bot_get_updates_json_error",
                    extra={"extra": {"error": f"{type(exc).__name__}: {exc}"}},
                )
                await asyncio.sleep(1.0)
                continue

            if not data.get("ok", False):
                await asyncio.sleep(1.0)
                continue
            for upd in data.get("result", []) or []:
                upd_id = int(upd.get("update_id", 0))
                try:
                    subscriber_store.advance_offset(upd_id)
                except Exception:
                    pass

                callback = upd.get("callback_query")
                if callback:
                    try:
                        cb_id = callback.get("id")
                        from_user = callback.get("from", {})
                        from_user_id = int(from_user.get("id")) if from_user and from_user.get("id") is not None else None
                        msg = callback.get("message") or {}
                        msg_chat = msg.get("chat") or {}
                        msg_chat_id = int(msg_chat.get("id")) if msg_chat and msg_chat.get("id") is not None else None
                        msg_id = int(msg.get("message_id")) if msg and msg.get("message_id") is not None else None
                        data_s = callback.get("data") or ""

                        if not data_s or not cb_id or msg_chat_id is None or msg_id is None:
                            continue

                        if ":" in data_s:
                            action, sid = data_s.split(":", 1)
                        else:
                            action, sid = data_s, ""

                        st = reply_ui_store.get(sid) if sid else None

                        callback_ctx = CallbackContext(
                            callback_id=str(cb_id),
                            user_id=int(from_user_id) if from_user_id is not None else None,
                            chat_id=int(msg_chat_id) if msg_chat_id is not None else None,
                            message_id=int(msg_id) if msg_id is not None else None,
                            data=str(data_s),
                            action=str(action),
                            sid=str(sid),
                            raw=callback,
                            message=msg,
                        )
                        if await account_flow.handle_callback(callback_ctx):
                            continue

                        if await warp.handle_callback(
                            callback_ctx,
                            get_telethon_client=get_telethon_client,
                        ):
                            continue

                        if await portrait.handle_callback(
                            callback_ctx,
                            get_telethon_client=get_telethon_client,
                        ):
                            continue

                        await bot_api.bot_answer_callback_query(cb_id, text="Неизвестное действие", show_alert=False)
                        continue
                    except Exception as e:  # noqa: BLE001
                        logger.error("callback_handle_error", extra={"extra": {"error": f"{type(e).__name__}: {e}"}})
                        continue

                message = upd.get("message") or upd.get("edited_message")
                if not message:
                    continue
                chat = message.get("chat", {})
                text = message.get("text")
                chat_type = chat.get("type")
                if chat_type == "private":
                    user_id = int(chat.get("id"))
                    text_s = (text or "").strip()
                    lower = text_s.lower()
                    contact = message.get("contact") or None

                    start_payload = None
                    if text_s.startswith("/start"):
                        parts = text_s.split(maxsplit=1)
                        if len(parts) == 2:
                            start_payload = parts[1].strip() or None

                    message_ctx = MessageContext(
                        user_id=user_id,
                        chat_id=user_id,
                        message_id=int(message.get("message_id")) if message.get("message_id") is not None else 0,
                        text=text_s,
                        entities=message.get("entities") or [],
                        raw=message,
                        start_payload=start_payload,
                        contact=contact,
                    )

                    if is_warp_chat_mode and lower == CMD_ACCOUNT:
                        # Warp Chat: list private dialogs when already linked, otherwise start onboarding
                        acc = get_user_account(user_id)
                        if acc:
                            list_ctx = MessageContext(
                                user_id=user_id,
                                chat_id=message_ctx.chat_id,
                                message_id=message_ctx.message_id,
                                text="/list_1",
                                entities=message_ctx.entities,
                                raw=message_ctx.raw,
                                start_payload=message_ctx.start_payload,
                                contact=message_ctx.contact,
                            )
                            list_handled = await warp.handle_command(
                                list_ctx,
                                get_telethon_client=get_telethon_client,
                            )
                            if list_handled:
                                continue
                            if await account_flow.handle_message(message_ctx):
                                continue
                        else:
                            if await account_flow.handle_message(message_ctx):
                                continue

                    if await account_flow.handle_message(message_ctx):
                        continue

                    # Forwarded-message handling removed

                    # Account onboarding is handled in tg.handlers.account_flow

# /stop subscription control
                    if lower.startswith("/stop"):
                        if subscriber_store.remove(user_id):
                            logger.info("subscriber_removed", extra={"extra": {"user_id": user_id}})
                        continue

                    # Subscription control is explicit: only /start adds, /stop removes
                    if lower.startswith("/start"):
                        # Always ensure subscription
                        added_before = subscriber_store.contains(user_id)
                        added_ok = subscriber_store.add(user_id)
                        if added_ok and (not added_before):
                            logger.info("subscriber_added", extra={"extra": {"user_id": user_id}})

                        # Deep-link payload handling: delegate to warp handler
                        if await warp.handle_start_payload(message_ctx, get_telethon_client=get_telethon_client):
                            continue

                    # Handle /unlink
                    # Handle /account status/help
                    # Handle /account: if linked, list private chats; otherwise start linking flow

                    # Fallback commands: delegate to warp handler
                    if await warp.handle_command(message_ctx, get_telethon_client=get_telethon_client):
                        continue

        except Exception as e:  # noqa: BLE001
            logger.error("bot_poller_error", extra={"extra": {"error": f"{type(e).__name__}: {e}"}})
            await asyncio.sleep(1.0)
