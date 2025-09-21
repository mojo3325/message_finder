import asyncio
import time
from typing import Any, Tuple, Dict

from telethon import events

from logging_config import logger
from core.dedup import DedupStore
from core.rate_limiter import rate_limiter, estimate_prompt_tokens
import services.classifier as classifier_service
from tg import context as tg_context
from tg.notifier import notifier_send


MessageItem = Tuple[int, int, str]


async def worker(
    queue: "asyncio.Queue[Tuple[events.NewMessage.Event, str]]",
    dedup_store: DedupStore,
    subscriber_store: Any,
) -> None:
    while True:
        event, text = await queue.get()
        try:
            # Deduplication is handled at enqueue time in the handler

            context_plain, context_html = await tg_context.collect_reply_context(event)

            estimated_tokens = estimate_prompt_tokens(text)
            if context_plain:
                estimated_tokens += max(1, len(context_plain) // 4)
            await rate_limiter.acquire(estimated_tokens)

            t0 = time.time()
            clf_result = await classifier_service.classify_with_openai(text, context=context_plain)
            latency_ms = int((time.time() - t0) * 1000)

            label = str(clf_result)

            logger.info(f"message: {text}\nlabel: {label}", extra={"plain": True})

            if label == "1":
                try:
                    author_user_id, author_reason = await tg_context.resolve_author_user_id(event)
                    link_all = await tg_context.build_message_link(event)
                    chat_obj = await event.get_chat()
                    chat_title = getattr(chat_obj, "title", None) or getattr(chat_obj, "username", None)
                    chat_username = getattr(chat_obj, "username", None)
                    chat_type = chat_obj.__class__.__name__ if chat_obj is not None else None
                    date_ts = None
                    try:
                        date_ts = int(getattr(event.message, "date", None).timestamp())  # type: ignore[union-attr]
                    except Exception:
                        date_ts = None

                    record: Dict[str, Any] = {
                        "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                        "message_id": int(event.id),
                        "author_user_id": int(author_user_id) if author_user_id is not None else None,
                        "author_reason": author_reason,
                        "text": text,
                        "label": label,
                        "link": link_all,
                        "context": context_plain,
                        "date_ts": date_ts,
                        "chat_title": chat_title,
                        "chat_username": chat_username,
                        "chat_type": chat_type,
                    }
                    await asyncio.to_thread(subscriber_store.save_classified_message, record)
                except Exception as e:  # noqa: BLE001
                    logger.warning("persist_message_failed", extra={"extra": {"error": str(e)}})

            if label == "1":
                link = link_all if 'link_all' in locals() else await tg_context.build_message_link(event)
                chat = chat_obj if 'chat_obj' in locals() else await event.get_chat()
                try:
                    from_user = await event.get_sender()
                except Exception:
                    from_user = None

                subscribers = subscriber_store.snapshot()
                if not subscribers:
                    logger.info(
                        "no_subscribers",
                        extra={
                            "extra": {
                                "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                "message_id": int(event.id),
                            }
                        },
                    )
                else:
                    for recipient_id in subscribers:
                        try:
                            await notifier_send(
                                int(recipient_id),
                                from_user,
                                chat,
                                text,
                                link,
                                context_html=context_html,
                                context_plain=context_plain,
                                classification_result=clf_result,
                                subscriber_store=subscriber_store,
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.error(
                                "notify_error",
                                extra={
                                    "extra": {
                                        "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                        "message_id": int(event.id),
                                        "recipient": int(recipient_id),
                                        "error": str(e),
                                    }
                                },
                            )

            # Already marked in handler to prevent duplicate enqueues
        except Exception as e:  # noqa: BLE001
            logger.error(
                "worker_error",
                extra={
                    "extra": {
                        "chat_id": int(getattr(event, "chat_id", 0)),
                        "message_id": int(getattr(event, "id", 0)),
                        "error": str(e),
                    }
                },
            )
        finally:
            queue.task_done()


