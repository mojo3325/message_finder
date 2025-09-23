import asyncio
import time
from typing import Any, Dict, List, Tuple

from telethon import events

from logging_config import logger
from core.dedup import DedupStore
from core.rate_limiter import rate_limiter
import services.classifier as classifier_service
from config import MISTRAL_BATCH_FLUSH_MS, MISTRAL_BATCH_MAX_SIZE
from tg import context as tg_context
from tg.notifier import notifier_send


MessageItem = Tuple[int, int, str]


async def worker(
    queue: "asyncio.Queue[Tuple[events.NewMessage.Event, str]]",
    dedup_store: DedupStore,
    subscriber_store: Any,
) -> None:
    while True:
        first_event, first_text = await queue.get()
        batch_items: List[Tuple[events.NewMessage.Event, str]] = [(first_event, first_text)]
        flush_deadline = time.monotonic() + (MISTRAL_BATCH_FLUSH_MS / 1000.0)

        while len(batch_items) < MISTRAL_BATCH_MAX_SIZE:
            remaining = flush_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_event, next_text = await asyncio.wait_for(queue.get(), timeout=remaining)
                batch_items.append((next_event, next_text))
            except asyncio.TimeoutError:
                break

        prepared_records: List[Dict[str, Any]] = []
        classifier_entries: List[Dict[str, Any]] = []

        try:
            for idx, (event, text) in enumerate(batch_items, start=1):
                try:
                    context_plain, context_html = await tg_context.collect_reply_context(event)
                except Exception as ctx_err:  # noqa: BLE001
                    logger.warning(
                        "context_collect_failed",
                        extra={
                            "extra": {
                                "chat_id": int(getattr(event, "chat_id", 0)),
                                "message_id": int(getattr(event, "id", 0)),
                                "error": str(ctx_err),
                            }
                        },
                    )
                    context_plain, context_html = None, None

                estimated_tokens = classifier_service.estimate_classifier_tokens(text, context_plain)
                await rate_limiter.acquire(estimated_tokens)

                entry: Dict[str, Any] = {"id": f"item_{idx}", "text": text}
                if context_plain:
                    entry["context"] = context_plain
                classifier_entries.append(entry)

                prepared_records.append(
                    {
                        "event": event,
                        "text": text,
                        "context_plain": context_plain,
                        "context_html": context_html,
                    }
                )

            results = await classifier_service.classify_batch_with_openai(classifier_entries)
            labels = [str(value) for value in results]
            if len(labels) < len(prepared_records):
                labels.extend("0" for _ in range(len(prepared_records) - len(labels)))

            for record, label in zip(prepared_records, labels):
                event = record["event"]
                text = record["text"]
                context_plain = record["context_plain"]
                context_html = record["context_html"]

                logger.info(f"message: {text}\nlabel: {label}", extra={"plain": True})

                if label == "1":
                    link_all = None
                    chat_obj = None
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

                        record_to_save: Dict[str, Any] = {
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
                        await asyncio.to_thread(subscriber_store.save_classified_message, record_to_save)
                    except Exception as persist_err:  # noqa: BLE001
                        logger.warning("persist_message_failed", extra={"extra": {"error": str(persist_err)}})

                    try:
                        from_user = await event.get_sender()
                    except Exception:
                        from_user = None

                    link = link_all if link_all is not None else await tg_context.build_message_link(event)
                    chat = chat_obj if chat_obj is not None else await event.get_chat()

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
                                    classification_result=label,
                                    subscriber_store=subscriber_store,
                                )
                            except Exception as notify_err:  # noqa: BLE001
                                logger.error(
                                    "notify_error",
                                    extra={
                                        "extra": {
                                            "chat_id": int(event.chat_id),  # type: ignore[arg-type]
                                            "message_id": int(event.id),
                                            "recipient": int(recipient_id),
                                            "error": str(notify_err),
                                        }
                                    },
                                )

        except Exception as e:  # noqa: BLE001
            for event, _ in batch_items:
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
            for _ in batch_items:
                queue.task_done()


