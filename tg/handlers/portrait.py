from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from logging_config import logger

from tg import bot_api, ui
from tg.context import (
    escape_html,
    fetch_author_messages_with_meta_from_history,
    fetch_author_texts_from_history,
    sanitize_telegram_html,
)
from tg.handlers.types import CallbackContext
from tg.helpers import (
    build_loading_frame,
    restore_reply_state_from_callback_message,
    start_loading_animation,
)
from tg.ui_state import reply_ui_store
import services.portrait as portrait_service


PORTRAIT_LOADING_SEQUENCE: tuple[str, ...] = (
    "👨🏻‍🦰 Собираю сообщения…",
    "👨🏻‍🦰 Анализирую сообщения…",
    "👨🏻‍🦰 Думаю над ответом…",
)
PORTRAIT_LOADING_INTERVAL_S: float = 1.6


async def handle_callback(
    ctx: CallbackContext,
    *,
    get_telethon_client: Callable[[], Any],
) -> bool:
    if ctx.action != "portrait":
        return False

    callback_id = ctx.callback_id
    chat_id = ctx.chat_id
    message_id = ctx.message_id
    sid = ctx.sid

    if callback_id is None or chat_id is None or message_id is None or not sid:
        return False

    await bot_api.bot_answer_callback_query(callback_id, text="Собираю сообщения…", show_alert=False)

    state = reply_ui_store.get(sid)
    if not state and sid:
        restored = await restore_reply_state_from_callback_message(
            ctx.message,
            get_telethon_client=get_telethon_client,
            requester_user_id=int(ctx.user_id) if ctx.user_id is not None else None,
            fallback_user_id=int(chat_id) if chat_id is not None else None,
        )
        if restored is not None:
            sid, state = restored

    if not state:
        await bot_api.bot_answer_callback_query(callback_id, text="Состояние не найдено", show_alert=True)
        return True

    reply_ui_store.clear_cancelled(str(sid))

    sid_local = str(sid)
    chat_id_local = int(chat_id)
    message_id_local = int(message_id)

    back_keyboard = {"inline_keyboard": [[{"text": "⬅ Назад", "callback_data": f"back:{sid_local}"}]]}

    reply_keyboard = ui.build_reply_keyboard(sid_local, None)
    headers = PORTRAIT_LOADING_SEQUENCE

    try:
        first_frame = build_loading_frame(0, headers[0])
        await bot_api.bot_edit_message_text(
            chat_id_local,
            message_id_local,
            first_frame,
            reply_markup=reply_keyboard,
        )
    except Exception:
        pass

    loading_animation: tuple[Optional[asyncio.Event], Optional[asyncio.Task[None]]] = (None, None)
    try:
        loading_animation = await start_loading_animation(
            chat_id_local,
            message_id_local,
            sid_local,
            header=headers,
            reply_markup=reply_keyboard,
            interval=PORTRAIT_LOADING_INTERVAL_S,
            freeze_on_last=True,
        )
    except Exception:
        loading_animation = (None, None)

    async def _background_job() -> None:
        nonlocal loading_animation

        async def _stop_loading() -> None:
            nonlocal loading_animation
            stop_evt, task = loading_animation
            if stop_evt is not None and not stop_evt.is_set():
                try:
                    stop_evt.set()
                except Exception:
                    pass
            if task is not None and not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            loading_animation = (None, None)

        try:
            author_id = getattr(state, "author_user_id", None)
            if not author_id:
                if reply_ui_store.is_cancelled(sid_local):
                    await _stop_loading()
                    return
                body_local = "К сожалению сообщений не достаточно для составления портрета человека"
                await _stop_loading()
                ok_local = await bot_api.bot_edit_message_text(
                    chat_id_local,
                    message_id_local,
                    body_local,
                    reply_markup=back_keyboard,
                )
                if not ok_local:
                    await bot_api.bot_send_html_message(
                        chat_id_local,
                        body_local,
                        reply_markup=back_keyboard,
                    )
                return

            texts: list[str] = []
            items_with_meta: list[dict] = []
            try:
                client_ref = get_telethon_client()
                source_chat_id = getattr(state, "source_chat_id", None)
                if client_ref is not None and source_chat_id is not None:
                    try:
                        items_with_meta = await fetch_author_messages_with_meta_from_history(
                            client_ref,
                            int(source_chat_id),
                            int(author_id),
                            limit_msgs=1500,
                            max_collect=200,
                            author_access_hash=getattr(state, "author_access_hash", None),
                        )
                        texts = [item.get("text", "") for item in items_with_meta if item.get("text")]
                    except Exception as meta_exc:
                        logger.warning(
                            "portrait_history_meta_error",
                            extra={"extra": {"error": str(meta_exc)}},
                        )
                        items_with_meta = []
                    if not texts:
                        try:
                            texts = await fetch_author_texts_from_history(
                                client_ref,
                                int(source_chat_id),
                                int(author_id),
                                limit_msgs=1500,
                                max_collect=200,
                                author_access_hash=getattr(state, "author_access_hash", None),
                            )
                        except Exception as plain_exc:
                            logger.warning(
                                "portrait_history_plain_error",
                                extra={"extra": {"error": str(plain_exc)}},
                            )
            except Exception as exc:
                logger.warning("portrait_history_error", extra={"extra": {"error": str(exc)}})

            if len([t for t in texts if t and t.strip()]) < 2:
                if reply_ui_store.is_cancelled(sid_local):
                    await _stop_loading()
                    return
                body_local = "К сожалению сообщений не достаточно для составления портрета человека"
                await _stop_loading()
                ok_local = await bot_api.bot_edit_message_text(
                    chat_id_local,
                    message_id_local,
                    body_local,
                    reply_markup=back_keyboard,
                )
                if not ok_local:
                    await bot_api.bot_send_html_message(
                        chat_id_local,
                        body_local,
                        reply_markup=back_keyboard,
                    )
                return

            def _build_analysis(items: list[dict]) -> str:
                if not items:
                    return ""
                lengths = [len(x.get("text", "")) for x in items if isinstance(x.get("text"), str)]
                hours = [int(x.get("hour_utc", 0) or 0) % 24 for x in items]
                weekdays = [int(x.get("weekday", 0) or 0) % 7 for x in items]
                timestamps = [int(x.get("date_ts", 0) or 0) for x in items if int(x.get("date_ts", 0) or 0) > 0]

                n = len(lengths)
                avg_len = int(sum(lengths) / n) if n else 0
                med_len = sorted(lengths)[n // 2] if n else 0

                hour_hist: dict[int, int] = {}
                for hour in hours:
                    hour_hist[hour] = hour_hist.get(hour, 0) + 1
                top_hours = sorted(hour_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

                weekday_hist: dict[int, int] = {}
                for wd in weekdays:
                    weekday_hist[wd] = weekday_hist.get(wd, 0) + 1
                top_weekdays = sorted(weekday_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                ru_weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

                night_hits = sum(1 for hour in hours if hour in {0, 1, 2, 3, 4, 5, 6, 23})
                night_share = round(100.0 * night_hits / len(hours), 1) if hours else 0.0

                gaps: list[float] = []
                ts_sorted = sorted(timestamps)
                for idx in range(1, len(ts_sorted)):
                    gaps.append((ts_sorted[idx] - ts_sorted[idx - 1]) / 3600.0)
                med_gap_h = round(sorted(gaps)[len(gaps) // 2], 2) if gaps else 0.0

                def _score_offset(offset: int) -> float:
                    if not hours:
                        return 0.0
                    day_hits = sum(1 for hour in hours if 9 <= ((hour + offset) % 24) <= 23)
                    return day_hits / len(hours)

                best_offset = 0
                best_score = -1.0
                for offset in range(-12, 15):
                    score = _score_offset(offset)
                    if score > best_score:
                        best_offset, best_score = offset, score

                local_hist: dict[int, int] = {}
                for hour in hours:
                    local_hour = (hour + best_offset) % 24
                    local_hist[local_hour] = local_hist.get(local_hour, 0) + 1
                top_local_hours = sorted(local_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

                from_ts = min(timestamps) if timestamps else 0
                to_ts = max(timestamps) if timestamps else 0
                try:
                    from_iso = __import__("datetime").datetime.utcfromtimestamp(from_ts).isoformat() + "Z" if from_ts else ""
                    to_iso = __import__("datetime").datetime.utcfromtimestamp(to_ts).isoformat() + "Z" if to_ts else ""
                except Exception:
                    from_iso, to_iso = "", ""

                def _fmt_pairs(pairs: list[tuple[int, int]], lookup: list[str] | None = None) -> str:
                    if not pairs:
                        return ""
                    formatted = []
                    for value, count in pairs:
                        label = lookup[value] if lookup and 0 <= value < len(lookup) else str(value)
                        formatted.append(f"{label}: {count}")
                    return ", ".join(formatted)

                parts = [
                    f"messages_collected: {len(items)}",
                    f"range_utc: {from_iso} .. {to_iso}",
                    f"avg_len: {avg_len}",
                    f"med_len: {med_len}",
                    f"hours_utc_top: {_fmt_pairs(top_hours)}",
                    f"weekdays_top: {_fmt_pairs(top_weekdays, ru_weekdays)}",
                    f"night_share_pct_utc: {night_share}",
                    f"median_gap_h: {med_gap_h}",
                    f"best_offset_guess: {best_offset:+d}",
                    f"top_local_hours: {_fmt_pairs(top_local_hours)}",
                ]
                return "\n".join(parts)

            analysis_block = _build_analysis(items_with_meta)

            portrait_text = await portrait_service.generate_portrait(texts, analysis=analysis_block)
            if reply_ui_store.is_cancelled(sid_local):
                await _stop_loading()
                return

            await _stop_loading()

            nickname = "отсутствует"
            try:
                client_ref = get_telethon_client()
                if client_ref is not None and author_id is not None:
                    entity = await client_ref.get_entity(int(author_id))
                    username = getattr(entity, "username", None)
                    if username:
                        nickname = f"@{username}"
            except Exception:
                pass

            sanitized = sanitize_telegram_html(portrait_text)
            body_final = "\n".join(
                [
                    "<b>👨🏻‍🦰 Портрет пользователя</b>",
                    f"• <b>Никнейм</b>: <i>{escape_html(nickname)}</i>",
                    "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                    sanitized,
                ]
            )
            ok_final = await bot_api.bot_edit_message_text(
                chat_id_local,
                message_id_local,
                body_final,
                reply_markup=ui.build_reply_keyboard(sid_local, None),
            )
            if not ok_final:
                await bot_api.bot_send_html_message(
                    chat_id_local,
                    body_final,
                    reply_markup=ui.build_reply_keyboard(sid_local, None),
                )
        except Exception as exc:
            await _stop_loading()
            logger.error("portrait_bg_error", extra={"extra": {"sid": sid_local, "error": str(exc)}})

    asyncio.create_task(_background_job())
    return True
