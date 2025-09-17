import asyncio
from typing import Any, Dict, Optional, Callable, Tuple

from logging_config import logger
from config import TELEGRAM_BOT_TOKEN
from services.clients import get_http_client
from tg import bot_api, ui
from tg.ui_state import reply_ui_store
from tg.context import escape_html, fetch_author_texts_from_history, fetch_author_messages_with_meta_from_history, sanitize_telegram_html
import services.replier as replier_service
import services.portrait as portrait_service
from services.feedback import save_feedback


async def bot_updates_poller(
    subscriber_store: Any,
    get_telethon_client: Callable[[], Any],
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning(
            "bot_token_missing",
            extra={"extra": {"msg": "TELEGRAM_BOT_TOKEN is not set; poller disabled"}},
        )
        return

    http_client = get_http_client()
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    timeout_s = 50
    try:
        await http_client.get(f"{base}/deleteWebhook", params={"drop_pending_updates": False})
    except Exception as e:  # noqa: BLE001
        logger.warning("delete_webhook_failed", extra={"extra": {"error": str(e)}})

    while True:
        try:
            _offset_val = subscriber_store.get_offset()
            offset = (_offset_val + 1) if _offset_val else None
            params: Dict[str, Any] = {"timeout": timeout_s}
            if offset is not None:
                params["offset"] = offset
            resp = await http_client.get(
                f"{base}/getUpdates",
                params=params,
                timeout=timeout_s + 10,
            )
            data = resp.json()
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

                        # ------------------------------
                        # On-demand state restoration for old messages (after restart)
                        # ------------------------------
                        # Animated loading helpers
                        def _build_progress_bar(step: int, width: int = 10) -> str:
                            pos = step % width
                            left = "░" * pos
                            right = "░" * (width - 1 - pos)
                            return f"{left}█{right}"

                        def _spinner_frames() -> list[str]:
                            return ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

                        def _wave_frames(base: str) -> list[str]:
                            # Simulate a wave with alternating emphasis and dots
                            dots = ["·  ", "·· ", "···", " ··", "  ·", "   "]
                            frames: list[str] = []
                            for d in dots:
                                frames.append(f"{base} {d}")
                            return frames

                        def _loading_frame(step: int, header: str) -> str:
                            sp = _spinner_frames()[step % len(_spinner_frames())]
                            bar = _build_progress_bar(step, 12)
                            wave = _wave_frames("В процессе")[step % 6]
                            return "\n".join([
                                f"<b>{header} {sp}</b>",
                                "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                                f"{escape_html(wave)}",
                                f"{escape_html(bar)}",
                            ])

                        def _start_loading_animation(chat_id: int, message_id: int, sid_local: str, header: str) -> tuple[asyncio.Event, asyncio.Task]:
                            stop_event: asyncio.Event = asyncio.Event()

                            async def _loop() -> None:
                                step = 0
                                try:
                                    while not stop_event.is_set() and not reply_ui_store.is_cancelled(sid_local):
                                        body_anim = _loading_frame(step, header)
                                        try:
                                            await bot_api.bot_edit_message_text(
                                                chat_id, message_id, body_anim, reply_markup=ui.build_reply_keyboard(sid_local, None)
                                            )
                                        except Exception:
                                            # Ignore animation edit errors
                                            pass
                                        step += 1
                                        await asyncio.sleep(0.9)
                                except Exception:
                                    pass

                            task = asyncio.create_task(_loop())
                            return stop_event, task

                        async def _restore_state_from_callback_message() -> Optional[Tuple[str, Any]]:
                            # Extract source link from message entities or text
                            def _extract_link(m: Dict[str, Any]) -> Optional[str]:
                                try:
                                    entities = m.get("entities") or []
                                    for ent in entities:
                                        if ent.get("type") == "text_link" and ent.get("url"):
                                            return str(ent.get("url"))
                                except Exception:
                                    pass
                                # Fallback: search plain text for t.me link
                                try:
                                    text = (m.get("text") or "").strip()
                                except Exception:
                                    text = ""
                                if not text:
                                    return None
                                for token in text.split():
                                    if token.startswith("http://t.me/") or token.startswith("https://t.me/") or token.startswith("t.me/"):
                                        return token if token.startswith("http") else f"https://{token}"
                                return None

                        # Themes are no longer rendered/used; no extraction needed

                            def _parse_link_to_ids(link: str) -> Optional[Tuple[int, int]]:
                                try:
                                    # Examples:
                                    # https://t.me/c/1234567/89 -> chat_id = -1001234567, message_id = 89
                                    # https://t.me/someusername/123 -> resolve username to chat_id
                                    if "/c/" in link:
                                        parts = link.rstrip("/").split("/c/")
                                        right = parts[1]
                                        internal_str, mid_str = right.split("/")[:2]
                                        chat_id = int(f"-100{int(internal_str)}")
                                        return chat_id, int(mid_str)
                                    # Username variant
                                    segments = link.rstrip("/").split("/")
                                    username = segments[-2]
                                    mid_str = segments[-1]
                                    return (username, int(mid_str))  # type: ignore[return-value]
                                except Exception:
                                    return None

                            # Resolve link
                            link = _extract_link(msg)
                            if not link:
                                return None

                            client_ref = get_telethon_client()
                            if client_ref is None:
                                return None

                            ids = _parse_link_to_ids(link)
                            if ids is None:
                                return None

                            # ids can be (int chat_id, int msg_id) or (str username, int msg_id)
                            try:
                                if isinstance(ids[0], str):
                                    entity = await client_ref.get_entity(ids[0])  # type: ignore[arg-type]
                                    source_chat_id = int(getattr(entity, "id", 0) or 0)
                                else:
                                    source_chat_id = int(ids[0])  # type: ignore[index]
                                source_msg_id = int(ids[1])  # type: ignore[index]
                            except Exception:
                                return None

                            # Fetch source message and sender
                            try:
                                src_msg = await client_ref.get_messages(entity=source_chat_id, ids=source_msg_id)
                            except Exception:
                                return None
                            if not src_msg:
                                return None

                            try:
                                raw_text = (getattr(src_msg, "message", None) or "").strip()
                            except Exception:
                                raw_text = ""
                            if not raw_text:
                                return None

                            try:
                                sender = await src_msg.get_sender()
                            except Exception:
                                sender = None
                            try:
                                chat_entity = await client_ref.get_entity(source_chat_id)
                            except Exception:
                                chat_entity = None

                            # Build context (plain and HTML) by traversing reply chain from the source message
                            async def _collect_context_from_message() -> Tuple[Optional[str], Optional[str]]:
                                try:
                                    chain = []
                                    current = src_msg
                                    steps = 0
                                    while current is not None and steps < 6:
                                        try:
                                            parent = await current.get_reply_message()
                                        except Exception:
                                            parent = None
                                        if not parent:
                                            break
                                        chain.append(parent)
                                        current = parent
                                        steps += 1
                                    if not chain:
                                        return None, None
                                    chain.reverse()

                                    parts_plain = []
                                    parts_html = ["<b>Контекст</b>", "<b>━━━━━━━━━━━━━━━━━━━━</b>"]
                                    for idx, m in enumerate(chain):
                                        try:
                                            t = (getattr(m, "message", None) or "").strip()
                                        except Exception:
                                            t = ""
                                        if not t:
                                            continue
                                        if idx == 0:
                                            parts_plain.append(f"Post: {t}")
                                        else:
                                            parts_plain.append(f"Reply{idx}: {t}")
                                        safe = escape_html(t)
                                        prefix = "Пост" if idx == 0 else f"Ответ {idx}"
                                        parts_html.append(f"• <i>{prefix}</i>:\n{safe}")
                                    if not parts_plain:
                                        return None, None
                                    return "\n".join(parts_plain), "\n".join(parts_html)
                                except Exception:
                                    return None, None

                            context_plain_local, context_html_local = await _collect_context_from_message()

                            # Compose a fresh original body HTML similar to notifier
                            def _build_body_html() -> str:
                                full_name = " ".join(filter(None, [getattr(sender, "first_name", None), getattr(sender, "last_name", None)])) or "отсутствует"
                                username_val = f"@{getattr(sender, 'username', '')}" if getattr(sender, "username", None) else "отсутствует"
                                chat_title_val = getattr(chat_entity, "title", None) or getattr(chat_entity, "username", None) or (chat_entity.__class__.__name__ if chat_entity is not None else "")
                                underlined = f"<u>{escape_html(raw_text)}</u>"
                                header = "<b>🔎 Обнаружен \"возможный диалог\"</b>"
                                divider = "<b>━━━━━━━━━━━━━━━━━━━━</b>"
                                parts_local = [
                                    header,
                                    divider,
                                    f"• <b>Имя</b>: {escape_html(full_name)}",
                                    f"• <b>Никнейм</b>: <i>{escape_html(username_val)}</i>",
                                    f"• <b>Чат</b>: <i>{escape_html(str(chat_title_val))}</i>",
                                    divider,
                                    f"<b>Сообщение</b>:\n{underlined}",
                                ]
                                if context_html_local:
                                    parts_local.extend([divider, context_html_local])
                                # Themes removed from UI as per new classifier schema
                                if link:
                                    parts_local.append(f"• <b>Ссылка</b>: <a href=\"{escape_html(link)}\">перейти</a>")
                                return "\n".join(parts_local)

                            rebuilt_body_html = _build_body_html()

                            # Save reconstructed state with a new sid
                            new_sid = reply_ui_store.create(
                                user_id=int(from_user_id) if from_user_id is not None else int(msg_chat_id),
                                original_body_html=rebuilt_body_html,
                                original_text=raw_text,
                                context_for_model=context_plain_local,
                                classification_result=None,
                                author_user_id=int(getattr(sender, "id", 0)) if sender is not None else None,
                                source_chat_id=int(source_chat_id) if source_chat_id is not None else None,
                            )
                            return new_sid, reply_ui_store.get(new_sid)

                        # Try restore if state is missing
                        if (not st) and sid:
                            restored = await _restore_state_from_callback_message()
                            if restored is not None:
                                sid, st = restored[0], restored[1]
                        

                        if action == "gen" or action == "regen":
                            await bot_api.bot_answer_callback_query(cb_id, text="Генерирую ответ…", show_alert=False)
                            if not st:
                                await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
                                continue
                            reply_ui_store.clear_cancelled(sid)

                            loading_body = "\n".join([
                                "<b>✍️ Генерация ответа…</b>",
                                "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                                "Подождите, пожалуйста…",
                            ])
                            await bot_api.bot_edit_message_text(
                                msg_chat_id, msg_id, loading_body, reply_markup=ui.build_reply_keyboard(sid, None)
                            )

                            # Start animated loading
                            stop_anim, _ = _start_loading_animation(
                                int(msg_chat_id), int(msg_id), sid, header="✍️ Генерация ответа…"
                            )

                            async def _bg_generate_and_update() -> None:
                                try:
                                    reply_text = await replier_service.generate_reply(
                                        st.original_text, context=st.context_for_model
                                    )
                                    if reply_ui_store.is_cancelled(sid):
                                        try:
                                            stop_anim.set()
                                        except Exception:
                                            pass
                                        return
                                    # Stop animation before final render
                                    try:
                                        stop_anim.set()
                                    except Exception:
                                        pass
                                    reply_ui_store.set_reply(sid, reply_text)
                                    safe = escape_html(reply_text)
                                    body = "\n".join([
                                        "<b>✍️ Предложенный ответ</b>",
                                        "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                                        safe,
                                    ])
                                    reply_markup: Dict[str, Any] = ui.build_reply_keyboard(sid, reply_text)
                                    ok = await bot_api.bot_edit_message_text(
                                        msg_chat_id, msg_id, body, reply_markup=reply_markup
                                    )
                                    if not ok and not reply_ui_store.is_cancelled(sid):
                                        fallback_markup = ui.build_reply_keyboard(sid, None)
                                        ok2 = await bot_api.bot_edit_message_text(
                                            msg_chat_id, msg_id, body, reply_markup=fallback_markup
                                        )
                                        if not ok2:
                                            await bot_api.bot_send_html_message(
                                                msg_chat_id, body, reply_markup=fallback_markup
                                            )
                                except Exception as e:
                                    try:
                                        stop_anim.set()
                                    except Exception:
                                        pass
                                    logger.error("gen_bg_error", extra={"extra": {"sid": sid, "error": str(e)}})

                            asyncio.create_task(_bg_generate_and_update())
                            continue

                        if action == "portrait":
                            await bot_api.bot_answer_callback_query(cb_id, text="Собираю сообщения…", show_alert=False)
                            if not st:
                                await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
                                continue
                            reply_ui_store.clear_cancelled(sid)

                            loading_body = "\n".join([
                                "<b>👨🏻‍🦰 Генерация портрета…</b>",
                                "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                                "Подождите, пожалуйста…",
                            ])
                            await bot_api.bot_edit_message_text(
                                msg_chat_id, msg_id, loading_body, reply_markup=ui.build_reply_keyboard(sid, None)
                            )

                            # Start animated loading
                            stop_anim, _ = _start_loading_animation(
                                int(msg_chat_id), int(msg_id), sid, header="👨🏻‍🦰 Генерация портрета…"
                            )

                            async def _bg_portrait_and_update() -> None:
                                try:
                                    author_id = getattr(st, "author_user_id", None)
                                    if not author_id:
                                        if reply_ui_store.is_cancelled(sid):
                                            try:
                                                stop_anim.set()
                                            except Exception:
                                                pass
                                            return
                                        body_local = "К сожалению сообщений не достаточно для составления портрета человека"
                                        ok_local = await bot_api.bot_edit_message_text(
                                            msg_chat_id, msg_id, body_local, reply_markup=ui.build_reply_keyboard(sid, None)
                                        )
                                        if not ok_local:
                                            await bot_api.bot_send_html_message(
                                                msg_chat_id, body_local, reply_markup=ui.build_reply_keyboard(sid, None)
                                            )
                                        try:
                                            stop_anim.set()
                                        except Exception:
                                            pass
                                        return

                                    texts: list[str] = []
                                    items_with_meta: list[dict] = []
                                    try:
                                        client_ref = get_telethon_client()
                                        source_chat_id = getattr(st, "source_chat_id", None)
                                        if client_ref is not None and source_chat_id is not None:
                                            items_with_meta = await fetch_author_messages_with_meta_from_history(
                                                client_ref,
                                                int(source_chat_id),
                                                int(author_id),
                                                limit_msgs=1500,
                                                max_collect=200,
                                            )
                                            texts = [it.get("text", "") for it in items_with_meta if it.get("text")]
                                        else:
                                            # Fallback to texts-only if meta fetch is not possible
                                            if client_ref is not None and source_chat_id is not None:
                                                texts = await fetch_author_texts_from_history(
                                                    client_ref,
                                                    int(source_chat_id),
                                                    int(author_id),
                                                    limit_msgs=1500,
                                                    max_collect=200,
                                                )
                                    except Exception as e:  # noqa: BLE001
                                        logger.warning(
                                            "portrait_history_error", extra={"extra": {"error": str(e)}}
                                        )
                                    if len([t for t in texts if t and t.strip()]) < 2:
                                        if reply_ui_store.is_cancelled(sid):
                                            try:
                                                stop_anim.set()
                                            except Exception:
                                                pass
                                            return
                                        body_local = "К сожалению сообщений не достаточно для составления портрета человека"
                                        ok_local = await bot_api.bot_edit_message_text(
                                            msg_chat_id, msg_id, body_local, reply_markup=ui.build_reply_keyboard(sid, None)
                                        )
                                        if not ok_local:
                                            await bot_api.bot_send_html_message(
                                                msg_chat_id, body_local, reply_markup=ui.build_reply_keyboard(sid, None)
                                            )
                                        try:
                                            stop_anim.set()
                                        except Exception:
                                            pass
                                        return
                                    def _build_analysis(items: list[dict]) -> str:
                                        if not items:
                                            return ""
                                        # Stats
                                        lengths = [len(x.get("text", "")) for x in items if isinstance(x.get("text"), str)]
                                        hours = [int(x.get("hour_utc", 0) or 0) % 24 for x in items]
                                        weekdays = [int(x.get("weekday", 0) or 0) % 7 for x in items]
                                        ts_vals = [int(x.get("date_ts", 0) or 0) for x in items if int(x.get("date_ts", 0) or 0) > 0]
                                        # Length stats
                                        n = len(lengths)
                                        avg_len = int(sum(lengths) / n) if n else 0
                                        med_len = sorted(lengths)[n // 2] if n else 0
                                        # Hour histogram
                                        hour_cnt: dict[int, int] = {}
                                        for h in hours:
                                            hour_cnt[h] = hour_cnt.get(h, 0) + 1
                                        top_hours = sorted(hour_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                                        # Weekday histogram (Mon=0)
                                        wd_cnt: dict[int, int] = {}
                                        for w in weekdays:
                                            wd_cnt[w] = wd_cnt.get(w, 0) + 1
                                        top_wd = sorted(wd_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                                        ru_wd = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
                                        # Night activity share (UTC 0-6 and 23)
                                        night = sum(1 for h in hours if h in {0,1,2,3,4,5,6,23})
                                        night_share = round(100.0 * night / len(hours), 1) if hours else 0.0
                                        # Inter-message median gap (hours)
                                        gaps: list[float] = []
                                        ts_sorted = sorted(ts_vals)
                                        for i in range(1, len(ts_sorted)):
                                            gaps.append((ts_sorted[i] - ts_sorted[i-1]) / 3600.0)
                                        med_gap_h = round(sorted(gaps)[len(gaps)//2], 2) if gaps else 0.0
                                        # Best timezone offset guess: maximize daytime 9..23 local
                                        def score_offset(off: int) -> float:
                                            if not hours:
                                                return 0.0
                                            dayhits = sum(1 for h in hours if 9 <= ((h + off) % 24) <= 23)
                                            return dayhits / len(hours)
                                        best_off = 0
                                        best_score = -1.0
                                        for off in range(-12, 15):
                                            sc = score_offset(off)
                                            if sc > best_score:
                                                best_off, best_score = off, sc
                                        # Top local hours with best offset
                                        local_cnt: dict[int, int] = {}
                                        for h in hours:
                                            lh = (h + best_off) % 24
                                            local_cnt[lh] = local_cnt.get(lh, 0) + 1
                                        top_local_hours = sorted(local_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                                        # Range
                                        from_ts = min(ts_vals) if ts_vals else 0
                                        to_ts = max(ts_vals) if ts_vals else 0
                                        try:
                                            from_iso = __import__("datetime").datetime.utcfromtimestamp(from_ts).isoformat() + "Z" if from_ts else ""
                                            to_iso = __import__("datetime").datetime.utcfromtimestamp(to_ts).isoformat() + "Z" if to_ts else ""
                                        except Exception:
                                            from_iso, to_iso = "", ""

                                        def fmt_top_hours(pairs: list[tuple[int,int]]) -> str:
                                            return ", ".join([f"{h}: {c}" for h, c in pairs]) if pairs else ""

                                        def fmt_top_wd(pairs: list[tuple[int,int]]) -> str:
                                            return ", ".join([f"{ru_wd[w]}: {c}" for w, c in pairs]) if pairs else ""

                                        parts = [
                                            f"messages_collected: {len(items)}",
                                            f"range_utc: {from_iso} .. {to_iso}",
                                            f"avg_len: {avg_len}",
                                            f"med_len: {med_len}",
                                            f"hours_utc_top: {fmt_top_hours(top_hours)}",
                                            f"weekdays_top: {fmt_top_wd(top_wd)}",
                                            f"night_share_pct_utc: {night_share}",
                                            f"median_gap_h: {med_gap_h}",
                                            f"best_offset_guess: {best_off:+d}",
                                            f"top_local_hours: {fmt_top_hours(top_local_hours)}",
                                        ]
                                        return "\n".join(parts)

                                    analysis_block = _build_analysis(items_with_meta)

                                    portrait = await portrait_service.generate_portrait(texts, analysis=analysis_block)
                                    if reply_ui_store.is_cancelled(sid):
                                        try:
                                            stop_anim.set()
                                        except Exception:
                                            pass
                                        return
                                    # Stop animation before final render
                                    try:
                                        stop_anim.set()
                                    except Exception:
                                        pass
                                    # Resolve nickname (username)
                                    nickname_val = "отсутствует"
                                    try:
                                        if client_ref is not None and author_id is not None:
                                            entity = await client_ref.get_entity(int(author_id))
                                            uname = getattr(entity, "username", None)
                                            if uname:
                                                nickname_val = f"@{uname}"
                                    except Exception:
                                        pass
                                    sanitized_portrait = sanitize_telegram_html(portrait)
                                    body_final = "\n".join([
                                        "<b>👨🏻‍🦰 Портрет пользователя</b>",
                                        f"• <b>Никнейм</b>: <i>{escape_html(nickname_val)}</i>",
                                        f"• <b>user_id</b>: <i>{int(author_id)}</i>",
                                        "<b>━━━━━━━━━━━━━━━━━━━━</b>",
                                        sanitized_portrait,
                                    ])
                                    ok_final = await bot_api.bot_edit_message_text(
                                        msg_chat_id, msg_id, body_final, reply_markup=ui.build_reply_keyboard(sid, None)
                                    )
                                    if not ok_final:
                                        await bot_api.bot_send_html_message(
                                            msg_chat_id, body_final, reply_markup=ui.build_reply_keyboard(sid, None)
                                        )
                                except Exception as e:
                                    try:
                                        stop_anim.set()
                                    except Exception:
                                        pass
                                    logger.error("portrait_bg_error", extra={"extra": {"sid": sid, "error": str(e)}})

                            asyncio.create_task(_bg_portrait_and_update())
                            continue

                        if action == "dislike":
                            await bot_api.bot_answer_callback_query(cb_id, text="Спасибо за отзыв!", show_alert=False)
                            if not st:
                                logger.warning(
                                    "feedback_handler_missing_state", extra={"extra": {"sid": sid, "action": action}}
                                )
                                continue

                            feedback_data = {
                                "message": st.original_text,
                                "output": {"classification": "0"},
                                "label": 0,
                            }

                            try:
                                await asyncio.to_thread(save_feedback, feedback_data)
                                logger.info("feedback_saved", extra={"extra": {"sid": sid, "action": action}})
                            except Exception as e:
                                logger.error("save_feedback_failed", extra={"extra": {"error": str(e)}})

                            # Try to delete the bot message tied to this callback
                            try:
                                await bot_api.bot_delete_message(msg_chat_id, msg_id)
                            except Exception as e:
                                logger.error("delete_message_failed", extra={"extra": {"chat_id": msg_chat_id, "message_id": msg_id, "error": str(e)}})

                            continue

                        if action == "back":
                            await bot_api.bot_answer_callback_query(cb_id, text="Возвращаю…", show_alert=False)
                            if not st:
                                await bot_api.bot_answer_callback_query(cb_id, text="Состояние не найдено", show_alert=True)
                                continue
                            reply_ui_store.mark_cancelled(sid)
                            orig_keyboard = {
                                "inline_keyboard": [
                                    [
                                        {"text": "👎", "callback_data": f"dislike:{sid}"},
                                        {"text": "✨", "callback_data": f"gen:{sid}"},
                                        {"text": "👨🏻‍🦰", "callback_data": f"portrait:{sid}"},
                                    ]
                                ]
                            }
                            ok = await bot_api.bot_edit_message_text(
                                msg_chat_id, msg_id, st.original_body_html, reply_markup=orig_keyboard
                            )
                            if not ok:
                                await bot_api.bot_send_html_message(
                                    msg_chat_id, st.original_body_html, reply_markup=orig_keyboard
                                )
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
                    if isinstance(text, str) and text.strip().lower().startswith("/stop"):
                        if subscriber_store.remove(user_id):
                            logger.info("subscriber_removed", extra={"extra": {"user_id": user_id}})
                    else:
                        added_before = subscriber_store.contains(user_id)
                        added_ok = subscriber_store.add(user_id)
                        if added_ok and (not added_before):
                            logger.info("subscriber_added", extra={"extra": {"user_id": user_id}})
        except Exception as e:  # noqa: BLE001
            logger.error("bot_poller_error", extra={"extra": {"error": f"{type(e).__name__}: {e}"}})
            await asyncio.sleep(1.0)


