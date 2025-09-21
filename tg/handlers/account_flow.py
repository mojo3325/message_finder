from __future__ import annotations

import asyncio

from logging_config import logger
from telethon.errors import FloodWaitError

from const import (
    ACCOUNT_ERROR_CODE_EXPIRED,
    ACCOUNT_ERROR_CODE_INVALID,
    ACCOUNT_ERROR_GENERIC,
    ACCOUNT_ERROR_NEED_2FA,
    ACCOUNT_ERROR_SESSION_INVALID,
    ACCOUNT_PROMPT_2FA,
    ACCOUNT_PROMPT_CODE,
    ACCOUNT_PROMPT_PHONE,
    ACCOUNT_PROMPT_START,
    ACCOUNT_STATUS_ACTIVE,
    ACCOUNT_STATUS_NONE,
    ACCOUNT_SUCCESS,
    ACCOUNT_UNLINK_NONE,
    ACCOUNT_UNLINK_OK,
    CMD_ACCOUNT,
    CMD_UNLINK,
)
from services.user_sessions import (
    client_from_session as create_client_from_session,
    create_ephemeral_client,
    send_login_code,
    sign_in_with_code,
    sign_in_with_password,
)
from tg import bot_api, ui
from tg.handlers.types import CallbackContext, MessageContext
from tg.helpers import safe_delete_messages
from tg.ui_state import account_fsm
from utilities.accounts_store import (
    delete_account as delete_user_account,
    get_user_account,
    save_or_update_account,
)


async def handle_callback(ctx: CallbackContext) -> bool:
    action = ctx.action
    if action not in {"acc_phone", "acc_qr"}:
        return False

    try:
        await bot_api.bot_answer_callback_query(ctx.callback_id, text="", show_alert=False)
    except Exception:
        pass

    user_id = ctx.user_id or ctx.chat_id
    if user_id is None or ctx.message_id is None:
        return False
    st = account_fsm.get(int(user_id))
    st.ui_message_id = int(ctx.message_id)

    if action == "acc_phone":
        st.state = "ACCOUNT_AWAITING_PHONE"
        st.phone = None
        st.phone_code_hash = None
        st.tmp_client_session = None
        chat_id = int(ctx.chat_id or user_id)
        message_id = int(ctx.message_id)
        try:
            await bot_api.bot_edit_message_text(chat_id, message_id, ACCOUNT_PROMPT_PHONE)
        except Exception:
            try:
                mid = await bot_api.bot_send_html_message_with_id(chat_id, ACCOUNT_PROMPT_PHONE)
            except Exception:
                mid = None
            if mid is not None:
                st.ui_message_id = int(mid)
        try:
            mid_kb = await bot_api.bot_send_html_message_with_id(
                chat_id,
                "Нажмите кнопку ниже, чтобы поделиться номером",
                reply_markup=ui.build_request_contact_keyboard(),
            )
            if mid_kb is not None:
                st.ui_aux_message_ids.append(int(mid_kb))
        except Exception:
            pass
        return True

    # QR flow
    try:
        client = create_ephemeral_client()
        await client.connect()
        qr_login = await client.qr_login()
    except Exception as exc:
        logger.error("qr_flow_error_cb", extra={"extra": {"user_id": user_id, "error": str(exc)}})
        try:
            await bot_api.bot_edit_message_text(int(ctx.chat_id or user_id), int(ctx.message_id), ACCOUNT_ERROR_GENERIC)
        except Exception:
            pass
        try:
            await client.disconnect()  # type: ignore[arg-type]
        except Exception:
            pass
        return True

    try:
        url = getattr(qr_login, "url", None) or getattr(qr_login, "_url", None)
    except Exception:
        url = None
    if not url:
        try:
            token = getattr(qr_login, "token", None)
            url = f"tg://login?token={token}" if token else None
        except Exception:
            url = None

    st.tmp_client_session = client.session.save()
    st.state = "ACCOUNT_QR_WAIT"

    body_qr = "\n".join([
        "<b>🔳 Вход по QR</b>",
        "<b>━━━━━━━━━━━━━━━━━━━━</b>",
        "Сканируйте QR или используйте кнопку ниже:",
    ])
    try:
        await bot_api.bot_edit_message_text(
            int(ctx.chat_id or user_id),
            int(ctx.message_id),
            body_qr,
            reply_markup=ui.build_account_confirm_keyboard(url),
        )
    except Exception:
        pass

    try:
        import qrcode
        from io import BytesIO

        if url:
            img = qrcode.make(url)
            buf = BytesIO()
            img.save(buf, format="PNG")
            mid_img = await bot_api.bot_send_photo_with_id(int(user_id), buf.getvalue(), caption_html="Откройте Telegram → Устройства → Сканировать QR")
            if mid_img is not None:
                st.ui_aux_message_ids.append(int(mid_img))
    except Exception:
        pass

    async def _wait_qr_and_finalize(chat_id: int, message_id: int, uid: int) -> None:
        try:
            await asyncio.wait_for(qr_login.wait(), timeout=120)
        except asyncio.TimeoutError:
            try:
                await bot_api.bot_edit_message_text(chat_id, message_id, "QR истёк. Нажмите QR ещё раз для новой попытки.")
            except Exception:
                pass
            return
        try:
            session = client.session.save()
            save_or_update_account(uid, session, phone_e164=None, method="qr")
            account_fsm.clear(uid)
            await bot_api.bot_edit_message_text(chat_id, message_id, ACCOUNT_SUCCESS)
        except Exception as exc_inner:
            logger.error("qr_login_failed_cb", extra={"extra": {"user_id": uid, "error": str(exc_inner)}})
            try:
                await bot_api.bot_edit_message_text(chat_id, message_id, ACCOUNT_ERROR_GENERIC)
            except Exception:
                pass
            return
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

        aux_ids = [int(mid) for mid in list(st.ui_aux_message_ids)]
        st.ui_aux_message_ids = []
        if aux_ids:
            try:
                await safe_delete_messages(uid, aux_ids)
            except Exception:
                pass
        await safe_delete_messages(chat_id, [message_id], delay=5)

    asyncio.create_task(_wait_qr_and_finalize(int(ctx.chat_id or user_id), int(ctx.message_id), int(user_id)))
    return True


async def handle_message(ctx: MessageContext) -> bool:
    user_id = ctx.user_id
    chat_id = ctx.chat_id
    message_id = ctx.message_id
    text_s = ctx.text.strip()
    lower = ctx.lower_text
    contact = ctx.contact
    message = ctx.raw

    if contact:
        st = account_fsm.get(user_id)
        try:
            st.user_message_ids.append(message_id)
        except Exception:
            pass

        raw_phone = str(contact.get("phone_number", "")).strip()
        digits = "".join([ch for ch in raw_phone if ch.isdigit() or ch == "+"]) or raw_phone
        phone_val = digits if digits.startswith("+") else "+" + "".join([ch for ch in digits if ch.isdigit()])
        if st.state != "ACCOUNT_AWAITING_PHONE":
            st.state = "ACCOUNT_AWAITING_PHONE"

        client = create_ephemeral_client()
        await client.connect()
        try:
            code_hash = await send_login_code(client, phone_val)
            st.tmp_client_session = client.session.save()
            st.phone = phone_val
            st.phone_code_hash = code_hash
            st.state = "ACCOUNT_AWAITING_CODE"
        except FloodWaitError as fw:
            wait_s = int(getattr(fw, "seconds", 0) or 0)
            wait_min = max(1, (wait_s + 59) // 60)
            st.state = "ACCOUNT_AWAITING_PHONE"

            try:
                await safe_delete_messages(user_id, st.ui_aux_message_ids)
            except Exception:
                pass
            st.ui_aux_message_ids = []

            reply_markup_fw = None
            try:
                qr_login = await client.qr_login()
                url = None
                try:
                    url = getattr(qr_login, "url", None) or getattr(qr_login, "_url", None)
                except Exception:
                    url = None
                if not url:
                    try:
                        token = getattr(qr_login, "token", None)
                        url = f"tg://login?token={token}" if token else None
                    except Exception:
                        url = None
                reply_markup_fw = ui.build_account_confirm_keyboard(url)
            except Exception:
                reply_markup_fw = None

            body_fw = "\n".join([
                ACCOUNT_PROMPT_PHONE,
                f"⚠️ Слишком много попыток. Подождите {wait_min} мин и попробуйте снова.",
                "Либо используйте быстрый вход по кнопке ниже:",
            ])
            ui_mid_fw = getattr(st, "ui_message_id", None)
            if ui_mid_fw is not None:
                try:
                    await bot_api.bot_edit_message_text(user_id, int(ui_mid_fw), body_fw, reply_markup=reply_markup_fw)
                except Exception:
                    pass
            else:
                try:
                    mid_tmp = await bot_api.bot_send_html_message_with_id(user_id, body_fw, reply_markup=reply_markup_fw)
                    st.ui_message_id = int(mid_tmp) if mid_tmp is not None else None
                except Exception:
                    pass

            try:
                await bot_api.bot_delete_message(user_id, message_id)
            except Exception:
                pass
            try:
                st.user_message_ids = [mid for mid in st.user_message_ids if int(mid) != message_id]
            except Exception:
                pass
            return True
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

        try:
            await safe_delete_messages(user_id, st.ui_aux_message_ids)
        except Exception:
            pass
        st.ui_aux_message_ids = []

        ui_mid = getattr(st, "ui_message_id", None)
        body = ACCOUNT_PROMPT_CODE
        if ui_mid is not None:
            try:
                await bot_api.bot_edit_message_text(user_id, int(ui_mid), body)
            except Exception:
                try:
                    mid2 = await bot_api.bot_send_html_message_with_id(user_id, body)
                    st.ui_message_id = int(mid2) if mid2 is not None else None
                except Exception:
                    pass
        else:
            try:
                mid2 = await bot_api.bot_send_html_message_with_id(user_id, body)
                st.ui_message_id = int(mid2) if mid2 is not None else None
            except Exception:
                pass

        mid_rm_opt = None
        try:
            mid_rm_opt = await bot_api.bot_send_html_message_with_id(
                user_id,
                "Скрываю клавиатуру…",
                reply_markup=ui.hide_reply_keyboard(),
            )
            if mid_rm_opt is not None:
                st.ui_aux_message_ids.append(int(mid_rm_opt))
        except Exception:
            mid_rm_opt = None

        to_delete: list[int] = []
        if mid_rm_opt is not None:
            to_delete.append(int(mid_rm_opt))
        try:
            to_delete.extend([int(x) for x in st.ui_aux_message_ids])
        except Exception:
            pass
        if to_delete:
            asyncio.create_task(safe_delete_messages(user_id, list(set(to_delete)), delay=0.6))

        try:
            await bot_api.bot_delete_message(user_id, message_id)
        except Exception:
            pass
        try:
            st.user_message_ids = [mid for mid in st.user_message_ids if int(mid) != message_id]
        except Exception:
            pass
        return True

    # Commands
    if lower.startswith(CMD_UNLINK):
        acc = get_user_account(user_id)
        if not acc:
            await bot_api.bot_send_html_message(user_id, ACCOUNT_UNLINK_NONE)
            return True
        client = None
        try:
            client = create_client_from_session(acc.string_session)
            await client.connect()
            try:
                await client.log_out()
            except Exception:
                pass
        except Exception as exc:
            logger.warning("unlink_logout_failed", extra={"extra": {"user_id": user_id, "error": str(exc)}})
        finally:
            try:
                if client:
                    await client.disconnect()
            except Exception:
                pass
        delete_user_account(user_id)
        await bot_api.bot_send_html_message(user_id, ACCOUNT_UNLINK_OK)
        try:
            await safe_delete_messages(user_id, account_fsm.get(user_id).ui_aux_message_ids)
        except Exception:
            pass
        try:
            account_fsm.clear(user_id)
        except Exception:
            pass
        return True

    if lower.startswith("/account status"):
        acc = get_user_account(user_id)
        if not acc:
            await bot_api.bot_send_html_message(user_id, ACCOUNT_STATUS_NONE)
        else:
            await bot_api.bot_send_html_message(user_id, ACCOUNT_STATUS_ACTIVE)
        return True

    if lower.startswith("/account help"):
        await bot_api.bot_send_html_message(user_id, ACCOUNT_PROMPT_START)
        return True

    if lower == CMD_ACCOUNT:
        acc = get_user_account(user_id)
        if acc:
            await bot_api.bot_send_html_message(user_id, ACCOUNT_STATUS_ACTIVE)
            return True
        st = account_fsm.get(user_id)
        st.state = "ACCOUNT_IDLE"
        st.phone = None
        st.phone_code_hash = None
        st.tmp_client_session = None
        st.ui_aux_message_ids = []
        st.user_message_ids = []
        try:
            await bot_api.bot_send_html_message(
                user_id,
                "\n".join([
                    "Аккаунт не подключён.",
                    "Чтобы отправлять сообщения от вашего имени, привяжите аккаунт:",
                ]),
                reply_markup=ui.build_account_start_keyboard(),
            )
        except Exception:
            pass
        return True

    text_present = bool(text_s)
    if not text_present:
        return False

    st = account_fsm.get(user_id)
    try:
        st.user_message_ids.append(message_id)
    except Exception:
        pass

    if st.state == "ACCOUNT_AWAITING_PHONE":
        phone = text_s
        if not (phone.startswith("+") and len(phone) >= 10):
            ui_mid = getattr(st, "ui_message_id", None)
            if ui_mid is not None:
                await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_PROMPT_PHONE)
            else:
                await bot_api.bot_send_html_message(user_id, ACCOUNT_PROMPT_PHONE)
            try:
                await bot_api.bot_delete_message(user_id, message_id)
            except Exception:
                pass
            try:
                st.user_message_ids = [mid for mid in st.user_message_ids if int(mid) != message_id]
            except Exception:
                pass
            return True

        client = create_ephemeral_client()
        await client.connect()
        try:
            try:
                code_hash = await send_login_code(client, phone)
                st.tmp_client_session = client.session.save()
                st.phone = phone
                st.phone_code_hash = code_hash
                st.state = "ACCOUNT_AWAITING_CODE"
            except FloodWaitError as fw:
                wait_s = int(getattr(fw, "seconds", 0) or 0)
                wait_min = max(1, (wait_s + 59) // 60)
                st.state = "ACCOUNT_AWAITING_PHONE"

                try:
                    await safe_delete_messages(user_id, st.ui_aux_message_ids)
                except Exception:
                    pass
                st.ui_aux_message_ids = []

                reply_markup_fw = None
                try:
                    qr_login = await client.qr_login()
                    url = None
                    try:
                        url = getattr(qr_login, "url", None) or getattr(qr_login, "_url", None)
                    except Exception:
                        url = None
                    if not url:
                        try:
                            token = getattr(qr_login, "token", None)
                            url = f"tg://login?token={token}" if token else None
                        except Exception:
                            url = None
                    reply_markup_fw = ui.build_account_confirm_keyboard(url)
                except Exception:
                    reply_markup_fw = None

                body_fw = "\n".join([
                    ACCOUNT_PROMPT_PHONE,
                    f"⚠️ Слишком много попыток. Подождите {wait_min} мин и попробуйте снова.",
                    "Либо используйте быстрый вход по кнопке ниже:",
                ])
                ui_mid_fw = getattr(st, "ui_message_id", None)
                if ui_mid_fw is not None:
                    await bot_api.bot_edit_message_text(user_id, int(ui_mid_fw), body_fw, reply_markup=reply_markup_fw)
                else:
                    mid_tmp = await bot_api.bot_send_html_message_with_id(user_id, body_fw, reply_markup=reply_markup_fw)
                    st.ui_message_id = int(mid_tmp) if mid_tmp is not None else None
                try:
                    await bot_api.bot_delete_message(user_id, message_id)
                except Exception:
                    pass
                try:
                    st.user_message_ids = [mid for mid in st.user_message_ids if int(mid) != message_id]
                except Exception:
                    pass
                return True

            try:
                await safe_delete_messages(user_id, st.ui_aux_message_ids)
            except Exception:
                pass
            st.ui_aux_message_ids = []

            try:
                qr_login = await client.qr_login()
                try:
                    url = getattr(qr_login, "url", None) or getattr(qr_login, "_url", None)
                except Exception:
                    url = None
                if not url:
                    try:
                        token = getattr(qr_login, "token", None)
                        url = f"tg://login?token={token}" if token else None
                    except Exception:
                        url = None
                body = "\n".join([
                    ACCOUNT_PROMPT_CODE,
                    "Если код не приходит, нажмите кнопку ниже для быстрого входа:",
                ])
                reply_markup = ui.build_account_confirm_keyboard(url)
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, body, reply_markup=reply_markup)
                else:
                    mid_tmp = await bot_api.bot_send_html_message_with_id(user_id, body, reply_markup=reply_markup)
                    st.ui_message_id = int(mid_tmp) if mid_tmp is not None else None

                async def _wait_link_and_finalize() -> None:
                    try:
                        await asyncio.wait_for(qr_login.wait(), timeout=120)
                    except asyncio.TimeoutError:
                        return
                    try:
                        sess = client.session.save()
                        save_or_update_account(user_id, sess, phone_e164=st.phone, method="link")
                        account_fsm.clear(user_id)
                        ui_mid2 = getattr(st, "ui_message_id", None)
                        if ui_mid2 is not None:
                            await bot_api.bot_edit_message_text(user_id, int(ui_mid2), ACCOUNT_SUCCESS)
                            await safe_delete_messages(user_id, list(st.ui_aux_message_ids))
                            await safe_delete_messages(user_id, list(st.user_message_ids))
                            await asyncio.sleep(5)
                            try:
                                await bot_api.bot_delete_message(user_id, int(ui_mid2))
                            except Exception:
                                pass
                        else:
                            await bot_api.bot_send_html_message(user_id, ACCOUNT_SUCCESS)
                    except Exception:
                        pass

                asyncio.create_task(_wait_link_and_finalize())
            except Exception:
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_PROMPT_CODE)
                else:
                    mid_tmp = await bot_api.bot_send_html_message_with_id(user_id, ACCOUNT_PROMPT_CODE)
                    st.ui_message_id = int(mid_tmp) if mid_tmp is not None else None

            try:
                await bot_api.bot_delete_message(user_id, message_id)
            except Exception:
                pass
            try:
                st.user_message_ids = [mid for mid in st.user_message_ids if int(mid) != message_id]
            except Exception:
                pass
        except Exception as exc:
            logger.error("send_code_failed", extra={"extra": {"user_id": user_id, "error": str(exc)}})
            ui_mid = getattr(st, "ui_message_id", None)
            if ui_mid is not None:
                await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_GENERIC)
            else:
                await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_GENERIC)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass
        return True

    if st.state == "ACCOUNT_AWAITING_CODE":
        code = "".join([ch for ch in text_s if ch.isdigit()])
        client = create_client_from_session(st.tmp_client_session or "")
        await client.connect()
        try:
            sess, err = await sign_in_with_code(client, st.phone or "", code, st.phone_code_hash or "")
            if sess and not err:
                save_or_update_account(user_id, sess, phone_e164=st.phone, method="phone")
                account_fsm.clear(user_id)
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_SUCCESS)
                    await safe_delete_messages(user_id, list(st.user_message_ids))
                    await safe_delete_messages(user_id, list(st.ui_aux_message_ids))
                    await asyncio.sleep(5)
                    try:
                        await bot_api.bot_delete_message(user_id, int(ui_mid))
                    except Exception:
                        pass
                else:
                    await bot_api.bot_send_html_message(user_id, ACCOUNT_SUCCESS)
                return True

            if err == "need_2fa":
                st.state = "ACCOUNT_AWAITING_2FA"
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_PROMPT_2FA)
                else:
                    mid_tmp = await bot_api.bot_send_html_message_with_id(user_id, ACCOUNT_PROMPT_2FA)
                    st.ui_message_id = int(mid_tmp) if mid_tmp is not None else None
            elif err == "code_invalid":
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_CODE_INVALID)
                else:
                    await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_CODE_INVALID)
            elif err == "code_expired":
                try:
                    code_hash = await send_login_code(client, st.phone or "")
                    st.phone_code_hash = code_hash
                    ui_mid = getattr(st, "ui_message_id", None)
                    if ui_mid is not None:
                        await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_CODE_EXPIRED)
                    else:
                        await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_CODE_EXPIRED)
                except FloodWaitError as fw:
                    wait_s = int(getattr(fw, "seconds", 0) or 0)
                    wait_min = max(1, (wait_s + 59) // 60)
                    reply_markup_fw = None
                    try:
                        qr_login = await client.qr_login()
                        url = None
                        try:
                            url = getattr(qr_login, "url", None) or getattr(qr_login, "_url", None)
                        except Exception:
                            url = None
                        if not url:
                            try:
                                token = getattr(qr_login, "token", None)
                                url = f"tg://login?token={token}" if token else None
                            except Exception:
                                url = None
                        reply_markup_fw = ui.build_account_confirm_keyboard(url)
                    except Exception:
                        reply_markup_fw = None
                    body_fw = "\n".join([
                        ACCOUNT_PROMPT_CODE,
                        f"⚠️ Слишком много попыток. Подождите {wait_min} мин и попробуйте снова.",
                        "Либо используйте быстрый вход по кнопке ниже:",
                    ])
                    ui_mid = getattr(st, "ui_message_id", None)
                    if ui_mid is not None:
                        await bot_api.bot_edit_message_text(user_id, ui_mid, body_fw, reply_markup=reply_markup_fw)
                    else:
                        await bot_api.bot_send_html_message(user_id, body_fw, reply_markup=reply_markup_fw)
                except Exception:
                    ui_mid = getattr(st, "ui_message_id", None)
                    if ui_mid is not None:
                        await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_GENERIC)
                    else:
                        await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_GENERIC)
            else:
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_GENERIC)
                else:
                    await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_GENERIC)

            try:
                await bot_api.bot_delete_message(user_id, message_id)
            except Exception:
                pass
            try:
                st.user_message_ids = [mid for mid in st.user_message_ids if int(mid) != message_id]
            except Exception:
                pass
        except Exception as exc:
            logger.error("sign_in_code_failed", extra={"extra": {"user_id": user_id, "error": str(exc)}})
            ui_mid = getattr(st, "ui_message_id", None)
            if ui_mid is not None:
                await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_GENERIC)
            else:
                await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_GENERIC)
            try:
                await bot_api.bot_delete_message(user_id, message_id)
            except Exception:
                pass
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass
        return True

    if st.state == "ACCOUNT_AWAITING_2FA":
        password = text_s
        client = create_client_from_session(st.tmp_client_session or "")
        await client.connect()
        try:
            sess, err = await sign_in_with_password(client, password)
            if sess and not err:
                save_or_update_account(user_id, sess, phone_e164=st.phone, method="phone")
                account_fsm.clear(user_id)
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_SUCCESS)
                    await safe_delete_messages(user_id, list(st.user_message_ids))
                    await safe_delete_messages(user_id, list(st.ui_aux_message_ids))
                    await asyncio.sleep(5)
                    try:
                        await bot_api.bot_delete_message(user_id, int(ui_mid))
                    except Exception:
                        pass
                else:
                    await bot_api.bot_send_html_message(user_id, ACCOUNT_SUCCESS)
            else:
                ui_mid = getattr(st, "ui_message_id", None)
                if ui_mid is not None:
                    await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_GENERIC)
                else:
                    await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_GENERIC)
        except Exception as exc:
            logger.error("sign_in_2fa_failed", extra={"extra": {"user_id": user_id, "error": str(exc)}})
            ui_mid = getattr(st, "ui_message_id", None)
            if ui_mid is not None:
                await bot_api.bot_edit_message_text(user_id, ui_mid, ACCOUNT_ERROR_GENERIC)
            else:
                await bot_api.bot_send_html_message(user_id, ACCOUNT_ERROR_GENERIC)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass
        return True

    return False
