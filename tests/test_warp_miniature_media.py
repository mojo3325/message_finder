from types import SimpleNamespace

from tg.handlers.warp import (
    _build_dialog_preview_text,
    _build_message_preview_lines,
    _collect_media_descriptions,
)
from tg.ui import build_warp_miniature


def _make_message(**kwargs):
    defaults = {
        "photo": None,
        "voice": None,
        "video": None,
        "video_note": None,
        "document": None,
        "sticker": None,
        "gif": None,
        "message": "",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_collect_media_descriptions_photo_and_voice():
    voice = SimpleNamespace(duration=17, attributes=[])
    msg = _make_message(photo=object(), voice=voice)
    descriptions = _collect_media_descriptions(msg)
    assert ("🖼 Фото", "[Фото]") in descriptions
    assert ("🎙 Голосовое сообщение 0:17", "[Голосовое сообщение 0:17]") in descriptions


def test_collect_media_descriptions_document_with_filename():
    document = SimpleNamespace(mime_type="application/pdf", attributes=[SimpleNamespace(file_name="report.pdf")])
    msg = _make_message(document=document)
    descriptions = _collect_media_descriptions(msg)
    assert descriptions == [("📄 Документ (report.pdf)", "[Документ: report.pdf]")]


def test_build_message_preview_lines_combines_text_and_media():
    photo_msg = _make_message(message="Привет", photo=object())
    lines, context = _build_message_preview_lines(photo_msg)
    assert lines == ["Привет", "🖼 Фото"]
    assert context == "Привет [Фото]"


def test_build_message_preview_lines_media_only_voice():
    voice = SimpleNamespace(duration=5, attributes=[])
    msg = _make_message(voice=voice)
    lines, context = _build_message_preview_lines(msg)
    assert lines == ["🎙 Голосовое сообщение 0:05"]
    assert context == "[Голосовое сообщение 0:05]"


def test_build_dialog_preview_text_includes_media_placeholders():
    msg = _make_message(photo=object())
    preview = _build_dialog_preview_text(msg)
    assert preview == "🖼 Фото"


def test_build_dialog_preview_text_combines_text_and_media():
    voice = SimpleNamespace(duration=8, attributes=[])
    msg = _make_message(message="Привет", voice=voice)
    preview = _build_dialog_preview_text(msg)
    assert preview == "Привет · 🎙 Голосовое сообщение 0:08"


def test_build_warp_miniature_renders_multiline_messages():
    messages = [
        {"direction": "in", "author": "Аня", "text": "Привет\n🖼 Фото"},
        {"direction": "out", "author": "Вы", "text": "🎙 Голосовое сообщение 0:05"},
    ]
    body, _ = build_warp_miniature("Чат", "12:00", messages, chat_id=123)
    assert "Привет" in body
    assert "🖼 Фото" in body
    assert "🎙 Голосовое сообщение 0:05" in body
    # Ensure attachment line preserved with indentation
    assert "   🖼 Фото" in body
