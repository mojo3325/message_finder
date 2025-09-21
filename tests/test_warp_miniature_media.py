from types import SimpleNamespace

from tg.handlers.warp import (
    _build_dialog_preview_text,
    _build_message_preview_lines,
    _build_miniature_message_entry,
    _collect_media_descriptions,
    get_media_info,
    get_media_type,
)
from tg.ui import build_warp_miniature


def _make_message(**kwargs):
    defaults = {
        "photo": None,
        "voice": None,
        "audio": None,
        "video": None,
        "video_note": None,
        "document": None,
        "sticker": None,
        "gif": None,
        "file": None,
        "media": None,
        "geo": None,
        "message": "",
        "out": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_collect_media_descriptions_photo_and_voice():
    voice = SimpleNamespace(duration=17, attributes=[])
    file_obj = SimpleNamespace(name="image.jpg", ext=".jpg", mime_type="image/jpeg", size=2048)
    msg = _make_message(photo=object(), voice=voice, file=file_obj)
    descriptions = _collect_media_descriptions(msg)
    displays = {desc["display"] for desc in descriptions}
    contexts = {desc["context"] for desc in descriptions}
    assert any(display.startswith("🖼 изображение") for display in displays)
    assert any(display.startswith("🎙 Голосовое сообщение 0:17") for display in displays)
    assert "[Фото: изображение.jpg]" in contexts
    assert "[Голосовое сообщение 0:17]" in contexts


def test_collect_media_descriptions_document_with_filename():
    file_obj = SimpleNamespace(name="report.pdf", ext=".pdf", mime_type="application/pdf", size=4096)
    document = SimpleNamespace(
        mime_type="application/pdf",
        attributes=[SimpleNamespace(file_name="report.pdf")],
        size=4096,
    )
    msg = _make_message(document=document, file=file_obj)
    descriptions = _collect_media_descriptions(msg)
    assert len(descriptions) == 1
    entry = descriptions[0]
    assert entry["media_type"] == "document"
    assert entry["display"].startswith("📄 документ")
    assert "report.pdf" not in entry["display"]
    assert entry["context"] == "[Документ: документ.pdf]"
    assert entry["info"]["format"] == "PDF"
    assert entry["info"]["name"] == "report.pdf"


def test_build_message_preview_lines_combines_text_and_media():
    photo_msg = _make_message(message="Привет", photo=object())
    lines, context = _build_message_preview_lines(photo_msg)
    assert "Привет" in lines
    assert any(line.startswith("🖼 изображение") for line in lines)
    assert context is not None and context.startswith("Привет")
    assert "[Фото: изображение" in context


def test_build_message_preview_lines_media_only_voice():
    voice = SimpleNamespace(duration=5, attributes=[])
    msg = _make_message(voice=voice)
    lines, context = _build_message_preview_lines(msg)
    assert lines == ["🎙 Голосовое сообщение 0:05"]
    assert context == "[Голосовое сообщение 0:05]"


def test_build_dialog_preview_text_includes_media_placeholders():
    msg = _make_message(photo=object())
    preview = _build_dialog_preview_text(msg)
    assert preview is not None and preview.startswith("🖼 изображение")


def test_build_dialog_preview_text_combines_text_and_media():
    voice = SimpleNamespace(duration=8, attributes=[])
    msg = _make_message(message="Привет", voice=voice)
    preview = _build_dialog_preview_text(msg)
    assert preview.startswith("Привет")
    assert "🎙 Голосовое сообщение 0:08" in preview


def test_get_media_type_and_info_voice():
    voice = SimpleNamespace(duration=12, attributes=[])
    msg = _make_message(voice=voice)
    assert get_media_type(msg) == "voice"
    info = get_media_info(msg)
    assert info is not None
    assert info["duration"] == "0:12"


def test_get_media_type_and_info_document():
    file_obj = SimpleNamespace(name="contract.docx", ext=".docx", mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", size=8192)
    document = SimpleNamespace(
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        attributes=[SimpleNamespace(file_name="contract.docx")],
        size=8192,
    )
    msg = _make_message(document=document, file=file_obj)
    assert get_media_type(msg) == "document"
    info = get_media_info(msg)
    assert info is not None
    assert info["format"] == "DOCX"
    assert info["name"] == "contract.docx"


def test_build_miniature_message_entry_voice_only():
    voice = SimpleNamespace(duration=5, attributes=[])
    msg = _make_message(voice=voice)
    entry = _build_miniature_message_entry(msg, chat_title="Rudi")
    assert entry is not None
    assert entry["direction"] == "in"
    assert entry["author"] == "Rudi"
    assert entry["has_media"] is True
    assert entry["media_type"] == "voice"
    assert entry["media_info"]["duration"] == "0:05"
    assert entry["preview_lines"] == ["🎙 Голосовое сообщение 0:05"]
    assert entry["text"] == "🎙 Голосовое сообщение 0:05"


def test_build_miniature_message_entry_text_and_media_outgoing():
    msg = _make_message(message="Привет", photo=object(), out=True)
    entry = _build_miniature_message_entry(msg, chat_title="Аня")
    assert entry is not None
    assert entry["direction"] == "out"
    assert entry["author"] == "Вы"
    assert entry["text"] == "Привет"
    assert entry["text_lines"] == ["Привет"]
    assert any(line.startswith("🖼 изображение") for line in entry["media_lines"])
    assert entry["has_media"] is True
    assert entry["media_type"] == "photo"
    assert entry["media_info"] is None
    assert any(line.startswith("🖼 изображение") for line in entry["preview_lines"][1:])


def test_build_warp_miniature_renders_multiline_messages():
    messages = [
        {
            "direction": "in",
            "author": "Аня",
            "text": "Привет",
            "text_lines": ["Привет"],
            "media_lines": ["🖼 изображение"],
            "preview_lines": ["Привет", "🖼 изображение"],
            "has_media": True,
        },
        {
            "direction": "out",
            "author": "Вы",
            "text": "",
            "text_lines": [],
            "media_lines": ["🎙 Голосовое сообщение 0:05"],
            "preview_lines": ["🎙 Голосовое сообщение 0:05"],
            "has_media": True,
        },
    ]
    body, _ = build_warp_miniature("Чат", "12:00", messages, chat_id=123)
    assert "Привет" in body
    assert "🖼 изображение" in body
    assert "🎙 Голосовое сообщение 0:05" in body
    assert "   🖼 изображение" in body
