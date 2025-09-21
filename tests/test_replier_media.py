from types import SimpleNamespace

import pytest

import services.replier as replier
from services.replier import _media_to_content_parts, generate_reply_sync


def test_media_to_content_parts_adds_supported_audio():
    media = {
        "type": "voice",
        "data": "YmFzZTY0",  # "base64"
        "format": "mp3",
    }
    parts = _media_to_content_parts(media)
    assert parts == [
        {"type": "input_audio", "input_audio": {"data": "YmFzZTY0", "format": "mp3"}}
    ]


def test_media_to_content_parts_normalizes_mpeg_to_mp3():
    media = {
        "type": "audio",
        "data": "ZGF0YQ==",
        "format": "mpeg",
    }
    parts = _media_to_content_parts(media)
    assert parts and parts[0]["input_audio"]["format"] == "mp3"


def test_media_to_content_parts_skips_unsupported_audio():
    media = {
        "type": "voice",
        "data": "AAAA",
        "format": "oga",
    }
    parts = _media_to_content_parts(media)
    assert parts == []


def test_generate_reply_includes_media_analysis(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, list] = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    dummy_client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    monkeypatch.setattr(replier, "get_gemini_client", lambda: dummy_client)
    monkeypatch.setattr(replier.gemini_rate_limiter, "acquire_sync", lambda *a, **k: None)

    context = {
        "text": "",
        "entries": [
            {
                "role": "USER",
                "identifier": "@friend",
                "text": "[Голосовое сообщение 0:06] (Транскрипция: привет)",
                "media": [
                    {
                        "display": "🎙 Голосовое сообщение 0:06",
                        "context": "[Голосовое сообщение 0:06]",
                        "type": "voice",
                        "data": "Zg==",
                        "format": "mp3",
                        "analysis": {"label": "Транскрипция", "text": "привет"},
                    }
                ],
            }
        ],
    }

    result = generate_reply_sync("", context=context)
    assert result == "ok"

    messages = captured["messages"]
    assert len(messages) == 2
    user_content = messages[1]["content"]
    text_snippets = [
        part.get("text", "") for part in user_content if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert any("Транскрипция: привет" in snippet for snippet in text_snippets)
