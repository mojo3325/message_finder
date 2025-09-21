import asyncio

import pytest

from tg.handlers import warp


@pytest.fixture(autouse=True)
def _restore_to_thread(monkeypatch: pytest.MonkeyPatch):
    async def passthrough(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(warp.asyncio, "to_thread", passthrough)
    yield


def test_generate_media_analysis_for_audio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(warp.media_annotator, "transcribe_audio_base64", lambda data, fmt: "привет")
    payload = {"type": "voice", "data": "Zg==", "format": "mp3"}
    result = asyncio.run(warp._generate_media_analysis(payload))
    assert result == {"label": "Транскрипция", "text": "привет"}


def test_generate_media_analysis_for_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(warp.media_annotator, "describe_image_base64", lambda data, mime: "описание")
    payload = {"type": "photo", "data": "Zg==", "mime_type": "image/jpeg"}
    result = asyncio.run(warp._generate_media_analysis(payload))
    assert result == {"label": "Описание", "text": "описание"}


def test_generate_media_analysis_ignores_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(warp.media_annotator, "transcribe_audio_base64", lambda *a, **k: None)
    payload = {"type": "voice", "data": "Zg==", "format": "mp3"}
    result = asyncio.run(warp._generate_media_analysis(payload))
    assert result is None


def test_augment_context_with_media_analysis_appends():
    media_payloads = [
        {"analysis": {"label": "Транскрипция", "text": "привет"}},
        {"analysis": {"label": "Описание", "text": "Фото паспорта"}},
    ]
    result = warp._augment_context_with_media_analysis("[Голосовое]", media_payloads)
    assert (
        result
        == "[Голосовое] (Транскрипция: привет; Описание: Фото паспорта)"
    )


def test_augment_context_with_media_analysis_when_empty():
    media_payloads = [{"analysis": {"label": "Транскрипция", "text": "ок"}}]
    result = warp._augment_context_with_media_analysis("", media_payloads)
    assert result == "Транскрипция: ок"
