import asyncio
from typing import Optional

from tg.handlers import warp


def test_ensure_supported_audio_bytes_passthrough_mp3():
    sample = b"abc"
    result = asyncio.run(warp._ensure_supported_audio_bytes(sample, "mp3", "audio/mpeg"))
    assert result == (sample, "mp3", "audio/mpeg")


def test_ensure_supported_audio_bytes_converts_when_needed(monkeypatch):
    sample = b"raw"

    def fake_transcode(data: bytes, fmt: Optional[str]) -> Optional[bytes]:
        assert data == sample
        assert fmt == "ogg"
        return b"wav-data"

    monkeypatch.setattr(warp, "_transcode_audio_bytes_to_wav_sync", fake_transcode)

    result = asyncio.run(warp._ensure_supported_audio_bytes(sample, None, "audio/ogg"))
    assert result == (b"wav-data", "wav", "audio/wav")


def test_ensure_supported_audio_bytes_returns_none_when_conversion_fails(monkeypatch):
    def fake_transcode(_data: bytes, _fmt: Optional[str]) -> Optional[bytes]:
        return None

    monkeypatch.setattr(warp, "_transcode_audio_bytes_to_wav_sync", fake_transcode)

    result = asyncio.run(warp._ensure_supported_audio_bytes(b"raw", "oga", "audio/ogg"))
    assert result is None
