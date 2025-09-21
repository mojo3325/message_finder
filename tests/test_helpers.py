from __future__ import annotations

import asyncio

import pytest

from tg import helpers


def test_shorten_truncates_with_ellipsis() -> None:
    long_text = "foo    bar   baz   qux"
    assert helpers.shorten(long_text, limit=10) == "foo bar b…"


def test_shorten_removes_extra_whitespace_without_truncation() -> None:
    assert helpers.shorten("   a   b   c   ", limit=50) == "a b c"


def test_parse_link_to_ids_username() -> None:
    assert helpers.parse_link_to_ids("https://t.me/u/1") == ("u", 1)


def test_parse_link_to_ids_private_chat() -> None:
    assert helpers.parse_link_to_ids("https://t.me/c/100123/45") == (-100100123, 45)


def test_build_loading_frame_contains_header_and_bar() -> None:
    header = "Работа"
    frame = helpers.build_loading_frame(0, header)
    lines = frame.split("\n")
    assert len(lines) == 3
    assert lines[0].startswith(f"<b>{header}")
    assert lines[1] == "<b>━━━━━━━━━━━━━━━━━━━━</b>"
    assert lines[2].startswith("<code>")


def test_safe_delete_messages_no_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def fake_delete(chat_id: int, message_id: int) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(helpers.bot_api, "bot_delete_message", fake_delete)
    asyncio.run(helpers.safe_delete_messages(123, [], attempts=2))
    assert called is False


def test_safe_delete_messages_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def flaky_delete(chat_id: int, message_id: int) -> None:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("temporary failure")

    monkeypatch.setattr(helpers.bot_api, "bot_delete_message", flaky_delete)
    asyncio.run(helpers.safe_delete_messages(555, [42], attempts=3, retry_delay=0))
    assert call_count == 3
