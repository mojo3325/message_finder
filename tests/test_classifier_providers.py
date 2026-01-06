import importlib
from typing import Any

import pytest


def _reload_classifier(monkeypatch: pytest.MonkeyPatch, **env_overrides: str):
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    import config as config_module
    import services.clients as clients_module
    import services.classifier as classifier_module

    importlib.reload(config_module)
    importlib.reload(clients_module)
    classifier = importlib.reload(classifier_module)
    return classifier


class FakeReservation:
    def __init__(self) -> None:
        self.committed = False
        self.released = False

    def commit(self, *_args: Any, **_kwargs: Any) -> None:
        self.committed = True

    def release(self) -> None:
        self.released = True


def test_gemini_used_first(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _reload_classifier(
        monkeypatch,
        ENABLE_GEMINI="true",
        PROVIDER_ORDER="gemini,cerebras,local",
    )

    monkeypatch.setattr(classifier, "load_feedback_examples", lambda: [])

    def _reserve(*_args: Any, **_kwargs: Any) -> tuple[FakeReservation, None]:
        return FakeReservation(), None

    monkeypatch.setattr(classifier._gemini_quota_manager, "reserve", _reserve)  # type: ignore[attr-defined]

    gemini_calls: list[str] = []

    def _fake_gemini(
        _prompt: str,
        _payload_json: str,
        items: list[classifier._BatchWorkItem],
        reservation: FakeReservation,
        _est_prompt: int,
        _est_output: int,
    ) -> dict[str, str]:
        reservation.commit(_est_prompt, _est_output)
        gemini_calls.append("gemini")
        return {item.request_id: "1" for item in items}

    monkeypatch.setattr(classifier, "_call_gemini_batch", _fake_gemini)
    monkeypatch.setattr(
        classifier,
        "_call_cerebras_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Cerebras should not be used")),
    )
    monkeypatch.setattr(
        classifier,
        "_call_local_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Local fallback should not be used")),
    )

    result = classifier.classify_with_openai_sync("primary path")

    assert result == "1"
    assert gemini_calls == ["gemini"]


def test_cerebras_used_when_gemini_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _reload_classifier(
        monkeypatch,
        ENABLE_GEMINI="true",
        PROVIDER_ORDER="gemini,cerebras,local",
    )

    monkeypatch.setattr(classifier, "load_feedback_examples", lambda: [])
    monkeypatch.setattr(
        classifier._gemini_quota_manager,
        "reserve",
        lambda *_args, **_kwargs: (None, "rpm_guard"),
    )  # type: ignore[attr-defined]

    cerebras_reasons: list[str | None] = []

    def _fake_cerebras(
        _prompt: str,
        _payload_json: str,
        items: list[classifier._BatchWorkItem],
        *,
        fallback_reason: str | None = None,
    ) -> dict[str, str]:
        cerebras_reasons.append(fallback_reason)
        return {item.request_id: "1" for item in items}

    monkeypatch.setattr(classifier, "_call_cerebras_batch", _fake_cerebras)
    monkeypatch.setattr(
        classifier,
        "_call_local_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Local fallback should not run first")),
    )

    result = classifier.classify_with_openai_sync("guard path")

    assert result == "1"
    assert cerebras_reasons == ["gemini_rpm_guard"]


def test_local_used_when_cerebras_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _reload_classifier(
        monkeypatch,
        ENABLE_GEMINI="true",
        PROVIDER_ORDER="gemini,cerebras,local",
    )

    monkeypatch.setattr(classifier, "load_feedback_examples", lambda: [])
    monkeypatch.setattr(
        classifier._gemini_quota_manager,
        "reserve",
        lambda *_args, **_kwargs: (None, "tpm_guard"),
    )  # type: ignore[attr-defined]

    def _raise_cerebras(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("cerebras down")

    monkeypatch.setattr(classifier, "_call_cerebras_batch", _raise_cerebras)

    local_reasons: list[str | None] = []

    def _fake_local(
        _prompt: str,
        _payload_json: str,
        items: list[classifier._BatchWorkItem],
        *,
        fallback_reason: str | None = None,
    ) -> dict[str, str]:
        local_reasons.append(fallback_reason)
        return {item.request_id: "1" for item in items}

    monkeypatch.setattr(classifier, "_call_local_batch", _fake_local)

    result = classifier.classify_with_openai_sync("local path")

    assert result == "1"
    assert local_reasons == ["cerebras_error"]
