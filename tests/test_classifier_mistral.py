from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import NoReturn

import pytest


def _reload_with_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    import config as config_module
    import services.clients as clients_module
    import services.classifier as classifier_module

    importlib.reload(config_module)
    importlib.reload(clients_module)
    classifier = importlib.reload(classifier_module)

    return classifier


def test_mistral_used_before_cerebras(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _reload_with_env(monkeypatch)

    monkeypatch.setattr(classifier, "load_feedback_examples", lambda: [])
    monkeypatch.setattr(classifier.gemini_rate_limiter, "acquire_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(classifier.time, "sleep", lambda *_args, **_kwargs: None)

    class SuccessfulMistralClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        @staticmethod
        def _create(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="1"))],
                usage=SimpleNamespace(prompt_tokens=32, completion_tokens=8),
            )

    mistral_calls: list[bool] = []

    monkeypatch.setattr(
        classifier,
        "get_cerebras_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Cerebras should not be called when Mistral is available")),
    )

    def _fake_get_mistral_client() -> SuccessfulMistralClient:
        mistral_calls.append(True)
        return SuccessfulMistralClient()

    monkeypatch.setattr(classifier, "get_mistral_client", _fake_get_mistral_client)
    monkeypatch.setattr(
        classifier,
        "get_lmstudio_client",
        lambda: (_ for _ in ()).throw(AssertionError("LM Studio should not be used when Mistral is available")),
    )

    result = classifier.classify_with_openai_sync("test message")

    assert result == "1"
    assert mistral_calls == [True]
    assert classifier._mistral_fallback_active is True
    assert classifier._local_fallback_active is False


def test_cerebras_used_when_mistral_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _reload_with_env(monkeypatch)

    monkeypatch.setattr(classifier, "load_feedback_examples", lambda: [])
    monkeypatch.setattr(classifier.gemini_rate_limiter, "acquire_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(classifier.time, "sleep", lambda *_args, **_kwargs: None)

    class FailingMistralClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        @staticmethod
        def _create(*_args: object, **_kwargs: object) -> NoReturn:
            raise RuntimeError("bad request")

    class SuccessfulCerebrasClient:
        def __init__(self, calls: list[bool]) -> None:
            self._calls = calls
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, *_args: object, **_kwargs: object):
            self._calls.append(True)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="1"))],
            )

    cerebras_calls: list[bool] = []

    monkeypatch.setattr(classifier, "get_mistral_client", lambda: FailingMistralClient())
    monkeypatch.setattr(classifier, "get_cerebras_client", lambda *_args, **_kwargs: SuccessfulCerebrasClient(cerebras_calls))
    monkeypatch.setattr(
        classifier,
        "get_lmstudio_client",
        lambda: (_ for _ in ()).throw(AssertionError("LM Studio should not be used before Cerebras fallback")),
    )

    result = classifier.classify_with_openai_sync("test message")

    assert result == "1"
    assert cerebras_calls, "Cerebras should be invoked when Mistral repeatedly fails"
    assert classifier._local_fallback_active is False
