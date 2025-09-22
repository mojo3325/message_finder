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


def test_mistral_fallback_engaged_before_local(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = _reload_with_env(monkeypatch)

    monkeypatch.setattr(classifier, "load_feedback_examples", lambda: [])
    monkeypatch.setattr(classifier.gemini_rate_limiter, "acquire_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(classifier.time, "sleep", lambda *_args, **_kwargs: None)

    class FailingCerebrasClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        @staticmethod
        def _create(*_args: object, **_kwargs: object) -> NoReturn:
            raise Exception("Error code: 429 - {'code': 'token_quota_exceeded'}")

    class FailingGeminiClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        @staticmethod
        def _create(*_args: object, **_kwargs: object) -> NoReturn:
            raise Exception("quota failure generativelanguage.googleapis.com")

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

    monkeypatch.setattr(classifier, "get_cerebras_client", lambda *_args, **_kwargs: FailingCerebrasClient())
    monkeypatch.setattr(classifier, "get_gemini_client", lambda: FailingGeminiClient())

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
    assert mistral_calls, "Mistral client should be used prior to the LM Studio fallback"
    assert classifier._mistral_fallback_active is True
    assert classifier._local_fallback_active is False
