import httpx
from openai import OpenAI

from config import (
    REQUEST_TIMEOUT_S,
    CEREBRAS_BASE_URL,
    CEREBRAS_API_KEY,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_API_KEY,
    GROQ_BASE_URL,
    GROQ_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_API_KEY,
)


_oa_client: OpenAI | None = None
_oa_client_key: str | None = None
_groq_client: OpenAI | None = None
_gemini_client: OpenAI | None = None
_lm_client: OpenAI | None = None
_http_client: httpx.AsyncClient | None = None


def _normalize_base_url(raw: str) -> str:
    base = (raw or "http://127.0.0.1:1234").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def get_cerebras_client(api_key: str | None = None) -> OpenAI:
    global _oa_client, _oa_client_key
    desired_key = api_key or CEREBRAS_API_KEY
    if _oa_client is not None and _oa_client_key == desired_key:
        return _oa_client
    _oa_client = OpenAI(
        base_url=_normalize_base_url(CEREBRAS_BASE_URL),
        api_key=desired_key,
        timeout=REQUEST_TIMEOUT_S,
    )
    _oa_client_key = desired_key
    return _oa_client


def get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    _groq_client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY, timeout=REQUEST_TIMEOUT_S)
    return _groq_client


def get_gemini_client() -> OpenAI:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    _gemini_client = OpenAI(
        base_url=_normalize_base_url(GEMINI_BASE_URL), api_key=GEMINI_API_KEY, timeout=REQUEST_TIMEOUT_S
    )
    return _gemini_client


def get_lmstudio_client() -> OpenAI:
    global _lm_client
    if _lm_client is not None:
        return _lm_client
    _lm_client = OpenAI(
        base_url=_normalize_base_url(LMSTUDIO_BASE_URL), api_key=LMSTUDIO_API_KEY, timeout=REQUEST_TIMEOUT_S, max_retries=1
    )
    return _lm_client


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is not None:
        return _http_client
    _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        try:
            await _http_client.aclose()
        finally:
            _http_client = None


