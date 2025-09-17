def is_gemini_transient_or_rate_error(e: Exception) -> bool:
    try:
        msg = str(e).lower()
    except Exception:
        return False
    markers = [
        "429",
        "rate limit",
        "too many requests",
        "quota",
        "resource exhausted",
        "retry-after",
        "timeout",
        "timed out",
        "deadline",
        "connection reset",
        "temporarily unavailable",
        "unavailable",
        "backend error",
    ]
    return any(m in msg for m in markers)


def is_gemini_quota_exhausted_error(e: Exception) -> bool:
    try:
        msg = str(e).lower()
    except Exception:
        return False
    quota_markers = [
        "exceeded your current quota",
        "quota",
        "resource_exhausted",
        "resource exhausted",
        "quota failure",
        "retryinfo",
        "generativelanguage.googleapis.com",
        "ai.google.dev/gemini-api/docs/rate-limits",
    ]
    return any(m in msg for m in quota_markers)


def is_cerebras_tpd_limit_error(e: Exception) -> bool:
    try:
        msg = str(e).lower()
    except Exception:
        return False
    return (
        "token_quota_exceeded" in msg
        or "too_many_tokens_error" in msg
        or "tokens per day" in msg
        or "limit hit" in msg
        or "http 429" in msg
        or "rate limit" in msg
    )



def is_groq_tpd_limit_error(e: Exception) -> bool:
    try:
        msg = str(e).lower()
    except Exception:
        return False
    return (
        "rate limit reached for model" in msg
        and "tokens per day" in msg
        or "tpd" in msg
        or "rate_limit_exceeded" in msg
    )

