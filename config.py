import os


# ------------------------------
# Configuration
# ------------------------------
LOG_LEVEL = "INFO"
REQUEST_TIMEOUT_S = 20.0
RETRY_MAX_ATTEMPTS = 3


def _is_running_in_docker() -> bool:
    try:
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    except Exception:
        return False


# Hardcoded credentials per user request (TEST USE ONLY)
TELEGRAM_API_ID = ""
TELEGRAM_API_HASH = ""
TELEGRAM_STRING_SESSION = (
    "REDACTED_STRING_SESSION="
)
TELEGRAM_SESSION_PATH = "session"


# LM Studio (local OpenAI-compatible) configuration (Docker-aware default)
LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL",
    "http://host.docker.internal:1234" if _is_running_in_docker() else "http://127.0.0.1:1234",
)
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "not-needed")


# Cerebras (OpenAI-compatible) configuration for classification
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_API_KEY = "REDACTED_CEREBRAS_KEY"
# Extra API key to use when primary Cerebras key hits TPD
CEREBRAS_EXTRA_API = "REDACTED_CEREBRAS_KEY"


# Gemini API (primary for replies, fallback for classifier)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = "REDACTED_GEMINI_KEY"
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_REPLY_MODEL = "gemini-2.5-flash"


# Bot API for notifications
TELEGRAM_BOT_TOKEN = "REDACTED_TELEGRAM_BOT_TOKEN"


GROQ_API_KEY = "REDACTED_GROQ_KEY"


# Groq OpenAI-compatible settings for reply generation
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_REPLY_MODEL = "openai/gpt-oss-120b"

# Portrait generation fallback model (does not support reasoning)
PORTRAIT_FALLBACK_MODEL = os.getenv("PORTRAIT_FALLBACK_MODEL", "moonshotai/kimi-k2-instruct-0905")


# Optional: limit processing to a single chat during tests (disabled by default)
_TEST_ONLY_CHAT_ID_RAW = os.getenv("TEST_ONLY_CHAT_ID", "").strip()
TEST_ONLY_CHAT_ID = int(_TEST_ONLY_CHAT_ID_RAW) if _TEST_ONLY_CHAT_ID_RAW else None


# Rate limits (Cerebras quotas)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
RATE_LIMIT_RPH = int(os.getenv("RATE_LIMIT_RPH", "900"))
RATE_LIMIT_TPM = int(os.getenv("RATE_LIMIT_TPM", "60000"))


# Gemini quotas (per model docs)
GEMINI_RATE_RPM = int(os.getenv("GEMINI_RATE_RPM", "15"))
GEMINI_RATE_TPM = int(os.getenv("GEMINI_RATE_TPM", "250000"))  # input tokens per minute
GEMINI_RATE_RPD = int(os.getenv("GEMINI_RATE_RPD", "1000"))


# Groq quotas (conservative defaults; adjustable via env)
GROQ_RATE_RPM = int(os.getenv("GROQ_RATE_RPM", "20"))
GROQ_RATE_TPM = int(os.getenv("GROQ_RATE_TPM", "60000"))
GROQ_RATE_RPD = int(os.getenv("GROQ_RATE_RPD", "1000"))


# Supabase configuration (subscribers storage)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wswatojrgeknsfqujjvi.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "REDACTED_JWT",
)


