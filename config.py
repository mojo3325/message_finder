import os


# ------------------------------
# Configuration
# ------------------------------
LOG_LEVEL = "INFO"
REQUEST_TIMEOUT_S = 20.0
RETRY_MAX_ATTEMPTS = 3
CLASSIFIER_MAX_OUTPUT_TOKENS = max(8, int(os.getenv("CLASSIFIER_MAX_OUTPUT_TOKENS", "32")))
CLASSIFIER_CONTEXT_CHAR_LIMIT = max(256, int(os.getenv("CLASSIFIER_CONTEXT_CHAR_LIMIT", "1200")))
CLASSIFIER_MESSAGE_CHAR_LIMIT = max(128, int(os.getenv("CLASSIFIER_MESSAGE_CHAR_LIMIT", "800")))
CLASSIFIER_BATCH_MAX_SIZE = max(1, int(os.getenv("CLASSIFIER_BATCH_MAX_SIZE", "8")))
CLASSIFIER_BATCH_FLUSH_MS = max(50, int(os.getenv("CLASSIFIER_BATCH_FLUSH_MS", "400")))


def _getenv_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return val.strip() if val is not None else default


def _is_running_in_docker() -> bool:
    try:
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    except Exception:
        return False


TELEGRAM_API_ID = _getenv_str("TELEGRAM_API_ID")
TELEGRAM_API_HASH = _getenv_str("TELEGRAM_API_HASH")
TELEGRAM_STRING_SESSION = _getenv_str("TELEGRAM_STRING_SESSION")
TELEGRAM_SESSION_PATH = _getenv_str("TELEGRAM_SESSION_PATH", "session")

# Accounts storage (per-user MTProto sessions)
ACCOUNTS_FILE = _getenv_str("ACCOUNTS_FILE", "data/accounts.json")
# Secret for encrypting StringSession in accounts storage
# Prefer setting via env ACCOUNTS_SECRET; fallback to TELEGRAM_API_HASH for dev
ACCOUNTS_SECRET = _getenv_str("ACCOUNTS_SECRET", TELEGRAM_API_HASH)


# LM Studio (local OpenAI-compatible) configuration (Docker-aware default)
LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL",
    "http://host.docker.internal:1234" if _is_running_in_docker() else "http://127.0.0.1:1234",
)
LMSTUDIO_MODEL = _getenv_str("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
LMSTUDIO_API_KEY = _getenv_str("LMSTUDIO_API_KEY", "not-needed")


# Classifier provider orchestration
ENABLE_GEMINI = _getenv_str("ENABLE_GEMINI", "true").lower() not in {"0", "false", "no"}
PROVIDER_ORDER = tuple(
    provider.strip() for provider in os.getenv("PROVIDER_ORDER", "gemini,cerebras,local").split(",") if provider.strip()
)


# Cerebras (OpenAI-compatible) configuration for classification
CEREBRAS_BASE_URL = _getenv_str("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = _getenv_str("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_API_KEY = _getenv_str("CEREBRAS_API_KEY")
# Extra API key to use when primary Cerebras key hits TPD
CEREBRAS_EXTRA_API = _getenv_str("CEREBRAS_EXTRA_API_KEY", _getenv_str("CEREBRAS_EXTRA_API"))


# Gemini API (primary for classifier and replies)
GEMINI_BASE_URL = _getenv_str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_API_KEY = _getenv_str("GEMINI_API_KEY")
GEMINI_MODEL = _getenv_str("GEMINI_MODEL", "gemma-3-27b-it")
GEMINI_REPLY_MODEL = _getenv_str("GEMINI_REPLY_MODEL", "gemini-2.5-flash")


# Bot API for notifications
# message_finder bot token (aka message_fuckerr_bot)
MESSAGE_FUCKERR_TOKEN = _getenv_str("MESSAGE_FINDER_BOT_TOKEN", _getenv_str("MESSAGE_FUCKERR_TOKEN"))
# warp_chat_bot
WARP_CHAT_BOT_TOKEN = _getenv_str("WARP_CHAT_BOT_TOKEN")
WARP_BOT_USERNAME = _getenv_str("WARP_BOT_USERNAME", "warp_chat_bot")


GROQ_API_KEY = _getenv_str("GROQ_API_KEY")


# Groq OpenAI-compatible settings for reply generation
GROQ_BASE_URL = _getenv_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_REPLY_MODEL = _getenv_str("GROQ_REPLY_MODEL", "openai/gpt-oss-120b")

# Portrait generation fallback model (does not support reasoning)
PORTRAIT_FALLBACK_MODEL = _getenv_str("PORTRAIT_FALLBACK_MODEL", "moonshotai/kimi-k2-instruct-0905")


# Optional: limit processing to a single chat during tests (disabled by default)
_TEST_ONLY_CHAT_ID_RAW = os.getenv("TEST_ONLY_CHAT_ID", "").strip()
TEST_ONLY_CHAT_ID = int(_TEST_ONLY_CHAT_ID_RAW) if _TEST_ONLY_CHAT_ID_RAW else None


# Rate limits (Cerebras quotas)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
RATE_LIMIT_RPH = int(os.getenv("RATE_LIMIT_RPH", "900"))
RATE_LIMIT_TPM = int(os.getenv("RATE_LIMIT_TPM", "60000"))


# Gemini quotas (per model docs)
GEMINI_RATE_RPM = int(os.getenv("GEMINI_RATE_RPM", "30"))
GEMINI_RATE_TPM = int(os.getenv("GEMINI_RATE_TPM", "15000"))  # input tokens per minute
GEMINI_RATE_RPD = int(os.getenv("GEMINI_RATE_RPD", "14400"))
GEMINI_RPM_GUARD = float(os.getenv("GEMINI_RPM_GUARD", "0.9"))
GEMINI_TPM_GUARD = float(os.getenv("GEMINI_TPM_GUARD", "0.9"))
GEMINI_RPD_GUARD = float(os.getenv("GEMINI_RPD_GUARD", "0.95"))


# Groq quotas (conservative defaults; adjustable via env)
GROQ_RATE_RPM = int(os.getenv("GROQ_RATE_RPM", "20"))
GROQ_RATE_TPM = int(os.getenv("GROQ_RATE_TPM", "60000"))
GROQ_RATE_RPD = int(os.getenv("GROQ_RATE_RPD", "1000"))


# Supabase configuration (subscribers storage)
SUPABASE_URL = _getenv_str("SUPABASE_URL")
SUPABASE_ANON_KEY = _getenv_str("SUPABASE_ANON_KEY")
# Warp Chat configuration
# Context messages to index for generation (N)
INDEX_CONTEXT_LIMIT = int(os.getenv("INDEX_CONTEXT_LIMIT", "40"))
# Alias for clarity in Warp Chat code
WARP_CONTEXT_LIMIT = int(os.getenv("WARP_CONTEXT_LIMIT", str(INDEX_CONTEXT_LIMIT)))
# Number of last messages to show in miniature
WARP_MINIATURE_LAST = int(os.getenv("WARP_MINIATURE_LAST", "4"))
# Page size for private chats list
WARP_LIST_PAGE_SIZE = int(os.getenv("WARP_LIST_PAGE_SIZE", "10"))


