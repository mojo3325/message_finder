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
TELEGRAM_API_ID = "28738574"
TELEGRAM_API_HASH = "343d6e8a20de4a2f3cc265eafdd64f71"
TELEGRAM_STRING_SESSION = (
    "1AZWarzkBuzIOLASsWrjMkxeeZ5PaJoMtJSZSajB2lJEXQivilsJJHIPX6JQgSFfIVfi0dTf-LbaBHk_8N_kUyXWljBgsAPJVOL6qtTX1fgJAxTTNfkTZQq049Ad9PrlLwfU4AbNXgyYAfXV_tLobDQLALoTssGcqxXulW6b556iDc0xf7msg-QO8OIVzLI28ASxtXdbfTMrBOQ9gp3xaV5oZyp3XNCic9vtRYqPWxkCRBnlM4m8RNwaUZo86rYldDCiugbzRNZwrfqq9VYZtKQ2fTO5FFUKnMZADaBAePy7fsAKpee1IXraaBjMQeRUR6DM8iXAdqkV-ajEkyUPNmMS3IMomhOE="
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
CEREBRAS_API_KEY = "csk-5njr2er53p5dwh6ymrfj4jjf9tctx9rpr3xtw4mc6te6dd38"
# Extra API key to use when primary Cerebras key hits TPD
CEREBRAS_EXTRA_API = "csk-6p5w4vjhnk42vnvmwjncn85cvwrp3dj84n5fj26m5n56phex"


# Gemini API (primary for replies, fallback for classifier)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = "AIzaSyD9Ig7Isn6THvmClllKTUucoMWPuEpdziU"
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_REPLY_MODEL = "gemini-2.5-flash"


# Bot API for notifications
TELEGRAM_BOT_TOKEN = "8255044221:AAH_MKTuXbuWoLn0OlJRr6amaXMbdQ3jUlg"


GROQ_API_KEY = "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"


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
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indzd2F0b2pyZ2VrbnNmcXVqanZpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2Mjc3MTgsImV4cCI6MjA3MzIwMzcxOH0.zj6HBxOIkoJsZ3qByQfipeoR4S-EbAMH7tJ__SLajtI",
)


