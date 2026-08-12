import os
from dotenv import load_dotenv

load_dotenv()

# NVIDIA NIM speaks the OpenAI chat-completions dialect, so a plain HTTP call is
# all this needs — no extra SDK.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,https://suno-story.vercel.app",
    ).split(",")
    if origin.strip()
]

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

# Casting is one blocking LLM call before the first audio byte, so keep it short.
CASTING_TIMEOUT_SECONDS = float(os.getenv("CASTING_TIMEOUT_SECONDS", "45"))
CASTING_MAX_CHARS = int(os.getenv("CASTING_MAX_CHARS", "20000"))
TRANSLITERATE_MAX_CHARS = int(os.getenv("TRANSLITERATE_MAX_CHARS", "30000"))


# Finished audio is kept only long enough for the listener to press Download;
# past that it is dead weight. Nothing survives a restart either way.
AUDIO_TTL_SECONDS = float(os.getenv("AUDIO_TTL_SECONDS", "900"))
AUDIO_SWEEP_SECONDS = float(os.getenv("AUDIO_SWEEP_SECONDS", "120"))
AUDIO_CACHE_MAX_MB = float(os.getenv("AUDIO_CACHE_MAX_MB", "300"))

# Both upstreams are shared, free-tier services that get slower — or start
# refusing — when hit hard, and one listener can be mid-story for minutes. These
# caps queue work instead of letting a handful of listeners degrade each other.
MAX_CONCURRENT_TTS = int(os.getenv("MAX_CONCURRENT_TTS", "6"))
MAX_CONCURRENT_CASTING = int(os.getenv("MAX_CONCURRENT_CASTING", "3"))


def casting_enabled() -> bool:
    return bool(NVIDIA_API_KEY)
