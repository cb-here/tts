import os
from dotenv import load_dotenv

load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()

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

CASTING_TIMEOUT_SECONDS = float(os.getenv("CASTING_TIMEOUT_SECONDS", "90"))
CASTING_MAX_BATCHES = int(os.getenv("CASTING_MAX_BATCHES", "150"))
CASTING_REASONING_EFFORT = os.getenv("CASTING_REASONING_EFFORT", "low")
TRANSLITERATE_MAX_CHARS = int(os.getenv("TRANSLITERATE_MAX_CHARS", "30000"))


PAUSE_SENTENCE_SECONDS = float(os.getenv("PAUSE_SENTENCE_SECONDS", "0.45"))
PAUSE_CLAUSE_SECONDS = float(os.getenv("PAUSE_CLAUSE_SECONDS", "0.18"))
PAUSE_DEFAULT_SECONDS = float(os.getenv("PAUSE_DEFAULT_SECONDS", "0.25"))
PAUSE_SPEAKER_SECONDS = float(os.getenv("PAUSE_SPEAKER_SECONDS", "0.6"))


SESSION_TTL_SECONDS = float(os.getenv("SESSION_TTL_SECONDS", str(6 * 3600)))


AUDIO_TTL_SECONDS = float(os.getenv("AUDIO_TTL_SECONDS", "7200"))
AUDIO_SWEEP_SECONDS = float(os.getenv("AUDIO_SWEEP_SECONDS", "120"))
AUDIO_CACHE_MAX_MB = float(os.getenv("AUDIO_CACHE_MAX_MB", "300"))

MAX_CONCURRENT_TTS = int(os.getenv("MAX_CONCURRENT_TTS", "6"))
EDGE_BOUNDARY = os.getenv("EDGE_BOUNDARY", "WordBoundary")

GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
GOOGLE_TIMEOUT_SECONDS = float(os.getenv("GOOGLE_TIMEOUT_SECONDS", "180"))
MAX_CONCURRENT_CASTING = int(os.getenv("MAX_CONCURRENT_CASTING", "3"))


def casting_enabled() -> bool:
    return bool(NVIDIA_API_KEY)
