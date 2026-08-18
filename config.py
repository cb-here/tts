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


# NVIDIA's Magpie TTS, reached through the same key as the casting model. It
# reads more naturally than edge-tts and about 13x faster than real time, but it
# hands back a whole WAV rather than a stream, so the reading is cut into pieces
# and the pieces are synthesised ahead of the playhead.
MAGPIE_FUNCTION_ID = os.getenv(
    "MAGPIE_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969"
)
MAGPIE_URL = os.getenv(
    "MAGPIE_URL",
    f"https://{MAGPIE_FUNCTION_ID}.invocation.api.nvcf.nvidia.com/v1/audio/synthesize",
)
# 24000 is rejected outright and 44100 doubles the reply for no audible gain at
# 64 kbps mono.
MAGPIE_SAMPLE_RATE = int(os.getenv("MAGPIE_SAMPLE_RATE", "22050"))
# Two ceilings meet here. The model refuses text over 2000 of its own units,
# which Devanagari hits at about 1550 characters, and the gateway drops any
# reply over 4 MB, which 22 kHz audio hits at about 1000. Staying well under
# both also keeps the first piece quick, since nothing plays until it lands.
MAGPIE_MAX_CHARS = int(os.getenv("MAGPIE_MAX_CHARS", "700"))
# Nothing plays until the first piece lands, so the reading opens with a short
# one — about a second and a half — and settles into full-length pieces after.
MAGPIE_OPENING_CHARS = int(os.getenv("MAGPIE_OPENING_CHARS", "220"))
# Pieces kept in flight ahead of the one being written out.
MAGPIE_PREFETCH = int(os.getenv("MAGPIE_PREFETCH", "3"))
MAGPIE_TIMEOUT_SECONDS = float(os.getenv("MAGPIE_TIMEOUT_SECONDS", "90"))
# Consecutive refusals before a reading stops asking Magpie at all. The free
# tier tends to refuse in stretches rather than one-offs, and each refused piece
# costs its whole retry budget before falling back regardless.
MAGPIE_GIVE_UP_AFTER = int(os.getenv("MAGPIE_GIVE_UP_AFTER", "3"))
# The free tier answers a short burst and then throttles, and at 13x real time
# there is no need to push it — two in flight already outruns the listener.
MAX_CONCURRENT_MAGPIE = int(os.getenv("MAX_CONCURRENT_MAGPIE", "2"))

# Casting is one blocking LLM call before the first audio byte, so keep it short.
CASTING_TIMEOUT_SECONDS = float(os.getenv("CASTING_TIMEOUT_SECONDS", "45"))
# Cost is driven by how many model calls the text creates, not by its length:
# long prose is cheap, rapid dialogue is not. Capping the calls protects the
# free tier — but going over means the whole story is read in a single voice, so
# the cap has to clear the longest story anyone actually pastes. Measured at
# CAST_BATCH_SIZE 60: 100,000 characters of dialogue-heavy Hindi is 35 calls.
CASTING_MAX_BATCHES = int(os.getenv("CASTING_MAX_BATCHES", "60"))
# gpt-oss models think before answering. "medium" and "high" place lines with
# their speakers more reliably, but the thinking happens on the one call the
# first spoken word waits for, so the pause before audio roughly doubles a step.
CASTING_REASONING_EFFORT = os.getenv("CASTING_REASONING_EFFORT", "low")
TRANSLITERATE_MAX_CHARS = int(os.getenv("TRANSLITERATE_MAX_CHARS", "30000"))


# A reading is stitched from pieces that were each synthesised alone, so the
# silence between them is ours to place. Left at zero the last word of one piece
# runs straight into the first of the next: no breath at a full stop, no beat
# before someone answers, and the whole thing gallops. Lengths are taken from
# the punctuation the piece ended on.
PAUSE_SENTENCE_SECONDS = float(os.getenv("PAUSE_SENTENCE_SECONDS", "0.45"))
PAUSE_CLAUSE_SECONDS = float(os.getenv("PAUSE_CLAUSE_SECONDS", "0.18"))
PAUSE_DEFAULT_SECONDS = float(os.getenv("PAUSE_DEFAULT_SECONDS", "0.25"))
# Held before a different voice comes in, whatever the punctuation said.
PAUSE_SPEAKER_SECONDS = float(os.getenv("PAUSE_SPEAKER_SECONDS", "0.6"))


# How long a parked text stays playable, counted from the last request for it.
# It has to outlast the longest story anyone will read: 50,000 characters is
# over an hour of speech, and losing the session mid-reading sends the player
# back to the start.
SESSION_TTL_SECONDS = float(os.getenv("SESSION_TTL_SECONDS", str(6 * 3600)))


# Finished audio is kept only long enough for the listener to press Download;
# past that it is dead weight. Nothing survives a restart either way.
AUDIO_TTL_SECONDS = float(os.getenv("AUDIO_TTL_SECONDS", "7200"))
AUDIO_SWEEP_SECONDS = float(os.getenv("AUDIO_SWEEP_SECONDS", "120"))
AUDIO_CACHE_MAX_MB = float(os.getenv("AUDIO_CACHE_MAX_MB", "300"))

# Both upstreams are shared, free-tier services that get slower — or start
# refusing — when hit hard, and one listener can be mid-story for minutes. These
# caps queue work instead of letting a handful of listeners degrade each other.
MAX_CONCURRENT_TTS = int(os.getenv("MAX_CONCURRENT_TTS", "6"))
MAX_CONCURRENT_CASTING = int(os.getenv("MAX_CONCURRENT_CASTING", "3"))


def casting_enabled() -> bool:
    return bool(NVIDIA_API_KEY)


def magpie_enabled() -> bool:
    # Same key as casting, so a deployment either has both or neither.
    return bool(NVIDIA_API_KEY)
