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

# Keys the reading may speak on, tried in turn. Comma-separated.
#
# What a reading spends is requests, not characters. A multi-voice reading cannot
# merge across speakers — every turn is a different voice — so 2,370 characters
# of dialogue leaves as 61 requests averaging 36 characters each, against a
# 700-character cap. Raising the cap changes nothing: there is never enough text
# in one voice to fill a request.
#
# What the service caps is how many are in flight, and that is counted per key —
# four on each of two keys went through together where eight on one did not. So
# a second key is a second set of slots, and it also keeps speech from competing
# with the casting model, which draws on NVIDIA_API_KEY. Left unset, speech runs
# on that same key exactly as before.
MAGPIE_API_KEYS = [
    key.strip()
    for key in os.getenv("MAGPIE_API_KEY", "").split(",")
    if key.strip()
] or ([NVIDIA_API_KEY] if NVIDIA_API_KEY else [])


# NVIDIA's Magpie TTS, spoken on MAGPIE_API_KEYS above. It
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
MAGPIE_TIMEOUT_SECONDS = float(os.getenv("MAGPIE_TIMEOUT_SECONDS", "90"))
# In flight at once, per key — four, which is what the service allows: four at a
# time are answered and the fifth is refused. The cap is counted per key, not per
# account (four on each of two keys went through together), so two keys carry
# eight and that is what a reading gets.
#
# It has to be this high. A reading is not slow because it is throttled, it is
# slow because each request takes about twenty seconds and a 30,000-character
# story makes eight hundred of them: at four in flight that renders at 0.89x real
# time, which is slower than the listener plays it, so the audio runs out mid-
# story. Eight is what puts the reading back in front.
#
# Refusals do not cost what they used to. A 429 now sends that one piece to
# edge-tts and nothing more — no give-up flag, no rest-of-the-story fallback —
# and edge streams, so it is if anything quicker.
MAX_CONCURRENT_MAGPIE = int(os.getenv("MAX_CONCURRENT_MAGPIE", "4"))
# Pieces kept in flight ahead of the one being written out, and the real ceiling
# on concurrency: only this many are ever in the air, so left below what the keys
# allow it is the prefetch throttling the reading rather than the service.
MAGPIE_PREFETCH = int(
    os.getenv("MAGPIE_PREFETCH", str(MAX_CONCURRENT_MAGPIE * 2))
)

# Casting is one blocking call before the first spoken word, so it wants to be
# short — but a timeout is not a saving. Every batch of a 15,000-character story
# hit 45 seconds on a slow host and the whole reading went out in stand-in
# voices. Waiting longer costs a pause; giving up costs the cast.
CASTING_TIMEOUT_SECONDS = float(os.getenv("CASTING_TIMEOUT_SECONDS", "120"))
# Cost is driven by how many model calls the text creates, not by its length:
# long prose is cheap, rapid dialogue is not. Capping the calls protects the
# free tier — but going over means the whole story is read in a single voice, so
# the cap has to clear the longest story anyone actually pastes. Raised with
# CAST_BATCH_SIZE coming down to 24: the same story now makes more calls, and
# the cap is about what the API can bear, not about the batch size.
CASTING_MAX_BATCHES = int(os.getenv("CASTING_MAX_BATCHES", "150"))
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
    return bool(MAGPIE_API_KEYS)
