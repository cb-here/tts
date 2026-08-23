import asyncio
import base64
import json
import logging
import re
from config import (
    GOOGLE_CREDENTIALS_FILE,
    GOOGLE_CREDENTIALS_JSON,
    GOOGLE_TIMEOUT_SECONDS,
)
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

CHIRP_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
GEMINI_ENDPOINT = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"

GEMINI_PREFIX = "gemini/"
GEMINI_MODEL = "gemini-2.5-flash-tts"

GOOGLE_SAMPLE_RATE = 24000
GOOGLE_BITRATE = 32000

CHIRP_MAX_BYTES = 4500
GEMINI_MAX_BYTES = 3400

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

_token_lock = asyncio.Lock()


def google_enabled() -> bool:
    return bool(GOOGLE_CREDENTIALS_JSON or GOOGLE_CREDENTIALS_FILE)


def is_chirp_voice(voice: str) -> bool:
    return "-Chirp3-HD-" in voice


def is_gemini_voice(voice: str) -> bool:
    return voice.startswith(GEMINI_PREFIX)


def is_google_voice(voice: str) -> bool:
    return is_chirp_voice(voice) or is_gemini_voice(voice)


def chirp_equivalent(voice: str, language: str = "hi-IN") -> str:
    if not is_gemini_voice(voice):
        return voice

    return f"{language}-Chirp3-HD-{voice[len(GEMINI_PREFIX):]}"


def is_rate_limited(error: Exception) -> bool:
    return "429" in str(error)


@lru_cache(maxsize=1)
def _credentials():
    from google.oauth2 import service_account

    if GOOGLE_CREDENTIALS_JSON:
        return service_account.Credentials.from_service_account_info(
            json.loads(GOOGLE_CREDENTIALS_JSON), scopes=SCOPES
        )

    return service_account.Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_FILE, scopes=SCOPES
    )


def _refresh() -> str:
    from google.auth.transport.requests import Request

    credentials = _credentials()

    if not credentials.valid:
        credentials.refresh(Request())

    return credentials.token


async def _token() -> str:
    async with _token_lock:
        return await asyncio.to_thread(_refresh)


def request_limit(voice: str) -> int:
    return GEMINI_MAX_BYTES if is_gemini_voice(voice) else CHIRP_MAX_BYTES


def split_for_request(text: str, limit: int) -> list[str]:
    if len(text.encode()) <= limit:
        return [text]

    pieces: list[str] = []
    buffer = ""

    for word in text.split(" "):
        candidate = f"{buffer} {word}" if buffer else word

        if buffer and len(candidate.encode()) > limit:
            pieces.append(buffer)
            buffer = word
        else:
            buffer = candidate

    if buffer:
        pieces.append(buffer)

    return pieces


def _speaking_rate(rate: str) -> float:
    try:
        percent = int(rate.strip().rstrip("%"))
    except (AttributeError, ValueError):
        return 1.0

    return min(4.0, max(0.25, 1 + percent / 100))


def _semitones(pitch: str) -> float:
    try:
        hertz = float(pitch.strip().rstrip("Hz"))
    except (AttributeError, ValueError):
        return 0.0

    return max(-20.0, min(20.0, hertz / 8.0))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _language(text: str, voice: str) -> str:
    if is_gemini_voice(voice):
        return "hi-IN" if DEVANAGARI.search(text) else "en-US"

    return "-".join(voice.split("-")[:2])


def _gemini_body(text: str, voice: str, rate: str, mood: str) -> dict:
    source = {"text": text}

    if mood:
        source["prompt"] = mood

    return {
        "input": source,
        "voice": {
            "languageCode": _language(text, voice),
            "name": voice[len(GEMINI_PREFIX) :],
            "modelName": GEMINI_MODEL,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "sampleRateHertz": GOOGLE_SAMPLE_RATE,
            "speakingRate": _speaking_rate(rate),
        },
    }


def _chirp_body(text: str, voice: str, rate: str, pitch: str) -> dict:
    shift = _semitones(pitch)

    if shift:
        source = {
            "ssml": f"<speak><prosody pitch='{shift:+.1f}st'>{_escape(text)}</prosody></speak>"
        }
    else:
        source = {"text": text}

    return {
        "input": source,
        "voice": {"languageCode": _language(text, voice), "name": voice},
        "audioConfig": {
            "audioEncoding": "MP3",
            "sampleRateHertz": GOOGLE_SAMPLE_RATE,
            "speakingRate": _speaking_rate(rate),
        },
    }


async def speak(
    text: str, voice: str, rate: str, pitch: str, mood: str = ""
) -> bytes:
    if not google_enabled():
        raise RuntimeError(
            "Neither GOOGLE_CREDENTIALS_JSON nor GOOGLE_CREDENTIALS_FILE is set, "
            "so Cloud TTS voices cannot speak"
        )

    gemini = is_gemini_voice(voice)
    endpoint = GEMINI_ENDPOINT if gemini else CHIRP_ENDPOINT

    token = await _token()
    audio = bytearray()

    async with httpx.AsyncClient(timeout=GOOGLE_TIMEOUT_SECONDS) as client:
        for piece in split_for_request(text, request_limit(voice)):
            body = (
                _gemini_body(piece, voice, rate, mood)
                if gemini
                else _chirp_body(piece, voice, rate, pitch)
            )

            response = await client.post(
                endpoint, json=body, headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code != 200:
                detail = response.json().get("error", {}).get("message", "")
                raise RuntimeError(f"{response.status_code}: {detail[:200]}")

            audio += base64.b64decode(response.json()["audioContent"])

    if not audio:
        raise RuntimeError(f"Cloud TTS returned no audio for {len(text)} characters")

    return bytes(audio)
