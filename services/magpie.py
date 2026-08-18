"""NVIDIA Magpie TTS, as an alternative to edge-tts.

Reached over plain HTTP with the same key the casting model uses, so a
deployment that can cast can also speak. It is roughly thirteen times faster
than real time, which is what makes up for the one thing it will not do: stream.
Every call returns a finished WAV, so a reading has to be cut into pieces and
the pieces synthesised ahead of the one being played.
"""

import asyncio
import logging
import random
import re

import httpx

from config import (
    MAGPIE_MAX_CHARS,
    MAGPIE_SAMPLE_RATE,
    MAGPIE_TIMEOUT_SECONDS,
    MAGPIE_URL,
    MAX_CONCURRENT_MAGPIE,
    NVIDIA_API_KEY,
)
from services.audio import encode_mp3, read_wav, stretch

logger = logging.getLogger(__name__)

_magpie_slots = asyncio.Semaphore(MAX_CONCURRENT_MAGPIE)

VOICE_PREFIX = "Magpie-"

# "Magpie-Multilingual.HI-IN.Leo" and "...Leo.Angry" both name the same locale.
VOICE_LOCALE = re.compile(r"^Magpie-[^.]+\.([A-Za-z]{2}-[A-Za-z]{2})\.")

SENTENCE_BREAK = re.compile(r"(?<=[.!?।])\s+|\n+")
# A sentence longer than a whole piece has to give way somewhere; these are the
# next-best places to cut without landing mid-word.
CLAUSE_BREAK = re.compile(r"(?<=[,;:—–])\s+")

# Measured on Hindi at 22 kHz: 252, 504 and 1008 characters read for 19.1s,
# 39.0s and 78.4s — a little quicker than edge-tts.
SECONDS_PER_CHAR = 0.077

# The free tier allows a burst and then throttles, so a refusal is a wait rather
# than a no. It does not always say so politely: under the same load it answers
# 400 "Mapping failed" on text it speaks perfectly well once it has caught up —
# the identical sentence was confirmed to succeed on its own, seconds later.
RETRIES = 4
BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 12.0

# Spacing the calls out beats recovering from being refused: at a second apart
# the free tier answers steadily, while a burst buys a few quick replies and
# then a hold several times longer than it saved. The service is shared, so how
# much gets through also varies with who else is using it.
MIN_INTERVAL_SECONDS = 1.0

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_MESSAGE = "mapping failed"

# These describe the request itself, so repeating it changes nothing.
FATAL_MESSAGES = (
    "maximum input length",
    "received message larger than max",
)


class MagpieError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Magpie returned {status}: {body[:300]}")
        self.status = status
        self.body = body

    @property
    def retryable(self) -> bool:
        lowered = self.body.lower()

        if any(message in lowered for message in FATAL_MESSAGES):
            return False

        return self.status in RETRYABLE_STATUS or RETRYABLE_MESSAGE in lowered


class Pacer:
    """Holds every request in this process to one shared rate.

    A per-request backoff is not enough on its own. When the service starts
    refusing, each piece in flight backs off on its own clock, and they all come
    back at once — which is exactly the burst that was refused. Being throttled
    here pushes back everything queued behind it, not just the piece that was
    turned away.
    """

    def __init__(self, interval: float):
        self._interval = interval
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()

        async with self._lock:
            start = max(loop.time(), self._next)
            self._next = start + self._interval

        delay = start - loop.time()

        if delay > 0:
            await asyncio.sleep(delay)

    def throttled(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        self._next = max(self._next, loop.time() + seconds)


_pacer = Pacer(MIN_INTERVAL_SECONDS)


def is_magpie(voice: str) -> bool:
    return voice.startswith(VOICE_PREFIX)


def locale_of(voice: str) -> str:
    """The locale in a voice's name, e.g. "hi-IN" for ...HI-IN.Leo.

    Only a fallback for when the caller does not say what language the text is
    in. The locale in the name is a label on the speaker, not a restriction:
    every one of the 86 voices was confirmed to read Hindi.

    The casing is not cosmetic. "hi-in" and "HI-IN" are both rejected, and the
    refusal arrives as 400 "Mapping failed. Check that all pronunciations are
    specified properly." — which reads like a problem with the text.
    """
    match = VOICE_LOCALE.match(voice)

    if not match:
        return "en-US"

    language, _, region = match.group(1).partition("-")

    return f"{language.lower()}-{region.upper()}"


def _split_on(text: str, pattern: re.Pattern, limit: int) -> list[str]:
    """Pack the pieces `pattern` produces into runs no longer than `limit`."""
    pieces: list[str] = []
    buffer = ""

    for piece in pattern.split(text):
        piece = piece.strip()

        if not piece:
            continue

        candidate = f"{buffer} {piece}" if buffer else piece

        if buffer and len(candidate) > limit:
            pieces.append(buffer)
            buffer = piece
        else:
            buffer = candidate

    if buffer:
        pieces.append(buffer)

    return pieces


def split_text(text: str, limit: int = MAGPIE_MAX_CHARS) -> list[str]:
    """Cut a line into pieces the model will accept, at sentence boundaries.

    Cutting mid-sentence is audible — the voice drops as if it had finished —
    so a sentence is only broken when it is on its own too long to send.
    """
    pieces: list[str] = []

    for sentence in _split_on(text, SENTENCE_BREAK, limit):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue

        for clause in _split_on(sentence, CLAUSE_BREAK, limit):
            # Still too long, so there is nothing left to break on but length.
            while len(clause) > limit:
                pieces.append(clause[:limit])
                clause = clause[limit:]

            if clause:
                pieces.append(clause)

    return pieces or [text]


async def _request(
    client: httpx.AsyncClient, text: str, voice: str, language: str
) -> bytes:
    response = await client.post(
        MAGPIE_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        files={
            "text": (None, text),
            "language": (None, language),
            "voice": (None, voice),
            "encoding": (None, "LINEAR_PCM"),
            "sample_rate_hz": (None, str(MAGPIE_SAMPLE_RATE)),
        },
    )

    if response.status_code != 200:
        # The body is where the gateway explains itself — a burst limit, a reply
        # over the size cap, text past the model's own limit.
        raise MagpieError(response.status_code, response.text)

    return response.content


async def synthesize(
    text: str,
    voice: str,
    tone: float = 1.0,
    language: str | None = None,
    speed: float = 1.0,
) -> bytes:
    """Speak one piece, returning mp3 frames ready to splice into the stream.

    `language` describes the text, not the voice. Any speaker reads any of the
    supported languages, so this is what decides pronunciation — sending Hindi
    under en-US produces an English reading of Devanagari.

    `speed` is how fast it is read and leaves the voice alone; `tone` shifts the
    voice itself, and is what tells two characters apart once the pool runs out.
    Magpie offers neither, so both are done to the audio afterwards.
    """
    last: Exception | None = None
    language = language or locale_of(voice)

    async with _magpie_slots:
        async with httpx.AsyncClient(timeout=MAGPIE_TIMEOUT_SECONDS) as client:
            for attempt in range(RETRIES + 1):
                await _pacer.wait()

                try:
                    wav = await _request(client, text, voice, language)
                    pcm, rate = read_wav(wav)

                    return encode_mp3(stretch(pcm, speed), rate, tone)
                except MagpieError as error:
                    last = error

                    if not error.retryable:
                        break
                except (httpx.HTTPError, ValueError) as error:
                    last = error

                if attempt == RETRIES:
                    break

                # Jittered so that two processes sharing the key do not settle
                # into lockstep and refuse each other indefinitely.
                cooldown = min(
                    MAX_BACKOFF_SECONDS, BACKOFF_SECONDS * 2**attempt
                ) * random.uniform(0.7, 1.0)

                _pacer.throttled(cooldown)

                logger.info(
                    "Magpie deferred %d chars on %s (%s) — holding every request "
                    "for %.1fs",
                    len(text),
                    voice,
                    last,
                    cooldown,
                )

    raise RuntimeError(
        f"Magpie could not speak {len(text)} characters as {voice}"
    ) from last
