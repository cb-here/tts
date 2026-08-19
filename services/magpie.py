"""NVIDIA Magpie TTS, as an alternative to edge-tts.

Reached over plain HTTP with the same key the casting model uses, so a
deployment that can cast can also speak. It is roughly thirteen times faster
than real time, which is what makes up for the one thing it will not do: stream.
Every call returns a finished WAV, so a reading has to be cut into pieces and
the pieces synthesised ahead of the one being played.
"""

import asyncio
import logging
import itertools
import os
import random
import re

import httpx

from config import (
    MAGPIE_MAX_CHARS,
    MAGPIE_SAMPLE_RATE,
    MAGPIE_TIMEOUT_SECONDS,
    MAGPIE_URL,
    MAGPIE_API_KEYS,
    MAX_CONCURRENT_MAGPIE,
)
from services.audio import encode_mp3, read_wav, stretch

logger = logging.getLogger(__name__)

# Per key, since that is what the limit is counted against. Two keys are two
# budgets and can be in flight at once.
_magpie_slots = asyncio.Semaphore(
    MAX_CONCURRENT_MAGPIE * max(1, len(MAGPIE_API_KEYS))
)

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

# Magpie answers 200 OK with a well-formed but empty WAV when it is struggling,
# and an empty WAV encodes to a perfectly valid mp3 of nothing but header. Spliced
# into the stream that is a paragraph which is simply never read aloud — no
# exception, no retry, no fallback, and nothing in the log to say it happened.
# Checking the length of what came back is what turns that silent hole into an
# ordinary refusal the retry ladder already knows how to handle.
MIN_AUDIO_SECONDS = 0.05
# A reply far shorter than the text warrants was cut off rather than left empty.
# Kept generous: how long a piece takes to read varies with the script and the
# speaker, and a wrong rejection costs a retry and an audibly different voice.
MIN_AUDIO_FRACTION = 0.25
# Under this there is not enough text for the proportion to mean anything —
# "ठीक है।" is over in a moment however it is read.
FRACTION_MIN_CHARS = 40

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
#
# Probing this directly finds nothing — fifteen requests back to back on one key
# were all answered, and 429 is what too many in flight looks like rather than
# too fast. Whole readings are the only honest measurement and they are not
# cheap to repeat: five runs of the same story drew 5, 3, 6, 6 and 8 refusals in
# that order, and the count tracked how much the key had been used that hour
# rather than anything that was changed between them. So this is left where it
# was, and made settable for anyone with a fresh key and the patience to time it
# properly.
MIN_INTERVAL_SECONDS = float(os.getenv("MAGPIE_MIN_INTERVAL", "1.0"))

# Failures in a row before the service is left alone entirely, and for how long.
BREAKER_LIMIT = 4
BREAKER_COOLDOWN_SECONDS = 600.0

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


class Breaker:
    """Stops calling a service that has stopped answering.

    A burst limit clears in seconds and is worth waiting out. A spent quota does
    not, and every piece then pays the full retry budget — five attempts and
    half a minute of backoff — only to fall back to edge-tts anyway. Once enough
    pieces have failed in a row, this makes the rest fail instantly instead, so
    the reading starts on the other engine rather than crawling.
    """

    def __init__(self, limit: int, cooldown: float, label: str = "Magpie"):
        self._limit = limit
        self._cooldown = cooldown
        self._label = label
        self._misses = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        return asyncio.get_running_loop().time() < self._open_until

    def succeeded(self) -> None:
        self._misses = 0
        self._open_until = 0.0

    def failed(self) -> None:
        self._misses += 1

        if self._misses >= self._limit:
            self._open_until = (
                asyncio.get_running_loop().time() + self._cooldown
            )
            logger.warning(
                "Magpie has refused %d requests in a row on %s — leaving that "
                "key alone for %.0f minutes",
                self._misses,
                self._label,
                self._cooldown / 60,
            )
            self._misses = 0


class Key:
    """One API key, with the pacing and the circuit breaker that belong to it.

    Both have to be per key or a second key buys nothing. Sharing one pacer
    holds the pair to the rate of a single key, and sharing one breaker takes
    the healthy key out of service the moment the other one is throttled.
    """

    def __init__(self, secret: str):
        self.secret = secret
        self.pacer = Pacer(MIN_INTERVAL_SECONDS)
        self.breaker = Breaker(
            BREAKER_LIMIT, BREAKER_COOLDOWN_SECONDS, label=str(self)
        )

    def __str__(self) -> str:
        # Enough to tell two keys apart in a log, and no more than that.
        return f"key …{self.secret[-4:]}"


_keys = [Key(secret) for secret in MAGPIE_API_KEYS]
_turn = itertools.count()


def _take_key() -> Key | None:
    """The next key that is not in its cooldown, taken in rotation.

    Round robin rather than "first that works": spreading the requests is the
    whole point, and always starting at the front would exhaust one key before
    touching the next.
    """
    if not _keys:
        return None

    start = next(_turn)

    for offset in range(len(_keys)):
        key = _keys[(start + offset) % len(_keys)]

        if not key.breaker.is_open():
            return key

    return None


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


def _short_reply(pcm: bytes, rate: int, text: str) -> str | None:
    """Why this audio cannot be a reading of `text`, if it cannot be."""
    # 16-bit mono, which is what the request asks for.
    seconds = len(pcm) / (rate * 2) if rate else 0.0

    # Punctuation on its own is allowed to come back as next to nothing.
    if not any(character.isalnum() for character in text):
        return None

    if seconds < MIN_AUDIO_SECONDS:
        return f"{seconds * 1000:.0f}ms of audio for {len(text)} characters"

    if len(text) >= FRACTION_MIN_CHARS:
        expected = len(text) * SECONDS_PER_CHAR

        if seconds < expected * MIN_AUDIO_FRACTION:
            return (
                f"{seconds:.1f}s of audio for {len(text)} characters, "
                f"where about {expected:.0f}s was due"
            )

    return None


async def _request(
    client: httpx.AsyncClient, text: str, voice: str, language: str, secret: str
) -> bytes:
    response = await client.post(
        MAGPIE_URL,
        headers={"Authorization": f"Bearer {secret}"},
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
    key = _take_key()

    if key is None:
        raise RuntimeError("Magpie is not answering; not asking again yet")

    async with _magpie_slots:
        async with httpx.AsyncClient(timeout=MAGPIE_TIMEOUT_SECONDS) as client:
            for attempt in range(RETRIES + 1):
                await key.pacer.wait()

                try:
                    wav = await _request(
                        client, text, voice, language, key.secret
                    )
                    pcm, rate = read_wav(wav)

                    cut = _short_reply(pcm, rate, text)

                    if cut:
                        # A ValueError here is retried and then falls back, which
                        # is what a refusal deserves. Letting it through instead
                        # publishes silence as though it were the reading.
                        raise ValueError(f"Magpie returned {cut}")

                    key.breaker.succeeded()

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

                # Only this key is held back. The other one is on its own
                # budget and has no reason to wait.
                key.pacer.throttled(cooldown)

                logger.info(
                    "Magpie deferred %d chars on %s via %s (%s) — holding that "
                    "key for %.1fs",
                    len(text),
                    voice,
                    key,
                    last,
                    cooldown,
                )

    key.breaker.failed()

    raise RuntimeError(
        f"Magpie could not speak {len(text)} characters as {voice}"
    ) from last
