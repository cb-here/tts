import asyncio
import logging
from collections.abc import AsyncIterator
from config import (
    EDGE_BOUNDARY,
    MAX_CONCURRENT_TTS,
    PAUSE_CLAUSE_SECONDS,
    PAUSE_DEFAULT_SECONDS,
    PAUSE_SENTENCE_SECONDS,
    PAUSE_SPEAKER_SECONDS,
)
from io import BytesIO
from langgraph.config import get_stream_writer
from langgraph.graph import START, StateGraph, END
from dataclasses import dataclass
from typing import TypedDict
from edge_tts import Communicate
from mutagen.id3 import ID3, USLT
from services.casting import (
    Utterance,
    edge_equivalent,
    iter_cast,
    language_of,
    merge_stream,
    speakable,
)
from services.audio import MP3_BITRATE, silence
from services.google_tts import (
    GOOGLE_BITRATE,
    chirp_equivalent,
    is_gemini_voice,
    is_google_voice,
    is_rate_limited,
    speak as google_speak,
)
from uuid import uuid4
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

_tts_slots = asyncio.Semaphore(MAX_CONCURRENT_TTS)

class TTSState(TypedDict):
    text: str
    status: str
    voice: str
    rate: str
    multi_voice: bool
    cast_genders: dict[str, str]
    cast_voices: dict[str, str]
    cast_moods: dict[str, str]


SECONDS_PER_CHAR = 0.088
GOOGLE_SECONDS_PER_CHAR = 0.069

EDGE_SAMPLE_RATE = 24000

EDGE_BITRATE = 48000
SILENCE_BITRATE = MP3_BITRATE * 1000


@dataclass(frozen=True)
class Mark:
    at: float
    text: str


def _seconds(size: int, bitrate: int) -> float:
    return size * 8 / bitrate

SENTENCE_END = ("।", ".", "!", "?", "…")
CLAUSE_END = (",", ";", ":", "—", "–")
TRAILING_MARKS = "\"'”’»›」"


def _pause_after(text: str, speaker_changes: bool) -> float:
    ended = text.rstrip().rstrip(TRAILING_MARKS).rstrip()

    if ended.endswith(SENTENCE_END):
        gap = PAUSE_SENTENCE_SECONDS
    elif ended.endswith(CLAUSE_END):
        gap = PAUSE_CLAUSE_SECONDS
    else:
        gap = PAUSE_DEFAULT_SECONDS

    return max(gap, PAUSE_SPEAKER_SECONDS) if speaker_changes else gap


def _percent(rate: str) -> float:
    try:
        return int(rate.strip().rstrip("%")) / 100
    except (AttributeError, ValueError):
        return 0.0


def estimate_duration(text: str, rate: str, voice: str = "") -> float:
    per_char = (
        GOOGLE_SECONDS_PER_CHAR if is_google_voice(voice) else SECONDS_PER_CHAR
    )

    return len(text) * per_char / (1 + _percent(rate))


def build_lyrics_tag(text: str) -> bytes:
    buffer = BytesIO()

    tag = ID3()
    tag.add(
        USLT(
            encoding=3,
            lang="hin",
            desc="Lyrics",
            text=text,
        )
    )
    tag.save(buffer, v2_version=4, v1=0)

    return buffer.getvalue()


def embed_text(state: TTSState):
    writer = get_stream_writer()
    writer(build_lyrics_tag(state["text"]))

    return {
        "status": "Lyrics Embedded"
    }


EDGE_RETRIES = 1
EDGE_RETRY_SECONDS = 0.5

LAST_RESORT_VOICE = "en-US-EmmaMultilingualNeural"


async def _edge_bytes(
    text: str, utterance: Utterance, voice: str
) -> tuple[bytes, list[tuple[float, str]]]:
    for attempt in range(EDGE_RETRIES + 1):
        audio = bytearray()
        boundaries: list[tuple[float, str]] = []

        communicate = Communicate(
            text=text,
            voice=voice,
            rate=utterance.rate,
            pitch=utterance.pitch,
            boundary=EDGE_BOUNDARY,
        )

        try:
            async with _tts_slots:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio += chunk["data"]
                    elif chunk["type"] == EDGE_BOUNDARY:
                        boundaries.append((chunk["offset"] / 1e7, chunk["text"]))
        except Exception:
            if attempt == EDGE_RETRIES:
                raise

            await asyncio.sleep(EDGE_RETRY_SECONDS)
            continue

        if audio:
            return bytes(audio), boundaries

        if attempt == EDGE_RETRIES:
            raise RuntimeError(f"edge-tts returned no audio for {len(text)} characters")

        await asyncio.sleep(EDGE_RETRY_SECONDS)

    raise RuntimeError(f"edge-tts returned no audio for {len(text)} characters")


FALLBACK_ROUNDS = 4
FALLBACK_BACKOFF_SECONDS = 3.0
MAX_FALLBACK_BACKOFF_SECONDS = 20.0


async def _speak_fallback(
    piece: str, utterance: Utterance, writer, rate: int, started: float
) -> float:
    for attempt in range(FALLBACK_ROUNDS):
        for voice in (edge_equivalent(utterance.voice), LAST_RESORT_VOICE):
            try:
                audio, boundaries = await _edge_bytes(piece, utterance, voice)
                writer(audio)

                for at, text in boundaries:
                    writer(Mark(at=started + at, text=text))

                return _seconds(len(audio), EDGE_BITRATE)
            except Exception as error:
                logger.warning(
                    "%s could not speak %d chars (%s: %s)",
                    voice,
                    len(piece),
                    type(error).__name__,
                    error,
                )

        if attempt + 1 < FALLBACK_ROUNDS:
            wait = min(
                MAX_FALLBACK_BACKOFF_SECONDS,
                FALLBACK_BACKOFF_SECONDS * 2**attempt,
            )
            logger.warning(
                "No voice would speak %d chars — waiting %.0fs and asking again "
                "(round %d of %d)",
                len(piece),
                wait,
                attempt + 1,
                FALLBACK_ROUNDS,
            )
            await asyncio.sleep(wait)

    logger.error(
        "Dropped %d characters of the story — no voice would speak them: %r",
        len(piece),
        piece[:120],
    )

    gap = silence(PAUSE_SENTENCE_SECONDS, rate)
    writer(gap)

    return _seconds(len(gap), SILENCE_BITRATE)


async def _speak_edge(cast: AsyncIterator[Utterance], writer) -> int:
    spoken = 0
    previous: Utterance | None = None
    elapsed = 0.0

    async for utterance in cast:
        if previous is not None:
            gap = silence(
                _pause_after(previous.text, previous.speaker != utterance.speaker),
                EDGE_SAMPLE_RATE,
            )
            writer(gap)
            elapsed += _seconds(len(gap), SILENCE_BITRATE)

        previous = utterance
        started = elapsed

        carried = 0
        bitrate = EDGE_BITRATE
        marks: list[Mark] = []

        if is_google_voice(utterance.voice):
            bitrate = GOOGLE_BITRATE

            try:
                async with _tts_slots:
                    audio = await google_speak(
                        utterance.text,
                        utterance.voice,
                        utterance.rate,
                        utterance.pitch,
                        utterance.mood,
                    )

                writer(audio)
                carried = len(audio)
            except Exception as error:
                stand_in = chirp_equivalent(utterance.voice)

                if is_gemini_voice(utterance.voice) and is_rate_limited(error):
                    logger.info(
                        "%s is out of requests for now — %s reads these %d chars "
                        "instead, without the direction",
                        utterance.voice,
                        stand_in,
                        len(utterance.text),
                    )
                else:
                    logger.warning(
                        "%s could not speak %d chars (%s: %s)",
                        utterance.voice,
                        len(utterance.text),
                        type(error).__name__,
                        error,
                    )

                if is_gemini_voice(utterance.voice):
                    try:
                        async with _tts_slots:
                            audio = await google_speak(
                                utterance.text,
                                stand_in,
                                utterance.rate,
                                utterance.pitch,
                            )

                        writer(audio)
                        carried = len(audio)
                    except Exception as second:
                        logger.warning(
                            "%s could not stand in either (%s: %s)",
                            stand_in,
                            type(second).__name__,
                            second,
                        )
        else:
            communicate = Communicate(
                text=utterance.text,
                voice=utterance.voice,
                rate=utterance.rate,
                pitch=utterance.pitch,
                boundary=EDGE_BOUNDARY,
            )

            try:
                async with _tts_slots:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            writer(chunk["data"])
                            carried += len(chunk["data"])
                        elif chunk["type"] == EDGE_BOUNDARY:
                            marks.append(
                                Mark(
                                    at=started + chunk["offset"] / 1e7,
                                    text=chunk["text"],
                                )
                            )
            except Exception as error:
                if carried:
                    raise

                logger.warning(
                    "%s failed part-way through %d chars (%s: %s)",
                    utterance.voice,
                    len(utterance.text),
                    type(error).__name__,
                    error,
                )

        if carried:
            for mark in marks:
                writer(mark)

            elapsed = started + _seconds(carried, bitrate)
        else:
            logger.warning(
                "%s returned no audio for %d chars — trying again",
                utterance.voice,
                len(utterance.text),
            )
            elapsed = started + await _speak_fallback(
                utterance.text, utterance, writer, EDGE_SAMPLE_RATE, started
            )

        spoken += 1

    return spoken


async def generate_audio(state: TTSState):
    writer = get_stream_writer()

    voice = state["voice"]
    language = language_of(state["text"])

    voice = speakable(voice, language)

    cast = merge_stream(
        iter_cast(
            text=state["text"],
            voice=voice,
            rate=state["rate"],
            multi_voice=state.get("multi_voice", False),
            cast_genders=state.get("cast_genders") or None,
            cast_voices=state.get("cast_voices") or None,
            cast_moods=state.get("cast_moods") or None,
        )
    )

    spoken = await _speak_edge(cast, writer)

    return {
        "status": f"Done ({spoken} utterance{'' if spoken == 1 else 's'})"
    }


graph = StateGraph(TTSState)

graph.add_node("embed_text", embed_text)
graph.add_node("generate_audio", generate_audio)

graph.add_edge(START, "embed_text")
graph.add_edge("embed_text", "generate_audio")
graph.add_edge("generate_audio", END)

workflow = graph.compile()


async def stream_tts(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
    cast_genders: dict[str, str] | None = None,
    cast_voices: dict[str, str] | None = None,
    cast_moods: dict[str, str] | None = None,
) -> AsyncIterator[bytes | Mark]:
    async for chunk in workflow.astream(
        {
            "text": text,
            "voice": voice,
            "rate": rate,
            "multi_voice": multi_voice,
            "cast_genders": cast_genders or {},
            "cast_voices": cast_voices or {},
            "cast_moods": cast_moods or {},
        },
        stream_mode="custom",
    ):
        yield chunk


async def render_to_file(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
) -> Path:
    filename = AUDIO_DIR / f"{uuid4()}.mp3"

    with filename.open("wb") as audio_file:
        async for chunk in stream_tts(text, voice, rate, multi_voice):
            if isinstance(chunk, bytes):
                audio_file.write(chunk)

    return filename
