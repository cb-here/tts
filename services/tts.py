import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from config import (
    MAGPIE_GIVE_UP_AFTER,
    MAGPIE_OPENING_CHARS,
    MAGPIE_PREFETCH,
    MAGPIE_SAMPLE_RATE,
    MAX_CONCURRENT_TTS,
    PAUSE_CLAUSE_SECONDS,
    PAUSE_DEFAULT_SECONDS,
    PAUSE_SENTENCE_SECONDS,
    PAUSE_SPEAKER_SECONDS,
    magpie_enabled,
)
from io import BytesIO
from langgraph.config import get_stream_writer
from langgraph.graph import START, StateGraph, END
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
from services.audio import silence
from services.magpie import is_magpie, split_text, synthesize
from services import magpie
from uuid import uuid4
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Held per utterance rather than per request: a listener is mid-story for
# minutes, so locking for a whole stream would stall everyone behind them.
_tts_slots = asyncio.Semaphore(MAX_CONCURRENT_TTS)

class TTSState(TypedDict):
    text: str
    status: str
    voice: str
    rate: str
    multi_voice: bool
    # Who the listener said each character is, and any voice they pinned. Empty
    # when the reader was never opened on the cast, in which case the casting
    # model works it out as before.
    cast_genders: dict[str, str]
    cast_voices: dict[str, str]


# Measured against edge-tts Hindi output at +0%: 5407 chars produced 497s and
# 13480 chars produced 1168s, both about 0.088s per character.
SECONDS_PER_CHAR = 0.088

# Magpie takes no pitch argument, so a character's pitch offset is applied by
# resampling instead. This converts one to the other: +22Hz reads about 4%
# faster and higher, which is roughly what edge-tts does with the same number.
HZ_PER_TONE = 500.0

# edge-tts answers with audio-24khz-48kbitrate-mono-mp3. The silence spliced
# between its pieces has to be cut at the same rate, or players stumble over the
# sample rate changing mid-file.
EDGE_SAMPLE_RATE = 24000

SENTENCE_END = ("।", ".", "!", "?", "…")
CLAUSE_END = (",", ";", ":", "—", "–")
# Quotation marks are stripped from dialogue already, but a piece cut out of the
# middle of a speech can still end on one.
TRAILING_MARKS = "\"'”’»›」"


def _pause_after(text: str, speaker_changes: bool) -> float:
    """How long to hold before the next piece, read off the end of this one."""
    ended = text.rstrip().rstrip(TRAILING_MARKS).rstrip()

    if ended.endswith(SENTENCE_END):
        gap = PAUSE_SENTENCE_SECONDS
    elif ended.endswith(CLAUSE_END):
        gap = PAUSE_CLAUSE_SECONDS
    else:
        gap = PAUSE_DEFAULT_SECONDS

    # Someone else answering is a beat in its own right, even mid-sentence.
    return max(gap, PAUSE_SPEAKER_SECONDS) if speaker_changes else gap


def _percent(rate: str) -> float:
    try:
        return int(rate.strip().rstrip("%")) / 100
    except (AttributeError, ValueError):
        return 0.0


def _tone_of(utterance: Utterance) -> float:
    """How far to shift the voice itself, to tell this character from the rest."""
    try:
        hertz = int(utterance.pitch.strip().rstrip("Hz"))
    except (AttributeError, ValueError):
        hertz = 0

    return 1 + hertz / HZ_PER_TONE


def _speed_of(utterance: Utterance) -> float:
    """How fast to read, with the voice left where it is.

    Kept apart from the tone above because they are not the same request. Folded
    together, asking for a slower story also asked for a deeper narrator.
    """
    return 1 + _percent(utterance.rate)


def estimate_duration(text: str, rate: str, voice: str = "") -> float:
    """Roughly how long this will take to read aloud.

    A streamed response has no length, so the browser reports an infinite
    duration and its progress bar has nothing to scale against. This gives the
    player something honest to draw until the real duration is known.
    """
    seconds_per_char = (
        magpie.SECONDS_PER_CHAR if is_magpie(voice) else SECONDS_PER_CHAR
    )

    # "+50%" means half again as fast, so the reading is correspondingly shorter.
    return len(text) * seconds_per_char / (1 + _percent(rate))


def build_lyrics_tag(text: str) -> bytes:
    """Render the ID3 lyrics tag on its own, without touching a file.

    An ID3v2 tag lives at the *front* of an mp3, so it can be emitted before a
    single audio byte exists. That is what lets the whole pipeline stream: the
    lyrics go out first, then audio flows in behind them.
    """
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


# edge-tts occasionally closes a connection having sent nothing, on text it
# reads perfectly well a second later — the same short lines were confirmed to
# succeed three times out of three on their own. Since this path is already the
# fallback, one more attempt is the difference between a lost line and a
# recovered one.
EDGE_RETRIES = 1
EDGE_RETRY_SECONDS = 0.5

# Multilingual, so it reads Devanagari and Latin alike. The last thing tried
# before a line is given up on.
LAST_RESORT_VOICE = "en-US-EmmaMultilingualNeural"


async def _edge_bytes(text: str, utterance: Utterance, voice: str) -> bytes:
    """Speak one piece with edge-tts and hand back the whole thing at once."""
    for attempt in range(EDGE_RETRIES + 1):
        audio = bytearray()

        communicate = Communicate(
            text=text,
            voice=voice,
            rate=utterance.rate,
            pitch=utterance.pitch,
        )

        try:
            async with _tts_slots:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio += chunk["data"]
        except Exception:
            if attempt == EDGE_RETRIES:
                raise

            await asyncio.sleep(EDGE_RETRY_SECONDS)
            continue

        if audio:
            return bytes(audio)

        # A clean connection that carried no audio is the same failure wearing
        # a different hat, and would otherwise be published as a silent line.
        if attempt == EDGE_RETRIES:
            raise RuntimeError(f"edge-tts returned no audio for {len(text)} characters")

        await asyncio.sleep(EDGE_RETRY_SECONDS)

    raise RuntimeError(f"edge-tts returned no audio for {len(text)} characters")


async def _speak_fallback(
    piece: str, utterance: Utterance, writer, rate: int
) -> None:
    """Get this piece spoken by something, or say plainly that it was lost.

    The mapped stand-in first, then a voice known to read anything. A voice that
    cannot pronounce the script at all answers with silence rather than an error,
    and that is how whole passages went missing from a story before.
    """
    for voice in (edge_equivalent(utterance.voice), LAST_RESORT_VOICE):
        try:
            writer(await _edge_bytes(piece, utterance, voice))
            return
        except Exception as error:
            logger.warning(
                "%s could not speak %d chars (%s: %s)",
                voice,
                len(piece),
                type(error).__name__,
                error,
            )

    # Holding the beat keeps the reading moving instead of tearing the
    # connection down, but part of the story is genuinely gone — not a warning.
    logger.error(
        "Dropped %d characters of the story — no voice would speak them: %r",
        len(piece),
        piece[:120],
    )
    writer(silence(PAUSE_SENTENCE_SECONDS, rate))


async def _speak_edge(cast: AsyncIterator[Utterance], writer) -> int:
    """Stream each utterance as edge-tts produces it."""
    spoken = 0
    previous: Utterance | None = None

    async for utterance in cast:
        if previous is not None:
            writer(
                silence(
                    _pause_after(previous.text, previous.speaker != utterance.speaker),
                    EDGE_SAMPLE_RATE,
                )
            )

        previous = utterance

        communicate = Communicate(
            text=utterance.text,
            voice=utterance.voice,
            rate=utterance.rate,
            pitch=utterance.pitch,
        )

        carried = 0

        try:
            async with _tts_slots:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        writer(chunk["data"])
                        carried += len(chunk["data"])
        except Exception as error:
            # Only safe to retry while nothing has gone out yet; past that the
            # listener would hear the first half of the line twice.
            if carried:
                raise

            logger.warning(
                "%s failed part-way through %d chars (%s: %s)",
                utterance.voice,
                len(utterance.text),
                type(error).__name__,
                error,
            )

        if not carried:
            # An utterance that produced nothing at all used to pass silently,
            # and a merged run of narration is several sentences long — which is
            # how five sentences at a time went missing from a story with
            # nothing in the log to show it.
            logger.warning(
                "%s returned no audio for %d chars — trying again",
                utterance.voice,
                len(utterance.text),
            )
            await _speak_fallback(utterance.text, utterance, writer, EDGE_SAMPLE_RATE)

        spoken += 1

    return spoken


# Magpie needs to be told what it is reading, and the voice cannot say — every
# speaker reads every language, whatever locale its name carries.
MAGPIE_LOCALES = {"hi": "hi-IN", "en": "en-US"}


async def _speak_magpie(
    cast: AsyncIterator[Utterance], writer, language: str
) -> int:
    """Synthesise ahead of the playhead, then write the pieces out in order.

    Magpie returns a finished WAV rather than a stream, so a piece being spoken
    would otherwise be the only work in flight and the listener would hear the
    gap before every one. Several are requested at once instead, while the
    output stays strictly in reading order.
    """
    pending: deque[tuple[asyncio.Task | None, str, Utterance]] = deque()
    spoken = 0
    opening = True
    previous: tuple[str, str] | None = None
    # Once the free tier starts refusing it usually keeps refusing, and every
    # piece then spends its whole retry budget before falling back anyway. After
    # a few in a row the reading gives up on Magpie and simply reads in edge-tts
    # — far quicker than grinding through the backoff for every line left.
    misses = 0
    dropped = False

    async def speak_on_edge(piece: str, utterance: Utterance) -> None:
        await _speak_fallback(piece, utterance, writer, MAGPIE_SAMPLE_RATE)

    async def write_next() -> None:
        nonlocal previous, misses, dropped

        task, piece, utterance = pending.popleft()

        if previous is not None:
            last_text, last_speaker = previous
            writer(
                silence(
                    _pause_after(last_text, last_speaker != utterance.speaker),
                    MAGPIE_SAMPLE_RATE,
                )
            )

        previous = (piece, utterance.speaker)

        if task is None:
            await speak_on_edge(piece, utterance)
            return

        try:
            writer(await task)
            misses = 0
        except Exception as error:
            # One piece failing must not cost the rest of the story, so it is
            # read by the nearest edge-tts voice instead. The change is audible,
            # which is better than a hole where the sentence should be.
            misses += 1
            # The reason lives on the cause: every attempt failed, and the
            # RuntimeError raised at the end only says that it did. Logging the
            # cause is the difference between "Magpie failed" and "the quota is
            # gone", which are not the same problem.
            reason = error.__cause__ or error
            logger.warning(
                "Magpie could not speak %d chars as %s (%s) — reading it in "
                "edge-tts instead",
                len(piece),
                utterance.voice,
                reason,
            )

            if misses >= MAGPIE_GIVE_UP_AFTER and not dropped:
                dropped = True
                logger.warning(
                    "Magpie has refused %d pieces in a row — reading the rest of "
                    "this story in edge-tts",
                    misses,
                )

            await speak_on_edge(piece, utterance)

    try:
        async for utterance in cast:
            tone = _tone_of(utterance)
            speed = _speed_of(utterance)

            if opening:
                pieces = split_text(utterance.text, MAGPIE_OPENING_CHARS)

                if len(pieces) > 1:
                    # Only the opening is shortened. Splitting the whole reading
                    # this finely would multiply the round trips for nothing.
                    pieces = pieces[:1] + split_text(" ".join(pieces[1:]))

                opening = False
            else:
                pieces = split_text(utterance.text)

            for piece in pieces:
                task = (
                    None
                    if dropped
                    else asyncio.ensure_future(
                        synthesize(piece, utterance.voice, tone, language, speed)
                    )
                )
                pending.append((task, piece, utterance))

                # Awaiting the head is the backpressure: no more than
                # MAGPIE_PREFETCH pieces are ever being synthesised at once.
                while len(pending) > MAGPIE_PREFETCH:
                    await write_next()

            spoken += 1

        while pending:
            await write_next()
    finally:
        # Reached when the listener navigates away mid-story, which would
        # otherwise leave paid-for requests running with nowhere to go.
        for task, _, _ in pending:
            if task is None:
                continue

            if task.done():
                # Retrieving the exception is what stops asyncio reporting it as
                # never retrieved once the task is garbage collected.
                task.exception()
            else:
                task.cancel()

    return spoken


async def generate_audio(state: TTSState):
    writer = get_stream_writer()

    voice = state["voice"]
    language = language_of(state["text"])

    voice = speakable(voice, language)

    if is_magpie(voice) and not magpie_enabled():
        # Deciding this once is the difference between a reading that starts
        # normally and one where every single piece is requested, refused, and
        # then re-spoken by the stand-in.
        stand_in = edge_equivalent(voice)
        logger.warning(
            "NVIDIA_API_KEY is not set, so %s is unavailable — reading in %s",
            voice,
            stand_in,
        )
        voice = stand_in

    cast = merge_stream(
        iter_cast(
            text=state["text"],
            voice=voice,
            rate=state["rate"],
            multi_voice=state.get("multi_voice", False),
            cast_genders=state.get("cast_genders") or None,
            cast_voices=state.get("cast_voices") or None,
        )
    )

    # Each piece is its own mp3, and mp3 frames concatenate cleanly, so the
    # speakers stitch into one continuous track. The narrator's voice decides
    # the engine for the whole reading; the cast is drawn from the matching
    # pool, so the two are never spliced together.
    if is_magpie(voice):
        spoken = await _speak_magpie(
            cast, writer, MAGPIE_LOCALES.get(language, "en-US")
        )
    else:
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
) -> AsyncIterator[bytes]:
    """Yield mp3 bytes as edge-tts produces them.

    `stream_mode="custom"` surfaces whatever the nodes hand to their stream
    writer, so each chunk here is raw audio rather than a state update.
    """
    async for chunk in workflow.astream(
        {
            "text": text,
            "voice": voice,
            "rate": rate,
            "multi_voice": multi_voice,
            "cast_genders": cast_genders or {},
            "cast_voices": cast_voices or {},
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
    """Drain the stream into a complete mp3 on disk."""
    filename = AUDIO_DIR / f"{uuid4()}.mp3"

    with filename.open("wb") as audio_file:
        async for chunk in stream_tts(text, voice, rate, multi_voice):
            audio_file.write(chunk)

    return filename
