"""Beta: cast a story across several voices.

The split is done locally and the LLM only *labels* the pieces it is shown. The
story text never makes a round trip through the model, so nothing can be
paraphrased, reordered or dropped.

Labelling happens a batch at a time, and the next batch is requested while the
current one is still being spoken. Otherwise the whole story would have to be
analysed before the first word came out — and on a free NVIDIA tier that call
has been measured anywhere between 6 and 42 seconds.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from config import (
    CASTING_MAX_CHARS,
    CASTING_TIMEOUT_SECONDS,
    MAX_CONCURRENT_CASTING,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    casting_enabled,
)

logger = logging.getLogger(__name__)

# Casting calls arrive in bursts as several listeners start at once, and the
# free tier answers far slower under that load.
_casting_slots = asyncio.Semaphore(MAX_CONCURRENT_CASTING)

# Paired quote marks only: a bare ' is an apostrophe far more often than it is a
# dialogue opener. `।` is the Devanagari full stop.
QUOTE_PATTERN = re.compile(r'["“„]([^"“”„]{1,800})["”“]|‘([^‘’]{1,800})’')
SENTENCE_BREAK = re.compile(r"(?<=[.!?।])\s+|\n+")
NARRATION_CHUNK_CHARS = 350

# Screenplay format: a short speaker name, a colon, then the line. The name is
# capped so that ordinary prose using a colon mid-sentence is not mistaken for a
# cue, and a majority of lines must match before the text is treated as a script.
SCRIPT_LINE = re.compile(r"^[ \t]*([^\s:][^:\n]{0,39}?)[ \t]*:[ \t]*(.*)$")
MIN_SCRIPT_LINES = 4
SCRIPT_LINE_RATIO = 0.6

# edge-tts ships exactly one male and one female neural voice per Indian
# language, so characters are told apart by pitch and pace rather than by voice
# alone. The second value is a rate shift in percentage points.
CHARACTER_TONES = [
    ("+0Hz", 0),
    ("+22Hz", 4),
    ("-18Hz", -4),
    ("+38Hz", -6),
    ("-30Hz", 6),
    ("+12Hz", 8),
]

VOICE_POOLS = {
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
}

NARRATOR = "narrator"

# Small enough that the first batch is labelled quickly, large enough that the
# model can still see a whole exchange and tell who is answering whom.
CAST_BATCH_SIZE = 8
CAST_CONTEXT_SEGMENTS = 3

# Merging exists to stop one-word replies each costing their own connection to
# Microsoft. Once a run is this long it has already earned its round trip, so it
# is sent to be spoken rather than held back waiting for the voice to change.
MERGE_MAX_CHARS = 400

SYSTEM_PROMPT = """You cast voice actors for an audiobook.

You are given numbered segments of a story, in reading order. Segments tagged
DIALOGUE are speech taken from inside quotation marks; every other segment is
narration.

Reply with JSON only, in exactly this shape:
{"labels": [{"i": 0, "speaker": "narrator", "gender": "neutral"}]}

Rules:
- Exactly one entry per segment, reusing that segment's own number as "i".
- Every narration segment is {"speaker": "narrator", "gender": "neutral"}.
  Write narrator in lowercase English. Never translate that word.
- For a DIALOGUE segment, work out who is speaking from the narration directly
  before and after it, which usually names them ("मोनू ने पूछा", "the old man
  replied", "she said").
- Two dialogue segments in a row are usually two different people answering
  each other. Do not give every line to the same character.
- Name each character exactly as the story spells it, spelled identically every
  time that character speaks.
- "gender" must be "male", "female" or "neutral".
- Never repeat, translate or quote the story text. Labels only."""

# Small models translate "narrator" even when told not to, which would cast the
# narration as if it were a character.
NARRATOR_ALIASES = {"narrator", "the narrator", "नैरेटर", "नार्रेटर", "कथावाचक", "सूत्रधार"}

# When a line is clearly dialogue but nobody could be pinned to it, alternating
# between two stand-ins at least keeps a conversation sounding like two people.
UNNAMED_SPEAKERS = ("speaker one", "speaker two")


@dataclass(frozen=True)
class Segment:
    text: str
    is_dialogue: bool
    # Set when the text names its own speaker, as a screenplay does. No model
    # is needed to work out who is talking in that case.
    speaker: str | None = None


@dataclass(frozen=True)
class Utterance:
    text: str
    voice: str
    rate: str
    pitch: str
    speaker: str


def _split_narration(text: str) -> list[str]:
    """Group narration into sentence-aligned chunks."""
    chunks: list[str] = []
    buffer = ""

    for sentence in SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{buffer} {sentence}" if buffer else sentence

        if buffer and len(candidate) > NARRATION_CHUNK_CHARS:
            chunks.append(buffer)
            buffer = sentence
        else:
            buffer = candidate

    if buffer:
        chunks.append(buffer)

    return chunks


def parse_script(text: str) -> list[Segment] | None:
    """Read screenplay format — a speaker's name, a colon, then their line.

    Nothing has to be inferred here: the text says who is talking. Returns None
    when the text does not look like a script, so prose falls through to the
    quotation-mark split instead.
    """
    lines = [line for line in text.splitlines() if line.strip()]

    if len(lines) < MIN_SCRIPT_LINES:
        return None

    parsed = [SCRIPT_LINE.match(line) for line in lines]
    named = [match for match in parsed if match]

    if len(named) / len(lines) < SCRIPT_LINE_RATIO:
        return None

    segments: list[Segment] = []

    for match, line in zip(parsed, lines):
        if match is None:
            # Stage directions and stray prose between speeches.
            segments.append(Segment(text=line.strip(), is_dialogue=False))
            continue

        speaker, spoken = match.group(1).strip(), match.group(2).strip()

        # A bare "राम:" with nothing after it is a cast list at the top of the
        # script, not a line to be read out.
        if spoken:
            segments.append(
                Segment(text=spoken, is_dialogue=True, speaker=speaker)
            )

    return segments or None


def split_into_segments(text: str) -> list[Segment]:
    """Cut the story into speakable pieces, tagging which ones are dialogue."""
    scripted = parse_script(text)

    if scripted is not None:
        return scripted

    segments: list[Segment] = []
    cursor = 0

    for match in QUOTE_PATTERN.finditer(text):
        for narration in _split_narration(text[cursor : match.start()]):
            segments.append(Segment(text=narration, is_dialogue=False))

        spoken = (match.group(1) or match.group(2) or "").strip()
        if spoken:
            segments.append(Segment(text=spoken, is_dialogue=True))

        cursor = match.end()

    for narration in _split_narration(text[cursor:]):
        segments.append(Segment(text=narration, is_dialogue=False))

    return segments


def _extract_json(content: str) -> dict:
    """Pull the JSON object out of a reply that may be fenced or prefaced."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("no JSON object in casting reply")

    return json.loads(content[start : end + 1])


async def _ask(system_prompt: str, user_content: str, subject: str) -> dict:
    """One JSON round trip to the model, with every refusal mode named.

    `subject` only ever appears in log lines, to say what the call was about.
    """
    body = {
        "model": NVIDIA_MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    if "gpt-oss" in NVIDIA_MODEL:
        # These models think before answering; at the default effort the pause
        # before the first spoken word roughly doubles.
        body["reasoning_effort"] = "low"

    async with _casting_slots:
        async with httpx.AsyncClient(timeout=CASTING_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json=body,
            )

    if response.status_code != 200:
        # raise_for_status() throws away the body, which is the one place a
        # provider explains itself — a moderation block, a quota, a dead model.
        logger.warning(
            "Casting API returned %s for %s: %s",
            response.status_code,
            subject,
            response.text[:400],
        )
        response.raise_for_status()

    choice = response.json()["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")

    if finish_reason not in (None, "stop"):
        # "content_filter" lands here, as does "length" on a truncated reply.
        logger.warning(
            "Casting stopped early (finish_reason=%s) for %s",
            finish_reason,
            subject,
        )

    try:
        return _extract_json(content)
    except ValueError:
        # A model that declines answers in prose rather than JSON, so the reply
        # itself is the diagnosis.
        logger.warning(
            "Casting reply was not JSON for %s (model=%s, finish_reason=%s): %r",
            subject,
            NVIDIA_MODEL,
            finish_reason,
            content[:400],
        )
        raise


def _is_narrator(speaker: str) -> bool:
    lowered = speaker.strip().lower()
    return lowered in NARRATOR_ALIASES or "narrat" in lowered


async def label_segments(
    segments: list[Segment],
    context: list[str] | None = None,
    known_characters: list[str] | None = None,
) -> dict[int, tuple[str, str]]:
    """Ask the model who speaks each segment. Returns {index: (speaker, gender)}."""
    listing = "\n".join(
        f"[{index}]{' DIALOGUE' if segment.is_dialogue else ''} {segment.text[:400]}"
        for index, segment in enumerate(segments)
    )

    preamble = ""

    if known_characters:
        # Each batch is a fresh call, so without this the same person gets
        # spelled three different ways and ends up with three different voices.
        preamble += (
            "Characters already cast in this story: "
            + ", ".join(known_characters)
            + ".\nWhen one of them speaks again, reuse that exact spelling.\n\n"
        )

    if context:
        # Later batches open mid-scene, so the model is shown the run-up to
        # them. It is told not to label those lines, only to read them.
        preamble += (
            "Earlier lines, for context only — do not label these:\n"
            + "\n".join(context)
            + "\n\n"
        )

    if preamble:
        listing = f"{preamble}Segments to label:\n{listing}"

    parsed = await _ask(SYSTEM_PROMPT, listing, subject=f"{len(segments)} segments")

    labels: dict[int, tuple[str, str]] = {}

    for entry in parsed.get("labels", []):
        try:
            index = int(entry["i"])
        except (KeyError, TypeError, ValueError):
            continue

        if not 0 <= index < len(segments):
            continue

        speaker = str(entry.get("speaker") or NARRATOR).strip() or NARRATOR
        gender = str(entry.get("gender") or "neutral").strip().lower()

        labels[index] = (speaker, gender)

    if not labels:
        # Well-formed JSON carrying nothing: a soft refusal looks exactly like
        # this, and would otherwise pass silently as "everyone is the narrator".
        logger.warning(
            "Casting returned no usable labels for %d segments (model=%s): %r",
            len(segments),
            NVIDIA_MODEL,
            str(parsed)[:400],
        )

    return labels


GENDER_PROMPT = """You are casting voice actors for an audiobook.

You are given the character names from a script. Say whether each character is
male, female or neutral.

Reply with JSON only, in exactly this shape:
{"cast": [{"name": "राम", "gender": "male"}]}

Rules:
- One entry per name, copying the name back exactly as it was given.
- "gender" must be "male", "female" or "neutral".
- Use "neutral" only when the name really gives no indication.
- No other text."""


async def label_genders(names: list[str]) -> dict[str, str]:
    """Ask only who is male or female.

    A script already states who speaks each line, so the model is spared the
    whole story and answers a handful of names instead — seconds, not tens of
    seconds.
    """
    parsed = await _ask(
        GENDER_PROMPT,
        "\n".join(names),
        subject=f"{len(names)} character names",
    )

    genders: dict[str, str] = {}

    for entry in parsed.get("cast", []):
        name = str(entry.get("name") or "").strip()
        gender = str(entry.get("gender") or "neutral").strip().lower()

        if name:
            genders[name] = gender

    return genders


def _shift_rate(rate: str, delta: int) -> str:
    try:
        base = int(rate.strip().rstrip("%"))
    except ValueError:
        base = 0

    return f"{max(-50, min(50, base + delta)):+d}%"


class VoiceAssigner:
    """Hands out voices, remembering the cast across batches.

    A character introduced in the first batch has to keep the same voice when
    they speak again twenty segments later, so this state outlives any one call
    to the model.
    """

    def __init__(self, base_voice: str, base_rate: str, has_narration: bool = True):
        self.base_voice = base_voice
        self.base_rate = base_rate
        self.has_narration = has_narration
        self.pool = VOICE_POOLS.get(base_voice.split("-")[0], VOICE_POOLS["en"])
        self.cast: dict[str, tuple[str, str, str]] = {}
        self.counts = {"male": 0, "female": 0}
        self.unnamed_turns = 0

    def _voice_for(self, speaker: str, gender: str) -> tuple[str, str, str]:
        if speaker not in self.cast:
            if gender not in self.pool:
                # Unknown gender: take whichever side has fewer characters so
                # the two voices stay balanced across the story.
                gender = (
                    "female" if self.counts["female"] <= self.counts["male"] else "male"
                )

            # In a pure script nobody narrates, so the plain unshifted voice is
            # free for the first character to use.
            narrator_voice = (
                (self.base_voice, self.base_rate, "+0Hz")
                if self.has_narration
                else None
            )

            # Otherwise that first tone would reproduce the narrator exactly.
            # Step past any such collision so every character stays distinct.
            for _ in range(len(CHARACTER_TONES)):
                pitch, rate_delta = CHARACTER_TONES[
                    self.counts[gender] % len(CHARACTER_TONES)
                ]
                candidate = (
                    self.pool[gender],
                    _shift_rate(self.base_rate, rate_delta),
                    pitch,
                )
                self.counts[gender] += 1

                if candidate != narrator_voice:
                    break

            self.cast[speaker] = candidate

        return self.cast[speaker]

    def assign(
        self,
        segments: list[Segment],
        labels: dict[int, tuple[str, str]],
    ) -> list[Utterance]:
        utterances: list[Utterance] = []

        for index, segment in enumerate(segments):
            speaker, gender = labels.get(index, (NARRATOR, "neutral"))

            narrates = not segment.is_dialogue or (
                # A script cue of "narrator:" means exactly what it says.
                _is_narrator(speaker) and segment.speaker is not None
            )

            if narrates:
                voice, rate, pitch = self.base_voice, self.base_rate, "+0Hz"
                speaker = NARRATOR
            else:
                if _is_narrator(speaker):
                    # Quoted speech is never the narrator, so the model simply
                    # failed to place it. Hand it to a stand-in rather than let
                    # the line blend back into the narration.
                    speaker = UNNAMED_SPEAKERS[
                        self.unnamed_turns % len(UNNAMED_SPEAKERS)
                    ]
                    gender = "neutral"
                    self.unnamed_turns += 1

                voice, rate, pitch = self._voice_for(speaker, gender)

            utterances.append(
                Utterance(
                    text=segment.text,
                    voice=voice,
                    rate=rate,
                    pitch=pitch,
                    speaker=speaker,
                )
            )

        return utterances


def assign_voices(
    segments: list[Segment],
    labels: dict[int, tuple[str, str]],
    base_voice: str,
    base_rate: str,
) -> list[Utterance]:
    """One-shot casting for a whole story. Used by tests and by `render_to_file`."""
    return _merge_adjacent(VoiceAssigner(base_voice, base_rate).assign(segments, labels))


def _merge_adjacent(utterances: list[Utterance]) -> list[Utterance]:
    """Fold neighbouring lines from one voice together.

    Every utterance costs a websocket round trip to Microsoft, so a story split
    into eighty narration sentences would spend more time connecting than
    speaking.
    """
    merged: list[Utterance] = []

    for utterance in utterances:
        previous = merged[-1] if merged else None

        if (
            previous is not None
            and (previous.voice, previous.rate, previous.pitch) == (utterance.voice, utterance.rate, utterance.pitch)
        ):
            merged[-1] = Utterance(
                text=f"{previous.text} {utterance.text}",
                voice=previous.voice,
                rate=previous.rate,
                pitch=previous.pitch,
                speaker=previous.speaker,
            )
        else:
            merged.append(utterance)

    return merged


def single_voice(text: str, voice: str, rate: str) -> list[Utterance]:
    return [Utterance(text=text, voice=voice, rate=rate, pitch="+0Hz", speaker=NARRATOR)]


def _context_lines(segments: list[Segment], utterances: list[Utterance]) -> list[str]:
    """Describe the tail of the previous batch so the next one opens in scene."""
    lines = []

    for segment, utterance in zip(
        segments[-CAST_CONTEXT_SEGMENTS:], utterances[-CAST_CONTEXT_SEGMENTS:]
    ):
        excerpt = segment.text[:160]

        if segment.is_dialogue:
            excerpt = f'"{excerpt}"'

        lines.append(f"{excerpt} — {utterance.speaker}")

    return lines


def _cannot_cast(text: str) -> str | None:
    if not casting_enabled():
        return "NVIDIA_API_KEY is not set"

    if len(text) > CASTING_MAX_CHARS:
        return f"text is {len(text)} chars, over the {CASTING_MAX_CHARS} limit"

    return None


async def _cast_script(
    segments: list[Segment],
    voice: str,
    rate: str,
) -> list[Utterance]:
    """Cast a screenplay, where every line already names its speaker."""
    names = sorted({segment.speaker for segment in segments if segment.speaker})
    genders: dict[str, str] = {}

    if casting_enabled():
        try:
            genders = await label_genders(names)
        except Exception as error:
            logger.warning(
                "Gender lookup failed (%s: %s) — balancing the %d character "
                "voices instead",
                type(error).__name__,
                error,
                len(names),
            )
    else:
        logger.info(
            "No NVIDIA_API_KEY: casting %d script characters by balance alone",
            len(names),
        )

    assigner = VoiceAssigner(
        voice,
        rate,
        has_narration=any(not segment.is_dialogue for segment in segments),
    )
    labels = {
        index: (segment.speaker, genders.get(segment.speaker, "neutral"))
        for index, segment in enumerate(segments)
        if segment.speaker
    }

    utterances = _merge_adjacent(assigner.assign(segments, labels))

    logger.info(
        "Cast script: %d character(s) across %d line(s) — %s",
        len(assigner.cast),
        len(segments),
        ", ".join(
            f"{name}={genders.get(name, 'balanced')}" for name in names
        ),
    )

    return utterances


async def iter_cast(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
) -> AsyncIterator[Utterance]:
    """Plan the read-through, yielding lines as soon as they are cast.

    Only the first batch is waited on before speaking starts; every later batch
    is labelled while the previous one is still being read aloud.
    """
    if not multi_voice:
        for utterance in single_voice(text, voice, rate):
            yield utterance
        return

    segments = split_into_segments(text)

    if not segments:
        for utterance in single_voice(text, voice, rate):
            yield utterance
        return

    if any(segment.speaker for segment in segments):
        # Screenplay format. Who speaks is already written down, so the model is
        # asked nothing but the genders — and if it is unavailable the reading
        # is still fully cast, just with the voices balanced by guesswork.
        for utterance in await _cast_script(segments, voice, rate):
            yield utterance
        return

    reason = _cannot_cast(text)
    if reason:
        logger.warning("Multi-voice unavailable (%s) — narrating in one voice", reason)
        for utterance in single_voice(text, voice, rate):
            yield utterance
        return

    assigner = VoiceAssigner(voice, rate)

    # Everything ahead of the first quotation mark is narration by construction,
    # so it needs no model at all. Handing it over immediately means the reading
    # begins at once and the opening pages cover the casting call entirely — and
    # a story with no dialogue never touches the API.
    opening = next(
        (index for index, segment in enumerate(segments) if segment.is_dialogue),
        len(segments),
    )
    lead, remainder = segments[:opening], segments[opening:]

    lead_utterances: list[Utterance] = assigner.assign(lead, {}) if lead else []

    for utterance in lead_utterances:
        yield utterance

    if not remainder:
        logger.info("No dialogue found — narrating %d segments in one voice", len(lead))
        return

    batches = [
        remainder[start : start + CAST_BATCH_SIZE]
        for start in range(0, len(remainder), CAST_BATCH_SIZE)
    ]
    pending = asyncio.ensure_future(
        label_segments(batches[0], context=_context_lines(lead, lead_utterances))
    )

    try:
        for position, batch in enumerate(batches):
            try:
                labels = await pending
            except Exception as error:
                # An empty label set still reads correctly: narration keeps the
                # narrator, and dialogue falls through to the stand-in voices.
                logger.warning(
                    "Casting batch %d/%d failed (%s: %s) — its %d dialogue line(s) "
                    "fall back to stand-in voices",
                    position + 1,
                    len(batches),
                    type(error).__name__,
                    error,
                    sum(segment.is_dialogue for segment in batch),
                )
                labels = {}

            utterances = assigner.assign(batch, labels)

            if position + 1 < len(batches):
                pending = asyncio.ensure_future(
                    label_segments(
                        batches[position + 1],
                        context=_context_lines(batch, utterances),
                        known_characters=sorted(assigner.cast),
                    )
                )
            else:
                pending = None

            for utterance in utterances:
                yield utterance
    finally:
        if pending is not None and not pending.done():
            pending.cancel()

    named = sorted(set(assigner.cast) - set(UNNAMED_SPEAKERS))

    logger.info(
        "Cast %d segments: %d named character(s) [%s], %d line(s) on stand-in voices",
        len(segments),
        len(named),
        ", ".join(named) or "none",
        assigner.unnamed_turns,
    )


async def merge_stream(utterances: AsyncIterator[Utterance]) -> AsyncIterator[Utterance]:
    """Fold neighbouring lines from one voice together, as they arrive.

    Same purpose as `_merge_adjacent`, but it only ever holds one line back, so
    merging costs nothing in latency.
    """
    held: Utterance | None = None

    async for utterance in utterances:
        if (
            held is not None
            and len(held.text) < MERGE_MAX_CHARS
            and (held.voice, held.rate, held.pitch)
            == (utterance.voice, utterance.rate, utterance.pitch)
        ):
            held = Utterance(
                text=f"{held.text} {utterance.text}",
                voice=held.voice,
                rate=held.rate,
                pitch=held.pitch,
                speaker=held.speaker,
            )
            continue

        if held is not None:
            yield held

        held = utterance

    if held is not None:
        yield held
