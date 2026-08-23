import asyncio
import json
import logging
import re
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from config import (
    CASTING_MAX_BATCHES,
    CASTING_REASONING_EFFORT,
    CASTING_TIMEOUT_SECONDS,
    MAX_CONCURRENT_CASTING,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    casting_enabled,
)
from services.attribution import from_narration
from services.names import match_key

logger = logging.getLogger(__name__)

_casting_slots = asyncio.Semaphore(MAX_CONCURRENT_CASTING)

QUOTE_PATTERN = re.compile(
    r'["“„]([^"“”„]{1,3000})["”“]'
    r"|‘([^‘’]{1,3000})’"
    r"|«([^«»]{1,3000})»"
    r"|‹([^‹›]{1,3000})›"
    r"|「([^「」]{1,3000})」"
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?।])\s+|\n+")
NARRATION_CHUNK_CHARS = 350

SCRIPT_COLON = re.compile(r"^[ \t]*([^\s:][^:\n]{0,39}?)[ \t]*:[ \t]*(.*)$")
SCRIPT_DASH = re.compile(r"^[ \t]*(\S[^\n]{0,39}?)[ \t]+[-–—][ \t]+(\S.*)$")
SCRIPT_BULLET = re.compile(r"^[ \t]*[-*•·—]+[ \t]+")
MARKDOWN_LINE = re.compile(r"^(?:\*\*.*\*\*|__.*__|#{1,6}\s+.*)$")

MIN_SCRIPT_CUES = 4
MIN_SCRIPT_SPEAKERS = 2
MIN_SCRIPT_CUE_RATIO = 0.15


def _script_cue(line: str) -> tuple[str, str] | None:
    if MARKDOWN_LINE.match(line.strip()):
        return None

    for pattern in (SCRIPT_COLON, SCRIPT_DASH):
        match = pattern.match(line)

        if match:
            speaker = match.group(1).strip()

            if speaker.startswith(("*", "_", "#", ">")):
                continue

            return speaker, match.group(2).strip()

    return None

CHARACTER_TONES = [
    ("+0Hz", 0),
    ("+22Hz", 4),
    ("-18Hz", -4),
    ("+38Hz", -6),
    ("-30Hz", 6),
    ("+12Hz", 8),
]

VOICE_POOLS = {
    "hi": {
        "female": [
            "hi-IN-SwaraNeural",
            "en-US-EmmaMultilingualNeural",
            "fr-FR-VivienneMultilingualNeural",
            "pt-BR-ThalitaMultilingualNeural",
            "de-DE-SeraphinaMultilingualNeural",
        ],
        "male": [
            "hi-IN-MadhurNeural",
            "ko-KR-HyunsuMultilingualNeural",
            "it-IT-GiuseppeMultilingualNeural",
            "fr-FR-RemyMultilingualNeural",
        ],
    },
    "en": {
        "female": [
            "en-US-AriaNeural",
            "en-US-EmmaMultilingualNeural",
            "fr-FR-VivienneMultilingualNeural",
            "pt-BR-ThalitaMultilingualNeural",
            "de-DE-SeraphinaMultilingualNeural",
        ],
        "male": [
            "en-US-GuyNeural",
            "ko-KR-HyunsuMultilingualNeural",
            "it-IT-GiuseppeMultilingualNeural",
            "fr-FR-RemyMultilingualNeural",
        ],
    },
}

MONOLINGUAL_EDGE = {
    "en-US-AriaNeural": "en-US-EmmaMultilingualNeural",
    "en-US-GuyNeural": "ko-KR-HyunsuMultilingualNeural",
}

GOOGLE_POOLS = {
    "female": [
        "hi-IN-Chirp3-HD-Achernar",
        "hi-IN-Chirp3-HD-Kore",
        "hi-IN-Chirp3-HD-Leda",
        "hi-IN-Chirp3-HD-Aoede",
        "hi-IN-Chirp3-HD-Despina",
    ],
    "male": [
        "hi-IN-Chirp3-HD-Algenib",
        "hi-IN-Chirp3-HD-Charon",
        "hi-IN-Chirp3-HD-Puck",
        "hi-IN-Chirp3-HD-Orus",
        "hi-IN-Chirp3-HD-Fenrir",
    ],
}

GEMINI_POOLS = {
    "female": [
        "gemini/Kore",
        "gemini/Leda",
        "gemini/Aoede",
        "gemini/Despina",
        "gemini/Zephyr",
    ],
    "male": [
        "gemini/Puck",
        "gemini/Charon",
        "gemini/Orus",
        "gemini/Fenrir",
        "gemini/Algenib",
    ],
}

GOOGLE_FALLBACK = {
    "female": "hi-IN-SwaraNeural",
    "male": "hi-IN-MadhurNeural",
}


def google_gender(voice: str) -> str:
    if voice in GEMINI_POOLS["male"] or voice in GOOGLE_POOLS["male"]:
        return "male"

    return "female"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def speakable(voice: str, language: str) -> str:
    if language == "hi" and voice in MONOLINGUAL_EDGE:
        return MONOLINGUAL_EDGE[voice]

    return voice


def edge_equivalent(voice: str) -> str:
    if "-Chirp3-HD-" in voice or voice.startswith("gemini/"):
        return GOOGLE_FALLBACK[google_gender(voice)]

    return MONOLINGUAL_EDGE.get(voice, voice)


def pool_for(base_voice: str, language: str) -> dict[str, list[str]]:
    if base_voice.startswith("gemini/"):
        return GEMINI_POOLS

    if "-Chirp3-HD-" in base_voice:
        return GOOGLE_POOLS

    return VOICE_POOLS.get(language, VOICE_POOLS["en"])


def language_of(text: str) -> str:
    return "hi" if DEVANAGARI.search(text) else "en"

NARRATOR = "narrator"

CAST_BATCH_SIZE = 24
CAST_PREFETCH = 3
CAST_CONTEXT_SEGMENTS = 3

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
  replied", "she said"). A name in the narration touching a line beats any
  guess from the words themselves.
- A speech broken in half by its own attribution stays with one person. In
  "[3] DIALOGUE तुमने ये क्यों किया?", "[4] रवि चिल्लाया।", "[5] DIALOGUE
  मैंने तुम पर भरोसा किया था!", segments 3 and 5 are BOTH रवि — the narration
  between them says who was talking, it does not hand over to someone else.
  The same holds for "[n] बोली," and "[n] ने कहा," sitting mid-sentence.
- Two dialogue segments with nothing between them are usually two different
  people answering each other. Do not give every line to the same character.
- Name each character exactly as the story spells it, spelled identically every
  time that character speaks.
- "gender" must be "male", "female" or "neutral".
- Never repeat, translate or quote the story text. Labels only."""

NARRATOR_ALIASES = {"narrator", "the narrator", "नैरेटर", "नार्रेटर", "कथावाचक", "सूत्रधार"}

UNNAMED_SPEAKERS = ("speaker one", "speaker two")

QUALIFIERS = {
    "एक", "कोई", "वह", "वो", "उस", "यह", "ये", "the", "a", "an", "some",
}


def _bare_name(speaker: str) -> str:
    words = [
        word for word in speaker.split() if word.casefold() not in QUALIFIERS
    ]

    return " ".join(words) or speaker


@dataclass(frozen=True)
class Segment:
    text: str
    is_dialogue: bool
    speaker: str | None = None


@dataclass(frozen=True)
class Label:

    speaker: str
    gender: str = "neutral"


@dataclass(frozen=True)
class Utterance:
    text: str
    voice: str
    rate: str
    pitch: str
    speaker: str
    is_dialogue: bool = False
    mood: str = ""


def _split_narration(text: str) -> list[str]:
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
    lines = [line for line in text.splitlines() if line.strip()]
    parsed = [_script_cue(line) for line in lines]
    cues = [cue for cue in parsed if cue]
    speakers = {speaker for speaker, _ in cues}

    if (
        len(cues) < MIN_SCRIPT_CUES
        or len(speakers) < MIN_SCRIPT_SPEAKERS
        or len(cues) / len(lines) < MIN_SCRIPT_CUE_RATIO
        or len(speakers) > len(cues) * 0.75
    ):
        return None

    segments: list[Segment] = []
    speaking: str | None = None

    for cue, line in zip(parsed, lines):
        if cue:
            name, spoken = cue

            if spoken:
                segments.append(Segment(text=spoken, is_dialogue=True, speaker=name))
                speaking = None
            else:
                speaking = name

            continue

        body = SCRIPT_BULLET.sub("", line).strip()

        if not body:
            continue

        if speaking:
            segments.append(Segment(text=body, is_dialogue=True, speaker=speaking))
        else:
            segments.append(Segment(text=body, is_dialogue=False))

    return segments or None


def split_into_segments(text: str) -> list[Segment]:
    scripted = parse_script(text)

    if scripted is not None:
        return scripted

    segments: list[Segment] = []
    cursor = 0

    for match in QUOTE_PATTERN.finditer(text):
        for narration in _split_narration(text[cursor : match.start()]):
            segments.append(Segment(text=narration, is_dialogue=False))

        spoken = next((group for group in match.groups() if group), "").strip()
        if spoken:
            segments.append(Segment(text=spoken, is_dialogue=True))

        cursor = match.end()

    for narration in _split_narration(text[cursor:]):
        segments.append(Segment(text=narration, is_dialogue=False))

    return segments


def _extract_json(content: str) -> dict:
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
    body = {
        "model": NVIDIA_MODEL,
        "temperature": 0,
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    if "gpt-oss" in NVIDIA_MODEL and CASTING_REASONING_EFFORT:
        body["reasoning_effort"] = CASTING_REASONING_EFFORT

    async with _casting_slots:
        async with httpx.AsyncClient(timeout=CASTING_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json=body,
            )

    if response.status_code != 200:
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
        logger.warning(
            "Casting stopped early (finish_reason=%s) for %s",
            finish_reason,
            subject,
        )

    if not content:
        raise ValueError(
            f"casting reply had no content for {subject} "
            f"(finish_reason={finish_reason}, model={NVIDIA_MODEL})"
        )

    try:
        return _extract_json(content)
    except ValueError:
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

    return (
        lowered in NARRATOR_ALIASES
        or "narrat" in lowered
        or "arrator" in lowered
    )


async def label_segments(
    segments: list[Segment],
    context: list[str] | None = None,
    known_characters: list[str] | None = None,
) -> dict[int, Label]:
    listing = "\n".join(
        f"[{index}]{' DIALOGUE' if segment.is_dialogue else ''} {segment.text[:400]}"
        for index, segment in enumerate(segments)
    )

    preamble = ""

    if known_characters:
        preamble += (
            "Characters already cast in this story: "
            + ", ".join(known_characters)
            + ".\nWhen one of them speaks again, reuse that exact spelling.\n\n"
        )

    if context:
        preamble += (
            "Earlier lines, for context only — do not label these:\n"
            + "\n".join(context)
            + "\n\n"
        )

    if preamble:
        listing = f"{preamble}Segments to label:\n{listing}"

    parsed = await _ask(SYSTEM_PROMPT, listing, subject=f"{len(segments)} segments")

    labels: dict[int, Label] = {}

    for entry in parsed.get("labels", []):
        try:
            index = int(entry["i"])
        except (KeyError, TypeError, ValueError):
            continue

        if not 0 <= index < len(segments):
            continue

        speaker = str(entry.get("speaker") or NARRATOR).strip() or NARRATOR
        gender = str(entry.get("gender") or "neutral").strip().lower()

        labels[index] = Label(speaker=speaker, gender=gender)

    if not labels:
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


CAST_LIST_PROMPT = """You are casting voice actors for an audiobook.

Read the story and list every character who actually speaks aloud in it.

Reply with JSON only, in exactly this shape:
{"cast": [{"name": "सीता", "gender": "female", "evidence": "सीता बोली"}]}

Rules:
- Name each character exactly as the story spells them, in the story's own script.
- "gender" must be "male", "female" or "neutral".
- Decide the gender in this order:
  1. Gendered verbs and adjectives in the story ("बोली"/"बोला", "कहती"/"कहता",
     "गई"/"गया"), how others address them, and how they are described. This
     always wins, even when it disagrees with the name.
  2. If the story never settles it, use the name itself — most Hindi names are
     clearly gendered (सीता, गुड़िया, प्रिया are female; राम, मोनू, रमेश are male).
- "evidence" is the short phrase that told you, copied from the story, or the
  name if that is what you went on.
- Use "neutral" only when neither the story nor the name gives any indication.
- Do not list the narrator. Do not list people who are only mentioned.
- No other text."""

CAST_LIST_SAMPLE_CHARS = 8000


async def identify_cast(text: str) -> dict[str, str]:
    parsed = await _ask(
        CAST_LIST_PROMPT,
        text[:CAST_LIST_SAMPLE_CHARS],
        subject="the character list",
    )

    genders: dict[str, str] = {}

    for entry in parsed.get("cast", []):
        name = str(entry.get("name") or "").strip()
        gender = str(entry.get("gender") or "neutral").strip().lower()

        if name and not _is_narrator(name):
            genders[name] = gender

    return genders


async def label_genders(names: list[str]) -> dict[str, str]:
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

    def __init__(
        self,
        base_voice: str,
        base_rate: str,
        has_narration: bool = True,
        known_genders: dict[str, str] | None = None,
        language: str = "hi",
        pinned: dict[str, str] | None = None,
        moods: dict[str, str] | None = None,
    ):
        self.base_voice = base_voice
        self.base_rate = base_rate
        self.has_narration = has_narration
        self.pool = pool_for(base_voice, language)
        self.cast: dict[str, tuple[str, str, str]] = {}
        self.counts = {"male": 0, "female": 0}
        self.unnamed_turns = 0
        self.after_unnamed = False
        self.known_genders: dict[str, str] = {}
        self._gender_by_key: dict[str, str] = {}
        self._canonical: dict[str, str] = {}
        self._pinned: dict[str, str] = {}
        self._moods: dict[str, str] = {}

        for name, choice in (pinned or {}).items():
            key = match_key(name)

            if key:
                self._pinned[key] = speakable(choice, language)

        for name, mood in (moods or {}).items():
            key = match_key(name)

            if key and mood.strip():
                self._moods[key] = mood.strip()

        self.learn_cast(known_genders or {})

    def learn_cast(self, genders: dict[str, str]) -> None:
        self.known_genders = genders

        for name, gender in genders.items():
            key = match_key(name)

            if key:
                self._gender_by_key[key] = gender
                self._canonical.setdefault(key, name)

    def mood_for(self, speaker: str) -> str:
        return self._moods.get(match_key(speaker), "")

    def canonical(self, speaker: str) -> str:
        bare = _bare_name(speaker)
        key = match_key(bare)

        if not key:
            return speaker

        return self._canonical.setdefault(key, bare)

    def gender_of(self, speaker: str) -> str | None:
        return self._gender_by_key.get(match_key(speaker))

    def _gender_for(self, speaker: str, labelled: str) -> str:
        settled = self.gender_of(speaker)

        if settled in self.pool:
            return settled

        return labelled

    def _voice_for(self, speaker: str, gender: str) -> tuple[str, str, str]:
        if speaker not in self.cast:
            chosen = self._pinned.get(match_key(speaker))

            if chosen:
                self.cast[speaker] = (chosen, self.base_rate, "+0Hz")
                return self.cast[speaker]

            gender = self._gender_for(speaker, gender)

            if gender not in self.pool:
                gender = (
                    "female" if self.counts["female"] <= self.counts["male"] else "male"
                )

            narrator_voice = (
                (self.base_voice, self.base_rate, "+0Hz")
                if self.has_narration
                else None
            )

            voices = sorted(self.pool[gender], key=lambda v: v == self.base_voice)

            for _ in range(len(voices) * len(CHARACTER_TONES)):
                turn = self.counts[gender]
                pitch, rate_delta = CHARACTER_TONES[
                    (turn // len(voices)) % len(CHARACTER_TONES)
                ]
                candidate = (
                    voices[turn % len(voices)],
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
        labels: dict[int, Label],
    ) -> list[Utterance]:
        utterances: list[Utterance] = []

        for index, segment in enumerate(segments):
            label = labels.get(index) or Label(NARRATOR)
            speaker, gender = label.speaker, label.gender

            narrates = not segment.is_dialogue or (
                _is_narrator(speaker) and segment.speaker is not None
            )

            if narrates:
                voice, rate, pitch = self.base_voice, self.base_rate, "+0Hz"
                speaker = NARRATOR
                self.after_unnamed = False
            else:
                if _is_narrator(speaker):
                    if self.after_unnamed:
                        self.unnamed_turns += 1
                    else:
                        self.unnamed_turns = 0

                    speaker = UNNAMED_SPEAKERS[
                        self.unnamed_turns % len(UNNAMED_SPEAKERS)
                    ]
                    gender = "neutral"
                    self.after_unnamed = True
                else:
                    self.after_unnamed = False

                speaker = self.canonical(speaker)

                voice, rate, pitch = self._voice_for(speaker, gender)

            utterances.append(
                Utterance(
                    text=segment.text,
                    voice=voice,
                    rate=rate,
                    pitch=pitch,
                    speaker=speaker,
                    is_dialogue=not narrates,
                    mood="" if narrates else self.mood_for(speaker),
                )
            )

        return utterances


def assign_voices(
    segments: list[Segment],
    labels: dict[int, Label],
    base_voice: str,
    base_rate: str,
) -> list[Utterance]:
    language = language_of(" ".join(s.text for s in segments[:20]))
    return _merge_adjacent(
        VoiceAssigner(base_voice, base_rate, language=language).assign(segments, labels)
    )


def _merge_adjacent(utterances: list[Utterance]) -> list[Utterance]:
    merged: list[Utterance] = []

    for utterance in utterances:
        previous = merged[-1] if merged else None

        if (
            previous is not None
            and not (previous.is_dialogue and utterance.is_dialogue)
            and (previous.voice, previous.rate, previous.pitch, previous.mood)
            == (utterance.voice, utterance.rate, utterance.pitch, utterance.mood)
        ):
            merged[-1] = Utterance(
                text=f"{previous.text} {utterance.text}",
                voice=previous.voice,
                rate=previous.rate,
                pitch=previous.pitch,
                speaker=previous.speaker,
                is_dialogue=previous.is_dialogue,
                mood=previous.mood,
            )
        else:
            merged.append(utterance)

    return merged


def single_voice(text: str, voice: str, rate: str) -> list[Utterance]:
    return [
        Utterance(
            text=text, voice=voice, rate=rate, pitch="+0Hz", speaker=NARRATOR
        )
    ]


def _context_lines(segments: list[Segment], utterances: list[Utterance]) -> list[str]:
    lines = []

    for segment, utterance in zip(
        segments[-CAST_CONTEXT_SEGMENTS:], utterances[-CAST_CONTEXT_SEGMENTS:]
    ):
        excerpt = segment.text[:160]

        if segment.is_dialogue:
            excerpt = f'"{excerpt}"'

        lines.append(f"{excerpt} — {utterance.speaker}")

    return lines


def _cannot_cast(batches: int) -> str | None:
    if not casting_enabled():
        return "NVIDIA_API_KEY is not set"

    if batches > CASTING_MAX_BATCHES:
        return (
            f"casting it would take {batches} model calls, over the "
            f"{CASTING_MAX_BATCHES} allowed"
        )

    return None


async def _cast_script(
    segments: list[Segment],
    voice: str,
    rate: str,
    text: str,
    cast_genders: dict[str, str] | None = None,
    cast_voices: dict[str, str] | None = None,
    cast_moods: dict[str, str] | None = None,
) -> list[Utterance]:
    names = sorted({segment.speaker for segment in segments if segment.speaker})
    genders: dict[str, str] = dict(cast_genders or {})

    if not genders and casting_enabled():
        try:
            genders = await identify_cast(text)
        except Exception as error:
            logger.warning(
                "Gender lookup failed (%s: %s) — balancing the %d character "
                "voices instead",
                type(error).__name__,
                error,
                len(names),
            )
    assigner = VoiceAssigner(
        voice,
        rate,
        has_narration=any(not segment.is_dialogue for segment in segments),
        known_genders=genders,
        language=language_of(" ".join(s.text for s in segments[:20])),
        pinned=cast_voices,
        moods=cast_moods,
    )
    labels = {
        index: Label(segment.speaker)
        for index, segment in enumerate(segments)
        if segment.speaker
    }

    utterances = _merge_adjacent(assigner.assign(segments, labels))

    return utterances


def _other_speaker(labels: dict, around: int, avoid: str) -> str | None:
    reach = max(around, max(labels) - around if labels else 0)

    for step in range(1, reach + 1):
        for index in (around - step, around + step):
            label = labels.get(index)

            if (
                label
                and not _is_narrator(label.speaker)
                and label.speaker != avoid
            ):
                return label.speaker

    return None


def fill_gaps(batch: list, labels: dict) -> None:
    for index, segment in enumerate(batch):
        if not segment.is_dialogue:
            continue

        label = labels.get(index)

        if label is not None and not _is_narrator(label.speaker):
            continue

        neighbour = None

        for step in (index - 1, index + 1):
            if 0 <= step < len(batch) and batch[step].is_dialogue:
                candidate = labels.get(step)

                if candidate and not _is_narrator(candidate.speaker):
                    neighbour = candidate.speaker
                    break

        if neighbour is None:
            continue

        other = _other_speaker(labels, index, neighbour)

        if other is None:
            continue

        labels[index] = Label(speaker=other, gender="neutral")


def _runs(batch: list) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []

    for index, segment in enumerate(batch):
        if segment.is_dialogue:
            current.append(index)
        elif current:
            runs.append(current)
            current = []

    if current:
        runs.append(current)

    return [run for run in runs if len(run) > 1]


def take_turns(batch: list, labels: dict, stated: dict, offset: int) -> None:
    for run in _runs(batch):
        named = [
            labels[index].speaker
            for index in run
            if index in labels and not _is_narrator(labels[index].speaker)
        ]

        pair = list(dict.fromkeys(named))

        if len(pair) != 2:
            continue

        anchors = [
            (position, stated[offset + index])
            for position, index in enumerate(run)
            if offset + index in stated
        ]

        if not anchors:
            continue

        position, speaker = anchors[0]

        if speaker not in pair:
            continue

        first = pair.index(speaker) if position % 2 == 0 else 1 - pair.index(speaker)

        for step, index in enumerate(run):
            labels[index] = Label(
                speaker=pair[(first + step) % 2],
                gender=labels[index].gender if index in labels else "neutral",
            )



async def iter_cast(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
    cast_genders: dict[str, str] | None = None,
    cast_voices: dict[str, str] | None = None,
    cast_moods: dict[str, str] | None = None,
) -> AsyncIterator[Utterance]:
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
        for utterance in await _cast_script(
            segments, voice, rate, text, cast_genders, cast_voices, cast_moods
        ):
            yield utterance
        return

    opening = next(
        (index for index, segment in enumerate(segments) if segment.is_dialogue),
        len(segments),
    )
    lead, remainder = segments[:opening], segments[opening:]

    batches = [
        remainder[start : start + CAST_BATCH_SIZE]
        for start in range(0, len(remainder), CAST_BATCH_SIZE)
    ]

    reason = _cannot_cast(len(batches))
    if reason:
        logger.warning("Multi-voice unavailable (%s) — narrating in one voice", reason)
        for utterance in single_voice(text, voice, rate):
            yield utterance
        return

    cast_list = (
        None if cast_genders else asyncio.ensure_future(identify_cast(text))
    )
    assigner = VoiceAssigner(
        voice,
        rate,
        language=language_of(text),
        known_genders=cast_genders,
        pinned=cast_voices,
        moods=cast_moods,
    )

    lead_utterances: list[Utterance] = assigner.assign(lead, {}) if lead else []

    for utterance in lead_utterances:
        yield utterance

    if not remainder:
        if cast_list:
            cast_list.cancel()
        return

    if cast_list:
        try:
            assigner.learn_cast(await cast_list)
        except Exception as error:
            logger.warning(
                "Could not read the cast up front (%s: %s) — genders fall back "
                "to whatever each batch reports",
                type(error).__name__,
                error,
            )

    stated = from_narration(remainder, sorted(assigner.known_genders))

    batches = [
        remainder[start : start + CAST_BATCH_SIZE]
        for start in range(0, len(remainder), CAST_BATCH_SIZE)
    ]

    inflight: deque = deque()

    def request(position: int) -> None:
        inflight.append(
            asyncio.ensure_future(
                label_segments(
                    batches[position],
                    context=(
                        _context_lines(lead, lead_utterances) if position == 0 else None
                    ),
                    known_characters=sorted(
                        set(assigner.cast) | set(assigner.known_genders)
                    )
                    or None,
                )
            )
        )

    try:
        for position in range(min(CAST_PREFETCH, len(batches))):
            request(position)

        for position, batch in enumerate(batches):
            try:
                labels = await inflight.popleft()
            except Exception as error:
                logger.warning(
                    "Casting batch %d/%d failed (%s: %s) — asking once more",
                    position + 1,
                    len(batches),
                    type(error).__name__,
                    error,
                )

                try:
                    labels = await label_segments(
                        batch,
                        known_characters=sorted(
                            set(assigner.cast) | set(assigner.known_genders)
                        )
                        or None,
                    )
                except Exception as retry_error:
                    logger.warning(
                        "Casting batch %d/%d failed again (%s: %s) — its %d "
                        "dialogue line(s) fall back to stand-in voices",
                        position + 1,
                        len(batches),
                        type(retry_error).__name__,
                        retry_error,
                        sum(segment.is_dialogue for segment in batch),
                    )
                    labels = {}

            queued = position + len(inflight) + 1
            if queued < len(batches):
                request(queued)

            offset = position * CAST_BATCH_SIZE

            for index in range(len(batch)):
                speaker = stated.get(offset + index)

                if speaker:
                    guessed = labels.get(index)
                    labels[index] = Label(
                        speaker=speaker,
                        gender=guessed.gender if guessed else "neutral",
                    )

            take_turns(batch, labels, stated, offset)
            fill_gaps(batch, labels)

            for utterance in assigner.assign(batch, labels):
                yield utterance
    finally:
        for task in inflight:
            if not task.done():
                task.cancel()


async def merge_stream(utterances: AsyncIterator[Utterance]) -> AsyncIterator[Utterance]:
    held: Utterance | None = None

    async for utterance in utterances:
        if (
            held is not None
            and len(held.text) < MERGE_MAX_CHARS
            and not (held.is_dialogue and utterance.is_dialogue)
            and (held.voice, held.rate, held.pitch, held.mood)
            == (utterance.voice, utterance.rate, utterance.pitch, utterance.mood)
        ):
            held = Utterance(
                text=f"{held.text} {utterance.text}",
                voice=held.voice,
                rate=held.rate,
                pitch=held.pitch,
                speaker=held.speaker,
                is_dialogue=held.is_dialogue,
                mood=held.mood,
            )
            continue

        if held is not None:
            yield held

        held = utterance

    if held is not None:
        yield held
