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
from services.magpie import is_magpie
from services.names import match_key

logger = logging.getLogger(__name__)

# Casting calls arrive in bursts as several listeners start at once, and the
# free tier answers far slower under that load.
_casting_slots = asyncio.Semaphore(MAX_CONCURRENT_CASTING)

# Paired quote marks only: a bare ' is an apostrophe far more often than it is a
# dialogue opener. The length cap keeps a stray opening mark from swallowing the
# rest of the story, but has to be generous — a speech running several
# paragraphs is still one person talking. `।` is the Devanagari full stop.
QUOTE_PATTERN = re.compile(
    r'["“„]([^"“”„]{1,3000})["”“]'
    r"|‘([^‘’]{1,3000})’"
    r"|«([^«»]{1,3000})»"
    r"|‹([^‹›]{1,3000})›"
    r"|「([^「」]{1,3000})」"
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?।])\s+|\n+")
NARRATION_CHUNK_CHARS = 350

# Screenplay format: a short speaker name followed by a colon. The line may
# carry the speech itself, or the name may stand alone with the speech on the
# lines beneath it. The name length is capped so ordinary prose using a colon
# mid-sentence is not mistaken for a cue.
SCRIPT_COLON = re.compile(r"^[ \t]*([^\s:][^:\n]{0,39}?)[ \t]*:[ \t]*(.*)$")
# "राम - क्यों?" is just as common a convention. The spaces around the dash are
# required, so a compound word or an aside like लिखा था—"रुको।" is not a cue.
SCRIPT_DASH = re.compile(r"^[ \t]*(\S[^\n]{0,39}?)[ \t]+[-–—][ \t]+(\S.*)$")
# Pasted scripts often bullet their dialogue; the marker is not spoken.
SCRIPT_BULLET = re.compile(r"^[ \t]*[-*•·—]+[ \t]+")
# A whole line wrapped in emphasis, or a markdown heading — a sign or a title,
# never a person speaking.
MARKDOWN_LINE = re.compile(r"^(?:\*\*.*\*\*|__.*__|#{1,6}\s+.*)$")

# Counting cues beats measuring what share of lines are cues: in the layout
# where the name sits on its own line, only half the lines are ever cues.
MIN_SCRIPT_CUES = 4
MIN_SCRIPT_SPEAKERS = 2
MIN_SCRIPT_CUE_RATIO = 0.15


def _script_cue(line: str) -> tuple[str, str] | None:
    """Read one line as a speaker cue, in either the colon or dash convention."""
    # A line set entirely in bold is a sign or a heading, not someone speaking.
    # Without this, "**देवगढ़ - 12 किलोमीटर**" reads as a character called
    # "**देवगढ़" and the line is spoken in a voice of its own.
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

# Microsoft ships exactly one male and one female voice for hi-IN, which is why
# a cast used to be one voice pitched several ways. The multilingual voices
# speak Hindi from a newer model, so a story can now field five distinct women
# and four distinct men before anyone has to share. The native hi-IN voice leads
# each list; the rest are ordered by how they were judged to read Hindi.
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

# The locale in a Magpie voice's name labels the speaker, not what they can
# read: all 86 were confirmed to speak Hindi, and the language is sent
# separately. That leaves thirteen distinct people — six women and seven men,
# more than edge-tts can field — and the same pool serves either language.
#
_MAGPIE_FEMALE = [
    "Magpie-Multilingual.EN-US.Mia",
    "Magpie-Multilingual.EN-US.Aria",
    "Magpie-Multilingual.EN-US.Sofia",
    "Magpie-Multilingual.ES-US.Isabela",
    "Magpie-Multilingual.HI-IN.Siwei",
    "Magpie-Multilingual.FR-FR.Louise",
]
_MAGPIE_MALE = [
    "Magpie-Multilingual.EN-US.Jason",
    "Magpie-Multilingual.EN-US.Leo",
    "Magpie-Multilingual.EN-US.Ray",
    "Magpie-Multilingual.ES-US.Diego",
    "Magpie-Multilingual.FR-FR.Pascal",
    "Magpie-Multilingual.VI-VN.Long.Neutral",
    "Magpie-Multilingual.ZH-CN.HouZhen",
]

MAGPIE_VOICE_POOLS = {
    "hi": {"female": _MAGPIE_FEMALE, "male": _MAGPIE_MALE},
    "en": {"female": _MAGPIE_FEMALE, "male": _MAGPIE_MALE},
}

def _speaker_stem(voice: str) -> str:
    """The locale-and-name part of a voice, with any trailing variant dropped."""
    parts = voice.split(".")

    return ".".join(parts[1:3]) if len(parts) >= 3 else voice

# edge-tts voices that only speak their own language. Handed Devanagari they do
# not approximate it — they return no audio at all and the line is simply lost,
# so they must never be reached for as a stand-in or left narrating Hindi. Every
# other voice here was confirmed to read Hindi and English alike.
MONOLINGUAL_EDGE = {
    "en-US-AriaNeural": "en-US-EmmaMultilingualNeural",
    "en-US-GuyNeural": "ko-KR-HyunsuMultilingualNeural",
}

# Where a Magpie voice turns for help. Magpie is a hosted service on a free
# tier, so a piece of a reading can fail while the rest succeeds; when that
# happens the line is still spoken, by the closest edge-tts voice, rather than
# leaving a hole in the story.
EDGE_EQUIVALENT = {
    "EN-US.Mia": "en-US-EmmaMultilingualNeural",
    "EN-US.Aria": "fr-FR-VivienneMultilingualNeural",
    "EN-US.Sofia": "pt-BR-ThalitaMultilingualNeural",
    "ES-US.Isabela": "de-DE-SeraphinaMultilingualNeural",
    "HI-IN.Siwei": "hi-IN-SwaraNeural",
    "HI-IN.Sofia": "pt-BR-ThalitaMultilingualNeural",
    "FR-FR.Louise": "en-US-EmmaMultilingualNeural",
    "ZH-CN.Siwei": "hi-IN-SwaraNeural",
    "EN-US.Jason": "ko-KR-HyunsuMultilingualNeural",
    "EN-US.Leo": "ko-KR-HyunsuMultilingualNeural",
    "EN-US.Ray": "it-IT-GiuseppeMultilingualNeural",
    "ES-US.Diego": "fr-FR-RemyMultilingualNeural",
    "FR-FR.Pascal": "ko-KR-HyunsuMultilingualNeural",
    "VI-VN.Long": "it-IT-GiuseppeMultilingualNeural",
    "ZH-CN.HouZhen": "fr-FR-RemyMultilingualNeural",
    "HI-IN.Leo": "hi-IN-MadhurNeural",
    "HI-IN.Pascal": "ko-KR-HyunsuMultilingualNeural",
}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def speakable(voice: str, language: str) -> str:
    """The voice to actually use, given what language the story is in.

    A listener can pick an English-only voice and paste a Hindi story, and
    edge-tts answers that with silence rather than an error. Swapping it here
    turns a reading that produces nothing into one that simply uses a
    neighbouring voice.
    """
    if language == "hi" and voice in MONOLINGUAL_EDGE:
        return MONOLINGUAL_EDGE[voice]

    return voice


def edge_equivalent(voice: str) -> str:
    """The edge-tts voice standing in for a Magpie one that would not answer.

    Keyed on the speaker rather than the exact voice, so an emotional variant
    falls back to the same stand-in the plain voice would have used.
    """
    if not is_magpie(voice):
        return voice

    return EDGE_EQUIVALENT.get(
        _speaker_stem(voice), "en-US-EmmaMultilingualNeural"
    )


def pool_for(base_voice: str, language: str) -> dict[str, list[str]]:
    """The voices a cast may draw on.

    A reading stays on one engine throughout — the narrator's voice decides
    which — because the two are encoded differently and splicing between them
    mid-story would be audible.
    """
    pools = MAGPIE_VOICE_POOLS if is_magpie(base_voice) else VOICE_POOLS

    return pools.get(language, pools["en"])


def language_of(text: str) -> str:
    """Which voice pool suits the text.

    Taken from the writing itself rather than the narrator's voice, because a
    multilingual narrator is named en-US while still reading Hindi.
    """
    return "hi" if DEVANAGARI.search(text) else "en"

NARRATOR = "narrator"

# Each batch costs one round trip regardless of its size, and the reply is only
# a line per segment, so bigger batches are nearly free. Eight was small enough
# that a long story needed dozens of calls and the casting fell behind the
# reading; this covers several minutes of speech per call instead.
CAST_BATCH_SIZE = 30
CAST_PREFETCH = 3
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
class Label:
    """What the casting model decided about one segment."""

    speaker: str
    gender: str = "neutral"


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
    parsed = [_script_cue(line) for line in lines]
    cues = [cue for cue in parsed if cue]
    speakers = {speaker for speaker, _ in cues}

    if (
        len(cues) < MIN_SCRIPT_CUES
        or len(speakers) < MIN_SCRIPT_SPEAKERS
        or len(cues) / len(lines) < MIN_SCRIPT_CUE_RATIO
        # In a real script the same handful of people speak more than once.
        # Prose that happens to contain dashes throws up a different "speaker"
        # every time, so requiring some repetition separates the two — while
        # still allowing a short scene where four people each speak twice.
        or len(speakers) > len(cues) * 0.75
    ):
        return None

    segments: list[Segment] = []
    speaking: str | None = None

    for cue, line in zip(parsed, lines):
        if cue:
            name, spoken = cue

            if spoken:
                # The cue carried the line itself, so the speaker is done —
                # anything after it is narration until the next cue.
                segments.append(Segment(text=spoken, is_dialogue=True, speaker=name))
                speaking = None
            else:
                # A bare "राहुल:" hands the floor over instead; the speech is on
                # the lines beneath it.
                speaking = name

            continue

        body = SCRIPT_BULLET.sub("", line).strip()

        if not body:
            continue

        if speaking:
            segments.append(Segment(text=body, is_dialogue=True, speaker=speaking))
        else:
            # Narration between speeches, and stage directions.
            segments.append(Segment(text=body, is_dialogue=False))

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

        # Whichever pair of quote marks matched.
        spoken = next((group for group in match.groups() if group), "").strip()
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

    if "gpt-oss" in NVIDIA_MODEL and CASTING_REASONING_EFFORT:
        # Only these models take the setting; sending it to others is an error.
        body["reasoning_effort"] = CASTING_REASONING_EFFORT

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

    if not content:
        # gpt-oss spends the token budget on reasoning before it writes
        # anything, so a raised effort or a long story can use the lot and leave
        # the content empty. Saying that beats a TypeError from json.loads.
        raise ValueError(
            f"casting reply had no content for {subject} "
            f"(finish_reason={finish_reason}, model={NVIDIA_MODEL})"
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

    # "arrator" as well as "narrat", because the model sometimes answers in both
    # scripts at once — "नarrator" begins with a Devanagari न and so matches
    # neither the alias list nor a check anchored on the leading n.
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
    """Ask the model who speaks each segment, and how."""
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

# The opening usually introduces everyone, and sending the whole novel just to
# name its characters is wasteful.
CAST_LIST_SAMPLE_CHARS = 8000


async def identify_cast(text: str) -> dict[str, str]:
    """Work out the whole cast and their genders in one pass, up front.

    Deciding gender from whichever batch happens to mention a character first
    means judging on a fragment, and that judgement then sticks for the rest of
    the story. Reading the story as a whole gives the model the gendered verbs
    it needs.
    """
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
            # Keyed by the story's own spelling, not a folded one: this list is
            # what every later batch is told to name its characters by.
            genders[name] = gender

    return genders


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

    def __init__(
        self,
        base_voice: str,
        base_rate: str,
        has_narration: bool = True,
        known_genders: dict[str, str] | None = None,
        language: str = "hi",
    ):
        self.base_voice = base_voice
        self.base_rate = base_rate
        self.has_narration = has_narration
        self.pool = pool_for(base_voice, language)
        self.cast: dict[str, tuple[str, str, str]] = {}
        self.counts = {"male": 0, "female": 0}
        self.unnamed_turns = 0
        # Decided from the whole story, so it beats whatever a single batch
        # guessed from a fragment.
        self.known_genders: dict[str, str] = {}
        # Both keyed by match_key, so a character is found however this
        # particular batch chose to spell them.
        self._gender_by_key: dict[str, str] = {}
        self._canonical: dict[str, str] = {}

        self.learn_cast(known_genders or {})

    def learn_cast(self, genders: dict[str, str]) -> None:
        """Take the cast read from the whole story as the authority on names."""
        self.known_genders = genders

        for name, gender in genders.items():
            key = match_key(name)

            if key:
                self._gender_by_key[key] = gender
                self._canonical.setdefault(key, name)

    def canonical(self, speaker: str) -> str:
        """The one spelling of this character used for the rest of the story.

        Without it "रवि" in the first batch and "Ravi" in the second are two
        people: two voices for one character, and a gender that is looked up
        under a name nothing was filed under.
        """
        key = match_key(speaker)

        if not key:
            return speaker

        # Whoever names a character first sets the spelling — the up-front cast
        # list if it found them, otherwise the batch that met them first.
        return self._canonical.setdefault(key, speaker)

    def gender_of(self, speaker: str) -> str | None:
        """What the up-front cast list said about this character, if anything."""
        return self._gender_by_key.get(match_key(speaker))

    def _gender_for(self, speaker: str, labelled: str) -> str:
        settled = self.gender_of(speaker)

        if settled in self.pool:
            return settled

        return labelled

    def _voice_for(self, speaker: str, gender: str) -> tuple[str, str, str]:
        if speaker not in self.cast:
            gender = self._gender_for(speaker, gender)

            if gender not in self.pool:
                # Nothing in the story settled it, so balance the two voices.
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

            # Real voices first, and the narrator's own voice last of them, so
            # characters only start sharing a voice at different pitches once
            # every distinct one is taken.
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

                # Before the voice is chosen, so the same person is never asked
                # for twice under two spellings.
                speaker = self.canonical(speaker)

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
    labels: dict[int, Label],
    base_voice: str,
    base_rate: str,
) -> list[Utterance]:
    """One-shot casting for a whole story. Used by tests and by `render_to_file`."""
    language = language_of(" ".join(s.text for s in segments[:20]))
    return _merge_adjacent(
        VoiceAssigner(base_voice, base_rate, language=language).assign(segments, labels)
    )


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
) -> list[Utterance]:
    """Cast a screenplay, where every line already names its speaker."""
    names = sorted({segment.speaker for segment in segments if segment.speaker})
    genders: dict[str, str] = {}

    if casting_enabled():
        try:
            # Read the script itself rather than just the cast list: the lines
            # carry gendered verbs, and a name alone often settles nothing.
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
    )
    labels = {
        index: Label(segment.speaker)
        for index, segment in enumerate(segments)
        if segment.speaker
    }

    utterances = _merge_adjacent(assigner.assign(segments, labels))

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
        for utterance in await _cast_script(segments, voice, rate, text):
            yield utterance
        return

    # Everything ahead of the first quotation mark is narration by construction,
    # so it needs no model at all. Handing it over immediately means the reading
    # begins at once and the opening pages cover the casting call entirely — and
    # a story with no dialogue never touches the API.
    opening = next(
        (index for index, segment in enumerate(segments) if segment.is_dialogue),
        len(segments),
    )
    lead, remainder = segments[:opening], segments[opening:]

    batches = [
        remainder[start : start + CAST_BATCH_SIZE]
        for start in range(0, len(remainder), CAST_BATCH_SIZE)
    ]

    # Judged on the work the text actually creates, not on how long it is. A
    # novel of prose splits into a handful of batches; a short but relentless
    # exchange of one-line dialogue splits into dozens, and it is the second one
    # that floods the API and leaves the reading waiting.
    reason = _cannot_cast(len(batches))
    if reason:
        logger.warning("Multi-voice unavailable (%s) — narrating in one voice", reason)
        for utterance in single_voice(text, voice, rate):
            yield utterance
        return

    # Started before anything else and awaited only when the first character
    # actually needs a voice, so the opening narration covers its cost.
    cast_list = asyncio.ensure_future(identify_cast(text))
    assigner = VoiceAssigner(voice, rate, language=language_of(text))

    lead_utterances: list[Utterance] = assigner.assign(lead, {}) if lead else []

    for utterance in lead_utterances:
        yield utterance

    if not remainder:
        cast_list.cancel()
        return

    try:
        assigner.learn_cast(await cast_list)
    except Exception as error:
        logger.warning(
            "Could not read the cast up front (%s: %s) — genders fall back to "
            "whatever each batch reports",
            type(error).__name__,
            error,
        )

    # Wherever the narration names the speaker outright, that is not a judgement
    # call and the model does not get one. Read once over the whole remainder,
    # so a line is attributed by the narration beside it even when the two land
    # in different batches.
    stated = from_narration(remainder, sorted(assigner.known_genders))

    batches = [
        remainder[start : start + CAST_BATCH_SIZE]
        for start in range(0, len(remainder), CAST_BATCH_SIZE)
    ]

    # Casting has to stay ahead of the reading, and a batch of quickfire dialogue
    # can be spoken in seconds while its labels take half a minute to come back.
    # Several batches are therefore kept in flight at once; waiting to see the
    # previous one first would starve the stream on any dialogue-heavy story.
    inflight: deque = deque()

    def request(position: int) -> None:
        inflight.append(
            asyncio.ensure_future(
                label_segments(
                    batches[position],
                    context=(
                        _context_lines(lead, lead_utterances) if position == 0 else None
                    ),
                    # The cast read from the whole story goes in from the first
                    # batch onwards. Sending only the characters already cast
                    # left that first batch with no names at all, free to
                    # invent a spelling the rest of the story then disagreed
                    # with.
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

            queued = position + len(inflight) + 1
            if queued < len(batches):
                request(queued)

            offset = position * CAST_BATCH_SIZE

            for index in range(len(batch)):
                speaker = stated.get(offset + index)

                if speaker:
                    # The gender the batch guessed is kept only as a fallback;
                    # the cast list read from the whole story still wins.
                    guessed = labels.get(index)
                    labels[index] = Label(
                        speaker=speaker,
                        gender=guessed.gender if guessed else "neutral",
                    )

            for utterance in assigner.assign(batch, labels):
                yield utterance
    finally:
        for task in inflight:
            if not task.done():
                task.cancel()


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
