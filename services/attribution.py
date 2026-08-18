"""Read who is speaking straight off the page, where the page says so.

Most misattribution happens on lines the story is not vague about. "तुमने ये
क्यों किया?" रवि चिल्लाया। names its speaker outright, and asking a model to
infer it only gives the model a chance to be wrong — which it was, repeatedly,
handing men's lines to women.

So the narration touching a line is checked for a character's name first, and
the model's answer is only kept where the text genuinely leaves it open.
"""

import re

from services.names import match_key

# Verbs of speaking. A narration segment carrying one of these is an
# attribution rather than an action, which is what makes the name inside it the
# speaker rather than someone being spoken about.
SPEECH_VERB = re.compile(
    r"कह|बोल|पूछ|चिल्ला|चीख|फुसफुसा|बुदबुदा|पुकार|जवाब|उत्तर|टोक|बताया|समझाया"
    r"|दोहरा|गरज|बड़बड़ा|हँसते|रोते|सिसक|मुस्कुरा"
    r"|\b(?:said|asked|replied|shouted|whispered|cried|added|murmured|answered"
    r"|yelled|snapped|muttered|continued|called|repeated|laughed|sobbed)\b",
    re.IGNORECASE,
)

_WORDS = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)

# Grammar, not names. Without this a postposition or pronoun can reduce to the
# same consonants as a character and be read as one.
_FUNCTION_WORDS = {
    "ने", "को", "से", "में", "पर", "का", "के", "की", "और", "कि", "है", "था",
    "थी", "थे", "वह", "वो", "यह", "ये", "उसने", "उसका", "उसकी", "मैंने", "मैं",
    "तुम", "आप", "हम", "फिर", "तो", "भी", "ही", "एक", "अब", "जब", "तब",
    "the", "a", "an", "he", "she", "they", "it", "his", "her", "their", "then",
    "and", "but", "with", "was", "were", "had",
}

# A name has to survive being reduced to consonants without becoming a stub —
# a one-letter key collides with far too much ordinary text.
MIN_KEY_LENGTH = 2

# An interjected attribution — the "रवि ने कहा," in the middle of a speech — is
# short. A long narration between two lines is a scene, and the speaker after it
# is anyone's guess.
INTERJECTION_MAX_CHARS = 60


def _named_in_order(text: str, by_key: dict[str, str]) -> list[str]:
    """Every known character named here, in the order they appear."""
    words = [
        word for word in _WORDS.findall(text) if word.casefold() not in _FUNCTION_WORDS
    ]

    found: list[str] = []

    for position, word in enumerate(words):
        # Bigrams too, so a character called "बूढ़ा आदमी" is not missed for
        # being two words. The longer reading is tried first.
        pair = f"{word} {words[position + 1]}" if position + 1 < len(words) else None

        for candidate in (pair, word):
            if candidate is None:
                continue

            key = match_key(candidate)

            if len(key) >= MIN_KEY_LENGTH and key in by_key:
                found.append(by_key[key])
                break

    return found


def _speaker_beside(segment, by_key: dict[str, str], nearest_last: bool) -> str | None:
    """Who this narration hands the neighbouring line to, if it says at all.

    A narration can name two people — "रवि ने कहा और दरवाज़ा बंद कर दिया। अमन ने
    पूछा," sits between two lines and attributes one to each. Whichever name is
    closest to the line in question is that line's speaker, so the passage is
    read from the end nearest it.
    """
    if segment is None or segment.is_dialogue:
        return None

    if not SPEECH_VERB.search(segment.text):
        return None

    named = _named_in_order(segment.text, by_key)

    if not named:
        return None

    return named[-1] if nearest_last else named[0]


def from_narration(segments: list, names: list[str]) -> dict[int, str]:
    """Attribute what the narration attributes; stay silent about the rest.

    Returns {segment index: character}, covering only dialogue the surrounding
    text names a speaker for.
    """
    by_key: dict[str, str] = {}

    for name in names:
        key = match_key(name)

        if len(key) >= MIN_KEY_LENGTH:
            by_key.setdefault(key, name)

    if not by_key:
        return {}

    attributed: dict[int, str] = {}

    for index, segment in enumerate(segments):
        if not segment.is_dialogue:
            continue

        after = segments[index + 1] if index + 1 < len(segments) else None
        before = segments[index - 1] if index else None

        # "..." रवि ने कहा — the commonest shape in Hindi, and the strongest
        # signal, so it is tried before the narration that leads in.
        speaker = _speaker_beside(after, by_key, nearest_last=False) or (
            _speaker_beside(before, by_key, nearest_last=True)
        )

        if speaker:
            attributed[index] = speaker

    # A speech split by its own attribution belongs to one person throughout:
    # in "A," रवि ने कहा, "B", the narration in the middle says who was already
    # talking — it does not hand the floor to someone else.
    #
    # The middle has to name that same person for this to carry. Propagating on
    # position alone chains: one line correctly given to रवि passes him down
    # through every following exchange, and the whole scene ends up his.
    for index, segment in enumerate(segments):
        if index + 2 >= len(segments) or not segment.is_dialogue:
            continue

        speaker = attributed.get(index)
        joint, second = segments[index + 1], segments[index + 2]

        if (
            speaker
            and second.is_dialogue
            and index + 2 not in attributed
            and len(joint.text) <= INTERJECTION_MAX_CHARS
            and _speaker_beside(joint, by_key, nearest_last=False) == speaker
            and _speaker_beside(joint, by_key, nearest_last=True) == speaker
        ):
            attributed[index + 2] = speaker

    return attributed
