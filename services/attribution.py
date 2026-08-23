import re

from services.names import match_key

SPEECH_VERB = re.compile(
    r"कह|बोल|पूछ|चिल्ला|चीख|फुसफुसा|बुदबुदा|पुकार|जवाब|उत्तर|टोक|बताया|समझाया"
    r"|दोहरा|गरज|बड़बड़ा|हँसते|रोते|सिसक|मुस्कुरा"
    r"|\b(?:said|asked|replied|shouted|whispered|cried|added|murmured|answered"
    r"|yelled|snapped|muttered|continued|called|repeated|laughed|sobbed)\b",
    re.IGNORECASE,
)

_WORDS = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)

_FUNCTION_WORDS = {
    "ने", "को", "से", "में", "पर", "का", "के", "की", "और", "कि", "है", "था",
    "थी", "थे", "वह", "वो", "यह", "ये", "उसने", "उसका", "उसकी", "मैंने", "मैं",
    "तुम", "आप", "हम", "फिर", "तो", "भी", "ही", "एक", "अब", "जब", "तब",
    "the", "a", "an", "he", "she", "they", "it", "his", "her", "their", "then",
    "and", "but", "with", "was", "were", "had",
}

MIN_KEY_LENGTH = 2

INTERJECTION_MAX_CHARS = 60


def _named_in_order(text: str, by_key: dict[str, str]) -> list[str]:
    words = [
        word for word in _WORDS.findall(text) if word.casefold() not in _FUNCTION_WORDS
    ]

    found: list[str] = []

    for position, word in enumerate(words):
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
    if segment is None or segment.is_dialogue:
        return None

    if not SPEECH_VERB.search(segment.text):
        return None

    named = _named_in_order(segment.text, by_key)

    if not named:
        return None

    return named[-1] if nearest_last else named[0]


def from_narration(segments: list, names: list[str]) -> dict[int, str]:
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

        speaker = _speaker_beside(after, by_key, nearest_last=False) or (
            _speaker_beside(before, by_key, nearest_last=True)
        )

        if speaker:
            attributed[index] = speaker

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
