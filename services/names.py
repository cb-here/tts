"""Recognise a character by name when the spelling drifts.

Each batch is a separate call to the model, and the same person comes back as
"रवि" in one and "Ravi" in the next. Treated as written, that is two characters:
two entries in the cast, two different voices, and a gender lookup that misses
and falls through to guesswork. Folding the spellings together is what keeps one
character sounding like one person.
"""

import re

# Enough of Devanagari to romanise a name. Accuracy beyond that is wasted here —
# the result is only ever compared against another name, never shown.
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
_SIGNS = {"ं": "n", "ः": "h", "ँ": "n"}

_HALANT = "्"
_NUKTA = "़"

# Titles that attach to a name in one batch and not the next.
_HONORIFICS = {
    "जी", "साहब", "साहिब", "बाबू", "भाई", "बहन", "दीदी", "अंकल", "आंटी",
    "श्री", "श्रीमती", "पंडित", "चाचा", "चाची", "मामा", "मामी", "बुआ",
    "mr", "mrs", "ms", "miss", "sir", "madam", "uncle", "aunty", "ji",
    "the", "a", "an",
}

_WORD_SPLIT = re.compile(r"[\s.,'’\-_/\\|()\[\]{}\"“”]+")
_NON_LETTER = re.compile(r"[^\wऀ-ॿ]", re.UNICODE)


def romanise(name: str) -> str:
    """Write a Devanagari name in Latin letters. Latin input passes through."""
    out: list[str] = []
    index = 0

    while index < len(name):
        character = name[index]

        if character == _NUKTA:
            index += 1
            continue

        if character in _CONSONANTS:
            out.append(_CONSONANTS[character])

            # A nukta sits between the consonant and whatever follows it.
            ahead = index + 1
            while ahead < len(name) and name[ahead] == _NUKTA:
                ahead += 1

            following = name[ahead] if ahead < len(name) else ""

            if following in _MATRAS:
                out.append(_MATRAS[following])
                index = ahead + 1
            elif following == _HALANT:
                index = ahead + 1
            else:
                # A bare consonant carries an implicit "a": र + व + ि is "ravi",
                # not "rvi", and it is the Latin spelling it has to match.
                out.append("a")
                index += 1

            continue

        out.append(
            _VOWELS.get(character)
            or _MATRAS.get(character)
            or _SIGNS.get(character)
            or character
        )
        index += 1

    return "".join(out)


def match_key(name: str) -> str:
    """A form of a name that survives how it happened to be spelled.

    Reduced to consonants, because that is what the spellings agree on: Pooja
    and पूजा romanise to "poojaa" and "puujaa", but both are p-j. A leading
    vowel is kept, so अमन and मोनू do not collapse into each other.

    Two genuinely different names can still land on one key — राम and रमा both
    give r-m. That is the safer way round: the cost is two characters sharing a
    voice, against one character being split across two.
    """
    words = [
        word
        for word in _WORD_SPLIT.split(name.strip())
        if word and word.casefold() not in _HONORIFICS
    ]

    if not words:
        words = [name.strip()]

    letters = romanise(_NON_LETTER.sub("", "".join(words))).casefold()
    consonants: list[str] = []

    for letter in letters:
        if letter in "aeiou":
            continue

        # An ad-hoc Latin spelling doubles consonants where Devanagari does not
        # — "buddha" against बूढ़ा — so a run counts once.
        if consonants and consonants[-1] == letter:
            continue

        consonants.append(letter)

    if letters[:1] in ("a", "e", "i", "o", "u"):
        consonants.insert(0, letters[0])

    return "".join(consonants)
