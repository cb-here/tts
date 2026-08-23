import re

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

_HONORIFICS = {
    "जी", "साहब", "साहिब", "बाबू", "भाई", "बहन", "दीदी", "अंकल", "आंटी",
    "श्री", "श्रीमती", "पंडित", "चाचा", "चाची", "मामा", "मामी", "बुआ",
    "mr", "mrs", "ms", "miss", "sir", "madam", "uncle", "aunty", "ji",
    "the", "a", "an",
}

_WORD_SPLIT = re.compile(r"[\s.,'’\-_/\\|()\[\]{}\"“”]+")
_NON_LETTER = re.compile(r"[^\wऀ-ॿ]", re.UNICODE)


def romanise(name: str) -> str:
    out: list[str] = []
    index = 0

    while index < len(name):
        character = name[index]

        if character == _NUKTA:
            index += 1
            continue

        if character in _CONSONANTS:
            out.append(_CONSONANTS[character])

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

        if consonants and consonants[-1] == letter:
            continue

        consonants.append(letter)

    if letters[:1] in ("a", "e", "i", "o", "u"):
        consonants.insert(0, letters[0])

    return "".join(consonants)
