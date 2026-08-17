from enum import Enum

class VoiceEnum(str, Enum):
    # Microsoft ships only these two for hi-IN.
    swara = "hi-IN-SwaraNeural"
    madhur = "hi-IN-MadhurNeural"

    # Multilingual voices: a newer, more expressive model that speaks Hindi as
    # well, which is the only way past the two voices above.
    emma = "en-US-EmmaMultilingualNeural"
    vivienne = "fr-FR-VivienneMultilingualNeural"
    thalita = "pt-BR-ThalitaMultilingualNeural"
    seraphina = "de-DE-SeraphinaMultilingualNeural"
    hyunsu = "ko-KR-HyunsuMultilingualNeural"
    giuseppe = "it-IT-GiuseppeMultilingualNeural"
    remy = "fr-FR-RemyMultilingualNeural"

    aria = "en-US-AriaNeural"
    guy = "en-US-GuyNeural"

    # NVIDIA Magpie. A different engine entirely — warmer and far less flat than
    # edge-tts. Picking one of these switches the whole reading, cast included.
    #
    # The locale in each name labels the speaker, not what they can read: every
    # one of them speaks Hindi and English alike, because the language is sent
    # separately from the voice. That is thirteen distinct people, where
    # edge-tts fields nine.
    magpie_mia = "Magpie-Multilingual.EN-US.Mia"
    magpie_aria = "Magpie-Multilingual.EN-US.Aria"
    magpie_sofia = "Magpie-Multilingual.EN-US.Sofia"
    magpie_isabela = "Magpie-Multilingual.ES-US.Isabela"
    magpie_siwei = "Magpie-Multilingual.HI-IN.Siwei"
    magpie_louise = "Magpie-Multilingual.FR-FR.Louise"

    magpie_jason = "Magpie-Multilingual.EN-US.Jason"
    magpie_leo = "Magpie-Multilingual.EN-US.Leo"
    magpie_ray = "Magpie-Multilingual.EN-US.Ray"
    magpie_diego = "Magpie-Multilingual.ES-US.Diego"
    magpie_pascal = "Magpie-Multilingual.FR-FR.Pascal"
    magpie_long = "Magpie-Multilingual.VI-VN.Long.Neutral"
    magpie_houzhen = "Magpie-Multilingual.ZH-CN.HouZhen"
