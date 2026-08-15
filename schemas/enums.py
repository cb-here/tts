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
