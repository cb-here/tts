"""Turn raw PCM into the mp3 frames the stream is made of.

edge-tts hands back mp3 already, but Magpie returns a WAV, and the two have to
reach the listener as one continuous track. Encoding each piece separately is
fine: mp3 frames concatenate, which is the same property the multi-voice reading
already relies on.
"""

import io
import wave
from functools import lru_cache

import lameenc
import numpy as np

# Speech at 22 kHz mono; past this the bitrate is spent on room tone.
MP3_BITRATE = 64

# A tone shift moves pitch and tempo together, so it stays a nudge rather than a
# transformation. Beyond this a voice starts sounding like a cartoon.
MIN_TONE = 0.80
MAX_TONE = 1.25


def read_wav(data: bytes) -> tuple[bytes, int]:
    """Pull the samples and the sample rate out of a WAV file."""
    with wave.open(io.BytesIO(data)) as handle:
        width = handle.getsampwidth()

        if width != 2:
            raise ValueError(f"expected 16-bit audio, got {width * 8}-bit")

        channels = handle.getnchannels()
        pcm = handle.readframes(handle.getnframes())
        rate = handle.getframerate()

    if channels > 1:
        # Keep the first channel rather than mixing: Python 3.13 dropped
        # `audioop`, and a synthesised voice is the same signal on both sides.
        pcm = b"".join(
            pcm[index : index + 2] for index in range(0, len(pcm), 2 * channels)
        )

    return pcm, rate


def encode_mp3(pcm: bytes, rate: int, tone: float = 1.0) -> bytes:
    """Encode one piece of PCM as a self-contained run of mp3 frames.

    `tone` above 1 reads faster and higher, below 1 slower and lower. Magpie
    takes no rate or pitch argument, so this is the only way to honour the speed
    control — and to tell two characters apart once the voice pool runs out. It
    is done by lying to the encoder about the input rate and having it resample
    back to the real one, which costs nothing on top of the encode.
    """
    tone = max(MIN_TONE, min(MAX_TONE, tone))

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE)
    encoder.set_in_sample_rate(int(round(rate * tone)))
    encoder.set_out_sample_rate(rate)
    encoder.set_channels(1)
    encoder.set_quality(5)
    # Suppresses the Xing/LAME header. That header describes a whole file, and
    # several of them spliced into one stream would each claim to be its start —
    # players seek by the first one they find.
    encoder.silence()

    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())


# Roughly 46 ms at 22 kHz: long enough to hold a pitch period of any voice,
# short enough that a syllable is not smeared across the join.
_FRAME = 1024
_HOP = _FRAME // 2
# How far the next frame may be nudged to line its waveform up with the last
# one. Without the search, overlap-add cancels itself wherever the two are out
# of phase and the voice goes hollow.
_SEARCH = _FRAME // 4
_SEARCH_STEP = 8


def stretch(pcm: bytes, speed: float) -> bytes:
    """Change how fast this is spoken without changing the voice.

    Above 1 is faster, below 1 slower. Resampling would do the same job in two
    lines, but it drags the pitch along with it — a story slowed to sound
    unhurried comes back in a deeper voice than the one that was chosen. This
    keeps the pitch where it was and only moves the clock.
    """
    if abs(speed - 1.0) < 0.01 or not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)

    if len(samples) < _FRAME * 2:
        return pcm

    window = np.hanning(_FRAME).astype(np.float32)
    out = np.zeros(int(len(samples) / speed) + _FRAME * 2, dtype=np.float32)
    weight = np.zeros_like(out)

    # What would have come next had the audio not been cut here. Each following
    # frame is chosen to continue it as smoothly as possible.
    natural: np.ndarray | None = None
    position = 0.0
    written = 0

    while True:
        start = int(round(position))

        if natural is not None:
            low = max(0, start - _SEARCH)
            high = min(len(samples) - _FRAME, start + _SEARCH)

            if high > low:
                candidates = np.lib.stride_tricks.sliding_window_view(
                    samples[low : high + _FRAME], _FRAME
                )[::_SEARCH_STEP]
                start = low + int(np.argmax(candidates @ natural)) * _SEARCH_STEP

        if start + _FRAME > len(samples) or written + _FRAME > len(out):
            break

        out[written : written + _FRAME] += samples[start : start + _FRAME] * window
        weight[written : written + _FRAME] += window

        natural = samples[start + _HOP : start + _HOP + _FRAME]

        if len(natural) < _FRAME:
            break

        position += _HOP * speed
        written += _HOP

    end = written + _FRAME
    mixed = out[:end] / np.maximum(weight[:end], 1e-6)

    return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()


@lru_cache(maxsize=64)
def silence(seconds: float, rate: int) -> bytes:
    """A run of mp3 frames with nothing in them.

    Each piece of a reading is synthesised on its own and the frames are simply
    concatenated, so without this the last word of one runs straight into the
    first word of the next — no breath at a full stop, no beat before someone
    answers. Cached because a reading asks for the same few lengths over and
    over.
    """
    if seconds <= 0:
        return b""

    return encode_mp3(bytes(int(rate * seconds) * 2), rate)
