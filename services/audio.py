"""Turn raw PCM into the mp3 frames the stream is made of.

edge-tts hands back mp3 already, but Magpie returns a WAV, and the two have to
reach the listener as one continuous track. Encoding each piece separately is
fine: mp3 frames concatenate, which is the same property the multi-voice reading
already relies on.
"""

import io
import wave

import lameenc

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
