from functools import lru_cache

import lameenc

MP3_BITRATE = 64


def encode_mp3(pcm: bytes, rate: int, bitrate: int = MP3_BITRATE) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(rate)
    encoder.set_out_sample_rate(rate)
    encoder.set_channels(1)
    encoder.set_quality(5)
    encoder.silence()

    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())


@lru_cache(maxsize=64)
def silence(seconds: float, rate: int, bitrate: int = MP3_BITRATE) -> bytes:
    if seconds <= 0:
        return b""

    return encode_mp3(bytes(int(rate * seconds) * 2), rate, bitrate)
