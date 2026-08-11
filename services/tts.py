import asyncio
from collections.abc import AsyncIterator
from config import MAX_CONCURRENT_TTS
from io import BytesIO
from langgraph.config import get_stream_writer
from langgraph.graph import START, StateGraph, END
from typing import TypedDict
from edge_tts import Communicate
from mutagen.id3 import ID3, USLT
from services.casting import iter_cast, merge_stream
from uuid import uuid4
from pathlib import Path

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Held per utterance rather than per request: a listener is mid-story for
# minutes, so locking for a whole stream would stall everyone behind them.
_tts_slots = asyncio.Semaphore(MAX_CONCURRENT_TTS)

class TTSState(TypedDict):
    text: str
    status: str
    voice: str
    rate: str
    multi_voice: bool


def build_lyrics_tag(text: str) -> bytes:
    """Render the ID3 lyrics tag on its own, without touching a file.

    An ID3v2 tag lives at the *front* of an mp3, so it can be emitted before a
    single audio byte exists. That is what lets the whole pipeline stream: the
    lyrics go out first, then audio flows in behind them.
    """
    buffer = BytesIO()

    tag = ID3()
    tag.add(
        USLT(
            encoding=3,
            lang="hin",
            desc="Lyrics",
            text=text,
        )
    )
    tag.save(buffer, v2_version=4, v1=0)

    return buffer.getvalue()


def embed_text(state: TTSState):
    writer = get_stream_writer()
    writer(build_lyrics_tag(state["text"]))

    return {
        "status": "Lyrics Embedded"
    }


async def generate_audio(state: TTSState):
    writer = get_stream_writer()

    cast = merge_stream(
        iter_cast(
            text=state["text"],
            voice=state["voice"],
            rate=state["rate"],
            multi_voice=state.get("multi_voice", False),
        )
    )

    spoken = 0

    # Each utterance is its own mp3, and mp3 frames concatenate cleanly, so the
    # speakers stitch into one continuous track.
    async for utterance in cast:
        communicate = Communicate(
            text=utterance.text,
            voice=utterance.voice,
            rate=utterance.rate,
            pitch=utterance.pitch,
            )

        async with _tts_slots:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    writer(chunk["data"])

        spoken += 1

    return {
        "status": f"Done ({spoken} utterance{'' if spoken == 1 else 's'})"
    }



graph = StateGraph(TTSState)

graph.add_node("embed_text", embed_text)
graph.add_node("generate_audio", generate_audio)

graph.add_edge(START, "embed_text")
graph.add_edge("embed_text", "generate_audio")
graph.add_edge("generate_audio", END)

workflow = graph.compile()


async def stream_tts(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
) -> AsyncIterator[bytes]:
    """Yield mp3 bytes as edge-tts produces them.

    `stream_mode="custom"` surfaces whatever the nodes hand to their stream
    writer, so each chunk here is raw audio rather than a state update.
    """
    async for chunk in workflow.astream(
        {"text": text, "voice": voice, "rate": rate, "multi_voice": multi_voice},
        stream_mode="custom",
    ):
        yield chunk


async def render_to_file(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
) -> Path:
    """Drain the stream into a complete mp3 on disk."""
    filename = AUDIO_DIR / f"{uuid4()}.mp3"

    with filename.open("wb") as audio_file:
        async for chunk in stream_tts(text, voice, rate, multi_voice):
            audio_file.write(chunk)

    return filename
