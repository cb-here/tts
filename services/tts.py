from langgraph.graph import START, StateGraph, END
from typing import TypedDict
from edge_tts import Communicate
from mutagen.id3 import ID3, USLT
from uuid import uuid4
from pathlib import Path

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

class TTSState(TypedDict):
    text: str
    output_file: str
    status: str
    voice: str
    rate: str


async def generate_audio(state: TTSState):
    text = state["text"]
    voice=state["voice"]
    rate=state["rate"]

    communicate = Communicate(
        text=text,
        voice=voice,
        rate=rate
        )
    
    filename = AUDIO_DIR / f"{uuid4()}.mp3"

    await communicate.save(str(filename))

    return {
        "output_file": filename,
        "status": "Done"
    }

def embed_text(state: TTSState):
    text = state["text"]
    output_file = state["output_file"]

    audio = ID3()

    audio.add(
        USLT(
    encoding=3,
    lang="hin",
    desc="Lyrics",
    text=text
)
    )

    audio.save(output_file)

    return {
        "status": "Lyrics Embedded"
    }



graph = StateGraph(TTSState)

graph.add_node("generate_audio", generate_audio)
graph.add_node("embed_text", embed_text)

graph.add_edge(START, "generate_audio")
graph.add_edge("generate_audio", "embed_text")
graph.add_edge("embed_text", END)

workflow = graph.compile()
