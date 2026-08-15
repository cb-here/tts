from pydantic import BaseModel, Field
from schemas.enums import VoiceEnum

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: VoiceEnum = VoiceEnum.emma
    rate: str = "+0%"


class TTSStreamRequest(TTSRequest):
    # Beta: let an LLM split the text by speaker and cast a voice per character.
    # `voice` stays the fallback, and is used for narration.
    multi_voice: bool = False


class DevanagariRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DevanagariResponse(BaseModel):
    text: str


class TTSStreamSession(BaseModel):
    session_id: str
    stream_url: str
    # A streamed response has no length, so the player has nothing to scale its
    # progress bar against until the whole reading has arrived.
    estimated_seconds: float
