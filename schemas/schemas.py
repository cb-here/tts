from pydantic import BaseModel, Field
from schemas.enums import VoiceEnum

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: VoiceEnum = VoiceEnum.swara
    rate: str = "+0%"


class TTSStreamRequest(TTSRequest):
    # Beta: let an LLM split the text by speaker and cast a voice per character.
    # `voice` stays the fallback, and is used for narration.
    multi_voice: bool = False


class TTSStreamSession(BaseModel):
    session_id: str
    stream_url: str
