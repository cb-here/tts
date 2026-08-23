from typing import Literal
from pydantic import BaseModel, Field
from schemas.enums import VoiceEnum

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: VoiceEnum = VoiceEnum.emma
    rate: str = "+0%"


class CastMember(BaseModel):

    gender: Literal["male", "female", "neutral"] = "neutral"
    voice: VoiceEnum | None = None
    mood: str | None = Field(default=None, max_length=200)


class TTSStreamRequest(TTSRequest):
    multi_voice: bool = False
    cast: dict[str, CastMember] | None = None


class DevanagariRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DevanagariResponse(BaseModel):
    text: str


class TTSStreamSession(BaseModel):
    session_id: str
    stream_url: str
    estimated_seconds: float


class SpokenMarks(BaseModel):
    marks: list[tuple[float, str]]
    done: bool
