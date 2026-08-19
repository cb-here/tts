from typing import Literal
from pydantic import BaseModel, Field
from schemas.enums import VoiceEnum

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: VoiceEnum = VoiceEnum.emma
    rate: str = "+0%"


class CastMember(BaseModel):
    """Who a character is, as decided by the listener rather than the model."""

    gender: Literal["male", "female", "neutral"] = "neutral"
    # Pinned outright, overriding the pool. Left unset, the character still gets
    # a voice of their own — just one chosen for them.
    voice: VoiceEnum | None = None


class TTSStreamRequest(TTSRequest):
    # Beta: let an LLM split the text by speaker and cast a voice per character.
    # `voice` stays the fallback, and is used for narration.
    multi_voice: bool = False
    # A cast the listener settled themselves, keyed by character name. Sending
    # it removes the one thing the model was worst at — deciding whether a
    # character is a man or a woman — and skips the call that used to ask.
    cast: dict[str, CastMember] | None = None


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
