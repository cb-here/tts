from pydantic import BaseModel, Field
from schemas.enums import VoiceEnum

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: VoiceEnum = VoiceEnum.swara
    rate: str = "+0%"

