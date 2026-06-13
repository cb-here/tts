from fastapi import APIRouter, HTTPException
from schemas.schemas import TTSRequest
from services.tts import workflow
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from utils.delete_file import delete_file
import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/generate")
async def generate(payload: TTSRequest):  
    try:

        result = await workflow.ainvoke({
            "text": payload.text,
            "voice": payload.voice,
            "rate": payload.rate
            })

        return FileResponse(result["output_file"], media_type="audio/mpeg", filename="speech.mp3", background=BackgroundTask(delete_file, result["output_file"]))
    except Exception as e:
        logger.exception("Audio generation failed")

        raise HTTPException(
        status_code=500,
        detail="Audio generation failed"
    )
