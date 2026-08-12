from fastapi import APIRouter, HTTPException
from schemas.schemas import DevanagariRequest, DevanagariResponse
from services.transliterate import to_devanagari
import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/text", tags=["Text"])


@router.post("/devanagari", response_model=DevanagariResponse)
async def devanagari(payload: DevanagariRequest):
    try:
        converted = await to_devanagari(payload.text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=413, detail=str(error))
    except Exception:
        logger.exception("Devanagari conversion failed")
        raise HTTPException(status_code=500, detail="Conversion failed")

    return DevanagariResponse(text=converted)
