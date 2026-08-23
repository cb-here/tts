from fastapi import APIRouter, HTTPException, Query
from schemas.schemas import (
    SpokenMarks,
    TTSRequest,
    TTSStreamRequest,
    TTSStreamSession,
)
from services.tts import Mark, estimate_duration, render_to_file, stream_tts
from services.cache import cached_file, publish, reserve, touch
from services.sessions import create_session, get_session
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from utils.delete_file import delete_file
from utils.downloads import content_disposition
import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/generate")
async def generate(payload: TTSRequest):
    try:

        output_file = await render_to_file(
            text=payload.text,
            voice=payload.voice,
            rate=payload.rate,
        )

        return FileResponse(output_file, media_type="audio/mpeg", filename="speech.mp3", background=BackgroundTask(delete_file, output_file))
    except Exception as e:
        logger.exception("Audio generation failed")

        raise HTTPException(
        status_code=500,
        detail="Audio generation failed"
    )


@router.post("/stream", response_model=TTSStreamSession)
async def create_stream(payload: TTSStreamRequest):
    chosen = payload.cast or {}

    session_id = create_session(
        text=payload.text,
        voice=payload.voice,
        rate=payload.rate,
        multi_voice=payload.multi_voice,
        cast_genders={
            name: member.gender
            for name, member in chosen.items()
            if member.gender != "neutral"
        },
        cast_voices={
            name: member.voice for name, member in chosen.items() if member.voice
        },
        cast_moods={
            name: member.mood.strip()
            for name, member in chosen.items()
            if member.mood and member.mood.strip()
        },
    )

    return TTSStreamSession(
        session_id=session_id,
        stream_url=f"{router.prefix}/stream/{session_id}",
        estimated_seconds=estimate_duration(
            payload.text, payload.rate, payload.voice
        ),
    )


@router.get("/stream/{session_id}")
async def stream(
    session_id: str,
    download: bool = False,
    filename: str = Query(default="speech.mp3"),
):
    session = get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Stream session not found or expired",
        )

    headers = {"Cache-Control": "no-store"}

    if download:
        headers["Content-Disposition"] = content_disposition(filename)

    finished = cached_file(session_id)

    if finished.exists():
        touch(finished)
        session.marks_done = True

        return FileResponse(finished, media_type="audio/mpeg", headers=headers)

    scratch = reserve(session_id)

    session.marks = []
    session.marks_done = False

    async def audio_chunks():
        complete = False

        try:
            with scratch.open("wb") as handle:
                async for chunk in stream_tts(
                    text=session.text,
                    voice=session.voice,
                    rate=session.rate,
                    multi_voice=session.multi_voice,
                    cast_genders=session.cast_genders,
                    cast_voices=session.cast_voices,
                    cast_moods=session.cast_moods,
                ):
                    if isinstance(chunk, Mark):
                        session.marks.append((chunk.at, chunk.text))
                        continue

                    handle.write(chunk)
                    yield chunk

            complete = True
        except Exception:
            logger.exception("Audio streaming failed for session %s", session_id)
            raise
        finally:
            session.marks_done = complete

            if complete:
                publish(scratch, session_id)
            else:
                scratch.unlink(missing_ok=True)

    headers["Accept-Ranges"] = "none"

    headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(
        audio_chunks(),
        media_type="audio/mpeg",
        headers=headers,
    )


@router.get("/stream/{session_id}/marks", response_model=SpokenMarks)
async def marks(session_id: str, after: int = 0):
    session = get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Stream session not found or expired",
        )

    return SpokenMarks(
        marks=session.marks[max(after, 0):],
        done=session.marks_done or cached_file(session_id).exists(),
    )
