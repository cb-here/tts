from fastapi import APIRouter, HTTPException, Query
from schemas.schemas import TTSRequest, TTSStreamRequest, TTSStreamSession
from services.tts import estimate_duration, render_to_file, stream_tts
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
    """Park the text and hand back a URL an `<audio>` element can play."""
    session_id = create_session(
        text=payload.text,
        voice=payload.voice,
        rate=payload.rate,
        multi_voice=payload.multi_voice,
    )

    return TTSStreamSession(
        session_id=session_id,
        stream_url=f"{router.prefix}/stream/{session_id}",
        estimated_seconds=estimate_duration(payload.text, payload.rate),
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
        # Already rendered once, so it can be served as an ordinary file —
        # with a length, and with byte ranges for seeking.
        touch(finished)
        return FileResponse(finished, media_type="audio/mpeg", headers=headers)

    scratch = reserve(session_id)

    async def audio_chunks():
        complete = False

        try:
            with scratch.open("wb") as handle:
                async for chunk in stream_tts(
                    text=session.text,
                    voice=session.voice,
                    rate=session.rate,
                    multi_voice=session.multi_voice,
                ):
                    handle.write(chunk)
                    yield chunk

            complete = True
        except Exception:
            logger.exception("Audio streaming failed for session %s", session_id)
            # Headers are already on the wire, so there is no status code left to
            # send. Re-raising drops the connection, which at least tells the
            # client the mp3 is truncated instead of quietly handing it a
            # well-formed but incomplete file.
            raise
        finally:
            # Also reached when the listener navigates away mid-story, which
            # must not leave a half-written file to be served as the real thing.
            if complete:
                publish(scratch, session_id)
            else:
                scratch.unlink(missing_ok=True)

    # Nothing exists to seek into yet; the browser buffers forward instead.
    headers["Accept-Ranges"] = "none"

    # nginx and most managed proxies buffer a response until it completes, which
    # would hold the whole reading back and quietly undo the streaming. This is
    # the opt-out; it is ignored by proxies that do not buffer anyway.
    headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(
        audio_chunks(),
        media_type="audio/mpeg",
        headers=headers,
    )
