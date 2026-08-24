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
import asyncio
import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/audio", tags=["Audio"])

FOLLOW_POLL_SECONDS = 0.15


class Render:
    def __init__(self, scratch):
        self.scratch = scratch
        self.done = asyncio.Event()
        self.task: asyncio.Task | None = None


_renders: dict[str, Render] = {}


async def _render(session_id: str, session, render: Render) -> None:
    complete = False

    try:
        with render.scratch.open("wb", buffering=0) as handle:
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

        complete = True
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Reading %s aloud failed", session_id)
    finally:
        session.marks_done = complete

        if complete:
            publish(render.scratch, session_id)
        else:
            render.scratch.unlink(missing_ok=True)

        _renders.pop(session_id, None)
        render.done.set()


async def stop_renders() -> None:
    running = [
        render.task for render in _renders.values() if render.task is not None
    ]

    for task in running:
        task.cancel()

    for task in running:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    _renders.clear()


async def _follow(session_id: str, render: Render):
    position = 0

    while True:
        data = b""

        for path in (render.scratch, cached_file(session_id)):
            try:
                with path.open("rb") as handle:
                    handle.seek(position)
                    data = handle.read()
                break
            except FileNotFoundError:
                continue

        if data:
            position += len(data)
            yield data
            continue

        if render.done.is_set():
            return

        await asyncio.sleep(FOLLOW_POLL_SECONDS)


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

    render = _renders.get(session_id)

    if render is None:
        render = Render(reserve(session_id))
        _renders[session_id] = render

        session.marks = []
        session.marks_done = False

        render.task = asyncio.create_task(_render(session_id, session, render))
    else:
        logger.info(
            "Session %s is already being read aloud — following that instead of "
            "reading the whole story again",
            session_id,
        )

    headers["Accept-Ranges"] = "none"

    headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(
        _follow(session_id, render),
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
