from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes.text import router as text_route
from routes.tts import router as tts_route
from fastapi.middleware.cors import CORSMiddleware
from config import AUDIO_SWEEP_SECONDS, CORS_ORIGINS, LOG_LEVEL
from routes.tts import stop_renders
from services.cache import clear_all, sweep
import asyncio
import logging

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def _sweeper():
    while True:
        await asyncio.sleep(AUDIO_SWEEP_SECONDS)

        try:
            sweep()
        except Exception:
            logger.exception("Audio sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    clear_all()
    task = asyncio.create_task(_sweeper())

    try:
        yield
    finally:
        task.cancel()
        await stop_renders()
        clear_all()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to EDGE tts"
    }

@app.get("/status")
async def status():
    return {
        "message": "Server is healthy"
    }

app.include_router(tts_route)
app.include_router(text_route)
