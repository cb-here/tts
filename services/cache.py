import logging
from config import AUDIO_CACHE_MAX_MB, AUDIO_TTL_SECONDS
from pathlib import Path
from time import time
from uuid import uuid4

logger = logging.getLogger(__name__)

CACHE_DIR = Path("audio")
CACHE_DIR.mkdir(exist_ok=True)


def cached_file(session_id: str) -> Path:
    return CACHE_DIR / f"{session_id}.mp3"


def reserve(session_id: str) -> Path:
    return CACHE_DIR / f"{session_id}.{uuid4().hex}.part"


def publish(scratch: Path, session_id: str) -> None:
    try:
        scratch.replace(cached_file(session_id))
    except OSError as error:
        logger.warning(
            "Could not cache the finished audio for %s (%s) — it will be "
            "rendered again if asked for",
            session_id,
            error,
        )


def touch(path: Path) -> None:
    try:
        path.touch()
    except OSError:
        logger.warning("Could not refresh cached audio %s", path)


def discard(session_id: str) -> None:
    _remove(cached_file(session_id))


def _remove(path: Path) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning("Could not remove cached audio %s", path)
        return 0


def _files() -> list[Path]:
    return [path for path in CACHE_DIR.glob("*") if path.is_file()]


def clear_all() -> None:
    freed = sum(_remove(path) for path in _files())

    if freed:
        logger.info("Cleared %.1f MB of cached audio", freed / 1_000_000)


def sweep() -> None:
    now = time()
    freed = 0
    survivors: list[tuple[float, int, Path]] = []

    for path in _files():
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue

        if now - stat.st_mtime > AUDIO_TTL_SECONDS:
            freed += _remove(path)
        else:
            survivors.append((stat.st_mtime, stat.st_size, path))

    budget = AUDIO_CACHE_MAX_MB * 1_000_000
    held = sum(size for _, size, _ in survivors)

    for _, size, path in sorted(survivors):
        if held <= budget:
            break

        held -= size
        freed += _remove(path)

    if freed:
        logger.info("Swept %.1f MB of cached audio", freed / 1_000_000)
