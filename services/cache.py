"""Keep the finished mp3 for a session so it is only ever synthesised once.

Without this, every request replays the whole pipeline: pressing Download after
listening would re-run the casting model and re-synthesise several minutes of
speech before the browser saw a byte, which reads as a broken button.

A completed file also has a length and supports byte ranges, so downloading and
seeking behave normally — neither is possible while the audio is still being
made up as it is sent.
"""

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
    """A scratch file for one render attempt.

    Two listeners can start the same session at once, so each writes to its own
    file and the winner is published by rename.
    """
    return CACHE_DIR / f"{session_id}.{uuid4().hex}.part"


def publish(scratch: Path, session_id: str) -> None:
    """Adopt a finished render as the cached copy.

    Failing here costs a re-render later and nothing else: by the time this runs
    the listener already has every byte. It must not be allowed to tear down a
    response that has, from their side, completely succeeded.
    """
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
    """Mark a file as still in use.

    The sweeper works off modification time, and an hour-long reading easily
    outlives the cache window it started in. Without this the file could be
    deleted from under a listener who is still partway through it.
    """
    try:
        path.touch()
    except OSError:
        logger.warning("Could not refresh cached audio %s", path)


def discard(session_id: str) -> None:
    """Drop the finished audio for a session that is gone.

    Deliberately only the published file. A ".part" belongs to a render that is
    still running and cleans up after itself; deleting it here pulled the file
    out from under a reading that was still being listened to.
    """
    _remove(cached_file(session_id))


def _remove(path: Path) -> int:
    """Delete one file, returning the bytes reclaimed."""
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
    """Empty the directory outright.

    Run at startup and shutdown: nothing in here outlives the process, so a
    crash cannot leave audio sitting on disk indefinitely.
    """
    freed = sum(_remove(path) for path in _files())

    if freed:
        logger.info("Cleared %.1f MB of cached audio", freed / 1_000_000)


def sweep() -> None:
    """Drop anything expired, then anything over the size budget.

    Deleting only when a new session starts would let a server that goes quiet
    hold its last few hundred megabytes forever.

    A file being written still has a fresh modification time, so it is never
    swept out from under an active render. Removing one that is actively being
    downloaded is safe too — the reader keeps the open file until it is done.
    """
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

    # Oldest first, so the listener least likely to still be around loses their
    # copy before anyone else does.
    for _, size, path in sorted(survivors):
        if held <= budget:
            break

        held -= size
        freed += _remove(path)

    if freed:
        logger.info("Swept %.1f MB of cached audio", freed / 1_000_000)
