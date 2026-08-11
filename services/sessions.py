from dataclasses import dataclass
from services.cache import discard
from time import monotonic
from uuid import uuid4

# A browser cannot POST a body from `<audio src="...">`, so the request is split
# in two: POST the text once to park it here, then GET the stream by id.
SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 200


@dataclass(frozen=True)
class TTSSession:
    text: str
    voice: str
    rate: str
    multi_voice: bool
    created_at: float


_sessions: dict[str, TTSSession] = {}


def _forget(session_id: str) -> None:
    del _sessions[session_id]
    discard(session_id)


def _prune(now: float) -> None:
    for key in [
        key
        for key, session in _sessions.items()
        if now - session.created_at > SESSION_TTL_SECONDS
    ]:
        _forget(key)

    # Hard cap as well, so a burst of traffic inside one TTL window cannot grow
    # the dict without bound.
    while len(_sessions) > MAX_SESSIONS:
        _forget(min(_sessions, key=lambda key: _sessions[key].created_at))


def create_session(text: str, voice: str, rate: str, multi_voice: bool = False) -> str:
    now = monotonic()
    _prune(now)

    session_id = str(uuid4())
    _sessions[session_id] = TTSSession(
        text=text,
        voice=voice,
        rate=rate,
        multi_voice=multi_voice,
        created_at=now,
    )

    return session_id


def get_session(session_id: str) -> TTSSession | None:
    session = _sessions.get(session_id)

    if session is None:
        return None

    if monotonic() - session.created_at > SESSION_TTL_SECONDS:
        _forget(session_id)
        return None

    return session
