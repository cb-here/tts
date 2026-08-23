from config import SESSION_TTL_SECONDS
from dataclasses import dataclass, field
from services.cache import discard
from time import monotonic
from uuid import uuid4

MAX_SESSIONS = 200


@dataclass
class TTSSession:
    text: str
    voice: str
    rate: str
    multi_voice: bool
    created_at: float
    cast_genders: dict[str, str] = field(default_factory=dict)
    cast_voices: dict[str, str] = field(default_factory=dict)
    cast_moods: dict[str, str] = field(default_factory=dict)
    touched_at: float = field(default=0.0)
    marks: list[tuple[float, str]] = field(default_factory=list)
    marks_done: bool = field(default=False)


_sessions: dict[str, TTSSession] = {}


def _forget(session_id: str) -> None:
    del _sessions[session_id]
    discard(session_id)


def _prune(now: float) -> None:
    for key in [
        key
        for key, session in _sessions.items()
        if now - session.touched_at > SESSION_TTL_SECONDS
    ]:
        _forget(key)

    while len(_sessions) > MAX_SESSIONS:
        _forget(min(_sessions, key=lambda key: _sessions[key].touched_at))


def create_session(
    text: str,
    voice: str,
    rate: str,
    multi_voice: bool = False,
    cast_genders: dict[str, str] | None = None,
    cast_voices: dict[str, str] | None = None,
    cast_moods: dict[str, str] | None = None,
) -> str:
    now = monotonic()
    _prune(now)

    session_id = str(uuid4())
    _sessions[session_id] = TTSSession(
        text=text,
        voice=voice,
        rate=rate,
        multi_voice=multi_voice,
        created_at=now,
        touched_at=now,
        cast_genders=cast_genders or {},
        cast_voices=cast_voices or {},
        cast_moods=cast_moods or {},
    )

    return session_id


def get_session(session_id: str) -> TTSSession | None:
    session = _sessions.get(session_id)

    if session is None:
        return None

    now = monotonic()

    if now - session.touched_at > SESSION_TTL_SECONDS:
        _forget(session_id)
        return None

    session.touched_at = now

    return session
