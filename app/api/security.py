import hashlib
import hmac
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class User:
    user_id: str
    departments: tuple[str, ...]

    @property
    def has_access(self) -> bool:
        return bool(self.departments)

    def metadata_filter(self) -> dict[str, Any] | None:
        """The access filter. Built from the user's identity, never from the request."""
        if "*" in self.departments:
            return None
        return {"department": list(self.departments)}


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class Authenticator:
    """Looks up a user from an API key. Only key hashes are stored."""

    def __init__(self, users_by_hash: dict[str, User]) -> None:
        if not users_by_hash:
            raise ValueError("No API keys configured")
        self._users = users_by_hash

    @classmethod
    def from_file(cls, path: Path) -> "Authenticator":
        if not path.is_file():
            raise FileNotFoundError(
                f"API keys file not found: {path}. Create one with: "
                "python -m scripts.create_api_key <user_id> <department> ..."
            )
        entries = json.loads(path.read_text(encoding="utf-8"))
        users = {
            entry["key_sha256"]: User(
                user_id=entry["user_id"],
                departments=tuple(d.lower() for d in entry["departments"]),
            )
            for entry in entries
        }
        return cls(users)

    def authenticate(self, api_key: str | None) -> User | None:
        if not api_key:
            return None
        candidate = hash_key(api_key)
        found: User | None = None
        for stored_hash, user in self._users.items():
            if hmac.compare_digest(stored_hash, candidate):
                found = user
        return found


class RateLimiter:
    """Sliding-window limiter per key. In-memory, so per process only."""

    def __init__(
        self,
        max_calls: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_calls <= 0 or window_seconds <= 0:
            raise ValueError("max_calls and window_seconds must be positive")
        self._max = max_calls
        self._window = window_seconds
        self._clock = clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= self._window:
                calls.popleft()
            if len(calls) >= self._max:
                return False
            calls.append(now)
            return True