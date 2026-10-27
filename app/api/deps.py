from dataclasses import dataclass

from app.api.security import Authenticator, RateLimiter
from app.api.service import RagService


@dataclass(frozen=True)
class Deps:
    """Everything the API needs. Tests pass fakes; production builds the real ones."""

    service: RagService
    authenticator: Authenticator
    limiter: RateLimiter