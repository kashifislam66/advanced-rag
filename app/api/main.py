import hashlib
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import Deps
from app.api.schemas import AskRequest, AskResponse, CitationOut, UsageOut
from app.api.security import User
from app.api.service import ServiceBusy
from app.config.settings import settings
from app.generation.generator import GenerationError
from app.utils.logger import request_id_var, setup_logging

logger = logging.getLogger(__name__)
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def create_app(deps: Deps | None = None) -> FastAPI:
    """Build the app. Tests pass fake deps; production builds real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = None
        if app.state.deps is None:
            setup_logging(settings.log_level)
            from app.api.wiring import build_deps  # imported here to keep tests light

            app.state.deps, container = build_deps(settings)
        yield
        if container is not None:
            container.close()

    prod = settings.environment == "prod"
    app = FastAPI(
        title="Advanced RAG API",
        lifespan=lifespan,
        docs_url=None if prod else "/docs",
        redoc_url=None,
        openapi_url=None if prod else "/openapi.json",
    )
    app.state.deps = deps

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "%s %s -> %d (%.0f ms)",
                request.method, request.url.path, response.status_code,
                (time.perf_counter() - start) * 1000,
            )
            return response
        finally:
            request_id_var.reset(token)

    def get_user(request: Request, x_api_key: str | None = Header(default=None)) -> User:
        current: Deps = request.app.state.deps
        user = current.authenticator.authenticate(x_api_key)
        if user is None:
            raise HTTPException(401, "Invalid or missing API key",
                                headers={"WWW-Authenticate": "ApiKey"})
        if not user.has_access:
            raise HTTPException(403, "This user has no document access")
        if not current.limiter.allow(user.user_id):
            raise HTTPException(429, "Too many requests", headers={"Retry-After": "60"})
        return user

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        current: Deps | None = request.app.state.deps
        if current is None or not current.service.is_ready():
            return JSONResponse(status_code=503, content={"status": "not ready"})
        return {"status": "ready"}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request, user: User = Depends(get_user)):
        current: Deps = request.app.state.deps
        start = time.perf_counter()
        try:
            result = current.service.ask(body.question, user)
        except ServiceBusy:
            raise HTTPException(503, "Server is busy, please retry", headers={"Retry-After": "5"})
        except GenerationError:
            raise HTTPException(502, "The language model is unavailable right now")
        except Exception:
            logger.exception("Unhandled error while answering")
            raise HTTPException(500, "Internal server error")

        logger.info(
            "ask user=%s q_len=%d q_hash=%s status=%s grounded=%s citations=%d tokens_in=%s tokens_out=%s %.0f ms",
            user.user_id, len(body.question),
            hashlib.sha256(body.question.encode("utf-8")).hexdigest()[:12],
            result.status, result.grounded, len(result.citations),
            result.input_tokens, result.output_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return AskResponse(
            request_id=request_id_var.get(),
            answer=result.answer,
            grounded=result.grounded,
            status=result.status,
            citations=[CitationOut(**c) for c in result.citations],
            usage=UsageOut(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
        )

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request, user: User = Depends(get_user)):
        current: Deps = request.app.state.deps

        def events() -> Iterator[str]:
            try:
                for event in current.service.stream(body.question, user):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except ServiceBusy:
                yield 'data: {"event": "error", "detail": "Server is busy, please retry"}\n\n'
            except Exception:
                logger.exception("Streaming failed")
                yield 'data: {"event": "error", "detail": "Internal server error"}\n\n'

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()