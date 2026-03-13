"""FastAPI app — middleware, CORS, lifespan, static file serving."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from .config import CORS_ORIGIN, HOST, PORT
from .db.database import init_db
from .routers import ask, meta, resources, settings
from .security import get_or_create_token, require_token
from .services.riot_poller import poll_riot_api
from .ws.manager import ws_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static files path (React build output)
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self' ws://127.0.0.1:8765"
        )
        return response


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    get_or_create_token()
    logger.info("Session token ready")

    # Start Riot poller as background task
    poller_task = asyncio.create_task(poll_riot_api())
    logger.info("Riot poller started")

    yield

    # Shutdown
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="LoL Build Coach",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Rate limiter
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — loopback only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[CORS_ORIGIN],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-Session-Token", "Content-Type"],
    )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Routers
    app.include_router(ask.router)
    app.include_router(resources.router)
    app.include_router(settings.router)
    app.include_router(meta.router)

    # Bootstrap endpoint — delivers session token (no auth required)
    @app.get("/bootstrap")
    async def bootstrap():
        token = get_or_create_token()
        return JSONResponse({"token": token})

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket, token: str = ""):
        expected = get_or_create_token()
        if token != expected:
            await ws.close(code=1008, reason="Invalid token")
            return
        await ws_manager.connect(ws)
        try:
            while True:
                # Keep connection alive; client can send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    # Serve React SPA static files
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="spa")
    else:
        @app.get("/")
        async def index():
            return HTMLResponse(
                "<h1>LoL Build Coach</h1>"
                "<p>Frontend not built yet. Run <code>cd frontend && npm run build</code></p>"
            )

    return app


app = create_app()
