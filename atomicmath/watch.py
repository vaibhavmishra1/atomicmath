"""Local web dashboard: read-only view of pipeline_events + DB counts (same SQLite as the run)."""
from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from .config import load_config
from .db import Store


def _dashboard_html_path() -> Path:
    return Path(__file__).resolve().parent / "web" / "dashboard.html"


def build_app(db_path: str) -> Starlette:
    store = Store(db_path)

    async def api_summary(_: Request) -> JSONResponse:
        c = store.dashboard_counts()
        with store._conn() as conn:
            r = conn.execute(
                "SELECT phase, step, message FROM pipeline_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        c["latest_phase"] = r["phase"] if r else None
        c["latest_step"] = r["step"] if r else None
        c["latest_message"] = r["message"] if r else None
        return JSONResponse(c)

    async def api_events(request: Request) -> JSONResponse:
        after = int(request.query_params.get("after", "0"))
        limit = int(request.query_params.get("limit", "800"))
        events = store.list_pipeline_events(after_id=after, limit=limit)
        return JSONResponse({"events": events})

    async def index(_: Request) -> Response:
        p = _dashboard_html_path()
        if p.is_file():
            return FileResponse(p, media_type="text/html")
        return Response("dashboard.html not found in package", status=404)

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/summary", api_summary),
            Route("/api/events", api_events),
        ],
    )


def serve_forever(config_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    cfg = load_config(config_path)
    app = build_app(cfg.storage.db_path)
    print(f"atomicmath trace UI → http://{host}:{port}/  (db: {cfg.storage.db_path})")
    uvicorn.run(app, host=host, port=port, log_level="info")
