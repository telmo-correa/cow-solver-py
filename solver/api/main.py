"""FastAPI application for the CoW solver.

Note: Rate limiting is intentionally not implemented at the application level.
It should be handled at the infrastructure layer (reverse proxy / load balancer)
for better separation of concerns and easier configuration.
"""

import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from solver.api.endpoints import router

# Configuration from environment variables with sensible defaults
HOST = os.environ.get("SOLVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("SOLVER_PORT", "8000"))
DEBUG = os.environ.get("SOLVER_DEBUG", "false").lower() in ("true", "1", "yes")

# Maximum request body size (10 MB)
MAX_REQUEST_SIZE = 10 * 1024 * 1024

# Check scipy availability at startup
try:
    import scipy  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

app = FastAPI(
    title="CoW Solver (Python)",
    description="A Python implementation of a CoW Protocol solver",
    version="0.1.0",
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject requests with body larger than MAX_REQUEST_SIZE."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(status_code=413, content={"detail": "Request too large"})
    return await call_next(request)


app.include_router(router)


@app.get("/health")
async def health() -> dict[str, object]:
    """Health check endpoint."""
    return {"status": "ok", "scipy_available": SCIPY_AVAILABLE}


def run() -> None:
    """Run the solver API server.

    Configuration via environment variables:
    - SOLVER_HOST: Host to bind to (default: 0.0.0.0)
    - SOLVER_PORT: Port to bind to (default: 8000)
    - SOLVER_DEBUG: Enable debug/reload mode (default: false)
    """
    uvicorn.run(
        "solver.api.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
    )


if __name__ == "__main__":
    run()
