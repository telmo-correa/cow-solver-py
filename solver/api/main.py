"""FastAPI application for the CoW solver."""

import os

import uvicorn
from fastapi import FastAPI

from solver.api.endpoints import router

# Configuration from environment variables with sensible defaults
HOST = os.environ.get("SOLVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("SOLVER_PORT", "8000"))
DEBUG = os.environ.get("SOLVER_DEBUG", "false").lower() in ("true", "1", "yes")

app = FastAPI(
    title="CoW Solver (Python)",
    description="A Python implementation of a CoW Protocol solver",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


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
