"""
FastAPI server wrapping the Incident Response RL environment.

Start with:
    uvicorn src.server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from src.environment import IncidentResponseEnv, SCENARIO_MAP
from src.models import IncidentAction, IncidentObservation, IncidentState
from src.server.graders import grade

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("incident-env")

# ──────────────────────────────────────────────
# Global environment holder
# ──────────────────────────────────────────────

_env: IncidentResponseEnv | None = None


def _get_env() -> IncidentResponseEnv:
    """Return the current environment; raise 503 if not initialised."""
    if _env is None:
        raise HTTPException(
            status_code=503,
            detail="Environment not initialised. Server is starting up.",
        )
    return _env


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise the environment on startup, clean up on shutdown."""
    global _env
    task_name = os.environ.get("INCIDENT_TASK", "single_service_failure")
    logger.info("Initialising environment with task: %s", task_name)

    try:
        _env = IncidentResponseEnv(task_name)
        _env.reset()
        logger.info(
            "Environment ready  │  task=%s  episode=%s",
            _env.task_name,
            _env.episode_id,
        )
    except ValueError as exc:
        logger.error("Failed to initialise environment: %s", exc)
        raise

    yield  # ← app is running

    logger.info("Shutting down environment.")
    _env = None


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Incident Response Environment",
    description=(
        "OpenEnv-compliant RL environment for production incident response. "
        "An AI agent receives system alerts and must diagnose and remediate "
        "production failures across 3 difficulty levels."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (needed for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Request logging middleware
# ──────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s  →  %d  (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ──────────────────────────────────────────────
# Validation error handler
# ──────────────────────────────────────────────

@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
        },
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Incident Response Environment API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    env = _get_env()
    return {
        "status": "ok",
        "task": env.task_name,
        "episode_id": env.episode_id,
    }


@app.get("/tasks")
async def list_tasks():
    return [
        {
            "name": "single_service_failure",
            "difficulty": "easy",
            "description": (
                "A bad deploy caused one service to fail. "
                "Identify the service and roll back."
            ),
        },
        {
            "name": "database_latency",
            "difficulty": "medium",
            "description": (
                "DB overload is causing API latency. "
                "Trace the root cause across 3 services and remediate."
            ),
        },
        {
            "name": "cascade_failure",
            "difficulty": "hard",
            "description": (
                "A cascading failure across 4 services. "
                "Find root cause and fix in correct order or make it worse."
            ),
        },
    ]


@app.post("/reset")
async def reset_env():
    env = _get_env()
    obs = env.reset()
    logger.info("Episode reset  │  episode=%s  task=%s", env.episode_id, env.task_name)
    return obs.model_dump()


@app.post("/reset/{task_name}")
async def reset_with_task(task_name: str):
    global _env
    if task_name not in SCENARIO_MAP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown task '{task_name}'. "
                f"Available: {list(SCENARIO_MAP.keys())}"
            ),
        )
    _env = IncidentResponseEnv(task_name)
    obs = _env.reset()
    logger.info(
        "Task switched & reset  │  task=%s  episode=%s",
        _env.task_name,
        _env.episode_id,
    )
    return obs.model_dump()


@app.post("/step")
async def step_env(action: IncidentAction):
    env = _get_env()
    obs = env.step(action)
    logger.info(
        "Step %d  │  action=%s  reward=%+.2f  done=%s",
        env.step_count,
        action.action_key(),
        obs.reward,
        obs.done,
    )
    return obs.model_dump()


@app.get("/state")
async def get_state():
    env = _get_env()
    return env.state().model_dump()


# ──────────────────────────────────────────────
# Grading endpoint
# ──────────────────────────────────────────────

class GradeRequest(BaseModel):
    task_name: str


@app.post("/grade")
async def grade_episode(req: GradeRequest):
    env = _get_env()

    if req.task_name not in SCENARIO_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task_name}'. Available: {list(SCENARIO_MAP.keys())}",
        )

    # Use current episode state for grading
    current_state = env.state()

    # Ensure we're grading the right task
    if current_state.task_name != req.task_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Current episode is task '{current_state.task_name}', "
                f"but grading was requested for '{req.task_name}'. "
                f"Switch tasks with POST /reset/{req.task_name} first."
            ),
        )

    result = grade(
        task_name=req.task_name,
        state=current_state,
        actions_taken=current_state.actions_taken,
        total_reward=env.cumulative_reward,
    )

    logger.info(
        "Graded episode  │  task=%s  score=%.4f  passed=%s",
        req.task_name,
        result["score"],
        result["passed"],
    )
    return result


# ──────────────────────────────────────────────
# Inline smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import subprocess
    import sys
    import httpx

    async def _smoke_test():
        """Quick async smoke test that runs a full episode via HTTP."""

        # Start server as a subprocess using uvicorn CLI
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.server.app:app",
                "--host", "127.0.0.1",
                "--port", "7860",
                "--log-level", "warning",
            ],
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give server time to start
        await asyncio.sleep(2.5)
        base = "http://127.0.0.1:7860"

        try:
            async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
                print("\n" + "=" * 60)
                print("  SMOKE TEST — Incident Response Environment API")
                print("=" * 60)

                # Health check
                r = await client.get("/health")
                assert r.status_code == 200, f"Health failed: {r.text}"
                print(f"\n✅ GET /health → {r.json()}")

                # List tasks
                r = await client.get("/tasks")
                assert r.status_code == 200
                tasks = r.json()
                print(f"✅ GET /tasks → {len(tasks)} tasks")
                for t in tasks:
                    print(f"   • {t['name']} ({t['difficulty']})")

                # Reset
                r = await client.post("/reset")
                assert r.status_code == 200
                obs = r.json()
                print(f"\n✅ POST /reset → step={obs['step_count']}, done={obs['done']}")
                print(f"   Alert: {obs['observation_text'][:80]}...")

                # Play a correct sequence for task 1
                task = "single_service_failure"
                steps = [
                    {"action_type": "check_recent_deploys", "target": None, "task_name": task},
                    {"action_type": "read_logs", "target": "user-service", "task_name": task},
                    {"action_type": "rollback", "target": "dep-evil-123", "task_name": task},
                    {"action_type": "declare_resolved", "target": None, "task_name": task},
                ]

                print(f"\n  Playing {len(steps)} steps for '{task}':")
                for step_data in steps:
                    r = await client.post("/step", json=step_data)
                    assert r.status_code == 200, f"Step failed: {r.text}"
                    obs = r.json()
                    print(
                        f"   Step {obs['step_count']}: "
                        f"{step_data['action_type']:25s}  "
                        f"reward={obs['reward']:+.1f}  "
                        f"done={obs['done']}"
                    )

                # State
                r = await client.get("/state")
                assert r.status_code == 200
                state = r.json()
                print(f"\n✅ GET /state → diagnosed={state['correctly_diagnosed']}, resolved={state['resolved']}")

                # Grade
                r = await client.post("/grade", json={"task_name": task})
                assert r.status_code == 200
                grade_result = r.json()
                print(f"✅ POST /grade → score={grade_result['score']}, passed={grade_result['passed']}")
                print(f"   Breakdown: {grade_result['breakdown']}")

                # Switch task
                r = await client.post("/reset/cascade_failure")
                assert r.status_code == 200
                obs = r.json()
                print(f"\n✅ POST /reset/cascade_failure → new episode started")
                print(f"   Alert: {obs['observation_text'][:80]}...")

                print(f"\n{'='*60}")
                print("  ALL SMOKE TESTS PASSED ✅")
                print(f"{'='*60}\n")

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    asyncio.run(_smoke_test())

