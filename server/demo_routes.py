"""
demo_routes.py — FastAPI router for the live demo SSE streaming endpoint.

Provides:
  POST /demo/run?task=cascade_failure   — SSE stream of agent steps
  GET  /demo/tasks                      — task metadata for the UI dropdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from environment import IncidentResponseEnv, SCENARIO_MAP
from models import IncidentAction
from demo_agent import DEMO_SEQUENCES, TASK_METADATA

logger = logging.getLogger("incident-env.demo")

router = APIRouter(prefix="/demo", tags=["demo"])


# ──────────────────────────────────────────────
# GET /demo/tasks — metadata for UI dropdown
# ──────────────────────────────────────────────

@router.get("/tasks")
async def demo_tasks():
    """Return task metadata for the frontend scenario selector."""
    return list(TASK_METADATA.values())


# ──────────────────────────────────────────────
# POST /demo/run — SSE streaming demo run
# ──────────────────────────────────────────────

@router.post("/run")
async def demo_run(task: str = Query(..., description="Task name to run")):
    """
    Run the hardcoded demo agent for a task and stream results as SSE.

    Each event is a JSON object with type:
      - "reset"    : initial environment state (services, alert)
      - "thinking" : agent reasoning before an action
      - "step"     : action result with observation, reward, services
      - "resolved" : final resolution summary
      - "error"    : error information
    """

    if task not in SCENARIO_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Available: {list(SCENARIO_MAP.keys())}",
        )

    if task not in DEMO_SEQUENCES:
        raise HTTPException(
            status_code=400,
            detail=f"No demo sequence available for task '{task}'.",
        )

    async def event_stream():
        try:
            # Create a fresh, isolated environment for this demo run.
            # This does NOT touch the global _env used by /step, /reset etc.
            env = IncidentResponseEnv(task)
            obs = env.reset()

            start_time = time.time()

            # ── Emit reset event ──
            yield _sse(json.dumps({
                "type": "reset",
                "task": task,
                "task_meta": TASK_METADATA.get(task, {}),
                "alert": obs.observation_text,
                "services": obs.services_summary,
                "incident_description": obs.incident_description,
            }))

            await asyncio.sleep(1.0)  # dramatic pause after alert

            steps = DEMO_SEQUENCES[task]
            cumulative_reward = 0.0

            for i, demo_step in enumerate(steps):
                step_num = i + 1

                # ── Emit thinking event ──
                yield _sse(json.dumps({
                    "type": "thinking",
                    "step": step_num,
                    "reasoning": demo_step.agent_reasoning,
                }))

                await asyncio.sleep(0.8)  # simulate agent "thinking"

                # ── Execute the action against the real environment ──
                action = IncidentAction(
                    action_type=demo_step.action_type,
                    target=demo_step.target,
                    task_name=task,
                )
                obs = env.step(action)
                cumulative_reward += obs.reward

                # Classify action for color-coding
                action_category = _classify_action(demo_step.action_type)

                # ── Emit step event ──
                yield _sse(json.dumps({
                    "type": "step",
                    "step": step_num,
                    "total_steps": len(steps),
                    "action_type": demo_step.action_type,
                    "target": demo_step.target,
                    "action_category": action_category,
                    "observation_text": obs.observation_text,
                    "reward": round(obs.reward, 2),
                    "cumulative_reward": round(cumulative_reward, 2),
                    "services": obs.services_summary,
                    "done": obs.done,
                    "metadata": obs.metadata,
                    "reasoning": demo_step.agent_reasoning,
                }))

                await asyncio.sleep(0.6)  # pacing between steps

                if obs.done:
                    break

            elapsed = round(time.time() - start_time, 1)

            # ── Emit resolved event ──
            yield _sse(json.dumps({
                "type": "resolved",
                "total_steps": len(steps),
                "cumulative_reward": round(cumulative_reward, 2),
                "max_reward": TASK_METADATA.get(task, {}).get("max_reward", 0),
                "elapsed_seconds": elapsed,
                "task": task,
                "services": obs.services_summary,
            }))

        except Exception as exc:
            logger.exception("Demo run failed for task=%s", task)
            yield _sse(json.dumps({
                "type": "error",
                "message": str(exc),
            }))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _sse(data: str) -> str:
    """Format a string as an SSE data event."""
    return f"data: {data}\n\n"


def _classify_action(action_type: str) -> str:
    """Classify an action type for UI color-coding."""
    if action_type in ("read_logs", "check_metrics", "check_all_services",
                        "check_recent_deploys", "check_db_queries"):
        return "investigate"
    elif action_type in ("rollback", "restart_service", "scale_up"):
        return "fix"
    elif action_type == "declare_resolved":
        return "resolve"
    return "unknown"
