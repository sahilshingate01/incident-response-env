#!/usr/bin/env python3
"""
inference.py — OpenEnv-compliant inference script for Incident Response Environment.

Runs an LLM agent through all 3 incident response tasks, logging each step
in the exact format required by hackathon judges.

Usage:
    python inference.py

Required env vars:
    NVIDIA_API_KEY — NVIDIA NIM API token

Optional env vars:
    API_BASE_URL   — LLM base URL            (default: https://integrate.api.nvidia.com/v1)
    MODEL_NAME     — model to use            (default: deepseek-ai/deepseek-v3.1)
    INCIDENT_TASK  — override task list      (default: run all 3)
    ENV_BASE_URL   — environment server URL  (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if it exists
load_dotenv()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MAX_STEPS = 12
TEMPERATURE = 0.3
MAX_TOKENS = 512
LLM_TIMEOUT = 120.0  # seconds
SUCCESS_SCORE_THRESHOLD = 0.5

ALL_TASKS = [
    "single_service_failure",
    "database_latency",
    "cascade_failure",
]

VALID_ACTIONS = [
    "read_logs",
    "check_metrics",
    "check_all_services",
    "check_recent_deploys",
    "check_db_queries",
    "rollback",
    "restart_service",
    "scale_up",
    "declare_resolved",
]

# ──────────────────────────────────────────────
# Configuration from env vars
# ──────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-70b-instruct")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert SRE engineer responding to a production incident.
You will receive an alert and must diagnose and fix the issue.

Available actions (respond with ONLY valid JSON, no other text):
- {"action_type": "read_logs", "target": "<service_name>", "task_name": "<task>"}
- {"action_type": "check_metrics", "target": "<service_name>", "task_name": "<task>"}
- {"action_type": "check_all_services", "target": null, "task_name": "<task>"}
- {"action_type": "check_recent_deploys", "target": null, "task_name": "<task>"}
- {"action_type": "check_db_queries", "target": null, "task_name": "<task>"}
- {"action_type": "rollback", "target": "<deploy_id>", "task_name": "<task>"}
- {"action_type": "restart_service", "target": "<service_name>", "task_name": "<task>"}
- {"action_type": "scale_up", "target": "<service_name>", "task_name": "<task>"}
- {"action_type": "declare_resolved", "target": null, "task_name": "<task>"}

Services available: api-gateway, payment-service, user-service, db-primary, cache-redis

Strategy:
1. First check_all_services to see the overview
2. Read logs of the most critical service
3. Check recent deploys if you suspect a bad release
4. Apply the correct fix
5. Call declare_resolved when confident

Respond ONLY with valid JSON. No explanation. No markdown. Just the JSON object.\
""")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _env_url(path: str) -> str:
    """Build full URL for the environment API."""
    return f"{ENV_BASE_URL.rstrip('/')}{path}"


def _format_services_summary(summary: dict) -> str:
    """Build a compact text representation of the services summary."""
    lines = []
    for svc, info in summary.items():
        status = info.get("status", "unknown")
        err = info.get("error_rate", 0)
        lat = info.get("latency_p99_ms", 0)
        lines.append(f"  {svc}: status={status}, error_rate={err}%, latency={lat}ms")
    return "\n".join(lines)


def _parse_llm_action(raw: str, task_name: str) -> dict | None:
    """
    Try to extract a valid action JSON from the LLM's response.
    Returns the action dict or None if parsing fails.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            # Ensure task_name is set
            obj["task_name"] = task_name
            # Validate action_type
            if obj["action_type"] in VALID_ACTIONS:
                return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and "action_type" in obj:
                obj["task_name"] = task_name
                if obj["action_type"] in VALID_ACTIONS:
                    return obj
        except json.JSONDecodeError:
            pass

    return None


def _fallback_action(task_name: str) -> dict:
    """Default safe action when LLM output can't be parsed."""
    return {
        "action_type": "check_all_services",
        "target": None,
        "task_name": task_name,
    }


# ──────────────────────────────────────────────
# Environment client
# ──────────────────────────────────────────────

class EnvClient:
    """Thin HTTP client for the incident response environment."""

    def __init__(self):
        self.http = httpx.Client(base_url=ENV_BASE_URL, timeout=30.0)

    def reset(self, task_name: str) -> dict:
        """Reset (or switch to) the given task. Returns observation dict."""
        r = self.http.post(f"/reset/{task_name}")
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        """Execute one action. Returns observation dict."""
        r = self.http.post("/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        """Get current episode state."""
        r = self.http.get("/state")
        r.raise_for_status()
        return r.json()

    def grade(self, task_name: str) -> dict:
        """Grade the current episode."""
        r = self.http.post("/grade", json={"task_name": task_name})
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        """Health check."""
        r = self.http.get("/health")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.http.close()


# ──────────────────────────────────────────────
# Agent: single episode runner
# ──────────────────────────────────────────────

def run_episode(
    task_name: str,
    llm: OpenAI,
    env: EnvClient,
) -> dict:
    """
    Run a single episode for the given task.
    Returns: {task_name, score, steps, success, rewards}
    """
    # ── [START] ──
    print(f"[START] task={task_name} env=incident-response-env model={MODEL_NAME}")

    # Reset environment to this task
    obs = env.reset(task_name)

    conversation: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Build initial user message from the alert
    initial_msg = (
        f"INCIDENT ALERT for task '{task_name}':\n\n"
        f"{obs['observation_text']}\n\n"
        f"Current service status:\n"
        f"{_format_services_summary(obs['services_summary'])}\n\n"
        f"Available actions:\n"
        + "\n".join(f"  - {a}" for a in obs['available_actions'])
        + f"\n\nTask name to use in your JSON: \"{task_name}\""
    )
    conversation.append({"role": "user", "content": initial_msg})

    rewards: list[float] = []
    step_n = 0
    done = False

    while step_n < MAX_STEPS and not done:
        step_n += 1
        action_str = "check_all_services"
        error_msg = "null"

        try:
            # ── Call LLM (NVIDIA NIM with streaming) ──
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.2,
                top_p=0.7,
                max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                stream=False,
                timeout=LLM_TIMEOUT,
            )
            raw_content = completion.choices[0].message.content or ""

            conversation.append({"role": "assistant", "content": raw_content})

            # ── Parse action ──
            action = _parse_llm_action(raw_content, task_name)
            if action is None:
                action = _fallback_action(task_name)
                error_msg = "invalid_json_from_llm"

            action_str = action["action_type"]

            # ── Step environment ──
            obs = env.step(action)
            reward = obs["reward"]
            done = obs["done"]
            rewards.append(reward)

            # ── [STEP] log ──
            print(
                f"[STEP] step={step_n} action={action_str} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error_msg}"
            )

            # ── Feed observation back to LLM ──
            if not done:
                follow_up = (
                    f"Result of your action '{action_str}':\n\n"
                    f"{obs['observation_text']}\n\n"
                    f"Current service status:\n"
                    f"{_format_services_summary(obs['services_summary'])}\n\n"
                    f"Step {step_n}/{MAX_STEPS}. "
                    f"What do you do next? Respond with ONLY valid JSON."
                )
                conversation.append({"role": "user", "content": follow_up})

        except httpx.HTTPStatusError as exc:
            error_detail = exc.response.text[:200] if exc.response else str(exc)
            rewards.append(0.0)
            print(
                f"[STEP] step={step_n} action={action_str} "
                f"reward=0.00 done=false "
                f"error={error_detail}"
            )
        except Exception as exc:
            rewards.append(0.0)
            print(
                f"[STEP] step={step_n} action={action_str} "
                f"reward=0.00 done=false "
                f"error={str(exc)[:200]}"
            )

    # ── Calculate final score ──
    try:
        grade_result = env.grade(task_name)
    except Exception as exc:
        print(f"Warning: /grade endpoint failed: {exc}")
        grade_result = {}

    grader_score = grade_result.get("score", 0.0)

    if "passed" in grade_result:
        success = grade_result["passed"]
    else:
        if task_name == "single_service_failure":
            success = grader_score >= 0.5
        elif task_name == "database_latency":
            success = grader_score >= 0.6
        elif task_name == "cascade_failure":
            success = grader_score >= 0.7
        else:
            success = grader_score >= 0.5

    total_reward = sum(rewards)
    # Normalize: sum of rewards divided by MAX_STEPS, clamped to [0, 1]
    normalized = max(0.0, min(1.0, total_reward / MAX_STEPS))

    # Build rewards string
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── [END] ──
    print(
        f"[END] task={task_name} success={'true' if success else 'false'} "
        f"steps={step_n} grader_score={grader_score:.3f} "
        f"total_reward={total_reward:.2f} normalized={normalized:.3f} "
        f"rewards={rewards_str}"
    )

    return {
        "task_name": task_name,
        "score": round(grader_score, 4),
        "grader_score": round(grader_score, 4),
        "normalized_score": round(normalized, 4),
        "steps": step_n,
        "success": success,
        "rewards": rewards,
        "total_reward": round(total_reward, 4),
    }


# ──────────────────────────────────────────────
# Server lifecycle
# ──────────────────────────────────────────────

def _start_env_server() -> subprocess.Popen | None:
    """
    Start the environment server locally if ENV_BASE_URL points to localhost.
    Returns the Popen handle (or None if using a remote server).
    """
    if "localhost" not in ENV_BASE_URL and "127.0.0.1" not in ENV_BASE_URL:
        return None  # Remote server — don't start locally

    # ── Check if already running ──
    try:
        r = httpx.get(f"{ENV_BASE_URL.rstrip('/')}/health", timeout=1.0)
        if r.status_code == 200:
            print(f"Environment server already running at {ENV_BASE_URL}")
            return None
    except (httpx.ConnectError, httpx.ReadTimeout):
        pass

    project_root = Path(__file__).resolve().parent
    print(f"Starting environment server at {ENV_BASE_URL} ...")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app",
            "--host", "0.0.0.0",
            "--port", "7860",
            "--log-level", "warning",
        ],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to be ready
    for attempt in range(30):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{ENV_BASE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                print(f"Environment server ready (took {(attempt + 1) * 0.5:.1f}s)")
                return proc
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass

    # Server didn't start — print stderr for debugging
    proc.terminate()
    stderr_out = proc.stderr.read().decode() if proc.stderr else ""
    print(f"ERROR: Environment server failed to start.\n{stderr_out[:500]}")
    sys.exit(1)


def _stop_env_server(proc: subprocess.Popen | None):
    """Stop the local environment server if we started one."""
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Incident Response Environment — Inference Runner")
    print("=" * 65)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Env URL:    {ENV_BASE_URL}")
    print(f"  NVIDIA Key: {'set' if NVIDIA_API_KEY else 'NOT SET'}")
    print("=" * 65)
    print()

    # ── Determine which tasks to run ──
    override = os.getenv("INCIDENT_TASK")
    if override and override in ALL_TASKS:
        tasks_to_run = [override]
    else:
        tasks_to_run = ALL_TASKS

    # ── Start local env server if needed ──
    server_proc = _start_env_server()

    # ── Create clients ──
    if not NVIDIA_API_KEY:
        print("! WARNING: NVIDIA_API_KEY environment variable is not set.")
        print("! Inference will likely fail unless it's provided in the environment.")
        effective_key = "missing_api_key_placeholder"
    else:
        effective_key = NVIDIA_API_KEY

    llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=effective_key,
    )
    env = EnvClient()

    results: list[dict] = []

    try:
        for task_name in tasks_to_run:
            print(f"\n{'─' * 65}")
            result = run_episode(task_name, llm, env)
            results.append(result)
            print()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        _stop_env_server(server_proc)

    # ── Summary table ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Task':<28s} {'Score':>7s} {'Steps':>6s} {'Success':>8s}")
    print(f"  {'─' * 28} {'─' * 7} {'─' * 6} {'─' * 8}")

    total_score = 0.0
    all_success = True
    for r in results:
        success_str = "YES ✅" if r["success"] else "NO ❌"
        print(
            f"  {r['task_name']:<28s} "
            f"{r['score']:>7.3f} "
            f"{r['steps']:>6d} "
            f"{success_str:>8s}"
        )
        total_score += r["score"]
        if not r["success"]:
            all_success = False

    avg_score = total_score / len(results) if results else 0.0
    print(f"  {'─' * 28} {'─' * 7} {'─' * 6} {'─' * 8}")
    print(f"  {'AVERAGE':<28s} {avg_score:>7.3f} {'':>6s} {'ALL ✅' if all_success else 'SOME ❌':>8s}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
