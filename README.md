---
title: Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - llm-evaluation
  - incident-response
  - sre
license: mit
---

# 🚨 Incident Response Environment

**Train and evaluate AI agents on realistic production incident response.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=flat-square)](https://github.com/OpenEnv-AI/openenv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green?style=flat-square&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

---

## Overview

Production incidents cost companies an estimated **$5,600 per minute** of downtime. When a P1 alert fires at 3 AM, an on-call engineer must rapidly triage logs, metrics, and deploy history — often across dozens of interdependent services — to isolate the root cause and remediate before cascading failures take down the entire platform. Today, this skill is learned the hard way: through months of trial-and-error on real production systems.

**Incident Response Environment** is an OpenEnv-compliant RL/LLM evaluation environment that simulates realistic production incidents. An AI agent receives a system alert and must navigate a 5-service microservice architecture — reading logs, checking metrics, inspecting deploys, and executing remediations — to diagnose the root cause and resolve the incident. The environment provides deterministic, shaped rewards that incentivize proper diagnostic methodology (investigate *before* you fix) and penalize reckless actions that make cascading failures worse.

The environment ships with **3 progressively harder tasks** that test increasingly complex SRE reasoning: from a simple bad deploy (1 service, 2 diagnosis steps) to a full cascade failure across 4 services requiring correct fix ordering. Each task uses synthetic but realistic data engines for metrics, logs, and deploy history, making episodes fully deterministic and reproducible — ideal for benchmarking LLM agents against each other on real-world operational reasoning.

---

## Environment Design

### Architecture

```
┌─────────────┐     POST /step      ┌──────────────────────────────────┐
│   AI Agent   │ ──────────────────► │   Incident Response Environment  │
│  (LLM/RL)    │ ◄────────────────── │                                  │
└─────────────┘   IncidentObservation│  ┌────────────┐ ┌─────────────┐  │
                                     │  │ FakeMetrics │ │  FakeLogs   │  │
                                     │  └────────────┘ └─────────────┘  │
                                     │  ┌────────────┐ ┌─────────────┐  │
                                     │  │FakeDeploys │ │  Scenarios  │  │
                                     │  └────────────┘ └─────────────┘  │
                                     └──────────────────────────────────┘
```

### Observation Space

After every action (and on `reset()`), the agent receives an `IncidentObservation`:

| Field | Type | Description |
|---|---|---|
| `observation_text` | `string` | Human-readable output from the last action (log lines, metrics, deploy info) |
| `services_summary` | `object` | Current status of all 5 services: `{service: {status, error_rate, latency_p99_ms}}` |
| `available_actions` | `list[string]` | Human-readable hints for valid actions the agent can take |
| `step_count` | `int` | Current step number in the episode |
| `incident_description` | `string` | The original incident scenario description |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward received for the last action |
| `metadata` | `object` | Episode ID, cumulative reward, diagnosis state, reward reasoning |

**Services simulated:** `api-gateway`, `payment-service`, `user-service`, `db-primary`, `cache-redis`

Each service reports a status of `healthy`, `degraded`, or `critical` based on its error rate and p99 latency.

### Action Space

The agent sends an `IncidentAction` JSON with `action_type`, `target`, and `task_name`:

| Action | Target | Description | Example |
|---|---|---|---|
| `read_logs` | service name | Read the last 15 log lines from a service | `{"action_type": "read_logs", "target": "user-service", "task_name": "..."}` |
| `check_metrics` | service name | Get error rate, latency, CPU, memory, RPS for a service | `{"action_type": "check_metrics", "target": "db-primary", "task_name": "..."}` |
| `check_all_services` | `null` | Get a status overview of all 5 services | `{"action_type": "check_all_services", "target": null, "task_name": "..."}` |
| `check_recent_deploys` | `null` | List recent deployments (last 24h) with timestamps and authors | `{"action_type": "check_recent_deploys", "target": null, "task_name": "..."}` |
| `check_db_queries` | `null` | View the slow query log (last 30 min) | `{"action_type": "check_db_queries", "target": null, "task_name": "..."}` |
| `rollback` | deploy ID | Roll back a specific deployment | `{"action_type": "rollback", "target": "dep-evil-123", "task_name": "..."}` |
| `restart_service` | service name | Restart a service | `{"action_type": "restart_service", "target": "db-primary", "task_name": "..."}` |
| `scale_up` | service name | Scale a service from 3 to 6 replicas | `{"action_type": "scale_up", "target": "db-primary", "task_name": "..."}` |
| `declare_resolved` | `null` | Mark the incident as resolved (terminal action if correct) | `{"action_type": "declare_resolved", "target": null, "task_name": "..."}` |

### Reward Function

The reward function is **deterministic** and uses **priority-based evaluation** — the first matching rule fires:

| Priority | Action | Reward | Condition |
|---|---|---|---|
| 1 | `declare_resolved` | **+1.0** | Environment is actually resolved |
| 2 | `declare_resolved` | **−0.2** | Premature declaration — incident still active |
| 3 | Correct diagnosis action | **+0.2** | Action matches scenario's required diagnosis steps (first time) |
| 4 | Correct fix action | **+0.3** | Correct remediation *after* full diagnosis |
| 5 | Correct fix action | **+0.1** | "Lucky fix" — correct action but without proper diagnosis first |
| 6 | Wrong first action | **−0.1** | Incorrect remediation makes cascade worse (Task 3 only) |
| 7 | Repeated action (>2×) | **−0.1** | Same action taken more than twice |
| 8 | Any other valid action | **0.0** | Action executed but not informative for this scenario |

**Key design choice:** The agent receives partial credit (+0.2) for each diagnostic step, encouraging thorough investigation. Fixing without diagnosing first yields reduced reward (+0.1 vs +0.3), teaching agents that *understanding the problem* matters as much as fixing it.

### Episode Boundaries

An episode ends when:
- The agent calls `declare_resolved` **and** the incident is actually resolved → `done = true`
- The agent reaches `MAX_STEPS` (15 by default in inference) → implicit termination

Calling `declare_resolved` prematurely does **not** end the episode — the agent receives −0.2 but can continue investigating.

---

## Tasks

### Task 1 — Single Service Failure (Easy)

**Scenario:** A bad deploy to `user-service` is causing 500 errors. Users are reporting login failures.

| | |
|---|---|
| **Root Cause** | Bad deployment (`v1.3.0`) with new caching logic introduced a NullPointerException |
| **Affected Service** | `user-service` (error rate 40–80%, fast failures) |
| **Required Diagnosis** | `check_recent_deploys` → `read_logs` on `user-service` (2 steps) |
| **Required Fix** | `rollback` the bad deploy |
| **Baseline (Qwen2.5-72B)** | Total Reward: +1.70 · Normalized Score: 0.113 · Steps: 5 |

**What makes it easy:** Single service affected, clear error signal in logs (NullPointerException), suspicious deploy visible in recent history with `dep-evil-*` ID.

---

### Task 2 — Database Latency Cascade (Medium)

**Scenario:** The primary database is overloaded, causing API latency to cascade through dependent services.

| | |
|---|---|
| **Root Cause** | `db-primary` overloaded — 498/500 connections used, full table scans with no index |
| **Affected Services** | `db-primary` (critical), `payment-service` (degraded), `api-gateway` (degraded) |
| **Required Diagnosis** | `check_metrics` on `api-gateway` → `check_metrics` on `db-primary` → `check_db_queries` (3 steps) |
| **Required Fix** | `scale_up` on `db-primary` + `declare_resolved` |
| **Baseline (Qwen2.5-72B)** | Total Reward: +1.90 · Normalized Score: 0.127 · Steps: 7 |

**What makes it medium:** Multi-hop reasoning required — the agent sees latency on `api-gateway` but must trace it upstream to `db-primary`, then confirm with slow query logs. The fix is `scale_up`, not `restart_service` — a common mistake.

---

### Task 3 — Full Cascade Failure (Hard)

**Scenario:** DB connection pool exhaustion triggers a cascading failure: payment timeouts → API retry storm → OOM across multiple services.

| | |
|---|---|
| **Root Cause** | `db-primary` connection pool exhausted (200/200), causing deadlocks |
| **Affected Services** | `db-primary` (critical), `payment-service` (critical), `api-gateway` (critical) |
| **Required Diagnosis** | 5 actions: check metrics on 3 services + read API gateway logs + check DB queries |
| **Required Fix** | `restart_service` on `db-primary` → then `payment-service` → `declare_resolved` (**order matters**) |
| **Wrong First Actions** | Restarting `api-gateway` or `payment-service` before diagnosis makes the cascade **worse** (−0.1 penalty) |
| **Baseline (Qwen2.5-72B)** | Total Reward: +1.50 · Normalized Score: 0.100 · Steps: 12 |

**What makes it hard:**
1. **Deceptive signals** — `api-gateway` shows the highest visible error rate, tempting agents to restart it first. But it's a symptom, not the cause.
2. **Fix ordering** — DB must be restarted before payment-service, or the retry storm continues.
3. **Penalty trap** — Restarting `payment-service` before diagnosis triggers the wrong-first-action penalty *and* doesn't count as a correct fix.
4. **5-step diagnosis** — The agent must be thorough before acting.

---

## Baseline Scores

All baselines measured using `inference.py` with default settings.

Score = sum(step rewards) / MAX_STEPS. This is the inference.py metric. The /grade endpoint produces a separate weighted rubric score (0.0–1.0) used by judges.

| Task | Model | Normalized Score (total_reward/MAX_STEPS) | Grader Score (/grade) | Steps | Total Reward | Success |
|---|---|---|---|---|---|---|
| `single_service_failure` | Qwen2.5-72B-Instruct | 0.113 | 1.00 | 5 | +1.70 | ✅ |
| `database_latency` | Qwen2.5-72B-Instruct | 0.127 | 1.00 | 7 | +1.90 | ✅ |
| `cascade_failure` | Qwen2.5-72B-Instruct | 0.100 | 1.00 | 12 | +1.50 | ✅ |
| `single_service_failure` | GPT-4o | 0.113 | 1.00 | 4 | +1.70 | ✅ |
| `cascade_failure` | GPT-4o | 0.067 | 0.72 | 15 | +1.00 | ✅ |

> **Note on Metrics:** The **Normalized Score** column is calculated as `total_reward / MAX_STEPS`. We recommend judges evaluate performance based on **total_reward**, as it captures the agent's full diagnostic and remediation accuracy. A perfect sequence yields **+1.70** for Task 1, **+1.90** for Task 2, and **+1.50** for Task 3.

---

## Pre-Submission Validation

| Check | Status | Notes |
|---|---|---|
| HF Space deploys | ✅ | Space live at https://sahilshingate-incident-response-env.hf.space, /health returns 200 |
| OpenEnv spec compliance | ✅ | openenv.yaml validated, typed Pydantic models, step/reset/state endpoints all present |
| Dockerfile builds | ✅ | python:3.11-slim base, port 7860, clean build with no errors |
| Baseline reproduces | ✅ | inference.py completes all 3 tasks, outputs [START]/[STEP]/[END] logs, no errors |
| 3+ tasks with graders | ✅ | single_service_failure (easy), database_latency (medium), cascade_failure (hard) — all graders return 0.0–1.0 |

**Run the built-in deterministic test (no LLM required):**
```bash
python -m src.environment
```

**Verify task registration:**
```bash
curl https://sahilshingate-incident-response-env.hf.space/tasks
```

---

## Setup & Usage

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/sahilshingate01/incident-response-env.git
cd incident-response-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the environment server
uvicorn src.server.app:app --host 0.0.0.0 --port 7860

# 4. (In another terminal) Run inference
export HF_TOKEN="your-huggingface-token"
python inference.py
```

### Docker

```bash
# Build the image
docker build -t incident-env .

# Run the server (port 7860)
docker run -p 7860:7860 incident-env

# Run inference against it
ENV_BASE_URL=http://localhost:7860 HF_TOKEN=your-token python inference.py
```

### Running the Environment Tests

```bash
# Run the built-in deterministic test harness (no LLM needed)
python -m src.environment
```

This runs hardcoded "correct" action sequences for all 3 tasks and verifies every reward matches expectations.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for the LLM |
| `HF_TOKEN` | *(required)* | HuggingFace API token — used as `api_key` |
| `INCIDENT_TASK` | `single_service_failure` | Task to load on server startup / override in inference |
| `ENV_BASE_URL` | `http://localhost:7860` | URL where the environment server is running |

---

## API Reference

| Method | Path | Description | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/health` | Health check | — | `{status, task, episode_id}` |
| `GET` | `/tasks` | List all available tasks | — | `[{name, difficulty, description}]` |
| `POST` | `/reset` | Reset current task | — | `IncidentObservation` |
| `POST` | `/reset/{task_name}` | Switch to a different task and reset | — | `IncidentObservation` |
| `POST` | `/step` | Execute one action | `IncidentAction` | `IncidentObservation` |
| `GET` | `/state` | Get internal state snapshot | — | `IncidentState` |
| `POST` | `/grade` | Grade the current episode | `{task_name: string}` | `{score, max_score, breakdown, passed}` |

Interactive API docs available at `/docs` when the server is running.

---

## Project Structure

```
incident-response-env/
├── Dockerfile                  # Docker image — python:3.11-slim, port 7860
├── .dockerignore               # Excludes .git, __pycache__, .env
├── openenv.yaml                # OpenEnv spec — tasks, endpoints, metadata
├── requirements.txt            # Python deps: fastapi, pydantic, openai, httpx, uvicorn
├── inference.py                # LLM agent runner — [START]/[STEP]/[END] logging
├── README.md                   # This file
└── src/
    ├── __init__.py
    ├── models.py               # Pydantic models: IncidentAction, Observation, State, Reward
    ├── environment.py          # Core RL environment: reset(), step(), state(), reward logic
    ├── data/
    │   ├── __init__.py
    │   ├── fake_metrics.py     # Deterministic synthetic metrics engine (5 services)
    │   ├── fake_logs.py        # Realistic log generator with incident-specific anomalies
    │   ├── fake_deploys.py     # Deploy history with injected bad deploys
    │   └── incident_scenarios.py  # Task definitions: diagnosis steps, fix actions, penalties
    └── server/
        ├── __init__.py
        ├── app.py              # FastAPI server: endpoints, CORS, lifespan, logging
        └── graders.py          # Per-task grading rubrics with weighted score breakdowns
```

---

## Technical Details

### Determinism

All data engines (`FakeMetricsEngine`, `FakeLogEngine`, `FakeDeployHistory`) are seeded by the incident type, making every episode **fully deterministic** for a given task. The same sequence of actions always produces the same observations and rewards.

### Grading Rubrics

Each task has a multi-component grading rubric (in `graders.py`) with weighted scores:

- **Task 1:** 6 components — deploy check (0.15), log read (0.15), diagnosis (0.10), rollback (0.30), resolution (0.20), efficiency (0.10)
- **Task 2:** 7 components — API check (0.10), DB check (0.15), queries (0.15), diagnosis (0.10), scale-up (0.25), resolution (0.15), efficiency (0.10)
- **Task 3:** 12 components — 5 diagnostic checks (0.06 each), diagnosis (0.10), DB restart (0.15), payment restart (0.10), correct order (0.10), resolution (0.10), efficiency (0.10), wrong-first penalty (−0.15)

### Inference Log Format

The `inference.py` script outputs logs in the exact format required by OpenEnv judges:

```
[START] task=single_service_failure env=incident-response-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=check_all_services reward=0.00 done=false error=null
[STEP] step=2 action=check_recent_deploys reward=0.20 done=false error=null
[STEP] step=3 action=read_logs reward=0.20 done=false error=null
[STEP] step=4 action=rollback reward=0.30 done=false error=null
[STEP] step=5 action=declare_resolved reward=1.00 done=true error=null
[END] success=true steps=5 score=0.113 rewards=0.00,0.20,0.20,0.30,1.00
```


## License

MIT — see [LICENSE](LICENSE) for details.

Built by [Sahil Shingate](https://github.com/sahilshingate01) for the OpenEnv Hackathon.
