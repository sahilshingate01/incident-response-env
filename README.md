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
  - reasoning-models
license: mit
---

# 🚨 Incident Response Environment

**Evaluate AI agents on realistic production incident response with advanced reasoning support.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=flat-square)](https://github.com/OpenEnv-AI/openenv)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-green?style=flat-square&logo=nvidia)](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nim/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green?style=flat-square&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## ⚡ Better Reasoning for Ops

Modern production incidents aren't solved by simple keyword matching—they require deep, multi-hop deductive reasoning. **Incident Response Environment** is built to challenge the next generation of **Large Reasoning Models (LRMs)**. 

Whether using DeepSeek-R1's internal chain-of-thought or OpenAI's o1-series, this environment captures the true complexity of SRE:
- **Trace Correlation**: Agents must correlate errors across 5 microservices using realistic trace IDs.
- **Hypothesis Testing**: Deterministic logs and metrics allow agents to test hypotheses about root causes.
- **Anti-Hacking Trajectories**: Built-in graders penalize "lucky guessing" and reward agents that follow rigorous investigative methodology.
- **NVIDIA NIM Integration**: Baseline scripts are optimized for NVIDIA NIM, supporting high-throughput evaluation of frontier models like Llama 3.1 405B and DeepSeek-V3.

---

## ## Overview

Production incidents cost companies an estimated **$5,600 per minute** of downtime. When a P1 alert fires at 3 AM, an on-call engineer must rapidly triage logs, metrics, and deploy history — often across dozens of interdependent services — to isolate the root cause and remediate before cascading failures take down the entire platform.

**Incident Response Environment** is an OpenEnv-compliant RL/LLM evaluation environment that simulates realistic production incidents. An AI agent receives a system alert and must navigate a 5-service microservice architecture — reading logs, checking metrics, inspecting deploys, and executing remediations — to diagnose the root cause and resolve the incident. The environment provides deterministic, shaped rewards that incentivize proper diagnostic methodology (investigate *before* you fix) and penalize reckless actions that make cascading failures worse.

The environment ships with **3 progressively harder tasks** that test increasingly complex SRE reasoning: from a simple bad deploy (1 service, 2 diagnosis steps) to a full cascade failure across 4 services requiring correct fix ordering. Each task uses synthetic but realistic data engines for metrics, logs, and deploy history, making episodes fully deterministic and reproducible — ideal for benchmarking LLM agents against each other on real-world operational reasoning.

---

## ## Environment Design

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

#### Logs and Temporal Flow

To enhance realism, observations evolve across steps (temporal flow) instead of being static snapshots. Logs proactively returned by the environment include:
- Identifiable **timestamps** (in ISO format) showing events evolving over time.
- Distinct **log levels** (INFO, WARN, ERROR).
- Standardized `trace_id` / `request_id` markers to correlate traces.
- Partial **stack traces** on key application errors.

The environment challenges the agent to filter signal from noise by regularly injecting **30–50% irrelevant noise logs** containing unrelated service logs and benign warnings into the observations.

---

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

---

### Reward Function

The reward function shapes behavior by deeply rewarding thorough diagnosis *before* any action is taken. The environment implements a continuous dense reward tied to its internal state tracking mechanism.

**Diagnostic Rewards:**
- **+0.2** for a correct intermediate diagnostic step
- **+0.1** for a partially useful step
- **0.0** for neutral evaluation
- **-0.05** for duplicate actions
- **-0.1** for an irrelevant step

**Action & Resolution Rewards:**
- **+0.3** for a correct final fix AFTER proper diagnosis
- **+0.2** for a correct fix applied without full diagnosis
- **+1.0** for successful resolution declaration
- **-0.2** for an incorrect or harmful action or premature resolution.

---

## ## Tasks

### Task 1 — Single Service Failure (Easy)
**Scenario:** A bad deploy to `user-service` is causing 500 errors.
- **Reasoning**: Detect direct correlation between a deployment timestamp and an error spike.
- **Required**: `check_recent_deploys` → `read_logs` → `rollback`.

### Task 2 — Database Latency Cascade (Medium)
**Scenario:** DB overload causing API latency to cascade.
- **Reasoning**: Multi-hop latency tracing from Gateway → Downstream → Database.
- **Required**: Correlate high p99 latency with slow query logs and scale up.

### Task 3 — Full Cascade Failure (Hard)
**Scenario:** DB connection pool exhaustion triggering a cascade.
- **Reasoning**: **Non-linear root cause**. Root cause is invisible at the initial layer.
- **Required**: Complex investigative trajectory (5+ steps) and strictly ordered remediation.

---

## ## Trajectory Grader

The environment utilizes a dedicated, bulletproof grade implementation:
- **Deterministic scoring:** Identical trajectories produce identical scores.
- **Evaluates full trajectory:** Grading occurs over sequential history actions.
- **Prevents skipping diagnosis:** Directly guessing the resolution action before proper validation will flag an anti-hack sequence.
- **Prevents reward hacking:** Triggers a rigid cap score reduction (to `0.3`) if shortcut validation paths are identified.

---

## ## Baseline Scores

All baselines measured using `inference.py` (optimized for NVIDIA NIM).

> **Understanding the Metrics**
> - **Grader Score (Primary):** Official evaluation metric (0.0–1.0).
> - **Total Reward (Secondary):** Raw sum of rewards accumulated.

| Task | Model | Grader Score | Pass? | Steps | Total Reward |
|---|---|---|---|---|---|
| `single_service_failure` | DeepSeek-V3.1 | **0.900** | ✅ | 9 | +1.60 |
| `database_latency` | DeepSeek-V3.1 | **0.617** | ✅ | 15 | +0.25 |
| `cascade_failure` | Llama 3.1 70B | **0.450** | ⚠️ | 12 | +0.80 |

---

## ## Pre-Submission Validation

Before submitting your environment to the OpenEnv Hub, run these validation steps to ensure your submission passes all automated checks.

### 📋 Validation Checklist
| Check | Requirement | Status |
|---|---|---|
| **Port Exposure** | Container listens on port `7860` | 🟢 |
| **Task Registration** | `/tasks` returns 3 valid tasks | 🟢 |
| **Logic Consistency** | `python environment.py` passes all test episodes | 🟢 |
| **Inference Baseline** | `inference.py` produces `[END]` log lines for all tasks | 🟢 |
| **Key Handling** | System handles missing `NVIDIA_API_KEY` without crashing | 🟢 |

### 🛠️ Verification Commands

```bash
# 1. Run the deterministic environment test harness
python environment.py

# 2. Check task registration metadata
curl http://localhost:7860/tasks

# 3. Verify the inference script (dry-run)
python inference.py
```

---

## ## Setup & Usage

### 🚀 Quick Start (with `uv`)

The project is optimized for [uv](https://github.com/astral-sh/uv).

```bash
# 1. generate lockfile and sync
uv lock
uv sync

# 2. Start the OpenEnv server
uv run server

# 3. (In another terminal) Run inference (NVIDIA NIM)
export NVIDIA_API_KEY="your-key"
uv run python inference.py
```

### Local Development (Standard Pip)

```bash
pip install -r requirements.txt
python -m server.app
```

### Docker

```bash
docker build -t incident-env .
docker run -p 7860:7860 incident-env
```

---

## ## API Reference

| Method | Path | Description | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/health` | Health check | — | `{status, task, episode_id}` |
| `GET` | `/tasks` | List all available tasks | — | `[{name, difficulty, description}]` |
| `POST` | `/reset` | Reset current task | — | `IncidentObservation` |
| `POST` | `/reset/{task_name}` | Switch to a different task and reset | — | `IncidentObservation` |
| `POST` | `/step` | Execute one action | `IncidentAction` | `IncidentObservation` |
| `GET` | `/state` | Get internal state snapshot | — | `IncidentState` |
| `POST` | `/grade` | Grade the current episode | `{task_name: string}` | `{score, max_score, breakdown, passed}` |

---

## ## Project Structure

```
incident-response-env/
├── Dockerfile          # python:3.11-slim, port 7860
├── openenv.yaml        # OpenEnv registration & task config
├── pyproject.toml      # PEP 621 metadata & uv config
├── uv.lock             # Deterministic dependency lockfile
├── server/             # FastAPI entry point & graders
├── data/               # Deterministic data engines (Logs/Metrics)
├── environment.py      # Core RL Environment logic (Step/Reset)
├── models.py           # Pydantic schema definitions
└── inference.py        # NVIDIA NIM reasoning baseline
```

---

## ## License

MIT — Built for the OpenEnv Hackathon by [Sahil Shingate](https://github.com/sahilshingate01).
