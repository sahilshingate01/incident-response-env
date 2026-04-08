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

## 🏗️ Environment Design

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
| `observation_text` | `string` | Human-readable output (log lines, metrics, deploy info) |
| `services_summary` | `object` | Status of all 5 services: `{status, error_rate, latency_p99_ms}` |
| `available_actions` | `list[string]` | Hints for valid actions |
| `step_count` | `int` | Current step number |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward received for the last action |

**Services simulated:** `api-gateway`, `payment-service`, `user-service`, `db-primary`, `cache-redis`

---

## 🎮 Tasks & Difficulty

### Task 1 — Single Service Failure (Easy)
A bad deploy to `user-service`.
- **Reasoning**: Detect direct correlation between a deployment timestamp and an error spike.
- **Required**: `check_recent_deploys` → `read_logs` → `rollback`.

### Task 2 — Database Latency Cascade (Medium)
Primary DB overload causing dependent service timeouts.
- **Reasoning**: Multi-hop latency tracing from Gateway → Downstream → Database.
- **Required**: Correlate high p99 latency with slow query logs and scale up.

### Task 3 — Full Cascade Failure (Hard)
Connection pool exhaustion triggering a retry storm and OOM.
- **Reasoning**: **Non-linear root cause**. The apparent error (OOM) is 3 hops away from the true root cause (DB deadlocks).
- **Required**: Complex investigative trajectory (5+ steps) and strictly ordered remediation.

---

## 🏆 Baseline Scores

Baselines measured using `inference.py` (optimized for NVIDIA NIM).

| Task | Model | Grader Score | Pass? | Steps |
|---|---|---|---|---|
| `single_service_failure` | DeepSeek-V3.1 | **0.900** | ✅ | 9 |
| `database_latency` | DeepSeek-V3.1 | **0.617** | ✅ | 15 |
| `cascade_failure` | Llama 3.1 70B | **0.450** | ⚠️ | 12 |

---

## 🛠️ Setup & Usage

### 🚀 Quick Start (with `uv`)

The project is optimized for [uv](https://github.com/astral-sh/uv).

```bash
# 1. Install dependencies and generate lockfile
uv lock
uv sync

# 2. Start the OpenEnv server
uv run server

# 3. Run inference (NVIDIA NIM)
export NVIDIA_API_KEY="your_key_here"
uv run python inference.py
```

### 🐳 Docker

```bash
docker build -t incident-env .
docker run -p 7860:7860 incident-env
```

---

## 📂 Project Structure

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

## 📜 License

MIT — Built for the OpenEnv Hackathon by [Sahil Shingate](https://github.com/sahilshingate01).
