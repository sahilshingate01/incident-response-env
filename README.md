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

# 🚨 AI SRE Agent — Live Incident Response

> **Production is down. $5,600 per minute is bleeding.** An AI agent just diagnosed a cascading failure across 4 services, identified a deadlocked database connection pool as root cause, and fixed everything in 9 steps — without waking anyone up.

**An RL/LLM benchmark environment where AI agents autonomously diagnose and remediate production incidents across a 5-service microservice architecture.**

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-Watch_It_Happen-red?style=for-the-badge)](https://huggingface.co/spaces/Sahilshingate/incident-response-env)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=flat-square)](https://github.com/OpenEnv-AI/openenv)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green?style=flat-square&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## ⚡ The Numbers

| Metric | Value |
|---|---|
| **Incident types** | 3 (Easy → Medium → Hard) |
| **Services simulated** | 5 microservices |
| **Agent actions available** | 9 types |
| **Optimal Task 1 reward** | +1.70 (5 steps) |
| **Optimal Task 3 reward** | +2.60 (9 steps) |
| **Time to resolve (cascade)** | 9 steps, ~14 seconds |

---

## 🎬 Watch It Happen

**[▶ Launch the Live Demo](https://huggingface.co/spaces/Sahilshingate/incident-response-env)** — no login, no API key, one click.

Hit **TRIGGER INCIDENT** and watch an AI SRE agent:
1. 🚨 Receive a cascading failure alert across 4 services
2. 🔍 Systematically investigate metrics, logs, and DB queries
3. 🧠 Reason through deceptive signals to find the real root cause
4. 🔧 Apply fixes in the correct order (DB → payment-service)
5. ✅ Resolve the incident — all services flip from red to green

The demo runs a hardcoded optimal agent so you see the **perfect trajectory** every time. Plug in your own LLM to see how it performs.

---

## 🧠 Why This Is Hard

This isn't "find the red service and restart it." These incidents require **multi-hop deductive reasoning**:

### Deceptive Signals
`api-gateway` error rate hits 67% — but it's a **symptom**, not the cause. Agents that restart the gateway waste a step and get penalized. The real root cause is a deadlocked `db-primary` connection pool two layers down.

### Fix Ordering Matters
In the cascade failure, you **must** restart `db-primary` before `payment-service`. Reversing the order re-exhausts the connection pool and makes things worse. The environment penalizes wrong-order fixes.

### Penalty Traps
- Restarting `payment-service` before diagnosing? **-0.15** penalty
- Declaring resolved prematurely? **-0.20** penalty
- Repeating actions? **-0.05** per duplicate
- The optimal path requires reading signals across 5 services, correlating slow queries with connection pools, and applying fixes in strict dependency order

---

## 🏗️ Architecture

```
┌─────────────────┐    POST /step     ┌──────────────────────────────────┐
│    AI Agent      │ ────────────────► │  Incident Response Environment   │
│   (LLM / RL)    │ ◄──────────────── │                                  │
└─────────────────┘  IncidentObs      │  ┌────────────┐ ┌─────────────┐  │
                                      │  │ FakeMetrics │ │  FakeLogs   │  │
  ┌─────────────────┐                 │  └────────────┘ └─────────────┘  │
  │  Live Dashboard  │  SSE Stream    │  ┌────────────┐ ┌─────────────┐  │
  │  (index.html)    │ ◄──────────── │  │FakeDeploys │ │  Scenarios  │  │
  └─────────────────┘  /demo/run      │  └────────────┘ └─────────────┘  │
                                      └──────────────────────────────────┘
```

**Services simulated:** `api-gateway` · `payment-service` · `user-service` · `db-primary` · `cache-redis`

Each service reports: `status` (healthy/degraded/critical), `error_rate`, `latency_p99_ms`

---

## 📋 Tasks

### Task 1 — Single Service Failure (Easy)
A bad deploy to `user-service` is causing 500 errors.
- **Key insight:** Direct correlation between deploy timestamp and error spike
- **Optimal path:** `check_recent_deploys` → `read_logs` → `rollback` → `declare_resolved`
- **Optimal reward:** +1.70

### Task 2 — Database Latency Cascade (Medium)
DB overload causing API latency to cascade across services.
- **Key insight:** Multi-hop latency tracing from Gateway → Downstream → Database
- **Optimal path:** `check_metrics(api-gateway)` → `check_metrics(db-primary)` → `check_db_queries` → `scale_up(db-primary)` → `declare_resolved`
- **Optimal reward:** +1.90

### Task 3 — Full Cascade Failure (Hard)
DB connection pool exhaustion triggering cascading failure across 4 services.
- **Key insight:** Non-linear root cause — root cause is invisible at the initial layer
- **Optimal path:** 5 diagnostic steps → `restart_service(db-primary)` → `restart_service(payment-service)` → `declare_resolved`
- **Optimal reward:** +2.60

---

## 🎯 Reward Function

| Category | Reward | Description |
|---|---|---|
| ✅ Correct diagnosis step | **+0.20** | Each correct investigative action |
| 🔧 Correct fix (after diagnosis) | **+0.30** | Right remediation after proper investigation |
| 🔧 Correct fix (no diagnosis) | **+0.20** | Lucky guess, reduced reward |
| 🎉 Successful resolution | **+1.00** | All fixes applied, incident resolved |
| ⚠️ Duplicate action | **-0.05** | Repeating a previous action |
| ❌ Wrong first action | **-0.15** | Penalty for incorrect initial remediation |
| ❌ Premature resolution | **-0.20** | Declaring resolved while issues persist |

---

## 📈 Scoring & Evaluation

The final task grade evaluation generates a normalized score between `0.0` and `1.0`.  

The evaluation checks standard compliance correctly scaled up against maximum attainable threshold constraints on given incidents. The formula is strictly equivalent to:
`Score = Cumulative Reward / Max Possible Reward`

Where the **Max Possible Rewards** per specific scenario outline correctly matches:
* **`1.70`** Single Service Failure  
* **`1.90`** Database Latency 
* **`2.50`** Cascade Failure 
* **`1.70`** Memory Leak / OOM Failure  

| Task Level | Scenario Type | Max Possible Reward | Normalized Score Scale | Status / Minimum Success |
|---|---|---|---|---|
| Easy | Bad Deploy Rollout | `1.70` | 0.0 — 1.0 | Requires Score ≥ 0.5 |
| Medium | DB Outage Diagnostics | `1.90` | 0.0 — 1.0 | Requires Score ≥ 0.6 |
| Medium | Memory Leak / OOM | `1.70` | 0.0 — 1.0 | Requires Score ≥ 0.5 |
| Hard | Full Cascade Storm | `2.50` | 0.0 — 1.0 | Requires Score ≥ 0.7 |

Your agent logic is inherently tested against strict randomized properties within bounds across repeated invocations seamlessly provided by the generated metadata seeds (e.g., `seed: 123456`) directly through local tests or on the HuggingFace endpoint itself.

---

## 🔌 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | **Live demo dashboard** |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Reset current task |
| `POST` | `/reset/{task_name}` | Switch task and reset |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Get internal state |
| `POST` | `/grade` | Grade current episode |
| `POST` | `/demo/run?task={name}` | **SSE stream — live demo** |
| `GET` | `/demo/tasks` | Task metadata for UI |
| `GET` | `/docs` | Swagger API docs |

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone
git clone https://github.com/sahilshingate01/incident-response-env.git
cd incident-response-env

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Open http://localhost:7860 for the live demo
```

### With Docker

```bash
docker build -t incident-env .
docker run -p 7860:7860 incident-env
```

### Run Inference (with your own LLM)

```bash
export API_KEY="your-api-key"
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
python inference.py
```

---

## 📁 Project Structure

```
incident-response-env/
├── server/
│   ├── app.py              # FastAPI server & endpoints
│   ├── demo_routes.py      # SSE streaming demo endpoint
│   └── graders.py          # Trajectory grading logic
├── src/static/
│   └── index.html          # Live demo dashboard (single file, no build)
├── data/
│   ├── fake_metrics.py     # Deterministic metrics engine
│   ├── fake_logs.py        # Deterministic log engine
│   ├── fake_deploys.py     # Deterministic deploy history
│   └── incident_scenarios.py  # Task definitions
├── environment.py          # Core RL environment (step/reset)
├── models.py               # Pydantic schemas
├── demo_agent.py           # Hardcoded optimal agent for demos
├── inference.py            # LLM inference runner
├── Dockerfile              # python:3.11-slim, port 7860
└── openenv.yaml            # OpenEnv registration
```

---

## 📊 Baseline Scores

| Task | Model | Normalized Score | Steps | Total Reward |
|---|---|---|---|---|
| `single_service_failure` | Optimal Agent | **1.000** | 5 | +1.70 |
| `database_latency` | Optimal Agent | **1.000** | 6 | +1.90 |
| `cascade_failure` | Optimal Agent | **1.000** | 10 | +2.50 |
| `memory_leak_oom` | Optimal Agent | **1.000** | 6 | +1.70 |

**Demo agent (hardcoded optimal) scores perfectly normalized 1.0/1.0 across all scenarios.**

---

## License

MIT — Built for the OpenEnv Hackathon by [Sahil Shingate](https://github.com/sahilshingate01).
