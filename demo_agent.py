"""
demo_agent.py — Hardcoded optimal agent for live demo playback.

Provides deterministic, perfect action sequences for all 3 incident tasks.
Each step includes agent_reasoning that sounds like a senior SRE explaining
their thought process.  Used by the /demo/run SSE endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DemoStep:
    """A single step the demo agent will take."""
    action_type: str
    target: Optional[str]
    agent_reasoning: str


# ──────────────────────────────────────────────
# Task 1 — Single Service Failure (Easy)
# ──────────────────────────────────────────────

TASK_1_STEPS: List[DemoStep] = [
    DemoStep(
        action_type="check_all_services",
        target=None,
        agent_reasoning=(
            "First, I need a broad picture. Let me pull the service status dashboard "
            "to see which services are flashing red before I dive into anything specific."
        ),
    ),
    DemoStep(
        action_type="check_recent_deploys",
        target=None,
        agent_reasoning=(
            "user-service is in CRITICAL state — that's our prime suspect. "
            "Error spikes like this often correlate with recent deploys. "
            "Let me check the deploy history for the last 24 hours."
        ),
    ),
    DemoStep(
        action_type="read_logs",
        target="user-service",
        agent_reasoning=(
            "Found a suspicious deploy 'dep-evil-123' pushed to user-service recently. "
            "Let me pull the logs to confirm the error pattern matches the deploy timestamp."
        ),
    ),
    DemoStep(
        action_type="rollback",
        target="dep-evil-123",
        agent_reasoning=(
            "Logs confirm: NullPointerException stack traces started exactly after dep-evil-123 "
            "was deployed. Classic bad release. Rolling back to restore the last known-good version."
        ),
    ),
    DemoStep(
        action_type="declare_resolved",
        target=None,
        agent_reasoning=(
            "Rollback complete. Error rate on user-service should be normalizing. "
            "Root cause was a bad deploy — simple, clean resolution. Declaring incident resolved."
        ),
    ),
]


# ──────────────────────────────────────────────
# Task 2 — Database Latency Cascade (Medium)
# ──────────────────────────────────────────────

TASK_2_STEPS: List[DemoStep] = [
    DemoStep(
        action_type="check_all_services",
        target=None,
        agent_reasoning=(
            "Multiple services showing elevated latency. This smells like an upstream "
            "dependency problem. Let me get the full picture before jumping to conclusions."
        ),
    ),
    DemoStep(
        action_type="check_metrics",
        target="api-gateway",
        agent_reasoning=(
            "api-gateway is the entry point — if latency is high here, it's either the gateway "
            "itself or something downstream. Need to see its specific metrics to differentiate."
        ),
    ),
    DemoStep(
        action_type="check_metrics",
        target="db-primary",
        agent_reasoning=(
            "api-gateway latency is elevated but CPU is normal. That rules out the gateway itself. "
            "The latency pattern suggests a blocking downstream call. db-primary is the most likely "
            "bottleneck — let me check it."
        ),
    ),
    DemoStep(
        action_type="check_db_queries",
        target=None,
        agent_reasoning=(
            "Confirmed: db-primary CPU is pegged at 95%+ and latency is through the roof. "
            "This is a database overload scenario. Let me check the slow query log to understand "
            "what's hammering the DB."
        ),
    ),
    DemoStep(
        action_type="scale_up",
        target="db-primary",
        agent_reasoning=(
            "Slow queries show full table scans with no indexes, called 1200+ times. "
            "498/500 connections used. The DB needs immediate relief — scaling up replicas "
            "to distribute the read load while we address the query optimization separately."
        ),
    ),
    DemoStep(
        action_type="declare_resolved",
        target=None,
        agent_reasoning=(
            "Scale-up is distributing load across new replicas. Connection pool pressure "
            "is relieving. API gateway latency should cascade back to normal. "
            "Declaring incident resolved — will file a follow-up ticket for query optimization."
        ),
    ),
]


# ──────────────────────────────────────────────
# Task 3 — Full Cascade Failure (Hard)
# ──────────────────────────────────────────────

TASK_3_STEPS: List[DemoStep] = [
    DemoStep(
        action_type="check_all_services",
        target=None,
        agent_reasoning=(
            "Multiple services are degraded or critical. This looks like a cascade. "
            "I need to map the blast radius before touching anything — wrong first move "
            "in a cascade can make things exponentially worse."
        ),
    ),
    DemoStep(
        action_type="check_metrics",
        target="api-gateway",
        agent_reasoning=(
            "api-gateway is showing high error rate and latency. But the pattern looks like "
            "a retry storm — the gateway might be a victim, not the root cause. "
            "Need to trace the dependency chain downstream."
        ),
    ),
    DemoStep(
        action_type="check_metrics",
        target="db-primary",
        agent_reasoning=(
            "api-gateway metrics confirm it's retrying failed requests to downstream services. "
            "The error pattern points to the data layer. Let me check db-primary — "
            "connection pool exhaustion could explain why everything is backed up."
        ),
    ),
    DemoStep(
        action_type="check_metrics",
        target="payment-service",
        agent_reasoning=(
            "db-primary is at 95%+ CPU with massive error rates. But I need to verify "
            "payment-service too — it sits between the gateway and DB. If it's holding "
            "connections open during retries, it could be amplifying the cascade."
        ),
    ),
    DemoStep(
        action_type="check_db_queries",
        target=None,
        agent_reasoning=(
            "payment-service latency is 4-6 seconds with high error rates. It's blocking "
            "on DB operations. Let me check the slow query log to see what's actually "
            "happening at the database level — deadlocks or connection exhaustion?"
        ),
    ),
    DemoStep(
        action_type="read_logs",
        target="api-gateway",
        agent_reasoning=(
            "Connection pool: 200/200 EXHAUSTED with 128 deadlocks. This is the root cause. "
            "DB connection pool is completely drained, causing cascading timeouts. "
            "Let me check api-gateway logs to confirm the full failure chain before applying fixes."
        ),
    ),
    DemoStep(
        action_type="restart_service",
        target="db-primary",
        agent_reasoning=(
            "Root cause confirmed: db-primary connection pool exhaustion → payment-service "
            "timeouts → api-gateway retry storm. Fix order is critical here. "
            "Must restart db-primary FIRST to clear the deadlocked connections and "
            "drain the blocked thread pool."
        ),
    ),
    DemoStep(
        action_type="restart_service",
        target="payment-service",
        agent_reasoning=(
            "db-primary is back online with a fresh connection pool. Now restarting "
            "payment-service to clear its stale connections and retry queues. "
            "This has to happen AFTER db-primary or the retries will just re-exhaust the pool."
        ),
    ),
    DemoStep(
        action_type="declare_resolved",
        target=None,
        agent_reasoning=(
            "Both services restarted in correct order. The cascade should be breaking: "
            "db-primary has clean connections → payment-service clears retry backlog → "
            "api-gateway error rate normalizes. Incident resolved."
        ),
    ),
]


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

DEMO_SEQUENCES = {
    "single_service_failure": TASK_1_STEPS,
    "database_latency": TASK_2_STEPS,
    "cascade_failure": TASK_3_STEPS,
}

TASK_METADATA = {
    "single_service_failure": {
        "name": "single_service_failure",
        "label": "Single Service Failure",
        "difficulty": "Easy",
        "difficulty_color": "#22c55e",
        "description": "A bad deploy to user-service is causing 500 errors.",
        "expected_steps": len(TASK_1_STEPS),
        "max_reward": 1.70,
    },
    "database_latency": {
        "name": "database_latency",
        "label": "Database Latency Cascade",
        "difficulty": "Medium",
        "difficulty_color": "#f59e0b",
        "description": "DB overload causing API latency to cascade across services.",
        "expected_steps": len(TASK_2_STEPS),
        "max_reward": 1.90,
    },
    "cascade_failure": {
        "name": "cascade_failure",
        "label": "Full Cascade Failure",
        "difficulty": "Hard",
        "difficulty_color": "#ef4444",
        "description": "DB connection pool exhaustion triggers cascading failure across 4 services.",
        "expected_steps": len(TASK_3_STEPS),
        "max_reward": 2.45,
    },
}
