"""
Pydantic models for the Incident Response RL Environment.
All models use Pydantic BaseModel (not dataclasses).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from enum import Enum


# ──────────────────────────────────────────────
# Shared / utility enums & models
# ──────────────────────────────────────────────

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ServiceMetrics(BaseModel):
    error_rate: float
    latency_p99_ms: int
    cpu_percent: int
    memory_percent: int
    requests_per_sec: int


class ServiceSummary(BaseModel):
    status: ServiceStatus
    error_rate: float
    latency_p99_ms: int


class DeployInfo(BaseModel):
    id: str
    service: str
    version: str
    timestamp: str
    deployed_by: str
    status: str
    commit_message: str


# ──────────────────────────────────────────────
# Core RL models
# ──────────────────────────────────────────────

VALID_ACTION_TYPES = [
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

VALID_TASK_NAMES = [
    "single_service_failure",
    "database_latency",
    "cascade_failure",
]


class IncidentAction(BaseModel):
    """
    Represents a single action taken by the RL agent.

    - action_type: one of the VALID_ACTION_TYPES
    - target: service name, deploy_id, or None depending on action_type
    - task_name: the scenario being played
    """
    action_type: str
    target: Optional[str] = None
    task_name: str

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        if v not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action_type '{v}'. Must be one of: {VALID_ACTION_TYPES}"
            )
        return v

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, v: str) -> str:
        if v not in VALID_TASK_NAMES:
            raise ValueError(
                f"Invalid task_name '{v}'. Must be one of: {VALID_TASK_NAMES}"
            )
        return v

    def action_key(self) -> str:
        """
        Build the composite key used to match against scenario's
        correct_diagnosis_actions / correct_fix_actions.
        e.g. "read_logs" + "user-service" -> "read_logs_user-service"
             "check_recent_deploys" + None  -> "check_recent_deploys"
        """
        if self.target:
            return f"{self.action_type}_{self.target}"
        return self.action_type


class IncidentObservation(BaseModel):
    """What the agent sees after every step (or on reset)."""
    done: bool
    reward: float
    observation_text: str
    available_actions: List[str]
    services_summary: dict
    step_count: int
    incident_description: str
    metadata: dict = {}


class IncidentReward(BaseModel):
    """Detailed reward breakdown for a single step."""
    value: float = Field(ge=-1.0, le=1.0)
    reason: str


class IncidentState(BaseModel):
    """Full internal state snapshot (for debugging / logging)."""
    episode_id: str
    task_name: str
    step_count: int
    incident_type: str
    correctly_diagnosed: bool
    resolved: bool
    actions_taken: List[str]
    current_scenario: str
