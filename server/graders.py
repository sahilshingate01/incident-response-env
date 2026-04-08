"""
Grading functions for each incident response task using Bulletproof Evaluation.

Each grader takes a completed episode's state and action history and
returns a score dict with { score, max_score, breakdown, passed }.
"""

from __future__ import annotations
from typing import List, Dict, Any
from models import IncidentState

def _index_of(actions: list[str], key: str) -> int | None:
    for i, a in enumerate(actions):
        if a == key or a.startswith(key):
            return i
    return None

def grade_trajectory_generic(
    state: IncidentState,
    actions_taken: list[str],
    total_reward: float,
    required_diagnosis: list[str],
    required_fix: list[str],
    has_fixed_order: bool = False
) -> dict:
    """Bulletproof common grader implementation tracking trajectory and capping hacking."""
    
    diagnosis_score = 0.0
    fix_score = 0.0
    sequence_score = 0.0
    efficiency_score = 0.0

    diagnosis_items_found = sum(1 for d in required_diagnosis if any(a.startswith(d) for a in actions_taken))
    diagnosis_score = (diagnosis_items_found / len(required_diagnosis)) * 0.4 if required_diagnosis else 0.4

    fix_items_found = sum(1 for f in required_fix if any(a.startswith(f) for a in actions_taken))
    fix_score = (fix_items_found / len(required_fix)) * 0.3 if required_fix else 0.3

    # Sequence tracking
    if has_fixed_order and len(required_fix) >= 2:
        idx_1 = _index_of(actions_taken, required_fix[0])
        idx_2 = _index_of(actions_taken, required_fix[1])
        if idx_1 is not None and idx_2 is not None and idx_1 < idx_2:
            sequence_score += 0.1
    else:
        # Default sequence pass if fixed order not strictly required
        sequence_score += 0.1

    # Investigation before fix verification (anti-hacking)
    first_fix_idx = len(actions_taken)
    for f in required_fix:
        if "declare_resolved" in f: continue
        i = _index_of(actions_taken, f)
        if i is not None and i < first_fix_idx:
            first_fix_idx = i

    early_fix = False
    if first_fix_idx < len(actions_taken):
        inv_actions_before_fix = sum(
            1 for a in actions_taken[:first_fix_idx] 
            if a.startswith("check_metrics") or a.startswith("read_logs") or a.startswith("check_db_queries") or a.startswith("check_recent_deploys")
        )
        if inv_actions_before_fix < 2 and diagnosis_items_found < len(required_diagnosis):
            early_fix = True

    if early_fix:
        # Cap score due to jumping to fix
        capped_val = min(0.3, fix_score)
        # OpenEnv: Must be strictly > 0
        clamped_capped = max(0.01, min(0.99, capped_val))

        return {
            "score": clamped_capped,
            "max_score": 1.0,
            "breakdown": {
                "diagnosis_score": 0.01,
                "fix_score": round(fix_score, 4),
                "sequence_score": 0.01,
                "efficiency_score": 0.01,
                "capped_due_to_hacking": True
            },
            "passed": False
        }

    # Rest of sequence score
    if diagnosis_items_found >= 2 and not early_fix:
        sequence_score += 0.1

    # Efficiency (0.1)
    if len(actions_taken) <= len(required_diagnosis) + len(required_fix) + 3 and fix_items_found == len(required_fix):
        efficiency_score = 0.1

    total_score = round(diagnosis_score + fix_score + sequence_score + efficiency_score, 4)
    
    breakdown = {
        "diagnosis_score": round(diagnosis_score, 4),
        "fix_score": round(fix_score, 4),
        "sequence_score": round(sequence_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "capped_due_to_hacking": False
    }

    # OpenEnv Phase 2 Requirement: Scores must be strictly within (0, 1)
    # 0.00 and 1.00 are specifically prohibited by the validator.
    clamped_score = max(0.01, min(0.99, total_score))

    return {
        "score": clamped_score,
        "max_score": 1.0,
        "breakdown": breakdown,
        "passed": total_score >= 0.7,
    }

def grade_task_1(state: IncidentState, actions: list[str], reward: float) -> dict:
    return grade_trajectory_generic(
        state, actions, reward,
        required_diagnosis=["check_recent_deploys", "read_logs_user-service"],
        required_fix=["rollback", "declare_resolved"]
    )

def grade_task_2(state: IncidentState, actions: list[str], reward: float) -> dict:
    return grade_trajectory_generic(
        state, actions, reward,
        required_diagnosis=["check_metrics_api-gateway", "check_metrics_db-primary", "check_db_queries"],
        required_fix=["scale_up_db-primary", "declare_resolved"]
    )

def grade_task_3(state: IncidentState, actions: list[str], reward: float) -> dict:
    return grade_trajectory_generic(
        state, actions, reward,
        required_diagnosis=["check_metrics_api-gateway", "check_metrics_payment-service", "check_metrics_db-primary", "read_logs_api-gateway", "check_db_queries"],
        required_fix=["restart_service_db-primary", "restart_service_payment-service", "declare_resolved"],
        has_fixed_order=True
    )

GRADER_MAP = {
    "single_service_failure": grade_task_1,
    "database_latency": grade_task_2,
    "cascade_failure": grade_task_3,
}

def grade(task_name: str, state: IncidentState, actions_taken: list[str], total_reward: float) -> dict:
    grader_fn = GRADER_MAP.get(task_name)
    if grader_fn is None:
        return {"score": 0.01, "max_score": 1.0, "breakdown": {}, "passed": False}
    return grader_fn(state, actions_taken, total_reward)
