from __future__ import annotations
from typing import List, Dict, Any
from models import IncidentState
from data.incident_scenarios import SCENARIO_MAP

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
    
    scenario = SCENARIO_MAP.get(state.task_name)
    max_reward = scenario.max_possible_reward if scenario else 1.0
    success_threshold = scenario.success_threshold if scenario else 0.5
    
    diagnosis_score = 0.0
    fix_score = 0.0
    sequence_score = 0.0
    efficiency_score = 0.0

    diagnosis_items_found = sum(1 for d in required_diagnosis if any(a.startswith(d) for a in actions_taken))
    diagnosis_score = (diagnosis_items_found / len(required_diagnosis)) * 0.4 if required_diagnosis else 0.4

    fix_items_found = sum(1 for f in required_fix if any(a.startswith(f) for a in actions_taken))
    fix_score = (fix_items_found / len(required_fix)) * 0.3 if required_fix else 0.3

    if has_fixed_order and len(required_fix) >= 2:
        idx_1 = _index_of(actions_taken, required_fix[0])
        idx_2 = _index_of(actions_taken, required_fix[1])
        if idx_1 is not None and idx_2 is not None and idx_1 < idx_2:
            sequence_score += 0.1
    else:
        sequence_score += 0.1

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

    score = round(max(0.0, min(1.0, total_reward / max_reward)), 4)

    if early_fix:
        return {
            "score": score,
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

    if diagnosis_items_found >= 2 and not early_fix:
        sequence_score += 0.1

    if len(actions_taken) <= len(required_diagnosis) + len(required_fix) + 3 and fix_items_found == len(required_fix):
        efficiency_score = 0.1

    breakdown = {
        "diagnosis_score": round(diagnosis_score, 4),
        "fix_score": round(fix_score, 4),
        "sequence_score": round(sequence_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "capped_due_to_hacking": False
    }

    return {
        "score": score,
        "max_score": 1.0,
        "breakdown": breakdown,
        "passed": score >= success_threshold,
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

def grade_memory_leak_oom(state: IncidentState, actions_taken: list[str], total_reward: float) -> dict:
    scenario = SCENARIO_MAP.get(state.task_name)
    max_reward = scenario.max_possible_reward if scenario else 1.70
    success_threshold = scenario.success_threshold if scenario else 0.5
    
    all_services_check = 0.15 if any(a.startswith("check_all_services") for a in actions_taken) else 0.0
    metrics_check = 0.15 if any(a.startswith("check_metrics") and "user-service" in a for a in actions_taken) else 0.0
    logs_read = 0.15 if any(a.startswith("read_logs") and "user-service" in a for a in actions_taken) else 0.0
    deploy_check = 0.10 if any(a.startswith("check_recent_deploys") for a in actions_taken) else 0.0
    restart_action = 0.30 if any(a.startswith("restart_service") and "user-service" in a for a in actions_taken) else 0.0
    resolution = 0.15 if any(a.startswith("declare_resolved") for a in actions_taken) else 0.0

    breakdown = {
        "all_services_check": all_services_check,
        "metrics_check": metrics_check,
        "logs_read": logs_read,
        "deploy_check": deploy_check,
        "restart_action": restart_action,
        "resolution": resolution
    }

    score = round(max(0.0, min(1.0, total_reward / max_reward)), 4)
    return {
        "score": score,
        "max_score": 1.0,
        "breakdown": breakdown,
        "passed": score >= success_threshold,
    }

GRADER_MAP = {
    "single_service_failure": grade_task_1,
    "database_latency": grade_task_2,
    "cascade_failure": grade_task_3,
    "memory_leak_oom": grade_memory_leak_oom
}

def grade(task_name: str, state: IncidentState, actions_taken: list[str], total_reward: float) -> dict:
    grader_fn = GRADER_MAP.get(task_name)
    if grader_fn is None:
        return {"score": 0.01, "max_score": 1.0, "breakdown": {}, "passed": False}
    return grader_fn(state, actions_taken, total_reward)
