"""
Grading functions for each incident response task.

Each grader takes a completed episode's state and action history and
returns a score dict with { score, max_score, breakdown, passed }.
"""

from __future__ import annotations

from src.models import IncidentState


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _has_any(actions: list[str], *prefixes: str) -> bool:
    """Return True if any action starts with one of the given prefixes."""
    return any(a.startswith(p) for a in actions for p in prefixes)


def _has_exact(actions: list[str], key: str) -> bool:
    """Return True if the exact action key appears in the list."""
    return key in actions


def _has_prefix(actions: list[str], prefix: str) -> bool:
    """Return True if any action starts with `prefix`."""
    return any(a.startswith(prefix) for a in actions)


def _count(actions: list[str], prefix: str) -> int:
    """Count how many actions begin with `prefix`."""
    return sum(1 for a in actions if a.startswith(prefix))


def _index_of(actions: list[str], key: str) -> int | None:
    """Return the first index of `key` in actions, or None."""
    for i, a in enumerate(actions):
        if a == key or a.startswith(key):
            return i
    return None


# ──────────────────────────────────────────────────────────────────
# Task 1 — single_service_failure (EASY)
# ──────────────────────────────────────────────────────────────────

def grade_task_1(
    state: IncidentState,
    actions_taken: list[str],
    total_reward: float,
) -> dict:
    """
    Scoring rubric:
      0.0-0.3  Tried some actions but wrong service targeted
      0.3-0.5  Found the right service but didn't roll back
      0.5-0.8  Rolled back but didn't confirm resolution
      0.8-1.0  Full correct diagnosis + fix + declared resolved
    """
    breakdown: dict[str, float] = {}
    score = 0.0

    # --- Did the agent look at deploys? (0.15) ---
    checked_deploys = _has_exact(actions_taken, "check_recent_deploys")
    breakdown["checked_deploys"] = 0.15 if checked_deploys else 0.0
    score += breakdown["checked_deploys"]

    # --- Did the agent read logs from user-service? (0.15) ---
    read_user_logs = _has_exact(actions_taken, "read_logs_user-service")
    breakdown["read_user_service_logs"] = 0.15 if read_user_logs else 0.0
    score += breakdown["read_user_service_logs"]

    # --- Was diagnosis correct? (0.10) ---
    breakdown["correctly_diagnosed"] = 0.10 if state.correctly_diagnosed else 0.0
    score += breakdown["correctly_diagnosed"]

    # --- Did the agent roll back? (0.30) ---
    did_rollback = _has_prefix(actions_taken, "rollback")
    breakdown["rollback_executed"] = 0.30 if did_rollback else 0.0
    score += breakdown["rollback_executed"]

    # --- Did the agent declare resolved (and was actually resolved)? (0.20) ---
    declared = _has_exact(actions_taken, "declare_resolved")
    resolved_correctly = declared and state.resolved
    breakdown["declared_resolved"] = 0.20 if resolved_correctly else 0.0
    score += breakdown["declared_resolved"]

    # --- Efficiency bonus (0.10): solved in ≤ 6 steps ---
    efficient = state.step_count <= 6
    breakdown["efficiency_bonus"] = 0.10 if (efficient and state.resolved) else 0.0
    score += breakdown["efficiency_bonus"]

    score = round(min(score, 1.0), 4)

    return {
        "score": score,
        "max_score": 1.0,
        "breakdown": breakdown,
        "passed": score >= 0.5,
    }


# ──────────────────────────────────────────────────────────────────
# Task 2 — database_latency (MEDIUM)
# ──────────────────────────────────────────────────────────────────

def grade_task_2(
    state: IncidentState,
    actions_taken: list[str],
    total_reward: float,
) -> dict:
    """
    Scoring rubric:
      0.0-0.2  Only checked one service
      0.2-0.5  Checked metrics but missed DB connection
      0.5-0.7  Identified DB but wrong remediation
      0.7-1.0  Correct DB identification + scale_up + resolved
    """
    breakdown: dict[str, float] = {}
    score = 0.0

    # --- Checked api-gateway metrics (0.10) ---
    checked_api = _has_exact(actions_taken, "check_metrics_api-gateway")
    breakdown["checked_api_gateway"] = 0.10 if checked_api else 0.0
    score += breakdown["checked_api_gateway"]

    # --- Checked db-primary metrics (0.15) ---
    checked_db = _has_exact(actions_taken, "check_metrics_db-primary")
    breakdown["checked_db_primary"] = 0.15 if checked_db else 0.0
    score += breakdown["checked_db_primary"]

    # --- Checked DB queries (0.15) ---
    checked_queries = _has_exact(actions_taken, "check_db_queries")
    breakdown["checked_db_queries"] = 0.15 if checked_queries else 0.0
    score += breakdown["checked_db_queries"]

    # --- Correctly diagnosed? (0.10) ---
    breakdown["correctly_diagnosed"] = 0.10 if state.correctly_diagnosed else 0.0
    score += breakdown["correctly_diagnosed"]

    # --- Scaled up db-primary (0.25) ---
    scaled_db = _has_exact(actions_taken, "scale_up_db-primary")
    breakdown["scale_up_db"] = 0.25 if scaled_db else 0.0
    score += breakdown["scale_up_db"]

    # --- Declared resolved correctly (0.15) ---
    declared = _has_exact(actions_taken, "declare_resolved")
    resolved_correctly = declared and state.resolved
    breakdown["declared_resolved"] = 0.15 if resolved_correctly else 0.0
    score += breakdown["declared_resolved"]

    # --- Efficiency bonus (0.10): solved in ≤ 8 steps ---
    efficient = state.step_count <= 8
    breakdown["efficiency_bonus"] = 0.10 if (efficient and state.resolved) else 0.0
    score += breakdown["efficiency_bonus"]

    score = round(min(score, 1.0), 4)

    return {
        "score": score,
        "max_score": 1.0,
        "breakdown": breakdown,
        "passed": score >= 0.6,
    }


# ──────────────────────────────────────────────────────────────────
# Task 3 — cascade_failure (HARD)
# ──────────────────────────────────────────────────────────────────

def grade_task_3(
    state: IncidentState,
    actions_taken: list[str],
    total_reward: float,
) -> dict:
    """
    Scoring rubric:
      0.0-0.2  Restarted wrong services first (made it worse)
      0.2-0.4  Identified some affected services
      0.4-0.6  Identified root cause (DB pool exhaustion)
      0.6-0.8  Correct fix order (DB first, then payment, then API)
      0.8-1.0  Perfect diagnosis + correct order + verified resolution
    Penalty: -0.15 if agent triggered wrong_first_actions before correct diagnosis
    """
    breakdown: dict[str, float] = {}
    score = 0.0

    # --- Checked payment-service metrics (0.06) ---
    breakdown["checked_payment"] = 0.06 if _has_exact(
        actions_taken, "check_metrics_payment-service"
    ) else 0.0
    score += breakdown["checked_payment"]

    # --- Checked api-gateway metrics (0.06) ---
    breakdown["checked_api"] = 0.06 if _has_exact(
        actions_taken, "check_metrics_api-gateway"
    ) else 0.0
    score += breakdown["checked_api"]

    # --- Checked db-primary metrics (0.06) ---
    breakdown["checked_db"] = 0.06 if _has_exact(
        actions_taken, "check_metrics_db-primary"
    ) else 0.0
    score += breakdown["checked_db"]

    # --- Read api-gateway logs (0.06) ---
    breakdown["read_api_logs"] = 0.06 if _has_exact(
        actions_taken, "read_logs_api-gateway"
    ) else 0.0
    score += breakdown["read_api_logs"]

    # --- Checked DB queries (root cause identification) (0.06) ---
    breakdown["checked_db_queries"] = 0.06 if _has_exact(
        actions_taken, "check_db_queries"
    ) else 0.0
    score += breakdown["checked_db_queries"]

    # --- Correctly diagnosed? (0.10) ---
    breakdown["correctly_diagnosed"] = 0.10 if state.correctly_diagnosed else 0.0
    score += breakdown["correctly_diagnosed"]

    # --- Restarted db-primary (0.15) ---
    restart_db = _has_exact(actions_taken, "restart_service_db-primary")
    breakdown["restart_db"] = 0.15 if restart_db else 0.0
    score += breakdown["restart_db"]

    # --- Restarted payment-service (0.10) ---
    restart_payment = _has_exact(actions_taken, "restart_service_payment-service")
    breakdown["restart_payment"] = 0.10 if restart_payment else 0.0
    score += breakdown["restart_payment"]

    # --- Correct fix ORDER: db-primary before payment-service (0.10) ---
    idx_db = _index_of(actions_taken, "restart_service_db-primary")
    idx_pay = _index_of(actions_taken, "restart_service_payment-service")
    correct_order = (
        idx_db is not None
        and idx_pay is not None
        and idx_db < idx_pay
    )
    breakdown["correct_fix_order"] = 0.10 if correct_order else 0.0
    score += breakdown["correct_fix_order"]

    # --- Declared resolved correctly (0.10) ---
    declared = _has_exact(actions_taken, "declare_resolved")
    resolved_correctly = declared and state.resolved
    breakdown["declared_resolved"] = 0.10 if resolved_correctly else 0.0
    score += breakdown["declared_resolved"]

    # --- Efficiency bonus (0.10): solved in ≤ 12 steps ---
    efficient = state.step_count <= 12
    breakdown["efficiency_bonus"] = 0.10 if (efficient and state.resolved) else 0.0
    score += breakdown["efficiency_bonus"]

    # ── Penalty: wrong_first_actions before diagnosis ──
    wrong_first = ["restart_service_api-gateway", "restart_service_payment-service"]
    triggered_wrong_before_diag = False
    for i, action in enumerate(actions_taken):
        if action in wrong_first:
            # Check if diagnosis was NOT complete at that point
            # Heuristic: if correctly_diagnosed is False and the wrong action
            # appeared before all 5 diagnosis actions were taken
            diag_actions_before = sum(
                1 for a in actions_taken[:i]
                if a in [
                    "check_metrics_payment-service",
                    "check_metrics_api-gateway",
                    "check_metrics_db-primary",
                    "read_logs_api-gateway",
                    "check_db_queries",
                ]
            )
            if diag_actions_before < 5:
                triggered_wrong_before_diag = True
                break

    penalty = -0.15 if triggered_wrong_before_diag else 0.0
    breakdown["wrong_first_penalty"] = penalty
    score += penalty

    score = round(max(0.0, min(score, 1.0)), 4)

    return {
        "score": score,
        "max_score": 1.0,
        "breakdown": breakdown,
        "passed": score >= 0.7,
    }


# ──────────────────────────────────────────────────────────────────
# Grader dispatcher
# ──────────────────────────────────────────────────────────────────

GRADER_MAP = {
    "single_service_failure": grade_task_1,
    "database_latency": grade_task_2,
    "cascade_failure": grade_task_3,
}


def grade(
    task_name: str,
    state: IncidentState,
    actions_taken: list[str],
    total_reward: float,
) -> dict:
    """Route to the correct grader and return its result."""
    grader_fn = GRADER_MAP.get(task_name)
    if grader_fn is None:
        return {
            "score": 0.0,
            "max_score": 1.0,
            "breakdown": {},
            "passed": False,
            "error": f"No grader found for task '{task_name}'",
        }
    return grader_fn(state, actions_taken, total_reward)
