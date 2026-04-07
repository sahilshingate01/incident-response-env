"""
IncidentResponseEnv — the core OpenEnv-compliant RL environment.

Wraps FakeMetricsEngine, FakeLogEngine, FakeDeployHistory and the
scenario definitions to provide reset(), step(), state() API.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List
from uuid import uuid4

from src.data.fake_deploys import FakeDeployHistory
from src.data.fake_logs import FakeLogEngine
from src.data.fake_metrics import FakeMetricsEngine
from src.data.incident_scenarios import (
    TASK_1_EASY,
    TASK_2_MEDIUM,
    TASK_3_HARD,
    IncidentScenario,
)
from src.models import (
    VALID_ACTION_TYPES,
    IncidentAction,
    IncidentObservation,
    IncidentReward,
    IncidentState,
)

# ──────────────────────────────────────────────
# Scenario registry
# ──────────────────────────────────────────────

SCENARIO_MAP: Dict[str, IncidentScenario] = {
    "single_service_failure": TASK_1_EASY,
    "database_latency": TASK_2_MEDIUM,
    "cascade_failure": TASK_3_HARD,
}

ALERT_MESSAGES: Dict[str, str] = {
    "single_service_failure": (
        "🚨 ALERT: Error rate spike detected on user-service. "
        "Users reporting login failures. P1 incident declared."
    ),
    "database_latency": (
        "🚨 ALERT: High latency detected across API endpoints. "
        "Database queries timing out. P1 incident declared."
    ),
    "cascade_failure": (
        "🚨 ALERT: Error rate spike detected on multiple services. "
        "Users reporting checkout failures. P1 incident declared."
    ),
}

AVAILABLE_ACTIONS_DISPLAY: List[str] = [
    'read_logs(target=<service_name>)',
    'check_metrics(target=<service_name>)',
    'check_all_services()',
    'check_recent_deploys()',
    'check_db_queries()',
    'rollback(target=<deploy_id>)',
    'restart_service(target=<service_name>)',
    'scale_up(target=<service_name>)',
    'declare_resolved()',
]


# ──────────────────────────────────────────────
# Slow-query templates
# ──────────────────────────────────────────────

SLOW_QUERY_TEMPLATES: Dict[str, str] = {
    "db_overload": (
        "=== Slow Query Log (last 30 min) ===\n"
        "1. [45200ms] SELECT * FROM users JOIN sessions ON users.id = sessions.user_id "
        "WHERE sessions.active = true; -- called 1200 times\n"
        "2. [32100ms] SELECT o.*, p.* FROM orders o JOIN products p ON o.product_id = p.id "
        "WHERE o.status = 'pending'; -- called 800 times\n"
        "3. [28700ms] UPDATE inventory SET stock = stock - 1 WHERE product_id IN "
        "(SELECT product_id FROM cart WHERE user_id = ?); -- called 600 times\n"
        "4. [18900ms] SELECT COUNT(*) FROM analytics_events WHERE created_at > NOW() - INTERVAL '1 hour'; "
        "-- full table scan, no index\n"
        "⚠️  Active connections: 498/500 | Lock wait timeouts: 47 in last 5 min"
    ),
    "cascade_failure": (
        "=== Slow Query Log (last 30 min) ===\n"
        "1. [timeout] SELECT balance FROM wallets WHERE user_id = ? FOR UPDATE; "
        "-- connection pool exhausted, 312 waiting threads\n"
        "2. [82000ms] INSERT INTO payment_transactions (...) VALUES (...); "
        "-- deadlock retry x3\n"
        "3. [timeout] SELECT * FROM products WHERE id IN (...); "
        "-- blocked by lock on payment_transactions\n"
        "⚠️  Connection pool: 200/200 EXHAUSTED | Deadlocks: 128 in last 10 min\n"
        "⚠️  Oldest waiting connection: 94s | Thread stack depth: CRITICAL"
    ),
    "bad_deploy": (
        "=== Slow Query Log (last 30 min) ===\n"
        "1. [320ms] SELECT * FROM users WHERE id = ?; -- normal\n"
        "2. [180ms] SELECT * FROM sessions WHERE user_id = ?; -- normal\n"
        "No anomalies detected. DB performance is within normal parameters."
    ),
}

DEFAULT_SLOW_QUERY = (
    "=== Slow Query Log (last 30 min) ===\n"
    "No anomalies detected. DB performance is within normal parameters."
)


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class IncidentResponseEnv:
    """
    OpenEnv-compliant Incident Response environment.

    Lifecycle:
        env = IncidentResponseEnv("single_service_failure")
        obs = env.reset()
        while not obs.done:
            action = agent.decide(obs)
            obs = env.step(action)
        print(env.state())
    """

    def __init__(self, task_name: str) -> None:
        if task_name not in SCENARIO_MAP:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {list(SCENARIO_MAP.keys())}"
            )

        self.task_name: str = task_name
        self.scenario: IncidentScenario = SCENARIO_MAP[task_name]

        # Data engines
        self.metrics_engine = FakeMetricsEngine(self.scenario.incident_type)
        self.log_engine = FakeLogEngine(self.scenario.incident_type)
        self.deploy_history = FakeDeployHistory(self.scenario.incident_type)

        # Episode state (initialised properly in reset())
        self.episode_id: str = str(uuid4())
        self.step_count: int = 0
        self.actions_taken: List[str] = []
        self.correctly_diagnosed: bool = False
        self.resolved: bool = False
        self.done: bool = False
        self.cumulative_reward: float = 0.0

        # Track which correct diagnosis / fix actions have been seen
        self._diagnosis_hits: set[str] = set()
        self._fix_hits: set[str] = set()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def reset(self) -> IncidentObservation:
        """Reset all state and return the initial observation."""
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.actions_taken = []
        self.correctly_diagnosed = False
        self.resolved = False
        self.done = False
        self.cumulative_reward = 0.0
        self._diagnosis_hits = set()
        self._fix_hits = set()

        # Re-seed data engines for determinism
        self.metrics_engine = FakeMetricsEngine(self.scenario.incident_type)
        self.log_engine = FakeLogEngine(self.scenario.incident_type)
        self.deploy_history = FakeDeployHistory(self.scenario.incident_type)

        services_summary = self.metrics_engine.get_all_services_summary()

        return IncidentObservation(
            done=False,
            reward=0.0,
            observation_text=ALERT_MESSAGES[self.task_name],
            available_actions=AVAILABLE_ACTIONS_DISPLAY,
            services_summary=services_summary,
            step_count=0,
            incident_description=self.scenario.description,
            metadata={"episode_id": self.episode_id},
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        """Execute one action and return the resulting observation."""
        if self.done:
            return IncidentObservation(
                done=True,
                reward=0.0,
                observation_text="Episode already finished. Call reset() to start a new episode.",
                available_actions=[],
                services_summary=self.metrics_engine.get_all_services_summary(),
                step_count=self.step_count,
                incident_description=self.scenario.description,
                metadata={"episode_id": self.episode_id},
            )

        self.step_count += 1
        action_key = action.action_key()
        self.actions_taken.append(action_key)

        # Execute the action to get observation text
        observation_text = self._execute_action(action)

        # Calculate reward (may mutate correctly_diagnosed / resolved)
        reward_info = self._calculate_reward(action)
        self.cumulative_reward += reward_info.value

        # Check terminal conditions
        if self.resolved and action.action_type == "declare_resolved":
            self.done = True
        elif action.action_type == "declare_resolved" and not self.resolved:
            # Premature declaration — not terminal, agent can keep trying
            pass

        services_summary = self.metrics_engine.get_all_services_summary()

        return IncidentObservation(
            done=self.done,
            reward=reward_info.value,
            observation_text=observation_text,
            available_actions=AVAILABLE_ACTIONS_DISPLAY if not self.done else [],
            services_summary=services_summary,
            step_count=self.step_count,
            incident_description=self.scenario.description,
            metadata={
                "episode_id": self.episode_id,
                "reward_reason": reward_info.reason,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "correctly_diagnosed": self.correctly_diagnosed,
                "resolved": self.resolved,
            },
        )

    def state(self) -> IncidentState:
        """Return the current internal state snapshot."""
        return IncidentState(
            episode_id=self.episode_id,
            task_name=self.task_name,
            step_count=self.step_count,
            incident_type=self.scenario.incident_type,
            correctly_diagnosed=self.correctly_diagnosed,
            resolved=self.resolved,
            actions_taken=self.actions_taken,
            current_scenario=self.scenario.name,
        )

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _match_scenario_action(self, action_key: str, scenario_actions: List[str]) -> str | None:
        """
        Match an action_key against a list of scenario action strings.

        Scenario actions may be:
        - Exact: "check_recent_deploys", "read_logs_user-service"
        - Type-only: "rollback" (matches "rollback_dep-evil-123")
        - With target: "scale_up_db-primary" (exact match)

        Returns the matched scenario action string, or None.
        """
        # 1) Exact match
        if action_key in scenario_actions:
            return action_key

        # 2) Check if any scenario action is just the action_type (no target),
        #    and our action_key starts with it.  E.g. scenario has "rollback",
        #    action_key is "rollback_dep-evil-123".
        for sa in scenario_actions:
            # Only match if the scenario action has no underscore-separated
            # target (i.e. it IS a bare action_type like "rollback" or "declare_resolved")
            if "_" not in sa and action_key.startswith(sa + "_"):
                return sa
            if sa == action_key.split("_", 1)[0] and "_" not in sa:
                return sa

        return None

    def _calculate_reward(self, action: IncidentAction) -> IncidentReward:
        """
        Deterministic reward function.  Rules are evaluated in priority order;
        first match wins.
        """
        action_key = action.action_key()

        # ── Rule 1 & 2: declare_resolved ──
        # Special handling: if declare_resolved is also a correct_fix_action,
        # we first register it as a fix hit (potentially setting resolved=True),
        # then apply the standard Rule 1/2 check.
        if action.action_type == "declare_resolved":
            fix_match = self._match_scenario_action(
                action_key, self.scenario.correct_fix_actions
            )
            if fix_match and fix_match not in self._fix_hits:
                self._fix_hits.add(fix_match)
                # Check if all fix actions are now complete
                if self._fix_hits >= set(self.scenario.correct_fix_actions):
                    self.resolved = True

            if self.resolved:
                return IncidentReward(value=1.0, reason="Incident fully resolved")
            else:
                return IncidentReward(
                    value=-0.2,
                    reason="Declared resolved prematurely — incident still active",
                )

        # ── Rule 7 (checked early): repeated action > 2 times ──
        action_count = Counter(self.actions_taken)[action_key]
        if action_count > 2:
            return IncidentReward(value=-0.1, reason="Repeated action — not useful")

        # ── Rule 6: wrong_first_actions (Task 3 only) ──
        if (
            self.scenario.wrong_first_actions
            and action_key in self.scenario.wrong_first_actions
            and not self.correctly_diagnosed
        ):
            return IncidentReward(
                value=-0.1,
                reason="Incorrect remediation makes cascade worse",
            )

        # ── Rule 3: correct diagnosis action ──
        diag_match = self._match_scenario_action(
            action_key, self.scenario.correct_diagnosis_actions
        )
        if diag_match and diag_match not in self._diagnosis_hits:
            self._diagnosis_hits.add(diag_match)
            # Check if all diagnosis actions are now complete
            if self._diagnosis_hits >= set(self.scenario.correct_diagnosis_actions):
                self.correctly_diagnosed = True
            return IncidentReward(value=0.2, reason="Good diagnostic step")

        # ── Rule 4 & 5: correct fix action ──
        fix_match = self._match_scenario_action(
            action_key, self.scenario.correct_fix_actions
        )
        if fix_match:
            if fix_match not in self._fix_hits:
                self._fix_hits.add(fix_match)

                if self.correctly_diagnosed:
                    # Rule 4: proper fix after diagnosis
                    if self._fix_hits >= set(self.scenario.correct_fix_actions):
                        self.resolved = True
                    return IncidentReward(
                        value=0.3, reason="Correct remediation"
                    )
                else:
                    # Rule 5: lucky fix without diagnosis
                    if self._fix_hits >= set(self.scenario.correct_fix_actions):
                        self.resolved = True
                    return IncidentReward(
                        value=0.1,
                        reason="Lucky fix — correct action but not properly diagnosed first",
                    )
            # Already applied this fix — treat as neutral
            pass

        # ── Rule 8: any other valid action ──
        return IncidentReward(value=0.0, reason="Action taken but not informative")

    def _execute_action(self, action: IncidentAction) -> str:
        """
        Simulate the action and return a human-readable observation string.
        """
        action_type = action.action_type
        target = action.target or ""

        if action_type == "read_logs":
            if not target:
                return "Error: read_logs requires a target service name."
            return self.log_engine.get_logs(target, lines=15)

        if action_type == "check_metrics":
            if not target:
                return "Error: check_metrics requires a target service name."
            metrics = self.metrics_engine.get_service_metrics(target)
            lines = [f"=== Metrics for {target} ==="]
            lines.append(f"  Error Rate:     {metrics['error_rate']}%")
            lines.append(f"  Latency (p99):  {metrics['latency_p99_ms']}ms")
            lines.append(f"  CPU Usage:      {metrics['cpu_percent']}%")
            lines.append(f"  Memory Usage:   {metrics['memory_percent']}%")
            lines.append(f"  Requests/sec:   {metrics['requests_per_sec']}")
            return "\n".join(lines)

        if action_type == "check_all_services":
            summary = self.metrics_engine.get_all_services_summary()
            lines = ["=== All Services Status ==="]
            for svc, info in summary.items():
                status_icon = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "critical": "🔴",
                }.get(info["status"], "❓")
                lines.append(
                    f"  {status_icon} {svc:25s}  status={info['status']:10s}  "
                    f"error_rate={info['error_rate']:.2f}%  "
                    f"latency={info['latency_p99_ms']}ms"
                )
            return "\n".join(lines)

        if action_type == "check_recent_deploys":
            deploys = self.deploy_history.get_recent_deploys()
            if not deploys:
                return "No recent deploys found in the last 24 hours."
            lines = ["=== Recent Deploys (last 24h) ==="]
            for d in deploys:
                lines.append(
                    f"  [{d['timestamp']}] {d['id']}  {d['service']} -> "
                    f"{d['version']}  by {d['deployed_by']}  "
                    f"status={d['status']}  \"{d['commit_message']}\""
                )
            return "\n".join(lines)

        if action_type == "check_db_queries":
            return SLOW_QUERY_TEMPLATES.get(
                self.scenario.incident_type, DEFAULT_SLOW_QUERY
            )

        if action_type == "rollback":
            if not target:
                return "Error: rollback requires a target deploy_id."
            return (
                f"Rollback of deploy {target} initiated. "
                "Monitoring error rates..."
            )

        if action_type == "restart_service":
            if not target:
                return "Error: restart_service requires a target service name."
            return f"Restarting {target}... Done. Service healthy."

        if action_type == "scale_up":
            if not target:
                return "Error: scale_up requires a target service name."
            return (
                f"Scaling up {target} from 3 to 6 replicas. "
                "Load distributing..."
            )

        if action_type == "declare_resolved":
            if self.resolved:
                return "Incident status updated to RESOLVED."
            return (
                "⚠️ Cannot mark as resolved — monitoring still shows active issues. "
                "Continue investigating."
            )

        return f"Unknown action: {action_type}"


# ──────────────────────────────────────────────────────────────────
# Comprehensive test harness
# ──────────────────────────────────────────────────────────────────

def _run_test_episode(
    task_name: str,
    actions: List[IncidentAction],
    expected_rewards: List[float],
) -> bool:
    """
    Run a full episode and verify rewards match expectations.
    Returns True if all rewards match.
    """
    env = IncidentResponseEnv(task_name)
    obs = env.reset()

    print(f"\n{'='*70}")
    print(f"  TASK: {task_name}")
    print(f"  Scenario: {env.scenario.description}")
    print(f"  Initial alert: {obs.observation_text[:80]}...")
    print(f"{'='*70}")

    all_pass = True
    total_reward = 0.0

    for i, (action, expected_r) in enumerate(zip(actions, expected_rewards)):
        obs = env.step(action)
        total_reward += obs.reward
        match = abs(obs.reward - expected_r) < 1e-6
        status = "✅" if match else "❌"

        if not match:
            all_pass = False

        print(
            f"  Step {i+1}: {action.action_key():40s}  "
            f"reward={obs.reward:+.1f}  expected={expected_r:+.1f}  {status}  "
            f"| {obs.metadata.get('reward_reason', '')}"
        )

    final_state = env.state()
    print(f"\n  Final State:")
    print(f"    diagnosed={final_state.correctly_diagnosed}  "
          f"resolved={final_state.resolved}  "
          f"done={obs.done}  "
          f"total_reward={total_reward:+.2f}  "
          f"steps={final_state.step_count}")
    print(f"  {'PASS ✅' if all_pass else 'FAIL ❌'}")
    return all_pass


def _make_action(action_type: str, target: str | None, task_name: str) -> IncidentAction:
    return IncidentAction(action_type=action_type, target=target, task_name=task_name)


if __name__ == "__main__":
    results = []

    # ──────────────────────────────────────────
    # Task 1: single_service_failure (EASY)
    # correct_diagnosis: ["check_recent_deploys", "read_logs_user-service"]
    # correct_fix: ["rollback"]
    # ──────────────────────────────────────────
    task1_actions = [
        _make_action("check_all_services", None, "single_service_failure"),        # neutral (0.0)
        _make_action("check_recent_deploys", None, "single_service_failure"),      # diagnosis 1/2 (+0.2)
        _make_action("read_logs", "user-service", "single_service_failure"),       # diagnosis 2/2 (+0.2) → diagnosed
        _make_action("rollback", "dep-evil-123", "single_service_failure"),        # correct fix (+0.3) → resolved
        _make_action("declare_resolved", None, "single_service_failure"),          # resolved (+1.0) → done
    ]
    task1_expected = [0.0, 0.2, 0.2, 0.3, 1.0]
    results.append(_run_test_episode("single_service_failure", task1_actions, task1_expected))

    # ──────────────────────────────────────────
    # Task 2: database_latency (MEDIUM)
    # correct_diagnosis: ["check_metrics_api-gateway", "check_metrics_db-primary", "check_db_queries"]
    # correct_fix: ["scale_up_db-primary", "declare_resolved"]
    # ──────────────────────────────────────────
    task2_actions = [
        _make_action("check_metrics", "api-gateway", "database_latency"),          # diagnosis 1/3 (+0.2)
        _make_action("check_metrics", "payment-service", "database_latency"),      # neutral (0.0)
        _make_action("check_metrics", "db-primary", "database_latency"),           # diagnosis 2/3 (+0.2)
        _make_action("check_db_queries", None, "database_latency"),               # diagnosis 3/3 (+0.2) → diagnosed
        _make_action("scale_up", "db-primary", "database_latency"),               # correct fix 1/2 (+0.3)
        _make_action("declare_resolved", None, "database_latency"),               # fix 2/2 → resolved (+1.0) → done
    ]
    task2_expected = [0.2, 0.0, 0.2, 0.2, 0.3, 1.0]
    results.append(_run_test_episode("database_latency", task2_actions, task2_expected))

    # ──────────────────────────────────────────
    # Task 3: cascade_failure (HARD)
    # correct_diagnosis: ["check_metrics_payment-service", "check_metrics_api-gateway",
    #                      "check_metrics_db-primary", "read_logs_api-gateway", "check_db_queries"]
    # correct_fix: ["restart_service_db-primary", "restart_service_payment-service", "declare_resolved"]
    # wrong_first: ["restart_service_api-gateway", "restart_service_payment-service"]
    # ──────────────────────────────────────────
    task3_actions = [
        _make_action("check_all_services", None, "cascade_failure"),               # neutral (0.0)
        _make_action("restart_service", "api-gateway", "cascade_failure"),          # wrong first (-0.1)
        _make_action("check_metrics", "payment-service", "cascade_failure"),        # diagnosis 1/5 (+0.2)
        _make_action("check_metrics", "api-gateway", "cascade_failure"),            # diagnosis 2/5 (+0.2)
        _make_action("check_metrics", "db-primary", "cascade_failure"),             # diagnosis 3/5 (+0.2)
        _make_action("read_logs", "api-gateway", "cascade_failure"),               # diagnosis 4/5 (+0.2)
        _make_action("check_db_queries", None, "cascade_failure"),                 # diagnosis 5/5 (+0.2) → diagnosed
        _make_action("restart_service", "db-primary", "cascade_failure"),           # correct fix 1/3 (+0.3)
        _make_action("restart_service", "payment-service", "cascade_failure"),      # correct fix 2/3 (+0.3)
        _make_action("declare_resolved", None, "cascade_failure"),                 # correct fix 3/3 → resolved (+1.0)
    ]
    task3_expected = [0.0, -0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 1.0]
    results.append(_run_test_episode("cascade_failure", task3_actions, task3_expected))

    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    task_names = ["single_service_failure", "database_latency", "cascade_failure"]
    for name, passed in zip(task_names, results):
        print(f"  {name:30s}  {'PASS ✅' if passed else 'FAIL ❌'}")
    print(f"\n  Overall: {'ALL PASS ✅' if all(results) else 'SOME FAILED ❌'}")
