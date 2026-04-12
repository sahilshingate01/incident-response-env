from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional
from uuid import uuid4

from data.fake_deploys import FakeDeployHistory
from data.fake_logs import FakeLogEngine
from data.fake_metrics import FakeMetricsEngine
from data.incident_scenarios import (
    TASK_1_EASY,
    TASK_2_MEDIUM,
    TASK_3_HARD,
    TASK_4_OOM,
    IncidentScenario,
)
from models import (
    VALID_ACTION_TYPES,
    IncidentAction,
    IncidentObservation,
    IncidentReward,
    IncidentState,
)

SCENARIO_MAP: Dict[str, IncidentScenario] = {
    "single_service_failure": TASK_1_EASY,
    "database_latency": TASK_2_MEDIUM,
    "cascade_failure": TASK_3_HARD,
    "memory_leak_oom": TASK_4_OOM
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
    "memory_leak_oom": (
        "🚨 ALERT: user-service restarting frequently due to OOM kills. "
        "Memory consumption climbing. P1 incident declared."
    )
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
    )
}

DEFAULT_SLOW_QUERY = (
    "=== Slow Query Log (last 30 min) ===\n"
    "No anomalies detected. DB performance is within normal parameters."
)

class IncidentResponseEnv:
    def __init__(self, task_name: str) -> None:
        if task_name not in SCENARIO_MAP:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {list(SCENARIO_MAP.keys())}"
            )

        self.task_name: str = task_name
        self.scenario: IncidentScenario = SCENARIO_MAP[task_name]
        self.seed: int = random.randint(0, 999999)

        self.metrics_engine = FakeMetricsEngine(self.scenario.incident_type, self.seed)
        self.log_engine = FakeLogEngine(self.scenario.incident_type, self.seed)
        self.deploy_history = FakeDeployHistory(self.scenario.incident_type, self.seed)

        self.episode_id: str = str(uuid4())
        self.step_count: int = 0
        self.actions_taken: List[str] = []
        self.correctly_diagnosed: bool = False
        self.resolved: bool = False
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        
        self.internal_flags = {
            "checked_metrics": False,
            "checked_logs": False,
            "identified_root_cause": False
        }

        self._diagnosis_hits: set[str] = set()
        self._fix_hits: set[str] = set()

    def reset(self, seed: Optional[int] = None) -> IncidentObservation:
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.actions_taken = []
        self.correctly_diagnosed = False
        self.resolved = False
        self.done = False
        self.cumulative_reward = 0.0
        self.internal_flags = {
            "checked_metrics": False,
            "checked_logs": False,
            "identified_root_cause": False
        }
        self._diagnosis_hits = set()
        self._fix_hits = set()

        if seed is None:
            seed = random.randint(0, 999999)
        self.seed = seed

        self.metrics_engine = FakeMetricsEngine(self.scenario.incident_type, self.seed)
        self.log_engine = FakeLogEngine(self.scenario.incident_type, self.seed)
        self.deploy_history = FakeDeployHistory(self.scenario.incident_type, self.seed)

        services_summary = self.metrics_engine.get_all_services_summary()

        return IncidentObservation(
            done=False,
            reward=0.0,
            observation_text=ALERT_MESSAGES[self.task_name],
            available_actions=AVAILABLE_ACTIONS_DISPLAY,
            services_summary=services_summary,
            step_count=0,
            incident_description=self.scenario.description,
            metadata={"episode_id": self.episode_id, "seed": self.seed},
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        if self.done:
            return IncidentObservation(
                done=True,
                reward=0.0,
                observation_text="Episode already finished.",
                available_actions=[],
                services_summary=self.metrics_engine.get_all_services_summary(),
                step_count=self.step_count,
                incident_description=self.scenario.description,
                metadata={"episode_id": self.episode_id, "seed": self.seed},
            )

        self.step_count += 1
        self.metrics_engine.advance_time()
        self.log_engine.advance_time()
        
        action_key = action.action_key()
        self.actions_taken.append(action_key)

        observation_text = self._execute_action(action)
        reward_info = self._calculate_reward(action)
        self.cumulative_reward += reward_info.value

        if self.resolved and action.action_type == "declare_resolved":
            self.done = True

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
                "seed": self.seed,
                "reward_reason": reward_info.reason,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "correctly_diagnosed": self.correctly_diagnosed,
                "resolved": self.resolved,
            },
        )

    def state(self) -> IncidentState:
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

    def _match_scenario_action(self, action_key: str, scenario_actions: List[str]) -> str | None:
        if action_key in scenario_actions:
            return action_key
        for sa in scenario_actions:
            if "_" not in sa and action_key.startswith(sa + "_"):
                return sa
            if sa == action_key.split("_", 1)[0] and "_" not in sa:
                return sa
        return None

    def _calculate_reward(self, action: IncidentAction) -> IncidentReward:
        action_key = action.action_key()
        action_type = action.action_type
        
        reward = 0.0
        reason = "Efficiency penalty"

        if self.actions_taken.count(action_key) > 1:
            reward -= 0.05
            reason = "Repeated action penalty"
            return IncidentReward(value=round(reward, 4), reason=reason)

        diag_match = self._match_scenario_action(action_key, self.scenario.correct_diagnosis_actions)
        if diag_match:
            if diag_match not in self._diagnosis_hits:
                self._diagnosis_hits.add(diag_match)
                reward += 0.2
                reason = "Correct diagnostic step"
                if self._diagnosis_hits >= set(self.scenario.correct_diagnosis_actions):
                    self.correctly_diagnosed = True
                    self.internal_flags["identified_root_cause"] = True
            else:
                reward += 0.05
                reason = "Already performed this diagnostic step"

        fix_match = self._match_scenario_action(action_key, self.scenario.correct_fix_actions)
        if fix_match:
            if fix_match not in self._fix_hits:
                self._fix_hits.add(fix_match)
                if self.correctly_diagnosed:
                    reward += 0.3
                    reason = "Correct fix applied after proper diagnosis"
                else:
                    reward += 0.2
                    reason = "Correct fix applied without full diagnosis"
            else:
                reward += 0.0
                reason = "Fix already applied"
        
        if action_type == "declare_resolved":
            all_fixes_done = self._fix_hits >= set(self.scenario.correct_fix_actions)
            if all_fixes_done:
                self.resolved = True
                reward += 1.0
                reason = "Incident successfully resolved"
            else:
                reward -= 0.2
                reason = "Declared resolved prematurely"

        if not self.correctly_diagnosed and self.scenario.wrong_first_actions:
            if self._match_scenario_action(action_key, self.scenario.wrong_first_actions):
                reward -= 0.15
                reason = "Penalty for incorrect initial action"

        final_reward = max(-1.0, min(1.0, reward))
        return IncidentReward(value=round(final_reward, 4), reason=reason)

    def _execute_action(self, action: IncidentAction) -> str:
        action_type = action.action_type
        target = action.target or ""

        if action_type == "read_logs":
            if not target: return "Error: read_logs requires a target."
            return self.log_engine.get_logs(target, lines=15)

        if action_type == "check_metrics":
            if not target: return "Error: check_metrics requires a target."
            metrics = self.metrics_engine.get_service_metrics(target)
            lines = [f"=== Metrics for {target} ==="]
            lines.append(f"  Error Rate:     {metrics['error_rate']}%")
            lines.append(f"  Latency (p99):  {metrics['latency_p99_ms']}ms")
            lines.append(f"  CPU Usage:      {metrics['cpu_percent']}%")
            lines.append(f"  Memory Usage:   {metrics.get('memory_percent', 0)}%")
            if "oom_kill_count" in metrics and metrics["oom_kill_count"] > 0:
                lines.append(f"  OOM Kills:      {metrics['oom_kill_count']}")
            lines.append(f"  Requests/sec:   {metrics['requests_per_sec']}")
            return "\n".join(lines)

        if action_type == "check_all_services":
            summary = self.metrics_engine.get_all_services_summary()
            lines = ["=== All Services Status ==="]
            for svc, info in summary.items():
                status_icon = {"healthy": "✅", "degraded": "⚠️", "critical": "🔴"}.get(info["status"], "❓")
                lines.append(f"  {status_icon} {svc:25s}  status={info['status']:10s}  error_rate={info['error_rate']:.2f}%  latency={info['latency_p99_ms']}ms")
            return "\n".join(lines)

        if action_type == "check_recent_deploys":
            deploys = self.deploy_history.get_recent_deploys()
            if not deploys: return "No recent deploys found in the last 24 hours."
            lines = ["=== Recent Deploys (last 24h) ==="]
            for d in deploys:
                lines.append(f"  [{d['timestamp']}] {d['id']}  {d['service']} -> {d['version']}  by {d['deployed_by']}  status={d['status']}  \"{d['commit_message']}\"")
            return "\n".join(lines)

        if action_type == "check_db_queries":
            return SLOW_QUERY_TEMPLATES.get(self.scenario.incident_type, DEFAULT_SLOW_QUERY)

        if action_type == "rollback":
            if not target: return "Error: rollback requires a target deploy_id."
            return f"Rollback of deploy {target} initiated. Monitoring error rates..."

        if action_type == "restart_service":
            if not target: return "Error: restart_service requires a target service name."
            return f"Restarting {target}... Done. Service healthy."

        if action_type == "scale_up":
            if not target: return "Error: scale_up requires a target service name."
            return f"Scaling up {target} from 3 to 6 replicas. Load distributing..."

        if action_type == "declare_resolved":
            if self.resolved: return "Incident status updated to RESOLVED."
            return "⚠️ Cannot mark as resolved — monitoring still shows active issues. Continue investigating."

        return f"Unknown action: {action_type}"
