from typing import List, Optional
from pydantic import BaseModel

class IncidentScenario(BaseModel):
    name: str
    incident_type: str
    affected_service: str
    root_cause: str
    correct_diagnosis_actions: List[str]
    correct_fix_actions: List[str]
    wrong_first_actions: Optional[List[str]] = None
    description: str
    max_possible_reward: float
    success_threshold: float

TASK_1_EASY = IncidentScenario(
  name="single_service_failure",
  incident_type="bad_deploy",
  affected_service="user-service",
  root_cause="bad_deploy_to_user_service",
  correct_diagnosis_actions=["check_recent_deploys", "read_logs_user-service"],
  correct_fix_actions=["rollback"],
  description="A bad deploy to user-service is causing 500 errors",
  max_possible_reward=1.70,
  success_threshold=0.5
)

TASK_2_MEDIUM = IncidentScenario(
  name="database_latency",
  incident_type="db_overload",
  affected_service="db-primary",
  root_cause="db_primary_overloaded",
  correct_diagnosis_actions=["check_metrics_api-gateway", "check_metrics_db-primary", "check_db_queries"],
  correct_fix_actions=["scale_up_db-primary", "declare_resolved"],
  description="DB overload causing API latency cascade",
  max_possible_reward=1.90,
  success_threshold=0.6
)

TASK_3_HARD = IncidentScenario(
  name="cascade_failure",
  incident_type="cascade_failure",
  affected_service="multiple",
  root_cause="db_connection_pool_exhausted",
  correct_diagnosis_actions=["check_metrics_api-gateway", "check_metrics_payment-service", "check_metrics_db-primary", "read_logs_api-gateway", "check_db_queries"],
  correct_fix_actions=["restart_service_db-primary", "restart_service_payment-service", "declare_resolved"],
  wrong_first_actions=["restart_service_api-gateway", "restart_service_payment-service"],
  description="DB connection pool exhaustion causes retry storm in payment-service. This spikes latency in api-gateway. Misleading logs suggest cache issue. Correct solution requires identifying DB bottleneck, and restarting DB THEN dependent services in correct order.",
  max_possible_reward=2.50,
  success_threshold=0.7
)

TASK_4_OOM = IncidentScenario(
  name="memory_leak_oom",
  incident_type="memory_leak",
  affected_service="user-service",
  root_cause="memory_leak_in_user_service",
  correct_diagnosis_actions=["check_all_services", "check_metrics_user-service", "read_logs_user-service", "check_recent_deploys"],
  correct_fix_actions=["restart_service_user-service"],
  description="user-service has a memory leak causing gradual OOM. Agent must read logs, check metrics, and restart the service.",
  max_possible_reward=1.70,
  success_threshold=0.5
)

if __name__ == "__main__":
    print(TASK_1_EASY.model_dump_json(indent=2))
    print(TASK_2_MEDIUM.model_dump_json(indent=2))
    print(TASK_3_HARD.model_dump_json(indent=2))
    print(TASK_4_OOM.model_dump_json(indent=2))
