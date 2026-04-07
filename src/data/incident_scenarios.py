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

TASK_1_EASY = IncidentScenario(
  name="single_service_failure",
  incident_type="bad_deploy",
  affected_service="user-service",
  root_cause="bad_deploy_to_user_service",
  correct_diagnosis_actions=["check_recent_deploys", "read_logs_user-service"],
  correct_fix_actions=["rollback"],
  description="A bad deploy to user-service is causing 500 errors"
)

TASK_2_MEDIUM = IncidentScenario(
  name="database_latency",
  incident_type="db_overload",
  affected_service="db-primary",
  root_cause="db_primary_overloaded",
  correct_diagnosis_actions=["check_metrics_api-gateway", "check_metrics_db-primary", "check_db_queries"],
  correct_fix_actions=["scale_up_db-primary", "declare_resolved"],
  description="DB overload causing API latency cascade"
)

TASK_3_HARD = IncidentScenario(
  name="cascade_failure",
  incident_type="cascade_failure",
  affected_service="multiple",
  root_cause="db_connection_pool_exhausted",
  correct_diagnosis_actions=["check_metrics_payment-service", "check_metrics_api-gateway", "check_metrics_db-primary", "read_logs_api-gateway", "check_db_queries"],
  correct_fix_actions=["restart_service_db-primary", "restart_service_payment-service", "declare_resolved"],
  wrong_first_actions=["restart_service_api-gateway", "restart_service_payment-service"],
  description="Cascading failure: DB connection pool exhausted -> payment timeout -> API retry storm -> OOM"
)

if __name__ == "__main__":
    print(TASK_1_EASY.model_dump_json(indent=2))
    print(TASK_2_MEDIUM.model_dump_json(indent=2))
    print(TASK_3_HARD.model_dump_json(indent=2))
