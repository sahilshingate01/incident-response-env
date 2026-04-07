import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class FakeLogEngine:
    def __init__(self, incident_type: Optional[str] = "normal"):
        self.incident_type = incident_type
        self.current_time = datetime.utcnow()
        self.step = 0
        random.seed(f"logs-{incident_type}")

    def advance_time(self):
        self.step += 1
        self.current_time += timedelta(seconds=15)

    def get_logs(self, service_name: str, lines: int = 20) -> str:
        logs_output = []
        is_affected = False
        
        if self.incident_type == "cascade_failure":
            if service_name == "db-primary":
                is_affected = True
            elif service_name == "payment-service" and self.step >= 1:
                is_affected = True
            elif service_name == "api-gateway" and self.step >= 2:
                is_affected = True
                
        for _ in range(lines):
            ts = (self.current_time - timedelta(seconds=random.randint(0, 15))).isoformat() + "Z"
            trace_id = f"trace-{random.randint(100000, 999999)}"
            
            if random.random() < 0.4:
                logs_output.append({
                    "timestamp": ts, "level": "INFO", "trace_id": trace_id, "service": service_name,
                    "message": "Routine health check executed successfully.",
                    "noise": True
                })
                continue
                
            if is_affected and random.random() < 0.6:
                if service_name == "db-primary":
                    logs_output.append({
                        "timestamp": ts, "level": "ERROR", "trace_id": trace_id, "service": service_name,
                        "message": "FATAL: Connection pool exhausted. 500/500 active.",
                        "stack_trace": "at com.db.ConnectionManager.getConnection(ConnectionManager.java:120)\n  at com.db.QueryRunner.execute(QueryRunner.java:45)"
                    })
                elif service_name == "payment-service":
                    logs_output.append({
                        "timestamp": ts, "level": "WARN", "trace_id": trace_id, "service": service_name,
                        "message": "Retrying DB transaction due to timeout. Attempt 3/3.",
                        "error_code": "DB_CONN_TIMEOUT"
                    })
                elif service_name == "api-gateway":
                    logs_output.append({
                        "timestamp": ts, "level": "ERROR", "trace_id": trace_id, "service": service_name,
                        "message": "Upstream timeout from payment-service. Falling back to cache... Cache MISS.",
                        "stack_trace": "at api.GatewayFilter.doFilter(GatewayFilter.java:55)\n  at cache.RedisFallback.fetch(RedisFallback.java:23)"
                    })
            else:
                logs_output.append({
                    "timestamp": ts, "level": "INFO", "trace_id": trace_id, "service": service_name,
                    "message": f"Successfully processed request for user-{random.randint(1,500)}"
                })

        logs_output.sort(key=lambda x: x["timestamp"])
        return json.dumps(logs_output, indent=2)

if __name__ == "__main__":
    print("--- Normal Logs ---")
    engine = FakeLogEngine()
    print(engine.get_logs("api-gateway", lines=5))
    
    print("\n--- Cascade Failure Logs (db-primary) ---")
    engine_bad = FakeLogEngine("cascade_failure")
    print(engine_bad.get_logs("db-primary", lines=10))
