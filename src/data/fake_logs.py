import random
import datetime
from typing import List, Optional

class FakeLogEngine:
    def __init__(self, incident_type: Optional[str] = None):
        self.incident_type = incident_type
        if incident_type:
            random.seed(f"logs-{incident_type}")
        else:
            random.seed("logs-normal")

    def _generate_timestamp(self) -> str:
        now = datetime.datetime.now()
        # Randomize seconds/milliseconds for each log line
        return (now - datetime.timedelta(seconds=random.randint(0, 3600))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def get_logs(self, service_name: str, lines: int = 20) -> str:
        """
        Returns realistic-looking log lines as a string
        """
        log_lines = []
        templates = {
            "INFO": [
                "Handling request GET /api/v1/resource",
                "Processed transaction ID {txid}",
                "User {user_id} logged in from {ip}",
                "Successfully updated cache for service {service}",
                "Response sent: status 200, latency {latency}ms"
            ],
            "WARN": [
                "Slow response from dependency: {service} took {latency}ms",
                "Retrying request to {service} (attempt 2/3)",
                "Connection pool nearing capacity: {pool_utilization}% used",
                "GC overhead increasing, heap usage alert"
            ],
            "ERROR": [
                "Failed to process request: timeout connecting to {service}",
                "Database query failed: Deadlock detected",
                "NullPointerException in {controller}.handleRequest() line {line}",
                "Unexpected 503 from upstream {service}",
                "Connection refused: Could not connect to {service}:{port}"
            ]
        }

        # Select logs based on incident type and service
        is_affected = False
        incident_logs = []

        if self.incident_type == "db_overload" and service_name == "db-primary":
            is_affected = True
            incident_logs = ["ERROR: Connection timeout to db-primary:5432 after 30000ms", "WARN: Postgres connection limit reached (500/500)"]
        elif self.incident_type == "memory_leak" and service_name == "user-service":
            is_affected = True
            incident_logs = ["WARN: Heap usage at 94% (7.8GB/8GB). GC overhead increasing."]
        elif self.incident_type == "bad_deploy" and service_name == "user-service":
            is_affected = True
            incident_logs = ["ERROR: NullPointerException in UserController.getUserById() line 142"]
        elif self.incident_type == "cascade_failure":
            if service_name == "api-gateway":
                is_affected = True
                incident_logs = ["ERROR: Upstream payment-service returned 503 after 3 retries"]
            elif service_name == "db-primary":
                is_affected = True
                incident_logs = ["ERROR: Connection pool exhausted (max_connections=200)"]

        for _ in range(lines):
            ts = self._generate_timestamp()
            if is_affected and random.random() < 0.4: # 40% chance of an error log if affected
                level = "ERROR"
                msg = random.choice(incident_logs) if incident_logs else random.choice(templates["ERROR"])
            else:
                level = random.choices(["INFO", "WARN"], weights=[0.9, 0.1])[0]
                msg = random.choice(templates[level])
            
            # Fill placeholders
            msg = msg.format(
                txid=random.randint(10000, 99999),
                user_id=random.randint(1, 4000),
                ip=f"192.168.1.{random.randint(1, 255)}",
                service=service_name,
                latency=random.randint(50, 5000),
                pool_utilization=random.randint(80, 99),
                controller="UserController",
                line=random.randint(100, 500),
                port=5432
            )
            log_lines.append(f"{ts} {level} - {msg}")

        # Sort logs by timestamp
        log_lines.sort()
        return "\n".join(log_lines)

if __name__ == "__main__":
    print("--- Normal Logs ---")
    engine = FakeLogEngine()
    print(engine.get_logs("api-gateway", lines=5))
    
    print("\n--- DB Overload Logs (db-primary) ---")
    engine_bad = FakeLogEngine("db_overload")
    print(engine_bad.get_logs("db-primary", lines=10))
