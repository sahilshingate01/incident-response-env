import random
from typing import Dict, List, Optional, Any

class FakeMetricsEngine:
    SERVICES = ["api-gateway", "payment-service", "user-service", "db-primary", "cache-redis"]

    def __init__(self, incident_type: Optional[str] = "normal"):
        self.incident_type = incident_type
        self.step = 0
        random.seed(f"metrics-{incident_type}")

    def advance_time(self):
        self.step += 1

    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        metrics = {
            "p50_latency_ms": random.randint(20, 50),
            "p95_latency_ms": random.randint(50, 100),
            "p99_latency_ms": random.randint(100, 200),
            "error_rate_percent": round(random.uniform(0.1, 0.5), 2),
            "cpu_percent": random.randint(20, 40),
            "memory_percent": random.randint(30, 50),
            "request_throughput": random.randint(1000, 5000),
            "error_rate": round(random.uniform(0.1, 0.5), 2) # legacy support
        }

        if self.incident_type == "db_overload":
            if service_name == "db-primary":
                metrics["cpu_percent"] = random.randint(90, 100)
                metrics["error_rate_percent"] = round(random.uniform(10, 30), 2)
                metrics["error_rate"] = metrics["error_rate_percent"]
                metrics["p99_latency_ms"] = random.randint(1000, 5000)

        elif self.incident_type == "cascade_failure":
            if service_name == "db-primary":
                metrics["cpu_percent"] = random.randint(95, 100)
                metrics["error_rate_percent"] = round(random.uniform(50, 80), 2)
            elif service_name == "payment-service" and self.step >= 1:
                metrics["p99_latency_ms"] = random.randint(4000, 6000)
                metrics["error_rate_percent"] = round(random.uniform(20, 40), 2)
                metrics["request_throughput"] = random.randint(200, 500)
            elif service_name == "api-gateway" and self.step >= 2:
                metrics["p99_latency_ms"] = random.randint(5000, 7000)
                metrics["error_rate_percent"] = round(random.uniform(15, 30), 2)
            elif service_name == "cache-redis":
                metrics["cpu_percent"] = random.randint(85, 95)
                metrics["p95_latency_ms"] = random.randint(200, 300)
            
            metrics["error_rate"] = metrics["error_rate_percent"]

        return metrics

    def get_all_services_summary(self) -> Dict[str, Dict]:
        summary = {}
        for service in self.SERVICES:
            metrics = self.get_service_metrics(service)
            status = "healthy"
            if metrics["error_rate_percent"] > 10 or metrics["p99_latency_ms"] > 1000:
                status = "critical"
            elif metrics["error_rate_percent"] > 2 or metrics["p99_latency_ms"] > 400:
                status = "degraded"
            
            summary[service] = {
                "status": status,
                "error_rate": metrics["error_rate_percent"],
                "latency_p99_ms": metrics["p99_latency_ms"]
            }
        return summary
