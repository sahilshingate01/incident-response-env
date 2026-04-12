import random
from typing import Dict, List, Optional, Any

class FakeMetricsEngine:
    SERVICES = ["api-gateway", "payment-service", "user-service", "db-primary", "cache-redis"]

    def __init__(self, incident_type: Optional[str] = "normal", seed: Optional[int] = None):
        self.incident_type = incident_type
        self.step = 0
        if seed is None:
            seed = random.randint(0, 999999)
        self.seed = seed
        self.rng = random.Random(f"{seed}-metrics-{incident_type}")

    def advance_time(self):
        self.step += 1

    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        metrics = {
            "p50_latency_ms": self.rng.randint(20, 50),
            "p95_latency_ms": self.rng.randint(50, 100),
            "p99_latency_ms": self.rng.randint(100, 200),
            "error_rate_percent": round(self.rng.uniform(0.1, 0.5), 2),
            "cpu_percent": self.rng.randint(20, 40),
            "memory_percent": self.rng.randint(30, 50),
            "request_throughput": self.rng.randint(1000, 5000),
            "oom_kill_count": 0
        }

        if self.incident_type == "db_overload":
            if service_name == "db-primary":
                metrics["cpu_percent"] = self.rng.randint(90, 100)
                metrics["error_rate_percent"] = round(self.rng.uniform(10, 30), 2)
                metrics["p99_latency_ms"] = self.rng.randint(1000, 5000)

        elif self.incident_type == "cascade_failure":
            if service_name == "db-primary":
                metrics["cpu_percent"] = self.rng.randint(95, 100)
                metrics["error_rate_percent"] = round(self.rng.uniform(50, 80), 2)
            elif service_name == "payment-service" and self.step >= 1:
                metrics["p99_latency_ms"] = self.rng.randint(4000, 6000)
                metrics["error_rate_percent"] = round(self.rng.uniform(20, 40), 2)
                metrics["request_throughput"] = self.rng.randint(200, 500)
            elif service_name == "api-gateway" and self.step >= 2:
                metrics["p99_latency_ms"] = self.rng.randint(5000, 7000)
                metrics["error_rate_percent"] = round(self.rng.uniform(15, 30), 2)
            elif service_name == "cache-redis":
                metrics["cpu_percent"] = self.rng.randint(85, 95)
                metrics["p95_latency_ms"] = self.rng.randint(200, 300)

        elif self.incident_type == "memory_leak":
            if service_name == "user-service":
                metrics["memory_percent"] = 92
                metrics["oom_kill_count"] = 7
                metrics["cpu_percent"] = 78
                metrics["error_rate_percent"] = 25.0
                metrics["p99_latency_ms"] = 3200
            elif service_name == "api-gateway":
                metrics["p99_latency_ms"] = self.rng.randint(1000, 2000)

        # Apply random variance on metrics
        err_var = self.rng.uniform(0.9, 1.1)
        lat_var = self.rng.uniform(0.8, 1.2)
        
        metrics["error_rate_percent"] = round(metrics["error_rate_percent"] * err_var, 2)
        metrics["p99_latency_ms"] = int(metrics["p99_latency_ms"] * lat_var)
        
        metrics["latency_p99_ms"] = metrics["p99_latency_ms"]
        metrics["error_rate"] = metrics["error_rate_percent"]
        metrics["requests_per_sec"] = metrics["request_throughput"]

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
