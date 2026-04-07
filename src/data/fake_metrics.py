import random
from typing import Dict, List, Optional

class FakeMetricsEngine:
    SERVICES = ["api-gateway", "payment-service", "user-service", "db-primary", "cache-redis"]
    
    def __init__(self, incident_type: Optional[str] = None):
        self.incident_type = incident_type
        # Seed for determinism if incident_type is provided
        if incident_type:
            random.seed(incident_type)
        else:
            random.seed("normal")

    def get_service_metrics(self, service_name: str) -> Dict:
        """
        Returns dict with: error_rate (float 0-100), latency_p99_ms (int), 
        cpu_percent (int), memory_percent (int), requests_per_sec (int)
        """
        # Default Normal Ranges
        error_rate = random.uniform(0, 2)
        latency = random.randint(50, 300)
        cpu = random.randint(20, 60)
        memory = random.randint(30, 70)
        rps = random.randint(100, 500)

        # Apply incident-specific anomalies
        if self.incident_type == "db_overload":
            if service_name == "db-primary":
                error_rate = random.uniform(10, 30)
                latency = random.randint(1000, 5000)
                cpu = random.randint(90, 100)
            elif service_name == "payment-service":
                # Dependent on DB
                error_rate = random.uniform(5, 15)
                latency = random.randint(500, 1500)

        elif self.incident_type == "memory_leak":
            if service_name == "user-service":
                memory = random.randint(95, 100)
                error_rate = random.uniform(5, 20)
                latency = random.randint(400, 1200)

        elif self.incident_type == "cascade_failure":
            if service_name == "db-primary":
                error_rate = random.uniform(20, 50)
                latency = random.randint(2000, 8000)
            elif service_name in ["payment-service", "api-gateway"]:
                error_rate = random.uniform(15, 40)
                latency = random.randint(1000, 4000)

        elif self.incident_type == "bad_deploy":
            if service_name == "user-service":
                error_rate = random.uniform(40, 80)
                latency = random.randint(100, 400) # Fast failures

        elif self.incident_type == "network_partition":
            if service_name == "cache-redis":
                error_rate = 100.0
                latency = 10000 # Timeout
            elif service_name == "api-gateway":
                error_rate = random.uniform(10, 20)

        return {
            "error_rate": round(error_rate, 2),
            "latency_p99_ms": latency,
            "cpu_percent": cpu,
            "memory_percent": memory,
            "requests_per_sec": rps
        }

    def get_all_services_summary(self) -> Dict[str, Dict]:
        """
        Returns dict of {service_name: {status: "healthy"/"degraded"/"critical", error_rate, latency}}
        """
        summary = {}
        for service in self.SERVICES:
            metrics = self.get_service_metrics(service)
            status = "healthy"
            if metrics["error_rate"] > 10 or metrics["latency_p99_ms"] > 1000:
                status = "critical"
            elif metrics["error_rate"] > 2 or metrics["latency_p99_ms"] > 400:
                status = "degraded"
            
            summary[service] = {
                "status": status,
                "error_rate": metrics["error_rate"],
                "latency_p99_ms": metrics["latency_p99_ms"]
            }
        return summary

if __name__ == "__main__":
    print("--- Normal State ---")
    engine = FakeMetricsEngine()
    print(engine.get_all_services_summary())
    
    print("\n--- DB Overload Incident ---")
    engine_db = FakeMetricsEngine("db_overload")
    print(engine_db.get_all_services_summary())
    print("DB Metrics:", engine_db.get_service_metrics("db-primary"))
