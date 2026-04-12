import random
import datetime
from typing import List, Dict, Optional

class FakeDeployHistory:
    def __init__(self, incident_type: Optional[str] = None, seed: Optional[int] = None):
        self.incident_type = incident_type
        if seed is None:
            seed = random.randint(0, 999999)
        self.seed = seed
        self.rng = random.Random(f"{seed}-deploys-{incident_type if incident_type else 'normal'}")

    def get_recent_deploys(self, hours: int = 24) -> List[Dict]:
        deploys = []
        services = ["api-gateway", "payment-service", "user-service", "db-primary", "cache-redis"]
        engineers = ["sahil", "alice", "bob", "carol", "dave"]
        commit_messages = [
            "feat: add login session tracking",
            "fix: correct typo in logging statement",
            "chore: update dependencies",
            "refactor: simplify routing logic",
            "docs: update API documentation"
        ]

        for i in range(3, 6):
            service = self.rng.choice(services)
            ts = (datetime.datetime.now() - datetime.timedelta(hours=self.rng.randint(1, hours))).isoformat()
            deploys.append({
                "id": f"dep-{self.rng.randint(1000, 9999)}",
                "service": service,
                "version": f"v1.2.{self.rng.randint(0, 50)}",
                "timestamp": ts,
                "deployed_by": self.rng.choice(engineers),
                "status": "success",
                "commit_message": self.rng.choice(commit_messages)
            })

        if self.incident_type == "bad_deploy":
            ts = (datetime.datetime.now() - datetime.timedelta(minutes=self.rng.randint(20, 30))).isoformat()
            deploys.append({
                "id": f"dep-evil-{self.rng.randint(100,999)}",
                "service": "user-service",
                "version": "v1.3.0",
                "timestamp": ts,
                "deployed_by": self.rng.choice(engineers),
                "status": "success",
                "commit_message": "feat: new user profile caching logic"
            })
            
        if self.incident_type == "memory_leak":
            ts = (datetime.datetime.now() - datetime.timedelta(minutes=self.rng.randint(160, 200))).isoformat()
            deploys.append({
                "id": f"dep-mem-{self.rng.randint(100,999)}",
                "service": "user-service",
                "version": "v2.1.0",
                "timestamp": ts,
                "deployed_by": self.rng.choice(engineers),
                "status": "success",
                "commit_message": "feat: process heavy data chunks in memory"
            })

        deploys.sort(key=lambda x: x["timestamp"], reverse=True)
        return deploys
