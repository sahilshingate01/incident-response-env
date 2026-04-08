import random
import datetime
from typing import List, Dict, Optional

class FakeDeployHistory:
    def __init__(self, incident_type: Optional[str] = None):
        self.incident_type = incident_type
        if incident_type:
            random.seed(f"deploys-{incident_type}")
        else:
            random.seed("deploys-normal")

    def get_recent_deploys(self, hours: int = 24) -> List[Dict]:
        """
        Returns list of deploy dicts: {id, service, version, timestamp, deployed_by, status, commit_message}
        """
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

        # Standard random deploys
        for i in range(3, 6):
            service = random.choice(services)
            ts = (datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, hours))).isoformat()
            deploys.append({
                "id": f"dep-{random.randint(1000, 9999)}",
                "service": service,
                "version": f"v1.2.{random.randint(0, 50)}",
                "timestamp": ts,
                "deployed_by": random.choice(engineers),
                "status": "success",
                "commit_message": random.choice(commit_messages)
            })

        # Inject a suspicious deploy if needed
        if self.incident_type == "bad_deploy":
            # Deploy 20-30 min ago
            ts = (datetime.datetime.now() - datetime.timedelta(minutes=random.randint(20, 30))).isoformat()
            deploys.append({
                "id": f"dep-evil-{random.randint(100,999)}",
                "service": "user-service",
                "version": "v1.3.0",
                "timestamp": ts,
                "deployed_by": "alice",
                "status": "success", # Deploys often finish "successfully" initially
                "commit_message": "feat: new user profile caching logic"
            })

        # Sort by timestamp (most recent first)
        deploys.sort(key=lambda x: x["timestamp"], reverse=True)
        return deploys

if __name__ == "__main__":
    print("--- Normal Deploy History ---")
    history = FakeDeployHistory()
    for d in history.get_recent_deploys():
        print(f"[{d['timestamp']}] {d['service']} - {d['commit_message']}")
        
    print("\n--- Bad Deploy History ---")
    history_bad = FakeDeployHistory("bad_deploy")
    for d in history_bad.get_recent_deploys():
        print(f"[{d['timestamp']}] {d['service']} - {d['commit_message']}")
