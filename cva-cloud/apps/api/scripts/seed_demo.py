import json
from datetime import datetime

from app.db.session import SessionLocal
from app.models.organization import Organization
from app.models.user import User
from app.models.cluster import Cluster
from app.models.incident import Incident
from app.models.reasoning_trace import ReasoningTrace


def run():
    db = SessionLocal()
    org = Organization(id="demo-org", name="Demo Org", plan="pro")
    user = User(id="demo-user", org_id=org.id, email="demo@cva.io", role="admin")
    cluster = Cluster(
        id="cluster-demo",
        org_id=org.id,
        user_id=user.id,
        name="test-prod",
        api_key="demo-api-key",
        status="connected",
        agent_version="0.1.0",
        last_seen=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )
    incident = Incident(
        id="inc-001",
        cluster_id=cluster.id,
        user_id=user.id,
        namespace="default",
        pod_name="payment-service-abc123",
        issue_type="ImagePullBackOff",
        severity="critical",
        status="pending",
        summary="nginx:v1.2.3 not found",
        action_type="fix_image_tag",
        action_config="{}",
        created_at=datetime.utcnow(),
    )
    trace = ReasoningTrace(
        id="trace-001",
        incident_id=incident.id,
        trace_json=json.dumps(
            [
                {
                    "stage": "OBSERVE",
                    "message": "Detected ImagePullBackOff on payment-service-abc123",
                    "evidence": ["manifest not found"],
                    "duration_ms": 100,
                },
                {
                    "stage": "ANALYZE",
                    "message": "Typical causes: wrong tag, registry auth",
                    "confidence": 0.8,
                    "duration_ms": 500,
                },
                {
                    "stage": "DECIDE",
                    "message": "Will attempt: fix_image_tag",
                    "confidence": 0.95,
                    "duration_ms": 2300,
                },
                {
                    "stage": "ACT",
                    "message": "Patched deployment payment-service",
                    "duration_ms": 5200,
                },
                {
                    "stage": "VERIFY",
                    "message": "New pod reached Ready state",
                    "duration_ms": 8400,
                },
            ]
        ),
        created_at=datetime.utcnow(),
    )

    db.add(org)
    db.add(user)
    db.add(cluster)
    db.add(incident)
    db.add(trace)
    db.commit()
    db.close()
    print("Seeded demo data")


if __name__ == "__main__":
    run()
