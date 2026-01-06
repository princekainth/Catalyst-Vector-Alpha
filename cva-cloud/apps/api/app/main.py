from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import api_router
import os

from app.db.session import SessionLocal
from app.models.cluster import Cluster

load_dotenv()

app = FastAPI(title=settings.app_name)

allowed_origins = os.getenv(
    "CVA_WEB_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/install/{cluster_id}/{api_key}")
def install_agent(cluster_id: str, api_key: str):
    db = SessionLocal()
    try:
        cluster = (
            db.query(Cluster)
            .filter(Cluster.id == cluster_id, Cluster.api_key == api_key)
            .first()
        )
        if not cluster:
            return Response(content="Cluster not found", status_code=404)

        api_url = os.getenv(
            "CVA_API_URL",
            os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8001"),
        )
        agent_image = os.getenv("CVA_AGENT_IMAGE", "cva-agent:local")
        agent_pull_policy = os.getenv("CVA_AGENT_PULL_POLICY", "IfNotPresent")
        ollama_url = os.getenv("CVA_OLLAMA_URL", "http://host.minikube.internal:11434")
        manifest = f"""apiVersion: v1
kind: Namespace
metadata:
  name: cva-system
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cva-agent
  namespace: cva-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cva-agent
rules:
  - apiGroups: [\"\"]
    resources: [\"pods\", \"events\", \"configmaps\", \"secrets\"]
    verbs: [\"get\", \"list\", \"watch\", \"patch\", \"delete\"]
  - apiGroups: [\"apps\"]
    resources: [\"deployments\", \"replicasets\"]
    verbs: [\"get\", \"list\", \"watch\", \"patch\"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cva-agent
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cva-agent
subjects:
  - kind: ServiceAccount
    name: cva-agent
    namespace: cva-system
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cva-agent
  namespace: cva-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cva-agent
  template:
    metadata:
      labels:
        app: cva-agent
    spec:
      serviceAccountName: cva-agent
      containers:
        - name: cva-agent
          image: {agent_image}
          imagePullPolicy: {agent_pull_policy}
          env:
            - name: CVA_API_URL
              value: {api_url}
            - name: CVA_CLUSTER_ID
              value: {cluster_id}
            - name: CVA_API_KEY
              value: {api_key}
            - name: OLLAMA_URL
              value: {ollama_url}
"""
        return Response(content=manifest, media_type="application/yaml")
    finally:
        db.close()
