export type Cluster = {
  id: string;
  name: string;
  status: string;
  agent_version: string | null;
  last_seen: string | null;
  created_at: string;
};

export type Incident = {
  id: string;
  cluster_id: string;
  pod_name: string;
  issue_type: string;
  severity: string;
  status: string;
  summary: string;
  created_at: string;
};

export type ReasoningTrace = {
  id: string;
  incident_id: string;
  trace_json: string;
  created_at: string;
};
