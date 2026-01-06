export type Cluster = {
  id: string;
  name: string;
  status: string;
  agent_version: string | null;
  last_seen: string | null;
  pod_snapshot?: Array<{
    name: string;
    namespace: string;
    phase: string;
    ready: boolean;
    restarts: number;
    reason: string;
    message?: string;
  }> | null;
  created_at: string;
};

export type Incident = {
  id: string;
  cluster_id: string;
  namespace: string;
  pod_name: string;
  issue_type: string;
  severity: string;
  status: string;
  summary: string;
  action_type?: string | null;
  action_config?: string | null;
  outcome?: string | null;
  executed_at?: string | null;
  completed_at?: string | null;
  archived_at?: string | null;
  created_at: string;
};

export type Policy = {
  id: string;
  org_id: string;
  cluster_id: string | null;
  name: string;
  issue_type: string;
  auto_approve: boolean;
  max_memory_mb: number | null;
  allow_placeholder: boolean;
  status: string;
  created_at: string;
  updated_at: string;
};

export type ReasoningTrace = {
  id: string;
  incident_id: string;
  trace_json: string;
  created_at: string;
};
