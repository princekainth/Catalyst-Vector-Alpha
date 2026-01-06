import SectionHeader from "@/components/section-header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import IncidentActions from "@/components/incident-actions";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Incident, ReasoningTrace } from "@/lib/types";

export const dynamic = "force-dynamic";

async function getIncident(id: string) {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<Incident>(`/api/v1/incidents/${id}`, { headers });
  } catch {
    return null;
  }
}

async function getTrace(id: string) {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<ReasoningTrace[]>(`/api/v1/incidents/${id}/trace`, { headers });
  } catch {
    return [] as ReasoningTrace[];
  }
}

export default async function IncidentDetail({ params }: { params: { id: string } }) {
  const incident = await getIncident(params.id);
  const traces = await getTrace(params.id);

  if (!incident) {
    return <Card>Incident not found.</Card>;
  }

  const tracePayload = traces[0]?.trace_json ? JSON.parse(traces[0].trace_json) : null;
  const traceSteps = Array.isArray(tracePayload)
    ? tracePayload
    : tracePayload?.steps || [];
  const normalizedSteps = traceSteps.map((step: any) => ({
    stage: step.stage || step.step_type || step.type || "step",
    message: step.message || step.content || "",
    confidence: step.confidence,
    evidence: Array.isArray(step.evidence) ? step.evidence : [],
  }));
  let outcome: Record<string, any> | null = null;
  if (incident.outcome && typeof incident.outcome === "string") {
    try {
      outcome = JSON.parse(incident.outcome);
    } catch {
      outcome = null;
    }
  }
  let actionConfig: Record<string, any> | null = null;
  if (incident.action_config && typeof incident.action_config === "string") {
    try {
      actionConfig = JSON.parse(incident.action_config);
    } catch {
      actionConfig = null;
    }
  }
  const displayAction = outcome?.action || incident.action_type || "n/a";
  const recommendedActions = Array.isArray(actionConfig?.recommended_actions)
    ? actionConfig?.recommended_actions
    : [];
  const outcomeSummary =
    outcome?.note ||
    outcome?.error ||
    outcome?.output ||
    (typeof incident.outcome === "string" ? incident.outcome : "");

  return (
    <div className="space-y-8">
      <SectionHeader title="Incident Details">{incident.pod_name}</SectionHeader>
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <Card className="space-y-4">
          <h3 className="text-lg font-display">Reasoning Timeline</h3>
          <div className="space-y-4 text-sm">
            {normalizedSteps.map((step: any, idx: number) => (
              <div key={idx} className="rounded-lg border border-white/10 p-4">
                <div className="flex items-center justify-between">
                  <p className="font-semibold">{String(step.stage).toUpperCase()}</p>
                  {typeof step.confidence === "number" ? (
                    <Badge tone="success">{Math.round(step.confidence * 100)}%</Badge>
                  ) : null}
                </div>
                <p className="text-white/70">{step.message}</p>
                {step.evidence ? (
                  <ul className="mt-2 list-disc pl-4 text-white/50">
                    {step.evidence.map((ev: string, i: number) => (
                      <li key={i}>{ev}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            ))}
            {traceSteps.length === 0 ? (
              <p className="text-white/50">No trace recorded yet.</p>
            ) : null}
          </div>
        </Card>
        <div className="space-y-6">
          {outcomeSummary ? (
            <Card className="space-y-2 border-white/15 bg-white/5">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-display">Outcome</h3>
                <Badge tone={incident.status === "fixed" ? "success" : incident.status === "failed" ? "danger" : "info"}>
                  {incident.status}
                </Badge>
              </div>
              <p className="text-sm text-white/70">
                {incident.status === "fixed" ? "Resolved by" : "Action"}: {displayAction}
              </p>
              <p className="text-sm text-white/70">{outcomeSummary}</p>
            </Card>
          ) : null}
          <Card className="space-y-3">
            <h3 className="text-lg font-display">Evidence</h3>
            <p className="text-sm text-white/70">Issue Type: {incident.issue_type}</p>
            <p className="text-sm text-white/70">Summary: {incident.summary}</p>
            <p className="text-sm text-white/70">Status: {incident.status}</p>
            {recommendedActions.length > 0 ? (
              <div className="space-y-1 text-sm text-white/70">
                <p className="text-white/60">Other possible fixes:</p>
                <ul className="list-disc pl-4 text-white/60">
                  {recommendedActions.map((rec: any, idx: number) => (
                    <li key={idx}>{rec?.action || "unknown"}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            <IncidentActions incidentId={incident.id} />
          </Card>
        </div>
      </div>
    </div>
  );
}
