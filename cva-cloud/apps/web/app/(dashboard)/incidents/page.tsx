import Link from "next/link";
import IncidentActionsList from "@/components/incident-actions-list";
import IncidentsHistoryClient from "@/components/incidents-history-client";
import SectionHeader from "@/components/section-header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Incident } from "@/lib/types";

export const dynamic = "force-dynamic";

async function getIncidents() {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<Incident[]>("/api/v1/incidents", { headers });
  } catch {
    return [] as Incident[];
  }
}

async function getArchivedIncidents() {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<Incident[]>("/api/v1/incidents/archived", { headers });
  } catch {
    return [] as Incident[];
  }
}

export default async function IncidentsPage() {
  const incidents = await getIncidents();
  const historyStatuses = new Set(["dismissed", "fixed", "failed"]);
  const activeIncidents = incidents.filter((incident) => !historyStatuses.has(incident.status));
  const historyIncidents = incidents.filter((incident) => historyStatuses.has(incident.status));
  const archivedIncidents = await getArchivedIncidents();

  return (
    <div className="space-y-8">
      <SectionHeader title="Incidents">Review and approve remediation actions.</SectionHeader>
      <div className="space-y-4">
        <h3 className="text-xs uppercase tracking-[0.3em] text-white/50">Active</h3>
        {activeIncidents.map((incident) => (
          <Card key={incident.id} className="relative flex items-start justify-between gap-6">
            <div>
              <p className="text-sm text-white/50">{incident.cluster_id}</p>
              <h3 className="text-lg font-display">{incident.pod_name}</h3>
              <p className="text-sm text-white/70">{incident.issue_type} - {incident.summary}</p>
              <div className="mt-3 flex gap-2">
                <Badge tone="danger">{incident.severity}</Badge>
                <Badge tone="warning">{incident.status}</Badge>
              </div>
              <div className="mt-3">
                <IncidentActionsList incidentId={incident.id} />
              </div>
              <div className="mt-4">
                <Link
                  href={`/incidents/${incident.id}`}
                  className="rounded-md border border-white/20 px-3 py-2 text-xs text-white/80"
                >
                  View Details
                </Link>
              </div>
            </div>
          </Card>
        ))}
        {activeIncidents.length === 0 ? (
          <Card>No active incidents right now.</Card>
        ) : null}
      </div>
      <IncidentsHistoryClient items={historyIncidents} mode="history" />
      <IncidentsHistoryClient items={archivedIncidents} mode="archived" />
    </div>
  );
}
