import Link from "next/link";
import SectionHeader from "@/components/section-header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Incident } from "@/lib/types";

async function getIncidents() {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<Incident[]>("/api/v1/incidents", { headers });
  } catch {
    return [] as Incident[];
  }
}

export default async function IncidentsPage() {
  const incidents = await getIncidents();

  return (
    <div className="space-y-8">
      <SectionHeader title="Incidents">Review and approve remediation actions.</SectionHeader>
      <div className="space-y-4">
        {incidents.map((incident) => (
          <Card key={incident.id} className="flex items-start justify-between gap-6">
            <div>
              <p className="text-sm text-white/50">{incident.cluster_id}</p>
              <h3 className="text-lg font-display">{incident.pod_name}</h3>
              <p className="text-sm text-white/70">{incident.issue_type} - {incident.summary}</p>
              <div className="mt-3 flex gap-2">
                <Badge tone="danger">{incident.severity}</Badge>
                <Badge tone="warning">{incident.status}</Badge>
              </div>
            </div>
            <Link
              href={`/incidents/${incident.id}`}
              className="rounded-md border border-white/20 px-3 py-2 text-xs text-white/80"
            >
              View Details
            </Link>
          </Card>
        ))}
        {incidents.length === 0 ? (
          <Card>No incidents to show yet.</Card>
        ) : null}
      </div>
    </div>
  );
}
