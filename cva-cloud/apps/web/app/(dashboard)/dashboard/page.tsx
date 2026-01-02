import SectionHeader from "@/components/section-header";
import { Card, CardTitle, CardValue } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Cluster, Incident } from "@/lib/types";

async function getData() {
  try {
    const headers = { ...(await getAuthHeaders()) };
    const [clusters, incidents] = await Promise.all([
      fetcher<Cluster[]>("/api/v1/clusters", { headers }),
      fetcher<Incident[]>("/api/v1/incidents", { headers }),
    ]);
    return { clusters, incidents };
  } catch {
    return { clusters: [], incidents: [] };
  }
}

export default async function DashboardPage() {
  const { clusters, incidents } = await getData();
  const activeIncidents = incidents.filter((item) => item.status !== "fixed");

  return (
    <div className="space-y-8">
      <SectionHeader title="Dashboard">Realtime view across clusters.</SectionHeader>
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardTitle>Total Clusters</CardTitle>
          <CardValue>{clusters.length}</CardValue>
        </Card>
        <Card>
          <CardTitle>Active Incidents</CardTitle>
          <CardValue>{activeIncidents.length}</CardValue>
        </Card>
        <Card>
          <CardTitle>Actions Today</CardTitle>
          <CardValue>18</CardValue>
        </Card>
        <Card>
          <CardTitle>Estimated Savings</CardTitle>
          <CardValue>$4.2k</CardValue>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <Card className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-display">Live Activity Feed</h3>
            <Badge tone="success">Live</Badge>
          </div>
          <div className="space-y-3 text-sm text-white/80">
            {activeIncidents.slice(0, 5).map((incident) => (
              <div key={incident.id} className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-semibold">{incident.issue_type}</p>
                  <p className="text-white/60">
                    {incident.pod_name} - {incident.summary}
                  </p>
                </div>
                <Badge tone="danger">{incident.severity}</Badge>
              </div>
            ))}
            {activeIncidents.length === 0 ? (
              <p className="text-white/50">No incidents yet. Seed data to preview the feed.</p>
            ) : null}
          </div>
        </Card>

        <Card className="space-y-4">
          <h3 className="text-lg font-display">Cluster Health</h3>
          <div className="space-y-4">
            {clusters.map((cluster) => (
              <div key={cluster.id} className="flex items-center justify-between">
                <div>
                  <p className="font-semibold">{cluster.name}</p>
                  <p className="text-xs text-white/50">{cluster.status}</p>
                </div>
                <Badge tone={cluster.status === "connected" ? "success" : "warning"}>
                  {cluster.status}
                </Badge>
              </div>
            ))}
            {clusters.length === 0 ? (
              <p className="text-white/50">No clusters connected yet.</p>
            ) : null}
          </div>
        </Card>
      </div>
    </div>
  );
}
