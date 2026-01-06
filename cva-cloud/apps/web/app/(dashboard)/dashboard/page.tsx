import Link from "next/link";
import SectionHeader from "@/components/section-header";
import { Card, CardTitle, CardValue } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Cluster, Incident } from "@/lib/types";
import ClusterStatusBadge from "@/components/cluster-status-badge";
import { currentUser } from "@clerk/nextjs/server";

export const dynamic = "force-dynamic";

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

function normalizeTimestamp(value: string): string {
  const cleaned = value.includes("T") ? value : value.replace(" ", "T");
  if (cleaned.includes("Z") || cleaned.match(/[+-]\d\d:\d\d$/)) {
    return cleaned;
  }
  return `${cleaned}Z`;
}

function getEffectiveStatus(status: string, lastSeen?: string | null) {
  if (!lastSeen) return "disconnected";
  const normalized = normalizeTimestamp(lastSeen);
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "disconnected";
  const ageMs = Date.now() - date.getTime();
  if (ageMs > 300_000) return "disconnected";
  if (ageMs > 90_000) return "stale";
  return status === "connected" ? "connected" : "connected";
}

function greetingForHour(hour: number): string {
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

export default async function DashboardPage() {
  const { clusters, incidents } = await getData();
  const user = await currentUser();
  const displayName =
    user?.firstName ||
    user?.fullName ||
    user?.primaryEmailAddress?.emailAddress ||
    "there";
  const activeIncidents = incidents.filter(
    (item) => item.status !== "fixed" && item.status !== "dismissed"
  );
  const now = Date.now();
  const pendingApprovals = incidents.filter((incident) => incident.status === "pending");
  const impactedClusters = new Set(activeIncidents.map((incident) => incident.cluster_id)).size;
  const greeting = greetingForHour(new Date().getHours());

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <SectionHeader title={`${greeting}, ${displayName}.`}>
          Welcome to Mission Control. Connect a cluster to start remediation.
        </SectionHeader>
        {clusters.length > 0 ? (
          <Link
            href="/clusters/new"
            className="rounded-md border border-white/10 px-3 py-2 text-xs text-white/70"
          >
            Connect Cluster
          </Link>
        ) : null}
      </div>

      {clusters.length === 0 ? (
        <Card className="flex flex-col gap-4">
          <div>
            <h3 className="text-lg font-display">Connect your first cluster</h3>
            <p className="text-sm text-white/60">
              Install the CVA agent and start streaming incidents into your dashboard.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-white/60">
            <span>1. Generate install command</span>
            <span>2. Apply to cluster</span>
            <span>3. Approve your first fix</span>
          </div>
          <Link
            href="/clusters/new"
            className="w-fit rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black"
          >
            Connect Cluster
          </Link>
        </Card>
      ) : null}

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
          <CardTitle>Clusters Impacted</CardTitle>
          <CardValue>{impactedClusters}</CardValue>
        </Card>
        <Card>
          <CardTitle>Pending Approvals</CardTitle>
          <CardValue>{pendingApprovals.length}</CardValue>
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
                  <p className="text-xs text-white/50">
                    {getEffectiveStatus(cluster.status, cluster.last_seen)}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <ClusterStatusBadge status={cluster.status} lastSeen={cluster.last_seen} />
                  <Link
                    href={`/clusters/${cluster.id}`}
                    className="rounded-md border border-white/10 px-2 py-1 text-xs text-white/70"
                  >
                    Details
                  </Link>
                </div>
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
