import Link from "next/link";
import SectionHeader from "@/components/section-header";
import ClusterActions from "@/components/cluster-actions";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Cluster } from "@/lib/types";

async function getClusters() {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<Cluster[]>("/api/v1/clusters", { headers });
  } catch {
    return [] as Cluster[];
  }
}

export default async function ClustersPage({
  searchParams,
}: {
  searchParams?: { connected?: string };
}) {
  const clusters = await getClusters();
  const hasClusters = clusters.length > 0;
  const showConnected = searchParams?.connected === "1";

  const formatRelative = (value?: string | null) => {
    if (!value) return "never";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "unknown";
    const diffMs = date.getTime() - Date.now();
    const diffSeconds = Math.round(diffMs / 1000);
    const diffMinutes = Math.round(diffSeconds / 60);
    const diffHours = Math.round(diffMinutes / 60);
    const diffDays = Math.round(diffHours / 24);
    if (Math.abs(diffMinutes) < 60) {
      return `${Math.abs(diffMinutes)} minutes ago`;
    }
    if (Math.abs(diffHours) < 24) {
      return `${Math.abs(diffHours)} hours ago`;
    }
    return `${Math.abs(diffDays)} days ago`;
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <SectionHeader title="Clusters">Connect and manage your Kubernetes clusters.</SectionHeader>
          {showConnected ? (
            <p className="text-sm text-accent">Cluster connected successfully.</p>
          ) : null}
        </div>
        <Link
          href="/clusters/new"
          className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black"
        >
          + Connect Cluster
        </Link>
      </div>

      {!hasClusters ? (
        <Card>
          <p>No clusters connected. Connect your first cluster to get started.</p>
        </Card>
      ) : (
        <Card className="overflow-hidden p-0">
          <div className="grid grid-cols-5 gap-4 border-b border-white/10 px-6 py-3 text-xs uppercase text-white/50">
            <span>Name</span>
            <span>Status</span>
            <span>Agent Version</span>
            <span>Last Seen</span>
            <span>Actions</span>
          </div>
          <div className="divide-y divide-white/5">
            {clusters.map((cluster) => (
              <div key={cluster.id} className="grid grid-cols-5 gap-4 px-6 py-4 text-sm">
                <span className="font-semibold">{cluster.name}</span>
                <Badge
                  tone={
                    cluster.status === "connected"
                      ? "success"
                      : cluster.status === "pending"
                        ? "warning"
                        : "danger"
                  }
                >
                  {cluster.status}
                </Badge>
                <span className="text-white/70">{cluster.agent_version || "—"}</span>
                <span className="text-white/70">{formatRelative(cluster.last_seen)}</span>
                <ClusterActions clusterId={cluster.id} />
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
