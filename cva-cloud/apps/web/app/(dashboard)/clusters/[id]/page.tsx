import SectionHeader from "@/components/section-header";
import { Card } from "@/components/ui/card";
import ClusterStatusBadge from "@/components/cluster-status-badge";
import LocalTime from "@/components/local-time";
import ClusterPodsPanel from "@/components/cluster-pods-panel";
import ClusterInstallCard from "@/components/cluster-install-card";
import { fetcher } from "@/lib/api";
import { getAuthHeaders } from "@/lib/api.server";
import type { Cluster } from "@/lib/types";


async function getCluster(id: string) {
  try {
    const headers = { ...(await getAuthHeaders()) };
    return await fetcher<Cluster>(`/api/v1/clusters/${id}`, { headers });
  } catch {
    return null;
  }
}

export default async function ClusterDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const cluster = await getCluster(params.id);

  if (!cluster) {
    return <Card>Cluster not found.</Card>;
  }

  return (
    <div className="space-y-8">
      <SectionHeader title="Cluster Details">
        {cluster.name}
      </SectionHeader>
      <Card className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-white/60">Status</span>
          <ClusterStatusBadge status={cluster.status} lastSeen={cluster.last_seen} />
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Agent Version</span>
          <span>{cluster.agent_version || "Not reported"}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Last Seen</span>
          <LocalTime value={cluster.last_seen} mode="relative" />
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Created At</span>
          <LocalTime value={cluster.created_at} mode="absolute" />
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Cluster ID</span>
          <span className="text-xs text-white/70">{cluster.id}</span>
        </div>
      </Card>
      <Card className="space-y-4">
        <h3 className="text-lg font-display">Agent Status</h3>
        <ClusterInstallCard
          clusterId={cluster.id}
          status={cluster.status}
          lastSeen={cluster.last_seen}
        />
      </Card>

      <ClusterPodsPanel clusterId={cluster.id} />
    </div>
  );
}
