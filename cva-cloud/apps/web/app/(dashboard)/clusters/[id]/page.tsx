import SectionHeader from "@/components/section-header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
          <Badge tone={cluster.status === "connected" ? "success" : "warning"}>
            {cluster.status}
          </Badge>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Agent Version</span>
          <span>{cluster.agent_version || "—"}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Last Seen</span>
          <span>{cluster.last_seen || "never"}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/60">Cluster ID</span>
          <span className="text-xs text-white/70">{cluster.id}</span>
        </div>
      </Card>
    </div>
  );
}
