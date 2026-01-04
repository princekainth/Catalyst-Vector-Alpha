"use client";

import { useEffect, useState } from "react";

import ClusterActions from "@/components/cluster-actions";
import LocalTime from "@/components/local-time";
import { Card } from "@/components/ui/card";
import ClusterStatusBadge, {
  getEffectiveStatus,
} from "@/components/cluster-status-badge";
import { fetcher } from "@/lib/api";
import type { Cluster } from "@/lib/types";

const POLL_INTERVAL_MS = 10_000;
const TICK_INTERVAL_MS = 5_000;

export default function ClustersTableClient() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [loading, setLoading] = useState(true);
  const [nowTick, setNowTick] = useState(0);

  const loadClusters = async () => {
    try {
      const data = await fetcher<Cluster[]>("/api/v1/clusters");
      setClusters(data);
    } catch {
      setClusters([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadClusters();
    const poller = setInterval(loadClusters, POLL_INTERVAL_MS);
    return () => clearInterval(poller);
  }, []);

  useEffect(() => {
    const ticker = setInterval(() => {
      setNowTick((prev) => prev + 1);
    }, TICK_INTERVAL_MS);
    return () => clearInterval(ticker);
  }, []);

  const hasClusters = clusters.length > 0;
  const header = (
    <div className="grid grid-cols-5 items-center gap-4 border-b border-white/10 px-6 py-3 text-xs uppercase text-white/50">
      <span>Name</span>
      <span className="text-center">Status</span>
      <span>Agent Version</span>
      <span>Last Seen</span>
      <span>Actions</span>
    </div>
  );

  const rows = clusters.map((cluster) => (
    <div key={cluster.id} className="grid grid-cols-5 items-center gap-4 px-6 py-4 text-sm">
      {(() => {
        const effectiveStatus = getEffectiveStatus(cluster.status, cluster.last_seen);
        const tone =
          effectiveStatus === "connected"
            ? "success"
            : effectiveStatus === "stale"
              ? "warning"
              : "danger";
        return (
          <>
      <span className="font-semibold">{cluster.name}</span>
      <div className="flex items-center justify-center">
        <ClusterStatusBadge status={effectiveStatus} lastSeen={cluster.last_seen} />
      </div>
      <span className="text-white/70">{cluster.agent_version || "—"}</span>
      <span className="text-white/70">
        <LocalTime key={nowTick} value={cluster.last_seen} mode="relative" />
      </span>
      <ClusterActions clusterId={cluster.id} />
          </>
        );
      })()}
    </div>
  ));

  if (loading && !hasClusters) {
    return <Card>Loading clusters...</Card>;
  }

  if (!hasClusters) {
    return <Card>No clusters connected. Connect your first cluster to get started.</Card>;
  }

  return (
    <Card className="overflow-hidden p-0">
      {header}
      <div className="divide-y divide-white/5">{rows}</div>
    </Card>
  );
}
