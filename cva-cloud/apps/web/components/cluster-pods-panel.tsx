"use client";

import { useEffect, useMemo, useState } from "react";

import { fetcher } from "@/lib/api";
import type { Cluster } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

type PodSnapshot = NonNullable<Cluster["pod_snapshot"]>;

const POLL_INTERVAL_MS = 8000;
const TICK_INTERVAL_MS = 4000;
const PAGE_SIZE = 20;
const SYSTEM_NAMESPACES = new Set(["kube-system", "monitoring", "cva-system", "kubernetes-dashboard"]);

function phaseTone(phase: string, reason: string, ready: boolean) {
  if (reason && reason !== "Running") return "danger";
  if (!ready && phase === "Running") return "warning";
  if (phase === "Running") return "success";
  if (phase === "Pending") return "warning";
  if (phase === "Failed") return "danger";
  return "default";
}

export default function ClusterPodsPanel({ clusterId }: { clusterId: string }) {
  const [pods, setPods] = useState<PodSnapshot>([]);
  const [updatedAt, setUpdatedAt] = useState(Date.now());
  const [search, setSearch] = useState("");
  const [namespaceFilter, setNamespaceFilter] = useState("all");
  const [hideSystem, setHideSystem] = useState(true);
  const [page, setPage] = useState(0);

  const loadPods = async () => {
    try {
      const cluster = await fetcher<Cluster>(`/api/v1/clusters/${clusterId}`);
      setPods(cluster.pod_snapshot || []);
      setUpdatedAt(Date.now());
    } catch {
      setPods([]);
    }
  };

  useEffect(() => {
    loadPods();
    const poller = setInterval(loadPods, POLL_INTERVAL_MS);
    return () => clearInterval(poller);
  }, [clusterId]);

  useEffect(() => {
    const ticker = setInterval(() => setUpdatedAt(Date.now()), TICK_INTERVAL_MS);
    return () => clearInterval(ticker);
  }, []);

  useEffect(() => {
    setPage(0);
  }, [search, namespaceFilter, hideSystem]);

  const namespaces = useMemo(() => {
    const set = new Set<string>();
    pods.forEach((pod) => {
      if (pod.namespace) set.add(pod.namespace);
    });
    return Array.from(set).sort();
  }, [pods]);

  const filteredPods = useMemo(() => {
    const query = search.trim().toLowerCase();
    return pods.filter((pod) => {
      if (hideSystem && SYSTEM_NAMESPACES.has(pod.namespace)) {
        return false;
      }
      if (namespaceFilter !== "all" && pod.namespace !== namespaceFilter) {
        return false;
      }
      if (!query) return true;
      return (
        pod.name.toLowerCase().includes(query) ||
        pod.namespace.toLowerCase().includes(query)
      );
    });
  }, [pods, search, namespaceFilter, hideSystem]);

  const pageCount = Math.max(1, Math.ceil(filteredPods.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  const pageStart = currentPage * PAGE_SIZE;
  const pageEnd = pageStart + PAGE_SIZE;
  const pagePods = filteredPods.slice(pageStart, pageEnd);

  const summary = useMemo(() => {
    const counts = { Running: 0, Pending: 0, Failed: 0, Other: 0 };
    pods.forEach((pod) => {
      if (pod.phase === "Running") counts.Running += 1;
      else if (pod.phase === "Pending") counts.Pending += 1;
      else if (pod.phase === "Failed") counts.Failed += 1;
      else counts.Other += 1;
    });
    return counts;
  }, [pods]);

  return (
    <Card className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-display">Live Pods</h3>
          <p className="text-xs text-white/50">Last refreshed {Math.round((Date.now() - updatedAt) / 1000)}s ago</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-white/60">
          <span>Running {summary.Running}</span>
          <span>Pending {summary.Pending}</span>
          <span>Failed {summary.Failed}</span>
          <span>Other {summary.Other}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-white/10 bg-black/30 px-3 py-2 text-xs text-white/70">
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search pods or namespaces"
            className="w-48 rounded-md border border-white/10 bg-black/40 px-2 py-1 text-xs text-white/80"
          />
          <select
            value={namespaceFilter}
            onChange={(event) => setNamespaceFilter(event.target.value)}
            className="rounded-md border border-white/10 bg-black/40 px-2 py-1 text-xs text-white/80"
          >
            <option value="all">All namespaces</option>
            {namespaces.map((ns) => (
              <option key={ns} value={ns}>
                {ns}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-xs text-white/60">
            <input
              type="checkbox"
              checked={hideSystem}
              onChange={(event) => setHideSystem(event.target.checked)}
            />
            Hide system namespaces
          </label>
        </div>
        <span className="text-xs text-white/50">
          Showing {filteredPods.length} of {pods.length}
        </span>
      </div>

      {pods.length === 0 ? (
        <p className="text-sm text-white/60">No pod snapshot reported yet.</p>
      ) : (
        <div className="space-y-2 text-sm">
          {pagePods.map((pod) => (
            <div
              key={`${pod.namespace}-${pod.name}`}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-white/10 bg-black/40 px-3 py-2"
            >
              <div>
                <p className="font-semibold">{pod.name}</p>
                <p className="text-xs text-white/50">{pod.namespace}</p>
              </div>
              <div className="flex items-center gap-2">
                <Badge tone={phaseTone(pod.phase, pod.reason, pod.ready)}>
                  {pod.reason || pod.phase}
                </Badge>
                <span className="text-xs text-white/50">
                  Restarts {pod.restarts ?? 0}
                </span>
                {!pod.ready ? (
                  <span className="text-xs text-warning">Not Ready</span>
                ) : null}
              </div>
            </div>
          ))}
          <div className="flex items-center justify-between text-xs text-white/60">
            <button
              type="button"
              onClick={() => setPage((prev) => Math.max(0, prev - 1))}
              disabled={currentPage === 0}
              className="rounded-md border border-white/10 px-2 py-1 disabled:opacity-50"
            >
              Previous
            </button>
            <span>
              Page {currentPage + 1} of {pageCount}
            </span>
            <button
              type="button"
              onClick={() => setPage((prev) => Math.min(pageCount - 1, prev + 1))}
              disabled={currentPage >= pageCount - 1}
              className="rounded-md border border-white/10 px-2 py-1 disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </Card>
  );
}
