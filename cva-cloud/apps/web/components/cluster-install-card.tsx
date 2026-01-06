"use client";

import { useState } from "react";
import { getEffectiveStatus, isHeartbeatFresh } from "@/components/cluster-status-badge";
import { Badge } from "@/components/ui/badge";

type InstallResponse = {
  cluster_id: string;
  api_key: string;
  install_command: string;
};

type Props = {
  clusterId: string;
  status: string;
  lastSeen?: string | null;
};

export default function ClusterInstallCard({ clusterId, status, lastSeen }: Props) {
  const [install, setInstall] = useState<InstallResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy command");
  const [error, setError] = useState("");
  const effectiveStatus = getEffectiveStatus(status, lastSeen);
  const connected = effectiveStatus === "connected";
  const statusTone = connected ? "success" : effectiveStatus === "stale" ? "warning" : "danger";

  const fetchInstall = async () => {
    setLoading(true);
    setError("");
    const res = await fetch(`/api/v1/clusters/${clusterId}/install`);
    setLoading(false);
    if (!res.ok) {
      const text = await res.text();
      setError(text || "Failed to fetch install command.");
      return;
    }
    const data = (await res.json()) as InstallResponse;
    setInstall(data);
  };

  const copyCommand = async () => {
    if (!install) return;
    try {
      await navigator.clipboard.writeText(install.install_command);
      setCopyLabel("Copied!");
      setTimeout(() => setCopyLabel("Copy command"), 2000);
    } catch {
      setCopyLabel("Copy failed");
      setTimeout(() => setCopyLabel("Copy command"), 2000);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-white/70">
          <span
            className={`h-2 w-2 rounded-full ${
              isHeartbeatFresh(lastSeen)
                ? "bg-accent animate-pulse"
                : effectiveStatus === "stale"
                  ? "bg-warning"
                  : "bg-white/30"
            }`}
          />
          <span>Agent status</span>
        </div>
        <Badge tone={statusTone}>{effectiveStatus}</Badge>
      </div>
      <div className="text-sm text-white/70">
        {connected
          ? "Agent connected. No action needed."
          : effectiveStatus === "stale"
            ? "Agent heartbeat is stale. Reconnect to refresh."
            : "Agent not reporting. Reinstall to reconnect."}
      </div>
      <button
        type="button"
        onClick={fetchInstall}
        disabled={loading}
        className={
          connected
            ? "w-fit rounded-md border border-white/10 px-4 py-2 text-sm text-white/70 disabled:opacity-60"
            : "w-fit rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black disabled:opacity-60"
        }
      >
        {loading
          ? "Preparing..."
          : connected
            ? "Show install command"
            : effectiveStatus === "stale"
              ? "Reconnect"
              : "Reinstall Agent"}
      </button>
      {error ? <p className="text-sm text-danger">{error}</p> : null}
      {install ? (
        <div className="rounded-md border border-white/10 bg-black/40 p-3">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <code className="text-xs text-white/80">{install.install_command}</code>
            <button
              type="button"
              className="rounded-md border border-white/10 px-3 py-1 text-xs text-white/70 hover:border-white/30"
              onClick={copyCommand}
            >
              {copyLabel}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
