"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "@/components/ui/toast";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Incident } from "@/lib/types";

type Props = {
  items: Incident[];
  mode: "history" | "archived";
};

export default function IncidentsHistoryClient({ items, mode }: Props) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const isArchived = mode === "archived";

  const allSelected = useMemo(
    () => items.length > 0 && selected.size === items.length,
    [items.length, selected.size]
  );

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelected(new Set());
      return;
    }
    setSelected(new Set(items.map((item) => item.id)));
  };

  const clearHistory = async (ids?: string[]) => {
    setLoading(true);
    const res = await fetch("/api/v1/incidents/history/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: ids ? JSON.stringify({ ids }) : JSON.stringify({}),
    });
    setLoading(false);
    if (!res.ok) {
      toast("Failed to clear history.", "error");
      return;
    }
    const data = await res.json().catch(() => ({}));
    const count = typeof data.deleted === "number" ? data.deleted : 0;
    toast(`History archived (${count}).`, "success");
    setSelected(new Set());
    router.refresh();
  };

  const restoreIncident = async (id: string) => {
    setLoading(true);
    const res = await fetch(`/api/v1/incidents/${id}/restore`, {
      method: "POST",
    });
    setLoading(false);
    if (!res.ok) {
      toast("Failed to restore incident.", "error");
      return;
    }
    toast("Incident restored to history.", "success");
    router.refresh();
  };

  const restoreSelected = async (ids: string[]) => {
    setLoading(true);
    const results = await Promise.all(
      ids.map((id) =>
        fetch(`/api/v1/incidents/${id}/restore`, {
          method: "POST",
        })
      )
    );
    setLoading(false);
    if (results.some((res) => !res.ok)) {
      toast("Failed to restore one or more incidents.", "error");
      return;
    }
    toast("Selected incidents restored.", "success");
    setSelected(new Set());
    router.refresh();
  };

  const handleClearSelected = () => {
    if (selected.size === 0) {
      toast("Select at least one incident to clear.", "info");
      return;
    }
    if (isArchived) {
      if (!window.confirm(`Restore ${selected.size} incident(s) to history?`)) {
        return;
      }
      restoreSelected(Array.from(selected));
      return;
    }
    if (!window.confirm(`Archive ${selected.size} incident(s) from history?`)) {
      return;
    }
    clearHistory(Array.from(selected));
  };

  const handleClearAll = () => {
    if (isArchived) {
      if (!window.confirm("Restore all archived incidents?")) {
        return;
      }
      restoreSelected(items.map((item) => item.id));
      return;
    }
    if (!window.confirm("Archive all history incidents?")) {
      return;
    }
    clearHistory();
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3 text-xs uppercase tracking-[0.3em] text-white/50">
          <span>{isArchived ? "Archived" : "History"}</span>
          <button
            type="button"
            onClick={toggleSelectAll}
            className="rounded-full border border-white/20 px-3 py-1 text-[10px] text-white/60 hover:border-white/40"
          >
            {allSelected ? "Clear selection" : "Select all"}
          </button>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleClearSelected}
            disabled={loading}
            className="rounded-full border border-white/20 px-4 py-2 text-xs uppercase tracking-[0.3em] text-white/70 hover:border-white/40 disabled:opacity-60"
          >
            {isArchived ? "Restore selected" : "Archive selected"}
          </button>
          <button
            type="button"
            onClick={handleClearAll}
            disabled={loading}
            className="rounded-full border border-danger/40 px-4 py-2 text-xs uppercase tracking-[0.3em] text-danger hover:border-danger/60 disabled:opacity-60"
          >
            {isArchived ? "Restore all" : "Archive all"}
          </button>
          {isArchived ? (
            <button
              type="button"
              onClick={async () => {
                setLoading(true);
                const res = await fetch("/api/v1/incidents/export?scope=archived");
                setLoading(false);
                if (!res.ok) {
                  toast("Failed to export history.", "error");
                  return;
                }
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = "cva-incident-history.json";
                link.click();
                window.URL.revokeObjectURL(url);
              }}
              disabled={loading}
              className="rounded-full border border-white/20 px-4 py-2 text-xs uppercase tracking-[0.3em] text-white/70 hover:border-white/40 disabled:opacity-60"
            >
              Export
            </button>
          ) : null}
        </div>
      </div>
      {items.map((incident) => {
        const statusTone =
          incident.status === "fixed"
            ? "success"
            : incident.status === "failed"
            ? "danger"
            : incident.status === "dismissed"
            ? "warning"
            : "info";
        const checked = selected.has(incident.id);
        return (
          <Card
            key={incident.id}
            className="relative flex items-start justify-between gap-6 border-white/10 bg-white/5 text-white/80"
          >
            <div className="flex items-start gap-4">
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleSelect(incident.id)}
                className="mt-2 h-4 w-4 rounded border-white/30 bg-transparent text-white accent-white/80"
              />
              <div>
                <p className="text-sm text-white/50">{incident.cluster_id}</p>
                <h3 className="text-lg font-display">{incident.pod_name}</h3>
                <p className="text-sm text-white/70">
                  {incident.issue_type} - {incident.summary}
                </p>
                {incident.action_type ? (
                  <p className="text-xs text-white/50">
                    {incident.status === "fixed" ? "Resolved by" : "Action"}:{" "}
                    {incident.action_type}
                  </p>
                ) : null}
                <div className="mt-3 flex gap-2">
                  <Badge tone="danger">{incident.severity}</Badge>
                  <Badge tone={statusTone}>{incident.status}</Badge>
                </div>
                <div className="mt-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <Link
                      href={`/incidents/${incident.id}`}
                      className="rounded-md border border-white/20 px-3 py-2 text-xs text-white/70"
                    >
                      View Details
                    </Link>
                    {isArchived ? (
                      <button
                        type="button"
                        onClick={() => restoreIncident(incident.id)}
                        className="rounded-md border border-white/20 px-3 py-2 text-xs text-white/70 hover:border-white/40 disabled:opacity-60"
                        disabled={loading}
                      >
                        Restore
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        );
      })}
      {items.length === 0 ? (
        <Card>{isArchived ? "No archived incidents yet." : "No incident history yet."}</Card>
      ) : null}
    </div>
  );
}
