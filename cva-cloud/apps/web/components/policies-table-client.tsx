"use client";

import { useEffect, useState } from "react";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetcher } from "@/lib/api";
import type { Policy } from "@/lib/types";

const ISSUE_TYPES = [
  "ImagePullBackOff",
  "CreateContainerConfigError",
  "CrashLoopBackOff",
  "OOMKilled",
  "Pending",
  "NotReady",
];

export default function PoliciesTableClient() {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({
    name: "",
    issue_type: ISSUE_TYPES[0],
    cluster_id: "",
    auto_approve: false,
    max_memory_mb: "",
    allow_placeholder: false,
  });

  const loadPolicies = async () => {
    try {
      const data = await fetcher<Policy[]>("/api/v1/policies");
      setPolicies(data);
    } catch {
      setPolicies([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPolicies();
  }, []);

  const submitPolicy = async () => {
    const payload: Record<string, unknown> = {
      name: form.name,
      issue_type: form.issue_type,
      cluster_id: form.cluster_id || null,
      auto_approve: form.auto_approve,
      allow_placeholder: form.allow_placeholder,
    };
    if (form.issue_type === "OOMKilled" && form.max_memory_mb) {
      payload.max_memory_mb = Number(form.max_memory_mb);
    }
    await fetcher<Policy>("/api/v1/policies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setForm({
      name: "",
      issue_type: ISSUE_TYPES[0],
      cluster_id: "",
      auto_approve: false,
      max_memory_mb: "",
      allow_placeholder: false,
    });
    loadPolicies();
  };

  const deletePolicy = async (policyId: string) => {
    await fetcher(`/api/v1/policies/${policyId}`, { method: "DELETE" });
    loadPolicies();
  };

  const toggleStatus = async (policy: Policy) => {
    const nextStatus = policy.status === "active" ? "disabled" : "active";
    await fetcher(`/api/v1/policies/${policy.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: nextStatus }),
    });
    loadPolicies();
  };

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <h3 className="text-lg font-display">Create Policy</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="text-xs text-white/60">Rule name</label>
            <input
              value={form.name}
              onChange={(event) => setForm({ ...form, name: event.target.value })}
              className="mt-2 w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm"
              placeholder="Auto-approve ImagePullBackOff"
            />
          </div>
          <div>
            <label className="text-xs text-white/60">Issue type</label>
            <select
              value={form.issue_type}
              onChange={(event) => setForm({ ...form, issue_type: event.target.value })}
              className="mt-2 w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm"
            >
              {ISSUE_TYPES.map((issue) => (
                <option key={issue} value={issue}>
                  {issue}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-white/60">Cluster scope (optional)</label>
            <input
              value={form.cluster_id}
              onChange={(event) => setForm({ ...form, cluster_id: event.target.value })}
              className="mt-2 w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm"
              placeholder="cluster-id or leave blank"
            />
          </div>
          <div className="flex items-center gap-3 pt-6">
            <input
              type="checkbox"
              checked={form.auto_approve}
              onChange={(event) => setForm({ ...form, auto_approve: event.target.checked })}
              className="h-4 w-4 rounded border-white/20 bg-black/40"
            />
            <span className="text-sm text-white/70">Auto-approve</span>
          </div>
          {form.issue_type === "OOMKilled" ? (
            <div>
              <label className="text-xs text-white/60">Max memory increase (MB)</label>
              <input
                value={form.max_memory_mb}
                onChange={(event) => setForm({ ...form, max_memory_mb: event.target.value })}
                className="mt-2 w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm"
                placeholder="512"
              />
            </div>
          ) : null}
          {form.issue_type === "CreateContainerConfigError" ? (
            <div className="flex items-center gap-3 pt-6">
              <input
                type="checkbox"
                checked={form.allow_placeholder}
                onChange={(event) =>
                  setForm({ ...form, allow_placeholder: event.target.checked })
                }
                className="h-4 w-4 rounded border-white/20 bg-black/40"
              />
              <span className="text-sm text-white/70">Allow placeholder config</span>
            </div>
          ) : null}
        </div>
        <button
          type="button"
          onClick={submitPolicy}
          disabled={!form.name}
          className="w-fit rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black disabled:opacity-60"
        >
          Save Policy
        </button>
      </Card>

      <Card className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-display">Active Policies</h3>
          <p className="text-xs text-white/50">{policies.length} rules</p>
        </div>
        {loading ? (
          <p className="text-sm text-white/60">Loading policies...</p>
        ) : null}
        {!loading && policies.length === 0 ? (
          <p className="text-sm text-white/60">No policies yet.</p>
        ) : null}
        {policies.map((policy) => (
          <div
            key={policy.id}
            className="grid gap-4 rounded-lg border border-white/10 bg-black/30 px-4 py-3 text-sm md:grid-cols-6"
          >
            <div className="md:col-span-2">
              <p className="text-xs text-white/50">Rule</p>
              <p className="font-semibold">{policy.name}</p>
            </div>
            <div>
              <p className="text-xs text-white/50">Scope</p>
              <p>{policy.cluster_id || "All clusters"}</p>
            </div>
            <div>
              <p className="text-xs text-white/50">Issue</p>
              <p>{policy.issue_type}</p>
            </div>
            <div>
              <p className="text-xs text-white/50">Mode</p>
              <Badge tone={policy.auto_approve ? "success" : "warning"}>
                {policy.auto_approve ? "Auto" : "Manual"}
              </Badge>
            </div>
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => toggleStatus(policy)}
                className="rounded-md border border-white/10 px-2 py-1 text-xs"
              >
                {policy.status === "active" ? "Disable" : "Enable"}
              </button>
              <button
                type="button"
                onClick={() => deletePolicy(policy.id)}
                className="rounded-md border border-white/10 px-2 py-1 text-xs text-danger"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}
