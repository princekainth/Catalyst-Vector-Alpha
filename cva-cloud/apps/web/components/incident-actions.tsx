"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";

import { API_BASE } from "@/lib/api";


export default function IncidentActions({ incidentId }: { incidentId: string }) {
  const router = useRouter();
  const { getToken } = useAuth();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const approveIncident = async () => {
    setLoading(true);
    setMessage("");
    const token = await getToken();
    const headers: Record<string, string> = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const res = await fetch(`${API_BASE}/api/v1/incidents/${incidentId}/approve/`, {
      method: "POST",
      headers,
    });
    setLoading(false);
    if (res.ok) {
      setMessage("Approved. Awaiting agent execution.");
      router.refresh();
    } else {
      setMessage("Failed to approve incident.");
    }
  };

  const rollbackIncident = async () => {
    setLoading(true);
    setMessage("");
    const token = await getToken();
    const headers: Record<string, string> = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const res = await fetch(`${API_BASE}/api/v1/incidents/${incidentId}/rollback/`, {
      method: "POST",
      headers,
    });
    setLoading(false);
    if (res.ok) {
      setMessage("Rollback requested.");
      router.refresh();
    } else {
      setMessage("Failed to request rollback.");
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={approveIncident}
          disabled={loading}
          className="rounded-md bg-warning/20 px-3 py-2 text-xs font-semibold text-warning disabled:opacity-60"
        >
          {loading ? "Approving..." : "Approve & Fix"}
        </button>
        <button
          type="button"
          onClick={rollbackIncident}
          disabled={loading}
          className="rounded-md border border-white/10 px-3 py-2 text-xs text-white/60 disabled:opacity-60"
        >
          {loading ? "Processing..." : "Rollback"}
        </button>
      </div>
      {message ? <p className="text-xs text-white/60">{message}</p> : null}
    </div>
  );
}
