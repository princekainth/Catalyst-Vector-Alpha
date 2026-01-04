"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";

import { API_BASE } from "@/lib/api";


export default function IncidentActionsList({ incidentId }: { incidentId: string }) {
  const router = useRouter();
  const { getToken } = useAuth();
  const [loading, setLoading] = useState(false);

  const callAction = async (path: string, method = "POST", body?: Record<string, unknown>) => {
    setLoading(true);
    const token = await getToken();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    setLoading(false);
    if (res.ok) {
      router.refresh();
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => callAction(`/api/v1/incidents/${incidentId}/approve/`)}
        disabled={loading}
        className="rounded-md bg-warning/20 px-3 py-1 text-xs font-semibold text-warning disabled:opacity-60"
      >
        Approve
      </button>
      <button
        type="button"
        onClick={() => callAction(`/api/v1/incidents/${incidentId}/rollback/`)}
        disabled={loading}
        className="rounded-md border border-white/10 px-3 py-1 text-xs text-white/60 disabled:opacity-60"
      >
        Rollback
      </button>
      <button
        type="button"
        onClick={() => callAction(`/api/v1/incidents/${incidentId}/`, "PATCH", { status: "dismissed" })}
        disabled={loading}
        className="absolute right-4 top-4 rounded-full border border-danger/40 px-2 py-1 text-xs text-danger hover:bg-danger/10 disabled:opacity-60"
        aria-label="Dismiss incident"
      >
        ✕
      </button>
    </div>
  );
}
