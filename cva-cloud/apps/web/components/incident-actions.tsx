"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "@/components/ui/toast";
export default function IncidentActions({ incidentId }: { incidentId: string }) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const approveIncident = async () => {
    setLoading(true);
    setMessage("");
    const res = await fetch(`/api/v1/incidents/${incidentId}/approve`, {
      method: "POST",
    });
    setLoading(false);
    if (res.ok) {
      setMessage("Approved. Awaiting agent execution.");
      toast("Approved. Awaiting agent execution.", "success");
      router.refresh();
    } else {
      setMessage("Failed to approve incident.");
      toast("Failed to approve incident.", "error");
    }
  };

  const rollbackIncident = async () => {
    setLoading(true);
    setMessage("");
    const res = await fetch(`/api/v1/incidents/${incidentId}/rollback`, {
      method: "POST",
    });
    setLoading(false);
    if (res.ok) {
      setMessage("Rollback requested.");
      toast("Rollback requested.", "info");
      router.refresh();
    } else {
      setMessage("Failed to request rollback.");
      toast("Failed to request rollback.", "error");
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
