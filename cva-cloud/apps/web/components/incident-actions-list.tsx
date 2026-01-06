"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "@/components/ui/toast";
export default function IncidentActionsList({ incidentId }: { incidentId: string }) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [messageTone, setMessageTone] = useState<"default" | "error">("default");

  const callAction = async (path: string, method = "POST", body?: Record<string, unknown>) => {
    setLoading(true);
    setMessage("");
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const res = await fetch(path, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    setLoading(false);
    if (res.ok) {
      router.refresh();
      return true;
    }
    return false;
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={async () => {
          const ok = await callAction(`/api/v1/incidents/${incidentId}/approve`);
          if (ok) {
            setMessageTone("default");
            setMessage("Approved. Awaiting agent execution.");
            toast("Approved. Awaiting agent execution.", "success");
          } else {
            setMessageTone("error");
            setMessage("Failed to approve incident.");
            toast("Failed to approve incident.", "error");
          }
        }}
        disabled={loading}
        className="rounded-md bg-warning/20 px-3 py-1 text-xs font-semibold text-warning disabled:opacity-60"
      >
        Approve
      </button>
      <button
        type="button"
        onClick={async () => {
          const ok = await callAction(`/api/v1/incidents/${incidentId}/rollback`);
          if (ok) {
            setMessageTone("default");
            setMessage("Rollback requested.");
            toast("Rollback requested.", "info");
          } else {
            setMessageTone("error");
            setMessage("Failed to request rollback.");
            toast("Failed to request rollback.", "error");
          }
        }}
        disabled={loading}
        className="rounded-md border border-white/10 px-3 py-1 text-xs text-white/60 disabled:opacity-60"
      >
        Rollback
      </button>
      <button
        type="button"
        onClick={async () => {
          const ok = await callAction(`/api/v1/incidents/${incidentId}/`, "PATCH", { status: "dismissed" });
          if (ok) {
            setMessageTone("default");
            setMessage("Dismissed.");
            toast("Incident dismissed.", "info");
          } else {
            setMessageTone("error");
            setMessage("Failed to dismiss incident.");
            toast("Failed to dismiss incident.", "error");
          }
        }}
        disabled={loading}
        className="absolute right-4 top-4 rounded-full border border-danger/40 px-2 py-1 text-xs text-danger hover:bg-danger/10 disabled:opacity-60"
        aria-label="Dismiss incident"
      >
        ✕
      </button>
      {message ? (
        <span className={messageTone === "error" ? "text-danger" : "text-white/60"}>
          {message}
        </span>
      ) : null}
    </div>
  );
}
