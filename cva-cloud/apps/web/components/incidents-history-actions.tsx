"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "@/components/ui/toast";

export default function IncidentsHistoryActions() {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const clearHistory = async () => {
    const confirmed = window.confirm(
      "Clear history? This will remove dismissed, fixed, and failed incidents."
    );
    if (!confirmed) {
      return;
    }
    setLoading(true);
    const res = await fetch("/api/v1/incidents/history/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    setLoading(false);
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      const count = typeof data.deleted === "number" ? data.deleted : 0;
      toast(`History cleared (${count}).`, "success");
      router.refresh();
      return;
    }
    toast("Failed to clear history.", "error");
  };

  return (
    <button
      type="button"
      onClick={clearHistory}
      disabled={loading}
      className="rounded-full border border-white/20 px-4 py-2 text-xs uppercase tracking-[0.3em] text-white/70 hover:border-white/40 disabled:opacity-60"
    >
      Clear history
    </button>
  );
}
