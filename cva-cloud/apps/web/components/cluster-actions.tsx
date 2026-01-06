"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
export default function ClusterActions({ clusterId }: { clusterId: string }) {
  const router = useRouter();
  const [deleting, setDeleting] = useState(false);
  const [message, setMessage] = useState("");

  const handleDelete = async () => {
    if (!confirm("Delete this cluster? This will remove related incidents.")) {
      return;
    }
    setMessage("");
    setDeleting(true);
    const res = await fetch(`/api/v1/clusters/${clusterId}/`, {
      method: "DELETE",
    });
    setDeleting(false);
    if (res.ok) {
      router.refresh();
    } else {
      const text = await res.text();
      setMessage(text || "Failed to delete cluster.");
    }
  };

  return (
    <div className="space-y-1 text-xs text-white/70">
      <div className="flex gap-2">
        <Link
          href={`/clusters/${clusterId}`}
          className="rounded-md border border-white/10 px-2 py-1"
        >
          View
        </Link>
        <button
          type="button"
          onClick={handleDelete}
          disabled={deleting}
          className="rounded-md border border-white/10 px-2 py-1 disabled:opacity-60"
        >
          {deleting ? "Deleting..." : "Delete"}
        </button>
      </div>
      {message ? <p className="text-danger">{message}</p> : null}
    </div>
  );
}
