"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";

import { API_BASE } from "@/lib/api";


export default function ClusterActions({ clusterId }: { clusterId: string }) {
  const router = useRouter();
  const { getToken } = useAuth();
  const [deleting, setDeleting] = useState(false);
  const [message, setMessage] = useState("");

  const handleDelete = async () => {
    if (!confirm("Delete this cluster? This will remove related incidents.")) {
      return;
    }
    setMessage("");
    setDeleting(true);
    const token = await getToken();
    if (!token) {
      setMessage("Sign in to delete clusters.");
      setDeleting(false);
      return;
    }
    const res = await fetch(`${API_BASE}/api/v1/clusters/${clusterId}/`, {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${token}`,
      },
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
