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

  const handleDelete = async () => {
    if (!confirm("Delete this cluster? This will remove related incidents.")) {
      return;
    }
    setDeleting(true);
    const token = await getToken();
    if (!token) {
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
    }
  };

  return (
    <div className="flex gap-2 text-xs text-white/70">
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
  );
}
