"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";

import { API_BASE } from "@/lib/api";
import { Card } from "@/components/ui/card";
import SectionHeader from "@/components/section-header";

type InstallResponse = {
  cluster_id: string;
  api_key: string;
  install_command: string;
};

export default function NewClusterPage() {
  const router = useRouter();
  const { getToken } = useAuth();
  const [name, setName] = useState("");
  const [install, setInstall] = useState<InstallResponse | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState("");

  const createCluster = async () => {
    setError("");
    const token = await getToken();
    if (!token) {
      setError("Please sign in to continue.");
      return;
    }
    const res = await fetch(`${API_BASE}/api/v1/clusters/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) {
      const text = await res.text();
      setError(text || "Failed to create cluster.");
      return;
    }
    const data = (await res.json()) as InstallResponse;
    setInstall(data);
  };

  const checkConnection = async () => {
    if (!install) return;
    setChecking(true);
    const token = await getToken();
    if (!token) {
      setError("Please sign in to continue.");
      setChecking(false);
      return;
    }
    const res = await fetch(`${API_BASE}/api/v1/clusters/${install.cluster_id}/`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    setChecking(false);
    if (!res.ok) {
      setError("Unable to check cluster status.");
      return;
    }
    const data = await res.json();
    if (data.status === "connected") {
      router.push("/clusters?connected=1");
    } else {
      setError(`Cluster status: ${data.status}`);
    }
  };

  return (
    <div className="space-y-8">
      <SectionHeader title="Connect a Cluster">
        Generate an install command for your Kubernetes cluster.
      </SectionHeader>

      <Card className="space-y-4">
        <label className="text-sm text-white/70">Cluster name</label>
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="prod-us-east"
          className="w-full rounded-md border border-white/10 bg-black/30 px-3 py-2 text-sm"
        />
        <button
          className="w-fit rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black"
          onClick={createCluster}
          disabled={!name}
        >
          Generate Install Command
        </button>
        {error ? <p className="text-sm text-danger">{error}</p> : null}
      </Card>

      {install ? (
        <Card className="space-y-4">
          <h3 className="text-lg font-display">Install Instructions</h3>
          <ol className="list-decimal space-y-2 pl-4 text-sm text-white/70">
            <li>Copy the command below.</li>
            <li>Run it in your cluster.</li>
            <li>Wait about 30 seconds for the agent to connect.</li>
          </ol>
          <code className="block rounded-md bg-black/40 p-3 text-xs text-white/80">
            {install.install_command}
          </code>
          <button
            className="w-fit rounded-md border border-white/10 px-4 py-2 text-sm"
            onClick={checkConnection}
            disabled={checking}
          >
            {checking ? "Checking..." : "Check Connection"}
          </button>
        </Card>
      ) : null}
    </div>
  );
}
