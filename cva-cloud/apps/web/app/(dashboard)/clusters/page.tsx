import Link from "next/link";
import SectionHeader from "@/components/section-header";
import ClustersTableClient from "@/components/clusters-table-client";

export default async function ClustersPage({
  searchParams,
}: {
  searchParams?: { connected?: string };
}) {
  const showConnected = searchParams?.connected === "1";

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <SectionHeader title="Clusters">Connect and manage your Kubernetes clusters.</SectionHeader>
          {showConnected ? (
            <p className="text-sm text-accent">Cluster connected successfully.</p>
          ) : null}
        </div>
        <Link
          href="/clusters/new"
          className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black"
        >
          + Connect Cluster
        </Link>
      </div>

      <ClustersTableClient />
    </div>
  );
}
