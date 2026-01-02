import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-black via-[#0b0f14] to-[#101726] flex items-center justify-center">
      <div className="max-w-xl text-center space-y-6">
        <h1 className="text-4xl font-display">CVA Mission Control</h1>
        <p className="text-white/70">
          AI-driven remediation for Kubernetes incidents. Start with the dashboard.
        </p>
        <Link
          href="/dashboard"
          className="inline-flex items-center justify-center rounded-md bg-accent px-4 py-2 text-sm font-semibold text-black"
        >
          Open Dashboard
        </Link>
      </div>
    </main>
  );
}
