import type { ReactNode } from "react";
import Link from "next/link";
import DashboardNav from "@/components/dashboard-nav";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-background text-white">
      <div className="grid grid-cols-[240px_1fr]">
        <aside className="min-h-screen border-r border-white/10 px-6 py-8">
          <Link
            href="/"
            className="block rounded-2xl border border-white/15 bg-white/5 px-4 py-3 backdrop-blur-lg transition hover:border-white/25 hover:bg-white/10"
          >
            <h1 className="text-2xl font-display font-semibold tracking-wide">CVA Cloud</h1>
            <p className="mt-1 text-sm text-white/70">Mission Control</p>
          </Link>
          <DashboardNav />
        </aside>
        <main className="px-8 py-8">{children}</main>
      </div>
    </div>
  );
}
