import type { ReactNode } from "react";
import Link from "next/link";

const nav = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/clusters", label: "Clusters" },
  { href: "/incidents", label: "Incidents" },
  { href: "/policies", label: "Policies" },
];

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-background text-white">
      <div className="grid grid-cols-[240px_1fr]">
        <aside className="min-h-screen border-r border-white/10 px-6 py-8">
          <Link href="/" className="block">
            <h1 className="text-xl font-display">CVA Cloud</h1>
            <p className="mt-2 text-xs text-white/60">Mission Control</p>
          </Link>
          <nav className="mt-8 space-y-2 text-sm">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="block rounded-md px-3 py-2 text-white/80 hover:bg-white/10"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </aside>
        <main className="px-8 py-8">{children}</main>
      </div>
    </div>
  );
}
