"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const nav = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/clusters", label: "Clusters" },
  { href: "/incidents", label: "Incidents" },
  { href: "/policies", label: "Policies" },
];

export default function DashboardNav() {
  const pathname = usePathname();

  return (
    <nav className="mt-8 space-y-2 text-sm">
      {nav.map((item) => {
        const isActive = pathname === item.href || pathname?.startsWith(`${item.href}/`);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`group flex items-center justify-between rounded-md border px-3 py-2 backdrop-blur-md transition ${
              isActive
                ? "border-white/30 bg-white/15 text-white shadow-[0_0_12px_rgba(255,255,255,0.08)]"
                : "border-white/10 bg-white/5 text-white/70 hover:border-white/20 hover:bg-white/10 hover:text-white"
            }`}
          >
            <span className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full transition ${
                  isActive ? "bg-accent" : "bg-white/30 group-hover:bg-white/60"
                }`}
              />
              {item.label}
            </span>
            <span className="text-xs text-white/40 group-hover:text-white/70">↗</span>
          </Link>
        );
      })}
    </nav>
  );
}
