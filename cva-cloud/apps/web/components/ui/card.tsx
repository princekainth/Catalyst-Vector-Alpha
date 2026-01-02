import type { ReactNode } from "react";
import clsx from "clsx";

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("rounded-xl bg-panel/80 border border-white/10 p-5", className)}>
      {children}
    </div>
  );
}

export function CardTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-sm font-semibold text-white/80">{children}</h3>;
}

export function CardValue({ children }: { children: ReactNode }) {
  return <p className="mt-3 text-3xl font-display">{children}</p>;
}
