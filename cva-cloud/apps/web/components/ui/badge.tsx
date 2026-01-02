import type { ReactNode } from "react";
import clsx from "clsx";

const tones: Record<string, string> = {
  success: "bg-accent/20 text-accent",
  warning: "bg-warning/20 text-warning",
  danger: "bg-danger/20 text-danger",
  neutral: "bg-white/10 text-white/70",
};

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: keyof typeof tones }) {
  return (
    <span className={clsx("rounded-full px-3 py-1 text-xs font-semibold", tones[tone])}>
      {children}
    </span>
  );
}
