import type { ReactNode } from "react";

export default function SectionHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h2 className="text-xl font-display">{title}</h2>
        {children ? <p className="text-sm text-white/60">{children}</p> : null}
      </div>
    </div>
  );
}
