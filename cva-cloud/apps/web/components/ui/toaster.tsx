"use client";

import { useEffect, useState } from "react";
import type { ToastTone } from "./toast";

type ToastItem = {
  id: string;
  message: string;
  tone: ToastTone;
};

const toneClasses: Record<ToastTone, string> = {
  success: "border-emerald-500/40 bg-emerald-500/15 text-emerald-100",
  error: "border-danger/40 bg-danger/20 text-danger",
  info: "border-white/15 bg-white/10 text-white/80",
};

export default function Toaster() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail || {};
      const message = String(detail.message || "").trim();
      if (!message) {
        return;
      }
      const tone = (detail.tone as ToastTone) || "info";
      const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      setToasts((prev) => [...prev, { id, message, tone }]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((item) => item.id !== id));
      }, 3200);
    };

    window.addEventListener("cva-toast", handler as EventListener);
    return () => window.removeEventListener("cva-toast", handler as EventListener);
  }, []);

  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="fixed right-4 top-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`rounded-lg border px-4 py-2 text-sm shadow-lg backdrop-blur ${toneClasses[toast.tone]}`}
        >
          {toast.message}
        </div>
      ))}
    </div>
  );
}
