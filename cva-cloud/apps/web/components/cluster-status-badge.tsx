"use client";

import { Badge } from "@/components/ui/badge";


export const HEARTBEAT_WINDOW_MS = 90_000;
export const DISCONNECT_WINDOW_MS = 300_000;

function normalizeTimestamp(value: string): string {
  const cleaned = value.includes("T") ? value : value.replace(" ", "T");
  if (cleaned.includes("Z") || cleaned.match(/[+-]\d\d:\d\d$/)) {
    return cleaned;
  }
  return `${cleaned}Z`;
}

export function getEffectiveStatus(status: string, lastSeen?: string | null) {
  if (!lastSeen) return "disconnected";
  const normalized = normalizeTimestamp(lastSeen);
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "disconnected";
  const ageMs = Date.now() - date.getTime();
  if (ageMs > DISCONNECT_WINDOW_MS) return "disconnected";
  if (ageMs > HEARTBEAT_WINDOW_MS) return "stale";
  return status === "connected" ? "connected" : "connected";
}

export function isHeartbeatFresh(lastSeen?: string | null): boolean {
  if (!lastSeen) return false;
  const normalized = normalizeTimestamp(lastSeen);
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return false;
  return Date.now() - date.getTime() <= HEARTBEAT_WINDOW_MS;
}

export default function ClusterStatusBadge({
  status,
  lastSeen,
}: {
  status: string;
  lastSeen?: string | null;
}) {
  const effectiveStatus = getEffectiveStatus(status, lastSeen);
  const tone =
    effectiveStatus === "connected"
      ? "success"
      : effectiveStatus === "stale"
        ? "warning"
        : "danger";

  return (
    <div className="flex items-center gap-2">
      <span
        className={`h-2 w-2 rounded-full ${
          isHeartbeatFresh(lastSeen)
            ? "bg-accent animate-pulse"
            : effectiveStatus === "stale"
              ? "bg-warning"
              : "bg-white/30"
        }`}
      />
      <Badge tone={tone}>{effectiveStatus}</Badge>
    </div>
  );
}
