"use client";

import { useMemo } from "react";


function formatRelative(date: Date): string {
  const diffMs = Date.now() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  if (diffMinutes <= 0) {
    return "just now";
  }
  if (Math.abs(diffMinutes) < 60) {
    return `${Math.abs(diffMinutes)} minutes ago`;
  }
  if (Math.abs(diffHours) < 24) {
    return `${Math.abs(diffHours)} hours ago`;
  }
  return `${Math.abs(diffDays)} days ago`;
}

function normalizeTimestamp(value: string): string {
  const cleaned = value.includes("T") ? value : value.replace(" ", "T");
  if (cleaned.includes("Z") || cleaned.match(/[+-]\d\d:\d\d$/)) {
    return cleaned;
  }
  return `${cleaned}Z`;
}


export default function LocalTime({
  value,
  mode = "relative",
}: {
  value?: string | null;
  mode?: "relative" | "absolute";
}) {
  const label = useMemo(() => {
    if (!value) return "never";
    const normalized = normalizeTimestamp(value);
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return "unknown";
    if (mode === "absolute") {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
    }
    return formatRelative(date);
  }, [value, mode]);

  return <span>{label}</span>;
}
