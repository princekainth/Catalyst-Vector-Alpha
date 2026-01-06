import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8001";

export async function GET() {
  const token = await auth().getToken();
  if (!token) {
    return NextResponse.json({ detail: "Unauthorized" }, { status: 401 });
  }
  const res = await fetch(`${API_BASE}/api/v1/incidents/archived`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "Content-Type": res.headers.get("content-type") || "application/json" },
  });
}
