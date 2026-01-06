import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8001";

export async function PATCH(
  request: Request,
  { params }: { params: { incidentId: string } },
) {
  const token = await auth().getToken();
  if (!token) {
    return NextResponse.json({ detail: "Unauthorized" }, { status: 401 });
  }
  const body = await request.text();
  const res = await fetch(`${API_BASE}/api/v1/incidents/${params.incidentId}/`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body,
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "Content-Type": res.headers.get("content-type") || "application/json" },
  });
}
