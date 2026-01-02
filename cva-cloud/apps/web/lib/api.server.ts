import "server-only";

import { auth } from "@clerk/nextjs/server";


export async function getAuthHeaders(): Promise<Record<string, string>> {
  const token = await auth().getToken();
  if (!token) {
    return {};
  }
  return { Authorization: `Bearer ${token}` };
}
