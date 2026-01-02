export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function ensureTrailingSlash(path: string): string {
  const [pathname, query] = path.split("?", 2);
  const normalized = pathname.endsWith("/") ? pathname : `${pathname}/`;
  return query ? `${normalized}?${query}` : normalized;
}

export async function fetcher<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${ensureTrailingSlash(path)}`, {
    cache: "no-store",
    credentials: "include",
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}`);
  }
  return res.json() as Promise<T>;
}
