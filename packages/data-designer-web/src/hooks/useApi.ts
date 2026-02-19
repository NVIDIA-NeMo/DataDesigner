const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Session
  getSession: () =>
    request<{
      file_name: string;
      file_path: string;
      row_count: number;
      columns: string[];
      all_columns: string[];
      version: number;
      finished: boolean;
    }>("/session"),
  getRows: () => request<Record<string, unknown>[]>("/session/rows"),
  getTrace: (row: number, column: string) =>
    request<Record<string, unknown>[]>(
      `/session/traces/${row}/${encodeURIComponent(column)}`
    ),
  reload: () => request<Record<string, unknown>>("/session/reload", { method: "POST" }),
  finish: () => request<Record<string, unknown>>("/session/finish", { method: "POST" }),

  // Annotations
  getAnnotations: () =>
    request<Record<string, { rating: string | null; note: string; columns: Record<string, { rating: string | null; note: string }> }>>(
      "/annotations"
    ),
  annotateRow: (row: number, rating: string | null, note: string) =>
    request<{ status: string }>("/annotations", {
      method: "POST",
      body: JSON.stringify({ row, rating, note }),
    }),
  annotateColumn: (row: number, column: string, rating: string | null, note: string) =>
    request<{ status: string }>("/annotations/column", {
      method: "POST",
      body: JSON.stringify({ row, column, rating, note }),
    }),
  getAnnotationsSummary: () =>
    request<{ good: number; bad: number; unreviewed: number; total: number }>(
      "/annotations/summary"
    ),

  // Health
  health: () => request<{ status: string }>("/health"),
};
