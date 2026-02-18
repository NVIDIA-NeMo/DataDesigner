const API_BASE = "/api";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
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
  // Schema
  getColumnSchemas: () => request<Record<string, unknown>>("/schema/columns"),
  getEnums: () => request<Record<string, string[]>>("/schema/enums"),

  // Config
  getConfig: () => request<Record<string, unknown>>("/config"),
  loadConfig: (config: Record<string, unknown>) =>
    request<Record<string, unknown>>("/config", {
      method: "PUT",
      body: JSON.stringify({ config }),
    }),
  resetConfig: () => request<{ status: string }>("/config", { method: "DELETE" }),
  exportConfig: (format: "yaml" | "json" = "yaml") =>
    request<{ format: string; content: string }>(
      `/config/export?format=${format}`
    ),

  // Columns
  listColumns: () => request<Record<string, unknown>[]>("/config/columns"),
  addColumn: (data: Record<string, unknown>) =>
    request<Record<string, unknown>>("/config/columns", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  updateColumn: (name: string, data: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/config/columns/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  deleteColumn: (name: string) =>
    request<{ status: string }>(`/config/columns/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  // Models
  listModels: () => request<Record<string, unknown>[]>("/config/models"),
  addModel: (data: Record<string, unknown>) =>
    request<Record<string, unknown>>("/config/models", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  deleteModel: (alias: string) =>
    request<{ status: string }>(`/config/models/${encodeURIComponent(alias)}`, {
      method: "DELETE",
    }),

  // References
  getReferences: () => request<string[]>("/references"),

  // Health
  health: () => request<{ status: string }>("/health"),
};
