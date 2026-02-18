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
  // Config discovery & loading
  listConfigs: () => request<{ name: string; path: string; active: boolean }[]>("/configs"),
  loadConfig: (path: string) =>
    request<Record<string, unknown>>("/config/load", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  getConfig: () => request<Record<string, unknown>>("/config"),
  getConfigYaml: () => request<{ content: string }>("/config/yaml"),
  saveConfig: (content: string) =>
    request<Record<string, unknown>>("/config/save", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  getConfigInfo: () =>
    request<{
      loaded: boolean;
      path: string | null;
      columns: { name: string; column_type: string; drop: boolean }[];
      models: Record<string, unknown>[];
      output_schema?: { name: string; column_type: string; drop: boolean; in_output: boolean; side_effect_of?: string }[];
    }>("/config/info"),
  exportConfig: (format: "yaml" | "json" = "yaml") =>
    request<{ format: string; content: string }>(`/config/export?format=${format}`),

  // Read-only lists
  listColumns: () => request<{ name: string; column_type: string; drop: boolean }[]>("/config/columns"),
  listModels: () => request<Record<string, unknown>[]>("/config/models"),

  // Execution
  validate: () => request<{ valid: boolean; message: string }>("/config/validate", { method: "POST" }),
  reviewConfig: (modelAlias: string) =>
    request<{
      static_issues: { level: string; type: string; column: string | null; message: string }[];
      llm_tips: { category: string; severity: string; column: string | null; tip: string }[];
      model_used: string;
    }>("/config/review", {
      method: "POST",
      body: JSON.stringify({ model_alias: modelAlias }),
    }),
  runPreview: (numRecords: number, debugMode: boolean = false) =>
    request<{ status: string }>("/preview", {
      method: "POST",
      body: JSON.stringify({ num_records: numRecords, debug_mode: debugMode }),
    }),
  runCreate: (numRecords: number, datasetName: string = "dataset", artifactPath?: string) =>
    request<{ status: string }>("/create", {
      method: "POST",
      body: JSON.stringify({ num_records: numRecords, dataset_name: datasetName, artifact_path: artifactPath }),
    }),
  getStatus: () =>
    request<{
      state: "idle" | "running" | "done" | "error";
      type: string | null;
      error: string | null;
      has_preview: boolean;
      has_create: boolean;
    }>("/status"),

  // Results
  getPreviewResults: () =>
    request<{
      columns: string[];
      rows: Record<string, unknown>[];
      analysis: unknown;
      row_count: number;
    }>("/preview/results"),
  getTrace: (row: number, column: string) =>
    request<Record<string, unknown>[]>(`/preview/traces/${row}/${encodeURIComponent(column)}`),
  getCreateResults: () =>
    request<{ num_records?: number; artifact_path?: string; columns?: string[] }>("/create/results"),

  // Logs
  getLogs: (since: number = 0) =>
    request<{ ts: number; level: string; name: string; message: string }[]>(
      `/logs?since=${since}`
    ),

  // Utility
  health: () => request<{ status: string }>("/health"),
};
