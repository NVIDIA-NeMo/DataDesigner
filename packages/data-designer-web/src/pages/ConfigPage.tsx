import { useEffect, useState, useCallback } from "react";
import {
  CheckCircle2,
  XCircle,
  RefreshCw,
  FileText,
  Loader2,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { COLUMN_TYPE_META, ColumnType } from "../types/config";

interface ConfigFile {
  name: string;
  path: string;
  active: boolean;
}

interface ColumnInfo {
  name: string;
  column_type: string;
  drop: boolean;
}

export default function ConfigPage() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [yaml, setYaml] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState<{
    valid: boolean;
    message: string;
  } | null>(null);
  const [validating, setValidating] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const info = await api.getConfigInfo();
      setLoaded(info.loaded);
      setActivePath(info.path);
      setColumns(info.columns);
      setModels(info.models);
      if (info.loaded) {
        const y = await api.getConfigYaml();
        setYaml(y.content);
      } else {
        setYaml("");
      }
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
    try {
      const cfgs = await api.listConfigs();
      setConfigs(cfgs);
    } catch {
      // config listing is non-critical
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleLoad = async (path: string) => {
    setLoading(true);
    setError(null);
    setValidation(null);
    try {
      await api.loadConfig(path);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    setValidating(true);
    setValidation(null);
    try {
      const result = await api.validate();
      setValidation(result);
    } catch (e: any) {
      setValidation({ valid: false, message: e.message });
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Configuration</h1>
          <p className="text-sm text-gray-500 mt-1">
            Load a config file and inspect its contents
          </p>
        </div>
        <button className="btn-ghost" onClick={refresh} title="Refresh">
          <RefreshCw size={16} />
        </button>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Config file picker */}
      <div className="card mb-6">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="label-text">Config File</label>
            <select
              className="select-field"
              value={activePath ?? ""}
              onChange={(e) => e.target.value && handleLoad(e.target.value)}
              disabled={loading}
            >
              <option value="">Select a config file...</option>
              {configs.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          {loaded && (
            <div className="pt-5">
              <button
                className="btn-primary flex items-center gap-2"
                onClick={handleValidate}
                disabled={validating}
              >
                {validating ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <CheckCircle2 size={14} />
                )}
                Validate
              </button>
            </div>
          )}
        </div>
        {loading && (
          <div className="flex items-center gap-2 mt-3 text-sm text-gray-400">
            <Loader2 size={14} className="animate-spin" />
            Loading config...
          </div>
        )}
        {validation && (
          <div
            className={`mt-3 flex items-center gap-2 text-sm ${
              validation.valid ? "text-green-400" : "text-red-400"
            }`}
          >
            {validation.valid ? (
              <CheckCircle2 size={14} />
            ) : (
              <XCircle size={14} />
            )}
            {validation.message}
          </div>
        )}
      </div>

      {!loaded ? (
        <div className="card text-center py-12">
          <FileText size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">
            Select a config file above to get started
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-6">
          {/* Column summary */}
          <div className="col-span-1 space-y-4">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
              Columns ({columns.length})
            </h2>
            <div className="space-y-1.5">
              {columns.map((col) => {
                const meta =
                  COLUMN_TYPE_META[col.column_type as ColumnType];
                return (
                  <div
                    key={col.name}
                    className="flex items-center gap-2 text-sm"
                  >
                    <span
                      className={`text-xs px-1.5 py-0.5 rounded border font-medium shrink-0 ${
                        meta?.color ?? "bg-surface-3 text-gray-400 border-border"
                      }`}
                    >
                      {meta?.emoji ?? "?"}{" "}
                      {meta?.label ?? col.column_type}
                    </span>
                    <span className="text-gray-200 truncate">{col.name}</span>
                    {col.drop && (
                      <span className="text-xs text-gray-600">(drop)</span>
                    )}
                  </div>
                );
              })}
            </div>

            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider pt-4">
              Models ({models.length})
            </h2>
            <div className="space-y-1.5">
              {models.map((m: any) => (
                <div key={m.alias} className="text-sm">
                  <span className="text-nvidia-green font-medium">
                    {m.alias}
                  </span>
                  <span className="text-gray-500 ml-2 text-xs truncate">
                    {m.model}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* YAML viewer */}
          <div className="col-span-2">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
              Config YAML
            </h2>
            <div className="card bg-surface-0 overflow-auto max-h-[65vh]">
              <pre className="text-xs font-mono text-gray-300 leading-relaxed whitespace-pre">
                {yaml}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
