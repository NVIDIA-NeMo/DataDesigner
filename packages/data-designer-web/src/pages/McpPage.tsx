import { useEffect, useState, useCallback } from "react";
import {
  Wrench,
  CheckCircle2,
  XCircle,
  Plus,
  Trash2,
  Terminal,
  Globe,
  Info,
  RefreshCw,
} from "lucide-react";
import { api } from "../hooks/useApi";

interface McpStatus {
  required: { name: string; configured: boolean }[];
  configured: Record<string, unknown>[];
  all_satisfied: boolean;
}

export default function McpPage() {
  const [status, setStatus] = useState<McpStatus>({
    required: [],
    configured: [],
    all_satisfied: true,
  });
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  // Form state
  const [formType, setFormType] = useState<"stdio" | "sse">("stdio");
  const [formName, setFormName] = useState("");
  const [formCommand, setFormCommand] = useState("");
  const [formArgs, setFormArgs] = useState("");
  const [formEnv, setFormEnv] = useState("");
  const [formEndpoint, setFormEndpoint] = useState("");
  const [formApiKey, setFormApiKey] = useState("");

  const refresh = useCallback(async () => {
    try {
      const s = await api.getMcpStatus();
      setStatus(s);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const resetForm = () => {
    setFormName("");
    setFormCommand("");
    setFormArgs("");
    setFormEnv("");
    setFormEndpoint("");
    setFormApiKey("");
  };

  const handleAdd = async () => {
    setError(null);
    try {
      const data: Record<string, unknown> = {
        provider_type: formType,
        name: formName,
      };
      if (formType === "sse") {
        data.endpoint = formEndpoint;
        if (formApiKey.trim()) data.api_key = formApiKey;
      } else {
        data.command = formCommand;
        if (formArgs.trim()) {
          data.args = formArgs.split(",").map((a) => a.trim()).filter(Boolean);
        }
        if (formEnv.trim()) {
          const env: Record<string, string> = {};
          formEnv.split(",").forEach((pair) => {
            const [k, ...v] = pair.split("=");
            if (k && v.length) env[k.trim()] = v.join("=").trim();
          });
          data.env = env;
        }
      }
      await api.addMcpProvider(data);
      resetForm();
      setShowForm(false);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleDelete = async (name: string) => {
    await api.deleteMcpProvider(name);
    await refresh();
  };

  const openFormForMissing = (name: string) => {
    setShowForm(true);
    setFormName(name);
    resetForm();
    setFormName(name);
  };

  const configured = status.configured as any[];

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">MCP Providers</h1>
          <p className="text-sm text-gray-500 mt-1">
            Connect to MCP tool servers for LLM tool-calling columns
          </p>
        </div>
        <button className="btn-ghost" onClick={refresh} title="Refresh">
          <RefreshCw size={16} />
        </button>
      </div>

      {/* Explanation banner */}
      <div className="card bg-surface-0 border-blue-700/20 mb-6 flex items-start gap-3">
        <Info size={16} className="text-blue-400 mt-0.5 shrink-0" />
        <div className="text-xs text-gray-400 space-y-1">
          <p>
            <span className="text-gray-300 font-medium">Local (stdio)</span> --
            Data Designer launches the server as a subprocess. You don't need to
            run it separately.
          </p>
          <p>
            <span className="text-gray-300 font-medium">Remote (SSE)</span> --
            Connects to an already-running MCP server endpoint.
          </p>
        </div>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Required by config */}
      {status.required.length > 0 && (
        <div className="mb-6">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">
            Required by Config
          </h2>
          {status.all_satisfied ? (
            <div className="card border-green-700/30 flex items-center gap-2">
              <CheckCircle2 size={16} className="text-green-400" />
              <span className="text-sm text-green-300">
                All required providers are configured
              </span>
            </div>
          ) : (
            <div className="space-y-2">
              {status.required.map((p) => (
                <div
                  key={p.name}
                  className={`card flex items-center gap-3 ${
                    p.configured ? "border-green-700/30" : "border-red-700/30"
                  }`}
                >
                  {p.configured ? (
                    <CheckCircle2 size={16} className="text-green-400 shrink-0" />
                  ) : (
                    <XCircle size={16} className="text-red-400 shrink-0" />
                  )}
                  <span
                    className={`text-sm font-medium font-mono ${
                      p.configured ? "text-green-300" : "text-red-300"
                    }`}
                  >
                    {p.name}
                  </span>
                  {!p.configured && (
                    <button
                      className="btn-primary ml-auto !py-1 !px-3 !text-xs"
                      onClick={() => openFormForMissing(p.name)}
                    >
                      Configure
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Configured providers */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-300">
            Configured Providers ({configured.length})
          </h2>
          <button
            className="btn-secondary flex items-center gap-1.5 !text-xs"
            onClick={() => {
              resetForm();
              setShowForm(true);
            }}
          >
            <Plus size={14} />
            Add Provider
          </button>
        </div>

        {configured.length === 0 ? (
          <div className="card text-center py-8">
            <Wrench size={28} className="text-gray-600 mx-auto mb-2" />
            <p className="text-sm text-gray-500">No MCP providers configured</p>
            <p className="text-xs text-gray-600 mt-1">
              Add a provider to enable tool calling in your LLM columns
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {configured.map((p: any) => (
              <div key={p.name} className="card flex items-start gap-3">
                {p.provider_type === "stdio" ? (
                  <Terminal size={16} className="text-cyan-400 mt-0.5 shrink-0" />
                ) : (
                  <Globe size={16} className="text-purple-400 mt-0.5 shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-200 font-mono">
                      {p.name}
                    </span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wider ${
                        p.provider_type === "stdio"
                          ? "bg-cyan-900/30 text-cyan-400 border border-cyan-700/30"
                          : "bg-purple-900/30 text-purple-400 border border-purple-700/30"
                      }`}
                    >
                      {p.provider_type === "stdio" ? "Local" : "Remote"}
                    </span>
                  </div>
                  {p.provider_type === "stdio" ? (
                    <p className="text-xs text-gray-500 mt-1 font-mono">
                      {p.command}
                      {p.args?.length > 0 && ` ${p.args.join(" ")}`}
                    </p>
                  ) : (
                    <p className="text-xs text-gray-500 mt-1 font-mono truncate">
                      {p.endpoint}
                    </p>
                  )}
                  {p.env && Object.keys(p.env).length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {Object.entries(p.env).map(([k, v]) => (
                        <span
                          key={k}
                          className="text-[10px] bg-surface-3 text-gray-500 px-1.5 py-0.5 rounded font-mono"
                        >
                          {k}={String(v)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  className="btn-ghost text-gray-600 hover:text-red-400"
                  onClick={() => handleDelete(p.name)}
                  title="Delete provider"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add provider form */}
      {showForm && (
        <div className="card border-nvidia-green/30">
          <h3 className="text-sm font-semibold text-gray-200 mb-4">
            Add MCP Provider
          </h3>

          {/* Type selector */}
          <div className="mb-4">
            <label className="label-text">Provider Type</label>
            <div className="flex gap-2 mt-1">
              <button
                className={`flex-1 flex items-center gap-2 px-3 py-2.5 rounded-md border text-sm transition-colors ${
                  formType === "stdio"
                    ? "border-cyan-700/50 bg-cyan-900/20 text-cyan-300"
                    : "border-border bg-surface-2 text-gray-400 hover:text-gray-200"
                }`}
                onClick={() => setFormType("stdio")}
              >
                <Terminal size={16} />
                <div className="text-left">
                  <div className="font-medium">Local subprocess</div>
                  <div className="text-[10px] opacity-70">
                    Data Designer launches the server for you
                  </div>
                </div>
              </button>
              <button
                className={`flex-1 flex items-center gap-2 px-3 py-2.5 rounded-md border text-sm transition-colors ${
                  formType === "sse"
                    ? "border-purple-700/50 bg-purple-900/20 text-purple-300"
                    : "border-border bg-surface-2 text-gray-400 hover:text-gray-200"
                }`}
                onClick={() => setFormType("sse")}
              >
                <Globe size={16} />
                <div className="text-left">
                  <div className="font-medium">Remote endpoint</div>
                  <div className="text-[10px] opacity-70">
                    Connect to an already-running server
                  </div>
                </div>
              </button>
            </div>
          </div>

          {/* Name */}
          <div className="mb-4">
            <label className="label-text">Provider Name</label>
            <input
              className="input-field"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
              placeholder="e.g. basic-tools"
            />
            <p className="text-[10px] text-gray-600 mt-1">
              Must match the provider name in your config's tool_configs
            </p>
          </div>

          {formType === "stdio" ? (
            <>
              {/* Command */}
              <div className="mb-4">
                <label className="label-text">Command</label>
                <input
                  className="input-field font-mono"
                  value={formCommand}
                  onChange={(e) => setFormCommand(e.target.value)}
                  placeholder="python"
                />
                <p className="text-[10px] text-gray-600 mt-1">
                  The executable to run, e.g. <code className="text-gray-500">python</code>,{" "}
                  <code className="text-gray-500">node</code>,{" "}
                  <code className="text-gray-500">uv run</code>
                </p>
              </div>

              {/* Args */}
              <div className="mb-4">
                <label className="label-text">Command Arguments</label>
                <input
                  className="input-field font-mono"
                  value={formArgs}
                  onChange={(e) => setFormArgs(e.target.value)}
                  placeholder="path/to/server.py, serve"
                />
                <p className="text-[10px] text-gray-600 mt-1">
                  Comma-separated. Each value becomes a separate argument.
                </p>
              </div>

              {/* Env */}
              <div className="mb-4">
                <label className="label-text">
                  Environment Variables{" "}
                  <span className="text-gray-600 font-normal">(optional)</span>
                </label>
                <input
                  className="input-field font-mono"
                  value={formEnv}
                  onChange={(e) => setFormEnv(e.target.value)}
                  placeholder="KEY=value, ANOTHER=value"
                />
                <p className="text-[10px] text-gray-600 mt-1">
                  Comma-separated KEY=VALUE pairs passed to the subprocess.
                </p>
              </div>
            </>
          ) : (
            <>
              {/* Endpoint */}
              <div className="mb-4">
                <label className="label-text">Server URL</label>
                <input
                  className="input-field font-mono"
                  value={formEndpoint}
                  onChange={(e) => setFormEndpoint(e.target.value)}
                  placeholder="http://localhost:8080/sse"
                />
                <p className="text-[10px] text-gray-600 mt-1">
                  The SSE endpoint URL of the running MCP server
                </p>
              </div>

              {/* API Key */}
              <div className="mb-4">
                <label className="label-text">
                  API Key{" "}
                  <span className="text-gray-600 font-normal">(optional)</span>
                </label>
                <input
                  className="input-field font-mono"
                  type="password"
                  value={formApiKey}
                  onChange={(e) => setFormApiKey(e.target.value)}
                  placeholder="sk-..."
                />
              </div>
            </>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2 border-t border-border">
            <button
              className="btn-primary flex items-center gap-1.5"
              onClick={handleAdd}
              disabled={
                !formName ||
                (formType === "sse" ? !formEndpoint : !formCommand)
              }
            >
              <Plus size={14} />
              Add Provider
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                setShowForm(false);
                resetForm();
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
