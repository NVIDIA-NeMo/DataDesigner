import { useState, useEffect, useRef } from "react";
import {
  Eye,
  HardDrive,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Terminal,
  Wrench,
} from "lucide-react";
import { api } from "../hooks/useApi";

type ExecState = "idle" | "running" | "done" | "error";

interface LogEntry {
  ts: number;
  level: string;
  name: string;
  message: string;
}

const LEVEL_COLORS: Record<string, string> = {
  ERROR: "text-red-400",
  WARNING: "text-amber-400",
  INFO: "text-gray-300",
  DEBUG: "text-gray-500",
};

export default function RunPage() {
  const [numRecords, setNumRecords] = useState(10);
  const [debugMode, setDebugMode] = useState(true);
  const [datasetName, setDatasetName] = useState("dataset");
  const [state, setState] = useState<ExecState>("idle");
  const [execType, setExecType] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [mcpMissing, setMcpMissing] = useState<string[]>([]);
  const [createResult, setCreateResult] = useState<{
    num_records?: number;
    artifact_path?: string;
  } | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const pollRef = useRef<number | null>(null);
  const logSinceRef = useRef<number>(0);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getConfigInfo().then((info) => setConfigLoaded(info.loaded));
    api.getMcpStatus().then((s) => {
      setMcpMissing(s.required.filter((p) => !p.configured).map((p) => p.name));
    }).catch(() => {});
    api.getStatus().then(async (s) => {
      setState(s.state);
      setExecType(s.type);
      setError(s.error);

      // Load existing logs from the backend buffer
      const existingLogs = await api.getLogs(0);
      if (existingLogs.length > 0) {
        setLogs(existingLogs);
        logSinceRef.current = existingLogs[existingLogs.length - 1].ts;
      }

      // Resume polling if a run is still in progress
      if (s.state === "running") {
        pollStatus();
      }
    });
  }, []);

  const pollStatus = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      const [s, newLogs] = await Promise.all([
        api.getStatus(),
        api.getLogs(logSinceRef.current),
      ]);

      if (newLogs.length > 0) {
        setLogs((prev) => [...prev, ...newLogs]);
        logSinceRef.current = newLogs[newLogs.length - 1].ts;
      }

      setState(s.state);
      setExecType(s.type);
      setError(s.error);

      if (s.state !== "running") {
        // One final log fetch to capture any remaining entries
        const finalLogs = await api.getLogs(logSinceRef.current);
        if (finalLogs.length > 0) {
          setLogs((prev) => [...prev, ...finalLogs]);
        }
        clearInterval(pollRef.current!);
        pollRef.current = null;
        if (s.state === "done" && s.type === "create") {
          const cr = await api.getCreateResults();
          setCreateResult(cr);
        }
      }
    }, 1000);
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Auto-scroll log panel
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handlePreview = async () => {
    setError(null);
    setCreateResult(null);
    setLogs([]);
    logSinceRef.current = 0;
    try {
      await api.runPreview(numRecords, debugMode);
      setState("running");
      setExecType("preview");
      pollStatus();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleCreate = async () => {
    setError(null);
    setCreateResult(null);
    setLogs([]);
    logSinceRef.current = 0;
    try {
      await api.runCreate(numRecords, datasetName);
      setState("running");
      setExecType("create");
      pollStatus();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const isRunning = state === "running";

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-gray-100">Run</h1>
        <p className="text-sm text-gray-500 mt-1">
          Execute preview or full dataset creation
        </p>
      </div>

      {!configLoaded && (
        <div className="card border-amber-700/50 bg-amber-900/20 mb-4 flex items-start gap-2">
          <AlertCircle size={16} className="text-amber-400 mt-0.5 shrink-0" />
          <p className="text-sm text-amber-300">
            No config loaded. Go to the Config page to load one first.
          </p>
        </div>
      )}

      {mcpMissing.length > 0 && (
        <div className="card border-amber-700/50 bg-amber-900/20 mb-4 flex items-start gap-2">
          <Wrench size={16} className="text-amber-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm text-amber-300">
              Missing MCP providers: <span className="font-mono font-semibold">{mcpMissing.join(", ")}</span>
            </p>
            <p className="text-xs text-amber-400/70 mt-1">
              Your config uses tool calling but these providers aren't configured.
              Go to the MCP page to set them up, or the preview will fail.
            </p>
          </div>
        </div>
      )}

      {error && state !== "running" && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4 flex items-start gap-2">
          <XCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Settings */}
      <div className="card mb-6">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Settings</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="label-text">Number of Records</label>
            <input
              className="input-field"
              type="number"
              min={1}
              value={numRecords}
              onChange={(e) => setNumRecords(parseInt(e.target.value) || 1)}
            />
          </div>
          <div>
            <label className="label-text">Dataset Name</label>
            <input
              className="input-field"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="dataset"
            />
          </div>
          <div>
            <label className="label-text">Debug Mode</label>
            <label className="flex items-center gap-2 mt-2">
              <input
                type="checkbox"
                checked={debugMode}
                onChange={(e) => setDebugMode(e.target.checked)}
                className="accent-[#76b900]"
              />
              <span className="text-sm text-gray-300">
                Capture full traces
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3 mb-6">
        <button
          className="btn-primary flex items-center gap-2 flex-1 justify-center py-3"
          onClick={handlePreview}
          disabled={isRunning || !configLoaded || mcpMissing.length > 0}
        >
          <Eye size={16} />
          Preview
        </button>
        <button
          className="btn-secondary flex items-center gap-2 flex-1 justify-center py-3"
          onClick={handleCreate}
          disabled={isRunning || !configLoaded || mcpMissing.length > 0}
        >
          <HardDrive size={16} />
          Create
        </button>
      </div>

      {/* Status banner */}
      {state === "done" && execType === "preview" && (
        <div className="card flex items-center gap-3 border-green-700/50 mb-4">
          <CheckCircle2 size={20} className="text-green-400" />
          <div>
            <p className="text-sm font-medium text-green-300">
              Preview complete
            </p>
            <p className="text-xs text-gray-400">
              Go to the Results page to inspect the generated data.
            </p>
          </div>
        </div>
      )}

      {state === "done" && execType === "create" && createResult && (
        <div className="card flex items-center gap-3 border-green-700/50 mb-4">
          <CheckCircle2 size={20} className="text-green-400" />
          <div>
            <p className="text-sm font-medium text-green-300">
              Dataset created ({createResult.num_records} records)
            </p>
            <p className="text-xs text-gray-400 font-mono">
              {createResult.artifact_path}
            </p>
          </div>
        </div>
      )}

      {state === "error" && (
        <div className="card flex items-center gap-3 border-red-700/50 mb-4">
          <XCircle size={20} className="text-red-400" />
          <div>
            <p className="text-sm font-medium text-red-300">
              {execType} failed
            </p>
            <p className="text-xs text-red-400">{error}</p>
          </div>
        </div>
      )}

      {/* Log panel */}
      {(isRunning || logs.length > 0) && (
        <div className="card bg-surface-0 p-0 overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-2 bg-surface-2 border-b border-border">
            <Terminal size={14} className="text-gray-400" />
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Logs
            </span>
            {isRunning && (
              <Loader2 size={12} className="animate-spin text-nvidia-green ml-auto" />
            )}
            <span className="text-xs text-gray-600 ml-auto">
              {logs.length} entries
            </span>
          </div>
          <div className="overflow-auto max-h-[400px] p-3 font-mono text-xs leading-relaxed">
            {logs.length === 0 && isRunning && (
              <p className="text-gray-600">Waiting for logs...</p>
            )}
            {logs.map((entry, i) => (
              <div key={i} className="flex gap-2 hover:bg-surface-2 px-1 rounded">
                <span className="text-gray-600 shrink-0 select-none w-16 text-right">
                  {new Date(entry.ts * 1000).toLocaleTimeString()}
                </span>
                <span
                  className={`shrink-0 w-8 text-right ${LEVEL_COLORS[entry.level] ?? "text-gray-400"}`}
                >
                  {entry.level === "WARNING" ? "WARN" : entry.level.slice(0, 4)}
                </span>
                <span className="text-gray-500 shrink-0 w-20 truncate">
                  {entry.name}
                </span>
                <span className={LEVEL_COLORS[entry.level] ?? "text-gray-300"}>
                  {entry.message}
                </span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
