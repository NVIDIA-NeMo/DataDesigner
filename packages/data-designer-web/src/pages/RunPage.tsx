import { useState, useEffect, useRef } from "react";
import {
  Play,
  Eye,
  HardDrive,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { api } from "../hooks/useApi";

type ExecState = "idle" | "running" | "done" | "error";

export default function RunPage() {
  const [numRecords, setNumRecords] = useState(10);
  const [debugMode, setDebugMode] = useState(true);
  const [datasetName, setDatasetName] = useState("dataset");
  const [state, setState] = useState<ExecState>("idle");
  const [execType, setExecType] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [createResult, setCreateResult] = useState<{
    num_records?: number;
    artifact_path?: string;
  } | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    api.getConfigInfo().then((info) => setConfigLoaded(info.loaded));
    api.getStatus().then((s) => {
      setState(s.state);
      setExecType(s.type);
      setError(s.error);
    });
  }, []);

  const pollStatus = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      const s = await api.getStatus();
      setState(s.state);
      setExecType(s.type);
      setError(s.error);
      if (s.state !== "running") {
        clearInterval(pollRef.current!);
        pollRef.current = null;
        if (s.state === "done" && s.type === "create") {
          const cr = await api.getCreateResults();
          setCreateResult(cr);
        }
      }
    }, 1500);
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handlePreview = async () => {
    setError(null);
    setCreateResult(null);
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
    <div className="p-6 max-w-3xl mx-auto">
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

      {error && (
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
          disabled={isRunning || !configLoaded}
        >
          <Eye size={16} />
          Preview
        </button>
        <button
          className="btn-secondary flex items-center gap-2 flex-1 justify-center py-3"
          onClick={handleCreate}
          disabled={isRunning || !configLoaded}
        >
          <HardDrive size={16} />
          Create
        </button>
      </div>

      {/* Status */}
      {state === "running" && (
        <div className="card flex items-center gap-3">
          <Loader2 size={20} className="animate-spin text-nvidia-green" />
          <div>
            <p className="text-sm font-medium text-gray-200">
              Running {execType}...
            </p>
            <p className="text-xs text-gray-500">
              Generating {numRecords} records. This may take a while.
            </p>
          </div>
        </div>
      )}

      {state === "done" && execType === "preview" && (
        <div className="card flex items-center gap-3 border-green-700/50">
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
        <div className="card flex items-center gap-3 border-green-700/50">
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
        <div className="card flex items-center gap-3 border-red-700/50">
          <XCircle size={20} className="text-red-400" />
          <div>
            <p className="text-sm font-medium text-red-300">
              {execType} failed
            </p>
            <p className="text-xs text-red-400">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
