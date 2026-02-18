import { useEffect, useState, useCallback, useRef } from "react";
import {
  CheckCircle2,
  XCircle,
  RefreshCw,
  FileText,
  Loader2,
  Save,
  Undo2,
} from "lucide-react";
import { api } from "../hooks/useApi";
import YamlHighlighter from "../components/YamlHighlighter";
import ConfigOverview from "../components/ConfigOverview";
import ReviewPanel from "../components/ReviewPanel";

type SubTab = "editor" | "overview" | "review";

interface ConfigFile {
  name: string;
  path: string;
  active: boolean;
}

export default function ConfigPage() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [columns, setColumns] = useState<{ name: string; column_type: string; drop: boolean }[]>([]);
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [outputSchema, setOutputSchema] = useState<
    { name: string; column_type: string; drop: boolean; in_output: boolean; side_effect_of?: string }[]
  >([]);
  const [loaded, setLoaded] = useState(false);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState<{ valid: boolean; message: string } | null>(null);
  const [validating, setValidating] = useState(false);

  const [savedYaml, setSavedYaml] = useState("");
  const [editYaml, setEditYaml] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [subTab, setSubTab] = useState<SubTab>("editor");

  const hasUnsavedChanges = editYaml !== savedYaml;

  const refresh = useCallback(async () => {
    try {
      const info = await api.getConfigInfo();
      setLoaded(info.loaded);
      setActivePath(info.path);
      setColumns(info.columns);
      setModels(info.models);
      setOutputSchema(info.output_schema ?? []);
      if (info.loaded) {
        const y = await api.getConfigYaml();
        setSavedYaml(y.content);
        setEditYaml(y.content);
      } else {
        setSavedYaml("");
        setEditYaml("");
      }
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
    try {
      setConfigs(await api.listConfigs());
    } catch {
      // non-critical
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleLoad = async (path: string) => {
    setLoading(true);
    setError(null);
    setValidation(null);
    setSaveMessage(null);
    try {
      await api.loadConfig(path);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMessage(null);
    setError(null);
    setValidation(null);
    try {
      await api.saveConfig(editYaml);
      await refresh();
      setSaveMessage("Saved and reloaded");
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (e: any) {
      setSaveMessage(null);
      setValidation({ valid: false, message: e.message });
    } finally {
      setSaving(false);
    }
  };

  const handleDiscard = () => {
    setEditYaml(savedYaml);
    setSaveMessage(null);
    setError(null);
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
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Configuration</h1>
          <p className="text-sm text-gray-500 mt-1">
            Load, edit, and save your config
          </p>
        </div>
        <button className="btn-ghost" onClick={refresh} title="Reload from disk">
          <RefreshCw size={16} />
        </button>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4 shrink-0">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Top bar: file picker + actions */}
      <div className="card mb-4 shrink-0">
        <div className="flex items-center gap-3">
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
                <option key={c.path} value={c.path}>{c.name}</option>
              ))}
            </select>
          </div>
          {loaded && (
            <div className="flex gap-2 pt-5">
              <button
                className="btn-primary flex items-center gap-1.5"
                onClick={handleSave}
                disabled={saving || !hasUnsavedChanges}
              >
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                Save
              </button>
              {hasUnsavedChanges && (
                <button className="btn-ghost text-gray-400" onClick={handleDiscard} title="Discard">
                  <Undo2 size={14} />
                </button>
              )}
              <button
                className="btn-secondary flex items-center gap-1.5"
                onClick={handleValidate}
                disabled={validating}
              >
                {validating ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
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
        <div className="flex items-center gap-4 mt-2 min-h-[20px]">
          {hasUnsavedChanges && <span className="text-xs text-amber-400">Unsaved changes</span>}
          {saveMessage && (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <CheckCircle2 size={12} />{saveMessage}
            </span>
          )}
          {validation && (
            <span className={`text-xs flex items-center gap-1 ${validation.valid ? "text-green-400" : "text-red-400"}`}>
              {validation.valid ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
              {validation.message}
            </span>
          )}
        </div>
      </div>

      {!loaded ? (
        <div className="card text-center py-12 flex-1">
          <FileText size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">Select a config file above to get started</p>
        </div>
      ) : (
        <>
          {/* Sub-tabs */}
          <div className="flex gap-1 mb-4 border-b border-border shrink-0">
            {([
              { id: "editor" as SubTab, label: "Editor" },
              { id: "overview" as SubTab, label: "Overview" },
              { id: "review" as SubTab, label: "Review" },
            ]).map((t) => (
              <button
                key={t.id}
                className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  subTab === t.id
                    ? "border-nvidia-green text-nvidia-green"
                    : "border-transparent text-gray-400 hover:text-gray-200"
                }`}
                onClick={() => setSubTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Editor tab */}
          {subTab === "editor" && (
            <div className="flex-1 flex flex-col min-h-0">
              {editing ? (
                <textarea
                  ref={textareaRef}
                  className="flex-1 bg-surface-0 border border-nvidia-green/40 rounded-lg p-4 font-mono text-xs text-gray-200 leading-relaxed resize-none focus:outline-none focus:ring-1 focus:ring-nvidia-green/30 transition-colors"
                  value={editYaml}
                  onChange={(e) => {
                    setEditYaml(e.target.value);
                    setSaveMessage(null);
                    setValidation(null);
                  }}
                  onBlur={() => setEditing(false)}
                  spellCheck={false}
                  autoFocus
                />
              ) : (
                <div
                  className="flex-1 bg-surface-0 border border-border rounded-lg p-4 overflow-auto cursor-text hover:border-border-hover transition-colors"
                  onClick={() => setEditing(true)}
                >
                  <YamlHighlighter code={editYaml} />
                </div>
              )}
            </div>
          )}

          {/* Overview tab */}
          {subTab === "overview" && (
            <div className="flex-1 overflow-auto">
              <ConfigOverview
                columns={columns}
                models={models}
                outputSchema={outputSchema}
              />
            </div>
          )}

          {/* Review tab */}
          {subTab === "review" && (
            <div className="flex-1 overflow-auto">
              <ReviewPanel models={models} loaded={loaded} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
