import { useEffect, useState, useCallback, useRef } from "react";
import {
  CheckCircle2,
  XCircle,
  RefreshCw,
  FileText,
  Loader2,
  Save,
  Undo2,
  TableProperties,
  Sparkles,
  AlertTriangle,
  Info,
  Lightbulb,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { COLUMN_TYPE_META, ColumnType } from "../types/config";
import YamlHighlighter from "../components/YamlHighlighter";

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
  const [outputSchema, setOutputSchema] = useState<
    { name: string; column_type: string; drop: boolean; in_output: boolean; side_effect_of?: string }[]
  >([]);
  const [loaded, setLoaded] = useState(false);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState<{
    valid: boolean;
    message: string;
  } | null>(null);
  const [validating, setValidating] = useState(false);

  // Editor state
  const [savedYaml, setSavedYaml] = useState("");
  const [editYaml, setEditYaml] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Review state
  const [reviewModel, setReviewModel] = useState("");
  const [reviewing, setReviewing] = useState(false);
  const [reviewResult, setReviewResult] = useState<{
    static_issues: { level: string; type: string; column: string | null; message: string }[];
    llm_tips: { category: string; severity: string; column: string | null; tip: string }[];
    model_used: string;
  } | null>(null);
  const [reviewOpen, setReviewOpen] = useState(true);

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
      const cfgs = await api.listConfigs();
      setConfigs(cfgs);
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

  const handleReview = async () => {
    if (!reviewModel) return;
    setReviewing(true);
    setReviewResult(null);
    try {
      const result = await api.reviewConfig(reviewModel);
      setReviewResult(result);
      setReviewOpen(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setReviewing(false);
    }
  };

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Configuration</h1>
          <p className="text-sm text-gray-500 mt-1">
            Load, edit, and save your config. Changes are written to disk.
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

      {/* Config file picker + action buttons */}
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
                <option key={c.path} value={c.path}>
                  {c.name}
                </option>
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
                {saving ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Save size={14} />
                )}
                Save
              </button>
              {hasUnsavedChanges && (
                <button
                  className="btn-ghost text-gray-400"
                  onClick={handleDiscard}
                  title="Discard changes"
                >
                  <Undo2 size={14} />
                </button>
              )}
              <button
                className="btn-secondary flex items-center gap-1.5"
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
              <div className="w-px h-6 bg-border mx-1" />
              <select
                className="select-field !w-40 !py-1.5 text-xs"
                value={reviewModel}
                onChange={(e) => setReviewModel(e.target.value)}
              >
                <option value="">Review model...</option>
                {models.map((m: any) => (
                  <option key={m.alias} value={m.alias}>
                    {m.alias}
                  </option>
                ))}
              </select>
              <button
                className="btn-secondary flex items-center gap-1.5 border-purple-700/50 text-purple-300 hover:bg-purple-900/20"
                onClick={handleReview}
                disabled={reviewing || !reviewModel}
              >
                {reviewing ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Sparkles size={14} />
                )}
                Review
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
        {/* Status messages */}
        <div className="flex items-center gap-4 mt-2 min-h-[20px]">
          {hasUnsavedChanges && (
            <span className="text-xs text-amber-400">Unsaved changes</span>
          )}
          {saveMessage && (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <CheckCircle2 size={12} />
              {saveMessage}
            </span>
          )}
          {validation && (
            <span
              className={`text-xs flex items-center gap-1 ${
                validation.valid ? "text-green-400" : "text-red-400"
              }`}
            >
              {validation.valid ? (
                <CheckCircle2 size={12} />
              ) : (
                <XCircle size={12} />
              )}
              {validation.message}
            </span>
          )}
        </div>
      </div>

      {/* Review results panel */}
      {reviewResult && (
        <div className="card mb-4 shrink-0 bg-surface-0 border-purple-700/30">
          <button
            className="w-full flex items-center gap-2 text-sm font-medium text-purple-300"
            onClick={() => setReviewOpen(!reviewOpen)}
          >
            {reviewOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <Sparkles size={14} />
            Config Review
            <span className="text-xs text-gray-500 font-normal ml-2">
              via {reviewResult.model_used}
            </span>
            <span className="text-xs text-gray-600 font-normal ml-auto">
              {reviewResult.static_issues.length} issues, {reviewResult.llm_tips.length} tips
            </span>
          </button>

          {reviewOpen && (
            <div className="mt-3 space-y-4">
              {/* Static issues */}
              {reviewResult.static_issues.length > 0 && (
                <div>
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Static Analysis
                  </h3>
                  <div className="space-y-1.5">
                    {reviewResult.static_issues.map((issue, i) => (
                      <div
                        key={i}
                        className={`flex items-start gap-2 text-xs rounded-md px-3 py-2 ${
                          issue.level === "ERROR"
                            ? "bg-red-900/20 border border-red-700/30"
                            : "bg-amber-900/15 border border-amber-700/30"
                        }`}
                      >
                        {issue.level === "ERROR" ? (
                          <XCircle size={12} className="text-red-400 mt-0.5 shrink-0" />
                        ) : (
                          <AlertTriangle size={12} className="text-amber-400 mt-0.5 shrink-0" />
                        )}
                        <div className="flex-1">
                          {issue.column && (
                            <span className="font-mono text-gray-400 mr-1">
                              {issue.column}:
                            </span>
                          )}
                          <span className={issue.level === "ERROR" ? "text-red-300" : "text-amber-300"}>
                            {issue.message}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* LLM tips */}
              {reviewResult.llm_tips.length > 0 && (
                <div>
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    AI Suggestions
                  </h3>
                  <div className="space-y-1.5">
                    {reviewResult.llm_tips.map((tip, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 text-xs bg-surface-2 border border-border rounded-md px-3 py-2"
                      >
                        {tip.severity === "warning" ? (
                          <AlertTriangle size={12} className="text-amber-400 mt-0.5 shrink-0" />
                        ) : tip.severity === "suggestion" ? (
                          <Lightbulb size={12} className="text-purple-400 mt-0.5 shrink-0" />
                        ) : (
                          <Info size={12} className="text-blue-400 mt-0.5 shrink-0" />
                        )}
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-0.5">
                            <span className="text-[10px] uppercase tracking-wider font-medium text-gray-500 bg-surface-3 px-1.5 py-0.5 rounded">
                              {tip.category.replace("_", " ")}
                            </span>
                            {tip.column && (
                              <span className="font-mono text-gray-500 text-[10px]">
                                {tip.column}
                              </span>
                            )}
                          </div>
                          <span className="text-gray-300">{tip.tip}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {reviewResult.static_issues.length === 0 && reviewResult.llm_tips.length === 0 && (
                <p className="text-xs text-green-400 flex items-center gap-1.5">
                  <CheckCircle2 size={12} />
                  Config looks good! No issues or suggestions.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {!loaded ? (
        <div className="card text-center py-12 flex-1">
          <FileText size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">
            Select a config file above to get started
          </p>
        </div>
      ) : (
        <div className="flex gap-4 flex-1 min-h-0">
          {/* Column + model summary sidebar */}
          <div className="w-56 shrink-0 space-y-4 overflow-auto">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Columns ({columns.length})
            </h2>
            <div className="space-y-1">
              {columns.map((col) => {
                const meta =
                  COLUMN_TYPE_META[col.column_type as ColumnType];
                return (
                  <div
                    key={col.name}
                    className="flex items-center gap-1.5 text-xs"
                  >
                    <span
                      className={`px-1 py-0.5 rounded border font-medium shrink-0 ${
                        meta?.color ?? "bg-surface-3 text-gray-400 border-border"
                      }`}
                    >
                      {meta?.emoji ?? "?"}
                    </span>
                    <span className="text-gray-300 truncate">{col.name}</span>
                  </div>
                );
              })}
            </div>

            {(() => {
              const used = models.filter((m: any) => m._used);
              const available = models.filter((m: any) => !m._used);
              return (
                <>
                  <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider pt-2">
                    Models in Use ({used.length})
                  </h2>
                  {used.length > 0 ? (
                    <div className="space-y-1.5">
                      {used.map((m: any) => (
                        <div key={m.alias} className="text-xs">
                          <div className="flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-nvidia-green shrink-0" />
                            <span className="text-nvidia-green font-semibold">
                              {m.alias}
                            </span>
                          </div>
                          <span className="text-gray-500 ml-3 truncate block text-[10px]">
                            {m.model}
                            {m.provider && (
                              <span className="text-gray-600"> via {m.provider}</span>
                            )}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[10px] text-gray-600">
                      No models referenced by columns
                    </p>
                  )}

                  {available.length > 0 && (
                    <>
                      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider pt-2">
                        Available ({available.length})
                      </h2>
                      <div className="space-y-1">
                        {available.map((m: any) => (
                          <div key={m.alias} className="text-xs">
                            <div className="flex items-center gap-1.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-gray-600 shrink-0" />
                              <span className="text-gray-500">
                                {m.alias}
                              </span>
                            </div>
                            <span className="text-gray-700 ml-3 truncate block text-[10px]">
                              {m.model}
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </>
              );
            })()}

            {/* Output Schema */}
            {outputSchema.length > 0 && (
              <>
                <div className="flex items-center gap-1.5 pt-3">
                  <TableProperties size={12} className="text-gray-400" />
                  <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Output Schema
                  </h2>
                </div>
                <div className="space-y-0.5 text-xs font-mono">
                  {outputSchema.map((field) => (
                    <div
                      key={field.name}
                      className={`flex items-center gap-1.5 ${
                        field.in_output ? "text-gray-300" : "text-gray-600 line-through"
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                          field.in_output ? "bg-nvidia-green" : "bg-gray-600"
                        }`}
                      />
                      <span className="truncate">{field.name}</span>
                      {field.side_effect_of && (
                        <span className="text-[10px] text-gray-600 shrink-0">
                          (from {field.side_effect_of})
                        </span>
                      )}
                    </div>
                  ))}
                  <div className="text-[10px] text-gray-600 pt-1">
                    {outputSchema.filter((f) => f.in_output).length} columns in output
                    {outputSchema.some((f) => !f.in_output) && (
                      <span>
                        , {outputSchema.filter((f) => !f.in_output).length} dropped
                      </span>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* YAML editor with syntax highlighting */}
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
        </div>
      )}
    </div>
  );
}
