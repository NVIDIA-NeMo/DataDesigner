import { useState, useEffect } from "react";
import { Save, X } from "lucide-react";
import { api } from "../hooks/useApi";
import { ColumnConfig, ColumnType, COLUMN_TYPE_META } from "../types/config";
import SamplerParamsForm from "./SamplerParamsForm";

interface Props {
  column?: ColumnConfig;
  onSave: () => void;
  onCancel: () => void;
}

const COLUMN_TYPES = Object.keys(COLUMN_TYPE_META) as ColumnType[];

export default function ColumnEditor({ column, onSave, onCancel }: Props) {
  const isNew = !column;
  const col = column as Record<string, any> | undefined;

  const [name, setName] = useState(column?.name ?? "");
  const [columnType, setColumnType] = useState<ColumnType>(
    (column?.column_type as ColumnType) ?? "sampler"
  );
  const [enums, setEnums] = useState<Record<string, string[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Sampler
  const [samplerType, setSamplerType] = useState(col?.sampler_type ?? "category");
  const [samplerParams, setSamplerParams] = useState<Record<string, unknown>>(
    col?.params ?? {}
  );

  // LLM / Image
  const [prompt, setPrompt] = useState(col?.prompt ?? "");
  const [modelAlias, setModelAlias] = useState(col?.model_alias ?? "");
  const [systemPrompt, setSystemPrompt] = useState(col?.system_prompt ?? "");

  // Expression
  const [expression, setExpression] = useState(col?.expr ?? "");

  // Embedding
  const [targetColumn, setTargetColumn] = useState(col?.target_column ?? "");

  useEffect(() => {
    api.getEnums().then(setEnums).catch(() => {});
  }, []);

  // Reset sampler params when sampler type changes
  const handleSamplerTypeChange = (newType: string) => {
    setSamplerType(newType);
    setSamplerParams({});
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const data: Record<string, unknown> = {
        name,
        column_type: columnType,
      };

      if (columnType === "sampler") {
        data.sampler_type = samplerType;
        data.params = samplerParams;
      } else if (
        columnType === "llm-text" ||
        columnType === "llm-code" ||
        columnType === "llm-structured" ||
        columnType === "llm-judge"
      ) {
        data.prompt = prompt;
        data.model_alias = modelAlias;
        if (systemPrompt) data.system_prompt = systemPrompt;
      } else if (columnType === "expression") {
        data.expr = expression;
      } else if (columnType === "embedding") {
        data.model_alias = modelAlias;
        data.target_column = targetColumn;
      } else if (columnType === "image") {
        data.model_alias = modelAlias;
        data.prompt = prompt;
      }

      if (isNew) {
        await api.addColumn(data);
      } else {
        await api.updateColumn(column!.name, data);
      }
      onSave();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const showPrompt = ["llm-text", "llm-code", "llm-structured", "llm-judge", "image"].includes(columnType);
  const showModelAlias = ["llm-text", "llm-code", "llm-structured", "llm-judge", "embedding", "image"].includes(columnType);
  const showSystemPrompt = ["llm-text", "llm-code", "llm-structured", "llm-judge"].includes(columnType);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-200">
          {isNew ? "Add Column" : `Edit: ${column!.name}`}
        </h3>
        <button className="btn-ghost" onClick={onCancel}>
          <X size={16} />
        </button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-700/50 rounded-md px-3 py-2 mb-4 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="label-text">Column Name</label>
          <input
            className="input-field"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. topic, question, answer"
          />
        </div>
        <div>
          <label className="label-text">Column Type</label>
          <select
            className="select-field"
            value={columnType}
            onChange={(e) => setColumnType(e.target.value as ColumnType)}
          >
            {COLUMN_TYPES.map((t) => (
              <option key={t} value={t}>
                {COLUMN_TYPE_META[t].emoji} {COLUMN_TYPE_META[t].label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Sampler fields */}
      {columnType === "sampler" && (
        <div className="space-y-4 mb-4">
          <div>
            <label className="label-text">Sampler Type</label>
            <select
              className="select-field"
              value={samplerType}
              onChange={(e) => handleSamplerTypeChange(e.target.value)}
            >
              {(enums.sampler_types ?? []).map((st) => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>
          </div>
          <div className="bg-surface-2 rounded-md p-4 border border-border">
            <p className="text-xs text-gray-500 mb-3 uppercase tracking-wider font-medium">
              Parameters
            </p>
            <SamplerParamsForm
              samplerType={samplerType}
              params={samplerParams}
              onChange={setSamplerParams}
            />
          </div>
        </div>
      )}

      {/* Prompt template */}
      {showPrompt && (
        <div className="mb-4">
          <label className="label-text">Prompt Template (Jinja2)</label>
          <textarea
            className="input-field min-h-[100px] font-mono text-xs"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Generate a {{ topic }} about..."
          />
        </div>
      )}

      {/* System prompt */}
      {showSystemPrompt && (
        <div className="mb-4">
          <label className="label-text">
            System Prompt{" "}
            <span className="text-gray-500 font-normal">(optional)</span>
          </label>
          <textarea
            className="input-field min-h-[60px] font-mono text-xs"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="You are a helpful assistant..."
          />
        </div>
      )}

      {/* Model alias */}
      {showModelAlias && (
        <div className="mb-4">
          <label className="label-text">Model Alias</label>
          <input
            className="input-field"
            value={modelAlias}
            onChange={(e) => setModelAlias(e.target.value)}
            placeholder="e.g. my_model"
          />
        </div>
      )}

      {/* Expression */}
      {columnType === "expression" && (
        <div className="mb-4">
          <label className="label-text">Jinja2 Expression</label>
          <input
            className="input-field font-mono"
            value={expression}
            onChange={(e) => setExpression(e.target.value)}
            placeholder="{{ column_a }} + {{ column_b }}"
          />
        </div>
      )}

      {/* Embedding target */}
      {columnType === "embedding" && (
        <div className="mb-4">
          <label className="label-text">Target Column</label>
          <input
            className="input-field"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="Name of column to embed"
          />
        </div>
      )}

      <div className="flex justify-end gap-2 pt-2 border-t border-border">
        <button className="btn-secondary" onClick={onCancel}>
          Cancel
        </button>
        <button
          className="btn-primary flex items-center gap-2"
          onClick={handleSave}
          disabled={saving || !name}
        >
          <Save size={14} />
          {saving ? "Saving..." : isNew ? "Add Column" : "Update Column"}
        </button>
      </div>
    </div>
  );
}
