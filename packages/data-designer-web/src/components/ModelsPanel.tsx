import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, AlertCircle, Cpu } from "lucide-react";
import { api } from "../hooks/useApi";
import { ModelConfig } from "../types/config";

export default function ModelsPanel() {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  // Form state
  const [alias, setAlias] = useState("");
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");

  const refresh = useCallback(async () => {
    try {
      const m = (await api.listModels()) as unknown as ModelConfig[];
      setModels(m);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAdd = async () => {
    setError(null);
    try {
      const data: Record<string, unknown> = { alias, model };
      if (provider) data.provider = provider;
      await api.addModel(data);
      setAlias("");
      setModel("");
      setProvider("");
      setAdding(false);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleDelete = async (a: string) => {
    try {
      await api.deleteModel(a);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">
            Model Configurations
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Configure LLM models used for data generation
          </p>
        </div>
        <button
          className="btn-primary flex items-center gap-2"
          onClick={() => setAdding(true)}
        >
          <Plus size={16} />
          Add Model
        </button>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4 flex items-start gap-2">
          <AlertCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {adding && (
        <div className="card mb-6 border-nvidia-green/30">
          <h3 className="text-sm font-semibold text-gray-200 mb-4">
            Add Model
          </h3>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="label-text">Alias</label>
              <input
                className="input-field"
                value={alias}
                onChange={(e) => setAlias(e.target.value)}
                placeholder="my_model"
              />
            </div>
            <div>
              <label className="label-text">Model Name</label>
              <input
                className="input-field"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="meta/llama-3.1-70b-instruct"
              />
            </div>
            <div>
              <label className="label-text">Provider (optional)</label>
              <input
                className="input-field"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                placeholder="nvidia"
              />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <button className="btn-secondary" onClick={() => setAdding(false)}>
              Cancel
            </button>
            <button
              className="btn-primary"
              onClick={handleAdd}
              disabled={!alias || !model}
            >
              Add
            </button>
          </div>
        </div>
      )}

      {models.length === 0 && !adding ? (
        <div className="card text-center py-12">
          <Cpu size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500 mb-2">No models configured</p>
          <button
            className="btn-primary inline-flex items-center gap-2"
            onClick={() => setAdding(true)}
          >
            <Plus size={16} />
            Add a model
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {models.map((m) => (
            <div key={m.alias} className="card-hover flex items-center gap-3 group">
              <Cpu size={16} className="text-nvidia-green shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">{m.alias}</div>
                <div className="text-xs text-gray-500 truncate">
                  {m.model}
                  {m.provider && (
                    <span className="ml-2 text-gray-600">
                      via {m.provider}
                    </span>
                  )}
                </div>
              </div>
              <button
                className="btn-ghost text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={() => handleDelete(m.alias)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
