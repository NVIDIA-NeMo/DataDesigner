import { useEffect, useState, useCallback } from "react";
import { Cpu, RefreshCw } from "lucide-react";
import { api } from "../hooks/useApi";

interface ModelConfig {
  alias: string;
  model: string;
  provider?: string;
  inference_parameters?: Record<string, unknown>;
  skip_health_check?: boolean;
}

export default function ModelsPanel() {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const info = await api.getConfigInfo();
      setLoaded(info.loaded);
      setModels(info.models as unknown as ModelConfig[]);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">
            Model Configurations
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Models defined in the loaded config (read-only)
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

      {!loaded ? (
        <div className="card text-center py-12">
          <Cpu size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">
            No config loaded. Go to the Config page to load one first.
          </p>
        </div>
      ) : models.length === 0 ? (
        <div className="card text-center py-12">
          <Cpu size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">No models defined in this config.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {models.map((m) => (
            <div key={m.alias} className="card">
              <div className="flex items-start gap-3">
                <Cpu
                  size={18}
                  className="text-nvidia-green mt-0.5 shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm text-gray-100">
                      {m.alias}
                    </span>
                    {m.provider && (
                      <span className="text-xs bg-surface-3 text-gray-400 px-2 py-0.5 rounded">
                        {m.provider}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1 font-mono truncate">
                    {m.model}
                  </p>
                  {m.inference_parameters &&
                    Object.keys(m.inference_parameters).length > 0 && (
                      <div className="mt-2 text-xs text-gray-500">
                        <span className="text-gray-600">Params: </span>
                        {Object.entries(m.inference_parameters)
                          .filter(
                            ([k]) =>
                              k !== "generation_type" &&
                              m.inference_parameters![k] != null
                          )
                          .map(([k, v]) => (
                            <span
                              key={k}
                              className="inline-flex items-center gap-1 bg-surface-3 px-1.5 py-0.5 rounded mr-1"
                            >
                              {k}={String(v)}
                            </span>
                          ))}
                      </div>
                    )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
