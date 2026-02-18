import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, Pencil, GripVertical, AlertCircle } from "lucide-react";
import { api } from "../hooks/useApi";
import { ColumnConfig, ColumnType, COLUMN_TYPE_META } from "../types/config";
import ColumnEditor from "../components/ColumnEditor";

export default function ConfigBuilder() {
  const [columns, setColumns] = useState<ColumnConfig[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<ColumnConfig | null>(null);
  const [adding, setAdding] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const cols = (await api.listColumns()) as ColumnConfig[];
      setColumns(cols);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDelete = async (name: string) => {
    try {
      await api.deleteColumn(name);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleSave = async () => {
    setEditing(null);
    setAdding(false);
    await refresh();
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">
            Config Builder
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Add and configure columns for your synthetic dataset
          </p>
        </div>
        <button className="btn-primary flex items-center gap-2" onClick={() => setAdding(true)}>
          <Plus size={16} />
          Add Column
        </button>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4 flex items-start gap-2">
          <AlertCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Column add/edit panel */}
      {(adding || editing) && (
        <div className="card mb-6 border-nvidia-green/30">
          <ColumnEditor
            column={editing ?? undefined}
            onSave={handleSave}
            onCancel={() => {
              setEditing(null);
              setAdding(false);
            }}
          />
        </div>
      )}

      {/* Column list */}
      {columns.length === 0 && !adding ? (
        <div className="card text-center py-12">
          <p className="text-gray-500 mb-2">No columns configured yet</p>
          <button
            className="btn-primary inline-flex items-center gap-2"
            onClick={() => setAdding(true)}
          >
            <Plus size={16} />
            Add your first column
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {columns.map((col, idx) => {
            const meta = COLUMN_TYPE_META[col.column_type as ColumnType];
            return (
              <div
                key={col.name}
                className="card-hover flex items-center gap-3 group"
              >
                <GripVertical
                  size={16}
                  className="text-gray-600 group-hover:text-gray-400 shrink-0"
                />
                <span className="text-gray-500 text-xs w-6 text-right shrink-0">
                  {idx + 1}
                </span>
                <span
                  className={`text-xs px-2 py-0.5 rounded border font-medium shrink-0 ${meta?.color ?? "bg-surface-3 text-gray-400 border-border"}`}
                >
                  {meta?.emoji} {meta?.label ?? col.column_type}
                </span>
                <span className="font-medium text-sm flex-1 truncate">
                  {col.name}
                </span>
                {col.drop && (
                  <span className="text-xs text-gray-500 bg-surface-3 px-2 py-0.5 rounded">
                    drop
                  </span>
                )}
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    className="btn-ghost"
                    title="Edit"
                    onClick={() => {
                      setAdding(false);
                      setEditing(col);
                    }}
                  >
                    <Pencil size={14} />
                  </button>
                  <button
                    className="btn-ghost text-red-400 hover:text-red-300"
                    title="Delete"
                    onClick={() => handleDelete(col.name)}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
