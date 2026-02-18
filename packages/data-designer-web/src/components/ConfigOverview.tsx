import { TableProperties } from "lucide-react";
import { COLUMN_TYPE_META, ColumnType } from "../types/config";

interface Props {
  columns: { name: string; column_type: string; drop: boolean }[];
  models: Record<string, unknown>[];
  outputSchema: {
    name: string;
    column_type: string;
    drop: boolean;
    in_output: boolean;
    side_effect_of?: string;
  }[];
}

export default function ConfigOverview({ columns, models, outputSchema }: Props) {
  const used = models.filter((m: any) => m._used);
  const available = models.filter((m: any) => !m._used);

  return (
    <div className="grid grid-cols-3 gap-6">
      {/* Columns */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-gray-300">
          Columns ({columns.length})
        </h2>
        <div className="space-y-1.5">
          {columns.map((col) => {
            const meta = COLUMN_TYPE_META[col.column_type as ColumnType];
            return (
              <div key={col.name} className="flex items-center gap-2 text-sm">
                <span
                  className={`text-xs px-1.5 py-0.5 rounded border font-medium shrink-0 ${
                    meta?.color ?? "bg-surface-3 text-gray-400 border-border"
                  }`}
                >
                  {meta?.emoji ?? "?"} {meta?.label ?? col.column_type}
                </span>
                <span className="text-gray-200 truncate">{col.name}</span>
                {col.drop && (
                  <span className="text-xs text-gray-600">(drop)</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Models */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-gray-300">
          Models in Use ({used.length})
        </h2>
        {used.length > 0 ? (
          <div className="space-y-2">
            {used.map((m: any) => (
              <div key={m.alias} className="text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-nvidia-green shrink-0" />
                  <span className="text-nvidia-green font-semibold">
                    {m.alias}
                  </span>
                </div>
                <p className="text-xs text-gray-500 ml-4 truncate">
                  {m.model}
                  {m.provider && (
                    <span className="text-gray-600"> via {m.provider}</span>
                  )}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-600">No models referenced</p>
        )}

        {available.length > 0 && (
          <>
            <h2 className="text-sm font-semibold text-gray-500 pt-2">
              Available ({available.length})
            </h2>
            <div className="space-y-1.5">
              {available.map((m: any) => (
                <div key={m.alias} className="text-sm">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-gray-600 shrink-0" />
                    <span className="text-gray-500">{m.alias}</span>
                  </div>
                  <p className="text-xs text-gray-700 ml-4 truncate">
                    {m.model}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Output Schema */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <TableProperties size={14} className="text-gray-400" />
          <h2 className="text-sm font-semibold text-gray-300">Output Schema</h2>
        </div>
        {outputSchema.length > 0 ? (
          <div className="space-y-1 font-mono text-sm">
            {outputSchema.map((field) => (
              <div
                key={field.name}
                className={`flex items-center gap-2 ${
                  field.in_output
                    ? "text-gray-300"
                    : "text-gray-600 line-through"
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full shrink-0 ${
                    field.in_output ? "bg-nvidia-green" : "bg-gray-600"
                  }`}
                />
                <span className="truncate">{field.name}</span>
                {field.side_effect_of && (
                  <span className="text-xs text-gray-600 shrink-0">
                    (from {field.side_effect_of})
                  </span>
                )}
              </div>
            ))}
            <p className="text-xs text-gray-600 pt-2">
              {outputSchema.filter((f) => f.in_output).length} columns in output
              {outputSchema.some((f) => !f.in_output) && (
                <span>
                  , {outputSchema.filter((f) => !f.in_output).length} dropped
                </span>
              )}
            </p>
          </div>
        ) : (
          <p className="text-xs text-gray-600">No columns configured</p>
        )}
      </div>
    </div>
  );
}
