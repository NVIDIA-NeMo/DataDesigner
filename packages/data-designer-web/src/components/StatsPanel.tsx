import { BarChart3 } from "lucide-react";
import { COLUMN_TYPE_META, ColumnType } from "../types/config";

interface ColumnStat {
  column_name: string;
  column_type: string;
  num_records: number;
  num_null: number;
  num_unique: number;
  simple_dtype: string;
  // LLM-specific
  input_tokens_mean?: number;
  input_tokens_median?: number;
  input_tokens_stddev?: number;
  output_tokens_mean?: number;
  output_tokens_median?: number;
  output_tokens_stddev?: number;
  // Sampler-specific
  sampler_type?: string;
  // Validation-specific
  num_valid_records?: number;
  [key: string]: unknown;
}

interface Props {
  analysis: { column_statistics: ColumnStat[] } | null;
  numRecords: number;
}

function fmt(val: unknown, decimals = 1): string {
  if (val === null || val === undefined || val === -1) return "--";
  if (typeof val === "number") return val.toFixed(decimals);
  return String(val);
}

export default function StatsPanel({ analysis, numRecords }: Props) {
  if (!analysis || !analysis.column_statistics || analysis.column_statistics.length === 0) {
    return null;
  }

  const stats = analysis.column_statistics;

  const llmStats = stats.filter(
    (s) => s.column_type.startsWith("llm-") && s.output_tokens_median !== undefined
  );
  const samplerStats = stats.filter((s) => s.column_type === "sampler");
  const otherStats = stats.filter(
    (s) => !s.column_type.startsWith("llm-") && s.column_type !== "sampler"
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <BarChart3 size={16} className="text-nvidia-green" />
        <h2 className="text-sm font-semibold text-gray-200">
          Dataset Statistics
        </h2>
        <span className="text-xs text-gray-500">
          {numRecords} records
        </span>
      </div>

      {/* Overview table */}
      <div className="overflow-auto rounded-lg border border-border">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-surface-2 border-b border-border">
              <th className="px-3 py-2 text-left text-gray-400 font-medium">Column</th>
              <th className="px-3 py-2 text-left text-gray-400 font-medium">Type</th>
              <th className="px-3 py-2 text-right text-gray-400 font-medium">Dtype</th>
              <th className="px-3 py-2 text-right text-gray-400 font-medium">Unique</th>
              <th className="px-3 py-2 text-right text-gray-400 font-medium">Nulls</th>
              {llmStats.length > 0 && (
                <>
                  <th className="px-3 py-2 text-right text-gray-400 font-medium">
                    Prompt Tokens
                  </th>
                  <th className="px-3 py-2 text-right text-gray-400 font-medium">
                    Completion Tokens
                  </th>
                </>
              )}
              {samplerStats.some((s) => s.sampler_type) && (
                <th className="px-3 py-2 text-right text-gray-400 font-medium">Sampler</th>
              )}
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => {
              const meta = COLUMN_TYPE_META[s.column_type as ColumnType];
              const isLLM = s.column_type.startsWith("llm-");
              return (
                <tr
                  key={s.column_name}
                  className="border-b border-border hover:bg-surface-2"
                >
                  <td className="px-3 py-1.5 text-gray-200 font-medium">
                    {s.column_name}
                  </td>
                  <td className="px-3 py-1.5">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${
                        meta?.color ?? "bg-surface-3 text-gray-400 border-border"
                      }`}
                    >
                      {meta?.emoji ?? "?"} {meta?.label ?? s.column_type}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right text-gray-400">
                    {s.simple_dtype}
                  </td>
                  <td className="px-3 py-1.5 text-right text-gray-300">
                    {s.num_unique}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    <span
                      className={
                        s.num_null > 0 ? "text-amber-400" : "text-gray-500"
                      }
                    >
                      {s.num_null}
                    </span>
                  </td>
                  {llmStats.length > 0 && (
                    <>
                      <td className="px-3 py-1.5 text-right text-gray-400 font-mono">
                        {isLLM
                          ? `${fmt(s.input_tokens_median)} ± ${fmt(s.input_tokens_stddev)}`
                          : ""}
                      </td>
                      <td className="px-3 py-1.5 text-right text-gray-400 font-mono">
                        {isLLM
                          ? `${fmt(s.output_tokens_median)} ± ${fmt(s.output_tokens_stddev)}`
                          : ""}
                      </td>
                    </>
                  )}
                  {samplerStats.some((ss) => ss.sampler_type) && (
                    <td className="px-3 py-1.5 text-right text-gray-400">
                      {s.sampler_type ?? ""}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
