import { useEffect, useState } from "react";
import { Table2, ChevronDown, ChevronRight, MessageSquare } from "lucide-react";
import { api } from "../hooks/useApi";
import { COLUMN_TYPE_META, ColumnType } from "../types/config";
import TraceViewer from "../components/TraceViewer";
import StatsPanel from "../components/StatsPanel";

type Tab = "data" | "stats";

export default function ResultsPage() {
  const [columns, setColumns] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [rowCount, setRowCount] = useState(0);
  const [analysis, setAnalysis] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [traceColumn, setTraceColumn] = useState<string | null>(null);
  const [trace, setTrace] = useState<Record<string, unknown>[]>([]);
  const [loadingTrace, setLoadingTrace] = useState(false);
  const [tab, setTab] = useState<Tab>("data");

  // Column metadata from config for type badges
  const [colMeta, setColMeta] = useState<
    Record<string, { column_type: string }>
  >({});

  useEffect(() => {
    Promise.all([api.getPreviewResults(), api.getConfigInfo()]).then(
      ([r, info]) => {
        const visibleCols = r.columns.filter(
          (c: string) =>
            !c.endsWith("__trace") && !c.endsWith("__reasoning_content")
        );
        setColumns(visibleCols);
        setRows(r.rows);
        setRowCount(r.row_count);
        setAnalysis(r.analysis);

        const meta: Record<string, { column_type: string }> = {};
        for (const col of info.columns) {
          meta[col.name] = { column_type: col.column_type };
        }
        setColMeta(meta);
      }
    ).catch((e) => setError(e.message));
  }, []);

  const traceColumns = columns.filter((c) => {
    return rows.length > 0 && rows[0][`${c}__trace`] !== undefined;
  });

  const handleRowClick = (rowIdx: number) => {
    if (expandedRow === rowIdx) {
      setExpandedRow(null);
      setTraceColumn(null);
      setTrace([]);
    } else {
      setExpandedRow(rowIdx);
      setTraceColumn(null);
      setTrace([]);
    }
  };

  const handleTraceSelect = async (rowIdx: number, col: string) => {
    setTraceColumn(col);
    setLoadingTrace(true);
    try {
      const t = await api.getTrace(rowIdx, col);
      setTrace(t);
    } catch {
      setTrace([]);
    } finally {
      setLoadingTrace(false);
    }
  };

  const formatCellValue = (val: unknown): string => {
    if (val === null || val === undefined) return "";
    if (typeof val === "object") return JSON.stringify(val);
    return String(val);
  };

  if (rowCount === 0) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <h1 className="text-xl font-semibold text-gray-100 mb-6">Results</h1>
        <div className="card text-center py-12">
          <Table2 size={32} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">
            No preview results yet. Run a preview from the Run page.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Results</h1>
          <p className="text-sm text-gray-500 mt-1">
            {rowCount} rows, {columns.length} columns. Click a row to inspect.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-border">
        <button
          className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
            tab === "data"
              ? "border-nvidia-green text-nvidia-green"
              : "border-transparent text-gray-400 hover:text-gray-200"
          }`}
          onClick={() => setTab("data")}
        >
          Data
        </button>
        <button
          className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
            tab === "stats"
              ? "border-nvidia-green text-nvidia-green"
              : "border-transparent text-gray-400 hover:text-gray-200"
          }`}
          onClick={() => setTab("stats")}
        >
          Statistics
        </button>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Stats tab */}
      {tab === "stats" && (
        <StatsPanel analysis={analysis} numRecords={rowCount} />
      )}

      {/* Data tab -- card-based layout: one card per row */}
      {tab === "data" && (
        <div className="space-y-2">
          {rows.map((row, idx) => {
            const isExpanded = expandedRow === idx;
            return (
              <div key={idx} className="card p-0 overflow-hidden">
                {/* Row header */}
                <button
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    isExpanded ? "bg-surface-3" : "hover:bg-surface-2"
                  }`}
                  onClick={() => handleRowClick(idx)}
                >
                  {isExpanded ? (
                    <ChevronDown size={14} className="text-gray-500 shrink-0" />
                  ) : (
                    <ChevronRight size={14} className="text-gray-500 shrink-0" />
                  )}
                  <span className="text-xs text-gray-500 w-6 shrink-0">
                    {idx + 1}
                  </span>
                  <div className="flex-1 flex flex-wrap gap-x-4 gap-y-1 min-w-0">
                    {columns.map((col) => {
                      const val = formatCellValue(row[col]);
                      const meta = colMeta[col];
                      const typeMeta = meta
                        ? COLUMN_TYPE_META[meta.column_type as ColumnType]
                        : null;
                      return (
                        <span key={col} className="text-xs max-w-[300px] truncate">
                          <span className="text-gray-500">
                            {typeMeta?.emoji ?? ""} {col}:{" "}
                          </span>
                          <span className="text-gray-300">
                            {val.length > 80 ? val.slice(0, 80) + "..." : val}
                          </span>
                        </span>
                      );
                    })}
                  </div>
                </button>

                {/* Expanded row detail */}
                {isExpanded && (
                  <div className="border-t border-border px-4 py-4 bg-surface-2 space-y-4">
                    {/* All column values */}
                    <div className="grid grid-cols-1 gap-3">
                      {columns.map((col) => {
                        const val = formatCellValue(row[col]);
                        const meta = colMeta[col];
                        const typeMeta = meta
                          ? COLUMN_TYPE_META[meta.column_type as ColumnType]
                          : null;
                        const isLong = val.length > 100;

                        return (
                          <div key={col}>
                            <div className="flex items-center gap-1.5 mb-1">
                              {typeMeta && (
                                <span
                                  className={`text-[10px] px-1 py-0.5 rounded border font-medium ${typeMeta.color}`}
                                >
                                  {typeMeta.emoji}
                                </span>
                              )}
                              <span className="text-xs font-medium text-gray-400">
                                {col}
                              </span>
                            </div>
                            {isLong ? (
                              <pre className="text-xs text-gray-200 bg-surface-1 rounded p-2.5 whitespace-pre-wrap leading-relaxed max-h-48 overflow-auto">
                                {val}
                              </pre>
                            ) : (
                              <p className="text-sm text-gray-200 pl-0.5">
                                {val || <span className="text-gray-600 italic">null</span>}
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {/* Traces */}
                    {traceColumns.length > 0 && (
                      <div className="border-t border-border pt-3">
                        <div className="flex items-center gap-2 mb-3">
                          <MessageSquare size={14} className="text-gray-400" />
                          <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                            Traces
                          </span>
                          <div className="flex gap-1 ml-2">
                            {traceColumns.map((tc) => (
                              <button
                                key={tc}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleTraceSelect(idx, tc);
                                }}
                                className={`text-xs px-2 py-1 rounded transition-colors ${
                                  traceColumn === tc
                                    ? "bg-nvidia-green/20 text-nvidia-green border border-nvidia-green/40"
                                    : "bg-surface-3 text-gray-400 hover:text-gray-200 border border-border"
                                }`}
                              >
                                {tc}
                              </button>
                            ))}
                          </div>
                        </div>
                        {traceColumn && (
                          <TraceViewer trace={trace} loading={loadingTrace} />
                        )}
                        {!traceColumn && (
                          <p className="text-xs text-gray-500">
                            Select a column above to view its generation trace.
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
