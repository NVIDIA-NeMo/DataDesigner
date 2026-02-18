import { useEffect, useState } from "react";
import { Table2, ChevronDown, ChevronRight, MessageSquare } from "lucide-react";
import { api } from "../hooks/useApi";
import TraceViewer from "../components/TraceViewer";

export default function ResultsPage() {
  const [columns, setColumns] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [rowCount, setRowCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [traceColumn, setTraceColumn] = useState<string | null>(null);
  const [trace, setTrace] = useState<Record<string, unknown>[]>([]);
  const [loadingTrace, setLoadingTrace] = useState(false);

  useEffect(() => {
    api
      .getPreviewResults()
      .then((r) => {
        const visibleCols = r.columns.filter(
          (c) => !c.endsWith("__trace") && !c.endsWith("__reasoning_content")
        );
        setColumns(visibleCols);
        setRows(r.rows);
        setRowCount(r.row_count);
      })
      .catch((e) => setError(e.message));
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

  const truncate = (val: unknown, max = 120): string => {
    if (val === null || val === undefined) return "";
    const s = typeof val === "object" ? JSON.stringify(val) : String(val);
    return s.length > max ? s.slice(0, max) + "..." : s;
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
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Results</h1>
          <p className="text-sm text-gray-500 mt-1">
            {rowCount} rows generated. Click a row to inspect traces.
          </p>
        </div>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      <div className="overflow-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-surface-2 border-b border-border">
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-400 w-8">
                #
              </th>
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left text-xs font-medium text-gray-400 max-w-[200px]"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <>
                <tr
                  key={idx}
                  className={`border-b border-border cursor-pointer transition-colors ${
                    expandedRow === idx
                      ? "bg-surface-3"
                      : "hover:bg-surface-2"
                  }`}
                  onClick={() => handleRowClick(idx)}
                >
                  <td className="px-3 py-2 text-gray-500">
                    <div className="flex items-center gap-1">
                      {expandedRow === idx ? (
                        <ChevronDown size={12} />
                      ) : (
                        <ChevronRight size={12} />
                      )}
                      {idx + 1}
                    </div>
                  </td>
                  {columns.map((col) => (
                    <td
                      key={col}
                      className="px-3 py-2 text-gray-300 max-w-[200px] truncate"
                      title={String(row[col] ?? "")}
                    >
                      {truncate(row[col])}
                    </td>
                  ))}
                </tr>
                {expandedRow === idx && (
                  <tr key={`${idx}-expand`} className="bg-surface-2">
                    <td colSpan={columns.length + 1} className="px-4 py-4">
                      {traceColumns.length > 0 ? (
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <MessageSquare
                              size={14}
                              className="text-gray-400"
                            />
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
                            <TraceViewer
                              trace={trace}
                              loading={loadingTrace}
                            />
                          )}
                          {!traceColumn && (
                            <p className="text-xs text-gray-500">
                              Select a column above to view its trace.
                            </p>
                          )}
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <p className="text-xs text-gray-500 mb-2">
                            No traces available. Enable debug mode when running
                            preview to capture LLM conversation traces.
                          </p>
                          <div className="grid grid-cols-2 gap-3">
                            {columns.map((col) => (
                              <div key={col}>
                                <span className="text-xs text-gray-500">
                                  {col}
                                </span>
                                <pre className="text-xs text-gray-300 bg-surface-1 rounded p-2 mt-1 max-h-32 overflow-auto whitespace-pre-wrap">
                                  {typeof row[col] === "object"
                                    ? JSON.stringify(row[col], null, 2)
                                    : String(row[col] ?? "")}
                                </pre>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
