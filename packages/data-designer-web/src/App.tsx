import { useEffect, useState, useCallback, useRef } from "react";
import { Database, CheckCircle2, RefreshCw } from "lucide-react";
import { api } from "./hooks/useApi";
import ResultsPage from "./pages/ResultsPage";

export default function App() {
  const [fileName, setFileName] = useState("");
  const [rowCount, setRowCount] = useState(0);
  const [version, setVersion] = useState(0);
  const [finished, setFinished] = useState(false);
  const [summary, setSummary] = useState({ good: 0, bad: 0, unreviewed: 0, total: 0 });
  const [finishMessage, setFinishMessage] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const refreshInfo = useCallback(async () => {
    try {
      const [session, sum] = await Promise.all([
        api.getSession(),
        api.getAnnotationsSummary(),
      ]);
      const versionChanged = session.version !== version && version !== 0;
      setFileName(session.file_name);
      setRowCount(session.row_count);
      setVersion(session.version);
      setFinished(session.finished);
      setSummary(sum);
      return versionChanged;
    } catch {
      return false;
    }
  }, [version]);

  useEffect(() => {
    refreshInfo();
    // Poll every 5s to detect agent-triggered reloads
    pollRef.current = window.setInterval(async () => {
      const changed = await refreshInfo();
      if (changed) {
        // Version changed (agent called /session/reload) -- trigger re-render
        window.dispatchEvent(new CustomEvent("session-reloaded"));
      }
    }, 5000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [refreshInfo]);

  const handleFinish = async () => {
    try {
      const result = await api.finish();
      setFinished(true);
      setFinishMessage(
        `Review saved to ${(result as any).file ?? "annotations file"}. Return to your coding agent to continue.`
      );
    } catch {
      // ignore
    }
  };

  const handleManualReload = async () => {
    try {
      await api.reload();
      await refreshInfo();
      window.dispatchEvent(new CustomEvent("session-reloaded"));
    } catch {
      // ignore
    }
  };

  const reviewed = summary.good + summary.bad;

  return (
    <div className="flex flex-col h-screen">
      {/* Top bar */}
      <header className="bg-surface-1 border-b border-border px-6 py-3 flex items-center gap-4 shrink-0">
        <Database size={20} className="text-nvidia-green" />
        <span className="font-semibold text-sm text-gray-100">
          Data Designer Review
        </span>
        <span className="text-xs text-gray-500 font-mono">{fileName}</span>
        <span className="text-xs text-gray-600">{rowCount} records</span>

        {/* Annotation progress */}
        <div className="flex items-center gap-2 ml-auto">
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-green-400">{summary.good} good</span>
            <span className="text-gray-600">/</span>
            <span className="text-red-400">{summary.bad} bad</span>
            <span className="text-gray-600">/</span>
            <span className="text-gray-500">{summary.unreviewed} left</span>
          </div>
          {rowCount > 0 && (
            <div className="w-24 h-1.5 bg-surface-3 rounded-full overflow-hidden">
              <div
                className="h-full bg-nvidia-green rounded-full transition-all"
                style={{ width: `${(reviewed / rowCount) * 100}%` }}
              />
            </div>
          )}
          <button
            className="btn-ghost text-gray-500"
            onClick={handleManualReload}
            title="Reload data from disk"
          >
            <RefreshCw size={14} />
          </button>
          <button
            className="btn-primary flex items-center gap-1.5 !py-1.5 !text-xs"
            onClick={handleFinish}
            disabled={finished}
          >
            <CheckCircle2 size={14} />
            {finished ? "Review Complete" : "Finish Review"}
          </button>
        </div>
      </header>

      {/* Finish message */}
      {finishMessage && (
        <div className="bg-green-900/20 border-b border-green-700/30 px-6 py-2 text-sm text-green-300">
          {finishMessage}
        </div>
      )}

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <ResultsPage />
      </main>
    </div>
  );
}
