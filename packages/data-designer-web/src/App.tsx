import { useState } from "react";
import { FileText, Play, Table2, Database } from "lucide-react";
import ConfigPage from "./pages/ConfigPage";
import RunPage from "./pages/RunPage";
import ResultsPage from "./pages/ResultsPage";

type Page = "config" | "run" | "results";

const NAV_ITEMS: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: "config", label: "Config", icon: <FileText size={18} /> },
  { id: "run", label: "Run", icon: <Play size={18} /> },
  { id: "results", label: "Results", icon: <Table2 size={18} /> },
];

export default function App() {
  const [page, setPage] = useState<Page>("config");

  return (
    <div className="flex h-screen">
      <aside className="w-56 bg-surface-1 border-r border-border flex flex-col shrink-0">
        <div className="p-4 border-b border-border flex items-center gap-2">
          <Database size={20} className="text-nvidia-green" />
          <span className="font-semibold text-sm">Data Designer</span>
        </div>
        <nav className="flex-1 p-2 space-y-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors ${
                page === item.id
                  ? "bg-nvidia-green/10 text-nvidia-green"
                  : "text-gray-400 hover:bg-surface-3 hover:text-gray-200"
              }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-border text-xs text-gray-500">
          NeMo Data Designer
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        {page === "config" && <ConfigPage />}
        {page === "run" && <RunPage />}
        {page === "results" && <ResultsPage />}
      </main>
    </div>
  );
}
