import { useState } from "react";
import {
  Columns3,
  Settings,
  FileDown,
  Database,
  LayoutDashboard,
} from "lucide-react";
import ConfigBuilder from "./pages/ConfigBuilder";
import ModelsPanel from "./components/ModelsPanel";
import ExportView from "./components/ExportView";

type Page = "builder" | "models" | "export";

const NAV_ITEMS: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: "builder", label: "Config Builder", icon: <LayoutDashboard size={18} /> },
  { id: "models", label: "Models", icon: <Settings size={18} /> },
  { id: "export", label: "Export", icon: <FileDown size={18} /> },
];

export default function App() {
  const [page, setPage] = useState<Page>("builder");

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
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

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {page === "builder" && <ConfigBuilder />}
        {page === "models" && <ModelsPanel />}
        {page === "export" && <ExportView />}
      </main>
    </div>
  );
}
