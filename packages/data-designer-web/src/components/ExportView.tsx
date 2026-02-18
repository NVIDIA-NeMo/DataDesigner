import { useState, useEffect } from "react";
import { Copy, Check, Download, FileText, FileJson } from "lucide-react";
import { api } from "../hooks/useApi";

export default function ExportView() {
  const [format, setFormat] = useState<"yaml" | "json">("yaml");
  const [content, setContent] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api
      .exportConfig(format)
      .then((res) => {
        setContent(res.content);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [format]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const ext = format === "yaml" ? "yaml" : "json";
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `data_designer_config.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">
            Export Configuration
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            View and download your config as YAML or JSON
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            className={`btn-secondary flex items-center gap-1.5 ${format === "yaml" ? "border-nvidia-green text-nvidia-green" : ""}`}
            onClick={() => setFormat("yaml")}
          >
            <FileText size={14} />
            YAML
          </button>
          <button
            className={`btn-secondary flex items-center gap-1.5 ${format === "json" ? "border-nvidia-green text-nvidia-green" : ""}`}
            onClick={() => setFormat("json")}
          >
            <FileJson size={14} />
            JSON
          </button>
        </div>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20 mb-4">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      <div className="card relative">
        <div className="absolute top-3 right-3 flex gap-1">
          <button
            className="btn-ghost text-gray-400"
            onClick={handleCopy}
            title="Copy to clipboard"
          >
            {copied ? <Check size={14} className="text-nvidia-green" /> : <Copy size={14} />}
          </button>
          <button
            className="btn-ghost text-gray-400"
            onClick={handleDownload}
            title="Download file"
          >
            <Download size={14} />
          </button>
        </div>
        <pre className="text-xs font-mono text-gray-300 overflow-auto max-h-[70vh] whitespace-pre leading-relaxed">
          {content || "# Empty configuration\n# Add columns to get started"}
        </pre>
      </div>
    </div>
  );
}
