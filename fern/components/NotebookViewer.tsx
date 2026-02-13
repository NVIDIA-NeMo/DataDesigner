/**
 * NotebookViewer - Renders Jupyter notebook content in Fern docs.
 *
 * Accepts notebook cells (markdown + code) and optionally a Colab URL.
 * Designed to work with Jupytext-generated notebooks from docs/notebook_source/*.py.
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Do NOT import React -- the `react` module is not resolvable in Fern's build.
 * This means class components (e.g. ErrorBoundary) are also not available.
 *
 * Usage in MDX:
 *   import { NotebookViewer } from "@/components/NotebookViewer";
 *   import notebook from "@/components/notebooks/1-the-basics";
 *
 *   <NotebookViewer
 *     notebook={notebook}
 *     colabUrl="https://colab.research.google.com/github/NVIDIA-NeMo/DataDesigner/blob/main/docs/colab_notebooks/1-the-basics.ipynb"
 *   />
 */

export interface CellOutput {
  type: "text" | "image";
  data: string;
  format?: "plain" | "html";
}

export interface NotebookCell {
  type: "markdown" | "code";
  source: string;
  language?: string;
  outputs?: CellOutput[];
}

export interface NotebookData {
  cells: NotebookCell[];
}

export interface NotebookViewerProps {
  /** Notebook data with cells array. If import fails, this may be undefined. */
  notebook?: NotebookData | null;
  /** Optional Colab URL for "Run in Colab" badge */
  colabUrl?: string;
  /** Show code cell outputs (default: true) */
  showOutputs?: boolean;
}

function NotebookViewerError({ message, detail }: { message: string; detail?: string }) {
  return (
    <div
      className="notebook-viewer__error"
      style={{
        padding: "1rem",
        margin: "1rem 0",
        background: "#fef2f2",
        border: "1px solid #fecaca",
        borderRadius: "8px",
        color: "#991b1b",
        fontFamily: "monospace",
        fontSize: "0.875rem",
      }}
    >
      <strong>NotebookViewer error:</strong> {message}
      {detail && (
        <pre style={{ marginTop: "0.5rem", overflow: "auto", whiteSpace: "pre-wrap" }}>
          {detail}
        </pre>
      )}
    </div>
  );
}

function escapeHtml(text: string): string {
  if (typeof text !== "string") return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function isSafeUrl(url: string): boolean {
  const trimmed = url.trim();
  return (
    trimmed.startsWith("http://") ||
    trimmed.startsWith("https://") ||
    trimmed.startsWith("mailto:") ||
    trimmed.startsWith("#") ||
    trimmed.startsWith("/")
  );
}

function renderMarkdown(markdown: string): string {
  if (typeof markdown !== "string") return "";
  let html = markdown
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) =>
      isSafeUrl(url)
        ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${text}</a>`
        : escapeHtml(`[${text}](${url})`)
    )
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html
    .split("\n")
    .map((line) => {
      if (/^#### (.*)$/.test(line)) return `<h4>${line.slice(5)}</h4>`;
      if (/^### (.*)$/.test(line)) return `<h3>${line.slice(4)}</h3>`;
      if (/^## (.*)$/.test(line)) return `<h2>${line.slice(3)}</h2>`;
      if (/^# (.*)$/.test(line)) return `<h1>${line.slice(2)}</h1>`;
      if (/^- (.*)$/.test(line)) return `<li>${line.slice(2)}</li>`;
      if (/^\d+\. (.*)$/.test(line)) return `<li>${line.replace(/^\d+\. /, "")}</li>`;
      if (line.trim() === "") return "";
      return `<p>${line}</p>`;
    })
    .join("\n");
  return html.replace(/(<li>.*?<\/li>\n?)+/gs, (m) => `<ul>${m}</ul>`);
}

function renderCell(cell: NotebookCell, index: number, showOutputs: boolean) {
  return (
    <div
      key={index}
      className={`notebook-viewer__cell notebook-viewer__cell--${cell.type}`}
    >
      {cell.type === "markdown" ? (
        <div
          className="notebook-viewer__markdown"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(cell.source) }}
        />
      ) : (
        <>
          <div className="notebook-viewer__code-wrapper">
            <pre className="notebook-viewer__code">
              <code
                className={`language-${cell.language || "python"}`}
                dangerouslySetInnerHTML={{ __html: escapeHtml(cell.source) }}
              />
            </pre>
          </div>
          {showOutputs && cell.outputs && cell.outputs.length > 0 && (
            <div className="notebook-viewer__outputs">
              {cell.outputs.map((out, i) =>
                out.type === "image" ? (
                  <img
                    key={i}
                    src={`data:image/png;base64,${out.data}`}
                    alt="Output"
                    className="notebook-viewer__output-image"
                  />
                ) : out.format === "html" ? (
                  <div
                    key={i}
                    className="notebook-viewer__output-html"
                    dangerouslySetInnerHTML={{ __html: out.data }}
                  />
                ) : (
                  <pre
                    key={i}
                    className="notebook-viewer__output-text"
                    dangerouslySetInnerHTML={{ __html: escapeHtml(out.data) }}
                  />
                )
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export const NotebookViewer = ({
  notebook,
  colabUrl,
  showOutputs = true,
}: NotebookViewerProps) => {
  if (notebook == null || typeof notebook !== "object") {
    return (
      <NotebookViewerError
        message="Notebook data is missing or invalid"
        detail={`Received: ${typeof notebook}. Run 'make generate-fern-notebooks' and ensure the import path is correct.`}
      />
    );
  }

  const cells = notebook?.cells;
  if (!Array.isArray(cells)) {
    return (
      <NotebookViewerError
        message="Notebook must have a 'cells' array"
        detail={`Received keys: ${Object.keys(notebook).join(", ")}`}
      />
    );
  }

  return (
    <div className="notebook-viewer">
      {colabUrl && (
        <div className="notebook-viewer__colab-banner">
          <a
            href={colabUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="notebook-viewer__colab-link"
          >
            <span className="notebook-viewer__colab-icon">&#9654;</span>
            Run in Google Colab
          </a>
        </div>
      )}

      <div className="notebook-viewer__cells">
        {cells.map((cell, index) => renderCell(cell, index, showOutputs))}
      </div>
    </div>
  );
};
