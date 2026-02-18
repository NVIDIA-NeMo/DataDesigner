import type { ReactNode } from "react";

/**
 * NotebookViewer - Renders Jupyter notebook content in Fern docs.
 *
 * Uses Fern's code block structure (fern-code, fern-code-block, etc.) so input
 * and output cells match the default Fern code block styling.
 *
 * Accepts notebook cells (markdown + code) and optionally a Colab URL.
 * Designed to work with Jupytext-generated notebooks from docs/notebook_source/*.py.
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Only type-only imports from "react" are used (erased at compile time).
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
  /** Pre-rendered syntax-highlighted HTML (from Pygments). When present, used instead of escaped source. */
  source_html?: string;
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

const UL_CLASS =
  "[&>li]:relative [&>li]:before:text-(color:--grayscale-a10) mb-3 list-none pl-3 [&>li]:pl-3 [&>li]:before:absolute [&>li]:before:ml-[-22px] [&>li]:before:mt-[-1px] [&>li]:before:content-['⦁'] [&>li]:before:self-center";
const OL_CLASS = "mb-3 list-outside list-decimal [&_ol]:!list-[lower-roman]";

function renderMarkdown(markdown: string): string {
  if (typeof markdown !== "string") return "";
  let html = markdown
    .replace(/<br\s*\/?>/gi, "\u0000BR\u0000")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\u0000BR\u0000/g, "<br />")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => {
      if (!isSafeUrl(url)) return escapeHtml(`[${text}](${url})`);
      const isInternal = url.startsWith("/") || url.startsWith("#");
      const attrs = isInternal
        ? `href="${escapeHtml(url)}" class="fern-mdx-link"`
        : `href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="fern-mdx-link"`;
      const icon =
        isInternal
          ? ""
          : '<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-external-link external-link-icon inline-block ml-0.5 align-middle" aria-hidden="true"><path d="M15 3h6v6"></path><path d="M10 14 21 3"></path><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path></svg>';
      return `<a ${attrs}>${text}${icon}</a>`;
    })
    .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html
    .split("\n")
    .map((line) => {
      if (/^#### (.*)$/.test(line)) return `<h4>${line.slice(5)}</h4>`;
      if (/^### (.*)$/.test(line)) return `<h3>${line.slice(4)}</h3>`;
      if (/^## (.*)$/.test(line)) return `<h2>${line.slice(3)}</h2>`;
      if (/^# (.*)$/.test(line)) return `<h1>${line.slice(2)}</h1>`;
      if (/^- (.*)$/.test(line)) return `<li data-ul>${line.slice(2)}</li>`;
      if (/^\d+\. (.*)$/.test(line)) return `<li data-ol>${line.replace(/^\d+\. /, "")}</li>`;
      if (line.trim() === "") return "";
      return `<p>${line}</p>`;
    })
    .join("\n");
  html = html.replace(
    /(<li data-ol>.*?<\/li>\s*)+/gs,
    (m) => `<ol class="${OL_CLASS}">${m.replace(/ data-ol/g, "").trim()}</ol>`
  );
  html = html.replace(
    /(<li data-ul>.*?<\/li>\s*)+/gs,
    (m) => `<ul class="${UL_CLASS}">${m.replace(/ data-ul/g, "").trim()}</ul>`
  );
  return html;
}

function handleCopy(content: string, button: HTMLButtonElement) {
  navigator.clipboard.writeText(content).catch(() => {});
  const originalHtml = button.innerHTML;
  const originalLabel = button.getAttribute("aria-label") ?? "Copy code";
  button.innerHTML = "Copied!";
  button.setAttribute("aria-label", "Copied to clipboard");
  setTimeout(() => {
    button.innerHTML = originalHtml;
    button.setAttribute("aria-label", originalLabel);
  }, 1500);
}

const FLAG_ICON = (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
    <line x1="4" x2="4" y1="22" y2="15" />
  </svg>
);

const SCROLL_AREA_STYLE = `[data-radix-scroll-area-viewport]{scrollbar-width:none;-ms-overflow-style:none;-webkit-overflow-scrolling:touch;}[data-radix-scroll-area-viewport]::-webkit-scrollbar{display:none}`;

const BUTTON_BASE_CLASS =
  "focus-visible:ring-(color:--accent) rounded-2 inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-colors hover:transition-none focus-visible:outline-none focus-visible:ring-1 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 text-(color:--grayscale-a11) hover:bg-(color:--accent-a3) hover:text-(color:--accent-11) pointer-coarse:size-9 size-7";

/** Fern code block structure – matches Fern docs (header with language + buttons, pre with scroll area). */
function FernCodeBlock({
  title,
  children,
  className = "",
  asPre = true,
  copyContent,
  showLineNumbers = false,
  codeHtml,
}: {
  title: string;
  children: ReactNode;
  className?: string;
  /** Use div instead of pre for content (needed when children include block elements like img/div). */
  asPre?: boolean;
  /** Raw text to copy when copy button is clicked. When provided, shows a copy button. */
  copyContent?: string;
  /** Show line numbers in a table layout (matches Fern's code block structure). */
  showLineNumbers?: boolean;
  /** Pre-rendered HTML for each line when showLineNumbers is true. Lines are split by newline. */
  codeHtml?: string;
}) {
  const headerLabel = title === "Output" ? "Output" : title.charAt(0).toUpperCase() + title.slice(1);
  const wrapperClasses =
    "fern-code fern-code-block bg-card-background border-card-border rounded-3 shadow-card-grayscale relative mb-6 mt-4 flex w-full min-w-0 max-w-full flex-col border first:mt-0";
  const preStyle = {
    backgroundColor: "rgb(255, 255, 255)",
    ["--shiki-dark-bg" as string]: "#212121",
    color: "rgb(36, 41, 46)",
    ["--shiki-dark" as string]: "#EEFFFF",
  };

  const scrollAreaContent = () => {
    if (codeHtml == null) return null;
    const lines = codeHtml.split("\n");
    return (
      <div
        dir="ltr"
        className="fern-scroll-area"
        style={{
          position: "relative",
          ["--radix-scroll-area-corner-width" as string]: "0px",
          ["--radix-scroll-area-corner-height" as string]: "0px",
        }}
      >
        <style dangerouslySetInnerHTML={{ __html: SCROLL_AREA_STYLE }} />
        <div
          data-radix-scroll-area-viewport=""
          className="fern-scroll-area-viewport"
          data-scrollbars="both"
          style={{ overflow: "scroll", maxHeight: "479px" }}
        >
          <div style={{ minWidth: "100%", display: "table" }}>
            <div className="code-block text-sm">
              <div className="code-block-inner">
                <table className="code-block-line-group">
                  <colgroup>
                    <col className="w-fit" />
                    <col />
                  </colgroup>
                  <tbody>
                    {lines.map((line, i) => (
                      <tr key={i} className="code-block-line">
                        <td className="code-block-line-gutter">
                          <span>{i + 1}</span>
                        </td>
                        <td className="code-block-line-content">
                          <span
                            className="line"
                            dangerouslySetInnerHTML={{
                              __html: line || " ",
                            }}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const codeBlockContent = showLineNumbers ? scrollAreaContent() : children;
  const isOutput = title === "Output";

  return (
    <div
      className={`${wrapperClasses} ${className}`}
      data-block-type={isOutput ? "output" : "code"}
    >
      <div className="fern-code-header fern-code-block-header bg-(color:--grayscale-a2) rounded-t-[inherit]">
        <div className="fern-code-header-inner fern-code-block-header-inner shadow-border-default mx-px flex min-h-10 items-center justify-between shadow-[inset_0_-1px_0_0]">
          <div className="fern-code-block-title flex min-h-10 overflow-x-auto">
            <div className="flex items-center px-3 py-1.5">
              <span className="fern-code-label fern-code-block-title-label text-(color:--grayscale-a11) rounded-1 text-sm font-semibold">
                {headerLabel}
              </span>
            </div>
          </div>
          <div className="fern-code-actions fern-code-block-actions flex items-center gap-1">
            <span className="inline-flex" role="button" aria-haspopup="dialog" aria-expanded="false" aria-label="Report incorrect code">
              <button type="button" className={`${BUTTON_BASE_CLASS} fern-feedback-button z-20`} aria-label="Report incorrect code">
                {FLAG_ICON}
              </button>
            </span>
            {copyContent != null && (
              <button
                type="button"
                className={`${BUTTON_BASE_CLASS} fern-copy-button group mr-1`}
                aria-label="Copy code"
                onClick={(e) => handleCopy(copyContent, e.currentTarget)}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                  <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
      {asPre ? (
        <pre
          className="code-block-root not-prose fern-code-content fern-code-block-content rounded-b-[inherit]"
          tabIndex={0}
          style={preStyle}
        >
          {codeBlockContent}
        </pre>
      ) : (
        <div className="code-block-root not-prose fern-code-content fern-code-block-content rounded-b-[inherit] notebook-viewer__output-content" tabIndex={0}>
          {codeBlockContent}
        </div>
      )}
    </div>
  );
}

function renderCell(cell: NotebookCell, index: number, showOutputs: boolean) {
  return (
    <div
      key={index}
      className={`notebook-viewer__cell notebook-viewer__cell--${cell.type}`}
    >
      {cell.type === "markdown" ? (
        <div
          className="notebook-viewer__markdown fern-prose prose break-words prose-h1:mt-[1.5em] first:prose-h1:mt-0 max-w-full"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(cell.source) }}
        />
      ) : (
        <>
          <FernCodeBlock
            title={cell.language || "python"}
            copyContent={cell.source}
            showLineNumbers
            codeHtml={cell.source_html ?? escapeHtml(cell.source)}
          >
            <code
              className={`language-${cell.language || "python"}`}
              dangerouslySetInnerHTML={{
                __html: cell.source_html ?? escapeHtml(cell.source),
              }}
            />
          </FernCodeBlock>
          {showOutputs && cell.outputs && cell.outputs.length > 0 && (
            <FernCodeBlock title="Output" className="notebook-viewer__output-block" asPre={false}>
              <div className="notebook-viewer__outputs-inner">
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
            </FernCodeBlock>
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
            className="fern-button success filled notebook-viewer__colab-link"
          >
            <span className="fern-button-content">
              <span aria-hidden="true">&#9654;</span>
              <span className="fern-button-text">Run in Google Colab</span>
            </span>
          </a>
        </div>
      )}

      <div className="notebook-viewer__cells">
        {cells.map((cell, index) => renderCell(cell, index, showOutputs))}
      </div>
    </div>
  );
};
