/**
 * ExpandableCode - Collapsible code block with summary and copy button.
 *
 * Used for "Full source" code snippets in dev notes.
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Do NOT import React -- the `react` module is not resolvable in Fern's build.
 *
 * Usage in MDX:
 *   import { ExpandableCode } from "@/components/ExpandableCode";
 *
 *   <ExpandableCode
 *     summary="Full source: openresearcher_demo.py"
 *     code={`...`}
 *     language="python"
 *     defaultOpen={false}
 *   />
 */

export interface ExpandableCodeProps {
  summary: string;
  code: string;
  language?: string;
  defaultOpen?: boolean;
}

function CopyButton({ text }: { text: string }) {
  return (
    <button
      type="button"
      className="expandable-code__copy"
      onClick={(e) => {
        navigator.clipboard?.writeText(text).then(() => {
          const btn = e.currentTarget as HTMLButtonElement;
          const orig = btn.textContent;
          btn.textContent = "Copied!";
          btn.classList.add("expandable-code__copy--copied");
          setTimeout(() => {
            btn.textContent = orig;
            btn.classList.remove("expandable-code__copy--copied");
          }, 1500);
        });
      }}
    >
      Copy
    </button>
  );
}

export const ExpandableCode = ({
  summary,
  code,
  language = "python",
  defaultOpen = false,
}: ExpandableCodeProps) => {
  const trimmed = code.trim();
  return (
    <details className="expandable-code" open={defaultOpen}>
      <summary className="expandable-code__summary">
        <strong>{summary}</strong>
        <span className="expandable-code__badge">{language}</span>
      </summary>
      <div className="expandable-code__content">
        <div className="expandable-code__toolbar">
          <CopyButton text={trimmed} />
        </div>
        <pre className="expandable-code__pre">
          <code className={`language-${language}`}>{trimmed}</code>
        </pre>
      </div>
    </details>
  );
};
