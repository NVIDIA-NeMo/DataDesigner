/**
 * ExpandableCode - Collapsible code block with summary.
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
 *     code={`
 *       import data_designer.config as dd
 *       ...
 *     `}
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

export const ExpandableCode = ({
  summary,
  code,
  language = "python",
  defaultOpen = false,
}: ExpandableCodeProps) => {
  return (
    <details className="expandable-code" open={defaultOpen}>
      <summary className="expandable-code__summary">
        <strong>{summary}</strong>
      </summary>
      <div className="expandable-code__content">
        <pre className="expandable-code__pre">
          <code className={`language-${language}`}>{code.trim()}</code>
        </pre>
      </div>
    </details>
  );
};
