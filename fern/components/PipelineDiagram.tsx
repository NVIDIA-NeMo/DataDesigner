/**
 * PipelineDiagram - Renders ASCII pipeline diagrams with monospace styling.
 *
 * Used for SDG stage diagrams in design-principles dev note.
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Do NOT import React -- the `react` module is not resolvable in Fern's build.
 *
 * Usage in MDX:
 *   import { PipelineDiagram } from "@/components/PipelineDiagram";
 *
 *   <PipelineDiagram diagram={`
 *         Seed Documents         Seed dataset column ingests...
 *               │
 *               ▼
 *     ┌─────────────────────────┐
 *     │  Artifact Extraction    │
 *     └───────────┬─────────────┘
 *   `} />
 */

export interface PipelineDiagramProps {
  diagram: string;
  title?: string;
  maxWidth?: string;
}

export const PipelineDiagram = ({
  diagram,
  title,
  maxWidth = "640px",
}: PipelineDiagramProps) => {
  return (
    <div className="pipeline-diagram" style={{ maxWidth }}>
      {title && <div className="pipeline-diagram__title">{title}</div>}
      <pre className="pipeline-diagram__pre">
        <code className="pipeline-diagram__code">{diagram.trim()}</code>
      </pre>
    </div>
  );
};
