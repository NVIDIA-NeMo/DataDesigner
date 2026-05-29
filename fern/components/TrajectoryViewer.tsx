/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * TrajectoryViewer - Renders multi-turn research trajectories with tool calls.
 *
 * Displays search, open, find, and answer steps with color-coded styling.
 * Used for deep research / MCP tool-use dev notes.
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Do NOT import React -- the `react` module is not resolvable in Fern's build.
 *
 * Usage in MDX:
 *   import { TrajectoryViewer } from "@/components/TrajectoryViewer";
 *   import trajectory from "@/components/devnotes/<post-slug>/<example>";
 *
 *   <TrajectoryViewer {...trajectory} defaultOpen />
 */

export interface ToolCall {
  fn: "search" | "open" | "find" | "answer";
  arg?: string;
  /**
   * Final-answer HTML body. Rendered via `dangerouslySetInnerHTML` — must be
   * pre-rendered HTML, NOT raw markdown. Use `<br />` for line breaks and
   * `<strong>...</strong>` for emphasis. Pure-text answers can be safely
   * passed too; HTML special chars are not escaped, so the fixture data is
   * the trust boundary (same model as NotebookViewer's HTML output cells).
   */
  body?: string;
  isGolden?: boolean;
}

export interface TrajectoryTurn {
  turnIndex: number;
  calls: ToolCall[];
}

export interface TrajectoryViewerProps {
  question: string;
  referenceAnswer?: string;
  goldenPassageHint?: string;
  turns: TrajectoryTurn[];
  summary?: string;
  defaultOpen?: boolean;
}

/**
 * Trajectory styles, injected by the component rather than loaded via docs.yml
 * `css:`. `css` is theme-owned, so under `global-theme: nvidia` a local `css:`
 * list is dropped at publish — styling ships with the component. See fern/docs.yml.
 */
const TRAJECTORY_VIEWER_CSS = `
.trajectory-viewer {
  font-family: -apple-system, system-ui, sans-serif;
  max-width: 960px;
  margin: 16px 0;
  padding: 0;
}
.trajectory-viewer__details {
  margin: 1rem 0;
}
.trajectory-viewer__summary {
  cursor: pointer;
  padding: 0.5rem 0;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.trajectory-viewer__stats {
  font-size: 0.8em;
  opacity: 0.6;
  font-weight: normal;
  font-family: "SF Mono", Menlo, Monaco, "Cascadia Code", monospace;
}
.trajectory-viewer__icon {
  margin-right: 0.35em;
  font-size: 0.9em;
}
.trajectory-viewer__question {
  background: rgba(66, 165, 245, 0.08);
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 8px;
}
.trajectory-viewer__question strong {
  color: #42a5f5;
}
.trajectory-viewer__ref {
  background: rgba(76, 175, 80, 0.08);
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  border-left: 4px solid #4caf50;
}
.trajectory-viewer__ref strong {
  color: #66bb6a;
}
.trajectory-viewer__hint {
  opacity: 0.5;
  font-size: 0.8em;
  margin-bottom: 12px;
}
.trajectory-viewer__turn {
  margin: 6px 0;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.trajectory-viewer__label {
  min-width: 48px;
  padding: 6px 0;
  opacity: 0.5;
  font-size: 0.75em;
  font-family: "SF Mono", Menlo, Monaco, "Cascadia Code", monospace;
  text-align: right;
  flex-shrink: 0;
}
.trajectory-viewer__body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.trajectory-viewer__group {
  display: flex;
  flex-direction: column;
  gap: 3px;
  position: relative;
}
.trajectory-viewer__group--multi {
  padding-left: 13px;
}
.trajectory-viewer__group--multi::before {
  content: "";
  position: absolute;
  left: 0;
  top: 4px;
  bottom: 4px;
  width: 3px;
  background: rgba(128, 128, 128, 0.3);
  border-radius: 2px;
}
.trajectory-viewer__call {
  padding: 5px 12px;
  border-radius: 5px;
  font-family: "SF Mono", Menlo, Monaco, "Cascadia Code", monospace;
  font-size: 0.82em;
  display: flex;
  gap: 8px;
}
.trajectory-viewer__call .trajectory-viewer__fn {
  font-weight: bold;
  min-width: 55px;
  flex-shrink: 0;
}
.trajectory-viewer__call .trajectory-viewer__arg {
  opacity: 0.85;
}
.trajectory-viewer__call--search {
  background: rgba(66, 165, 245, 0.1);
  border-left: 3px solid #42a5f5;
}
.trajectory-viewer__call--search .trajectory-viewer__fn {
  color: #42a5f5;
}
.trajectory-viewer__call--open {
  background: rgba(102, 187, 106, 0.1);
  border-left: 3px solid #66bb6a;
}
.trajectory-viewer__call--open .trajectory-viewer__fn {
  color: #66bb6a;
}
.trajectory-viewer__call--find {
  background: rgba(255, 167, 38, 0.1);
  border-left: 3px solid #ffa726;
}
.trajectory-viewer__call--find .trajectory-viewer__fn {
  color: #ffa726;
}
.trajectory-viewer__call--answer {
  background: rgba(76, 175, 80, 0.08);
  border-left: 3px solid #4caf50;
  padding: 10px 16px;
  border-radius: 6px;
  font-size: 0.88em;
  line-height: 1.5;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}
.trajectory-viewer__call--answer .trajectory-viewer__fn {
  color: #4caf50;
  font-weight: bold;
  font-family: "SF Mono", Menlo, Monaco, "Cascadia Code", monospace;
}
.trajectory-viewer__call--answer .trajectory-viewer__body {
  width: 100%;
  font-family: -apple-system, system-ui, sans-serif;
  font-size: 1em;
}
`;

const TOOL_ICONS: Record<string, string> = {
  search: "🔍",
  open: "📄",
  find: "🔎",
  answer: "✓",
};

function ToolCallBlock({ call }: { call: ToolCall }) {
  const isAnswer = call.fn === "answer";
  const argDisplay = call.arg ?? "";
  const cn = `trajectory-viewer__call trajectory-viewer__call--${call.fn}`;
  const icon = TOOL_ICONS[call.fn] ?? "";

  if (isAnswer && call.body) {
    return (
      <div className={cn}>
        <span className="trajectory-viewer__fn">
          {icon && <span className="trajectory-viewer__icon">{icon}</span>}
          {call.fn}
        </span>
        <div
          className="trajectory-viewer__body"
          dangerouslySetInnerHTML={{ __html: call.body }}
        />
      </div>
    );
  }

  return (
    <div className={cn}>
      <span className="trajectory-viewer__fn">
        {icon && <span className="trajectory-viewer__icon">{icon}</span>}
        {call.fn}
      </span>
      <span className="trajectory-viewer__arg">
        {argDisplay}
        {call.isGolden && " ⭐"}
      </span>
    </div>
  );
}

export const TrajectoryViewer = ({
  question,
  referenceAnswer,
  goldenPassageHint,
  turns,
  summary,
  defaultOpen = false,
}: TrajectoryViewerProps) => {
  const content = (
    <div className="trajectory-viewer">
      <style dangerouslySetInnerHTML={{ __html: TRAJECTORY_VIEWER_CSS }} />
      <div className="trajectory-viewer__question">
        <strong>Q:</strong> {question}
      </div>
      {referenceAnswer && (
        <div className="trajectory-viewer__ref">
          <strong>Reference:</strong> {referenceAnswer}
        </div>
      )}
      {goldenPassageHint && (
        <div className="trajectory-viewer__hint">{goldenPassageHint}</div>
      )}
      <div className="trajectory-viewer__turns">
        {turns.map((turn) => (
          <div key={turn.turnIndex} className="trajectory-viewer__turn">
            <div className="trajectory-viewer__label">T{turn.turnIndex}</div>
            <div className="trajectory-viewer__body">
              <div
                className={`trajectory-viewer__group ${
                  turn.calls.length > 1 ? "trajectory-viewer__group--multi" : ""
                }`}
              >
                {turn.calls.map((call, i) => (
                  <ToolCallBlock key={i} call={call} />
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const totalCalls = turns.reduce((acc, t) => acc + t.calls.length, 0);

  if (summary) {
    return (
      <details className="trajectory-viewer__details" open={defaultOpen}>
        <summary className="trajectory-viewer__summary">
          <strong>{summary}</strong>
          <span className="trajectory-viewer__stats">
            {turns.length} turns · {totalCalls} calls
          </span>
        </summary>
        {content}
      </details>
    );
  }

  return content;
};
