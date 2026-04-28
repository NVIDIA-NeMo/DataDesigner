/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Include - Extract and render sections from imported MDX/Markdown content.
 *
 * Replicates Sphinx's `.. include::` directive behavior within Fern's
 * client-side constraints. Content must be passed as a string prop
 * (resolved at import time or conversion time, not via filesystem reads).
 *
 * Supports three extraction modes:
 *   1. By heading name — grabs everything under a matching heading
 *   2. By marker comments — start-after / end-before string matching
 *   3. By line range — explicit start/end line numbers
 *
 * Usage in MDX:
 *
 *   // Option A: import raw content from another MDX file
 *   import rawContent from "./environments-overview.mdx?raw";
 *   import { Include } from "@/components/Include";
 *
 *   <Include content={rawContent} section="Variables" />
 *
 *   // Option B: import a pre-extracted JSON snippet
 *   import snippets from "@/data/snippets.json";
 *
 *   <Include content={snippets["env-variables"]} lang="yaml" />
 *
 *   // Option C: marker-based extraction (like Sphinx start-after/end-before)
 *   <Include
 *     content={rawContent}
 *     startAfter="<!-- BEGIN: install-steps -->"
 *     endBefore="<!-- END: install-steps -->"
 *   />
 *
 *   // Option D: line range extraction
 *   <Include content={rawContent} startLine={10} endLine={25} lang="python" />
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Only type-only imports from "react" are used (erased at compile time).
 */

// -- Types ------------------------------------------------------------------

export interface IncludeProps {
  /** The full source content string to extract from. */
  content: string;

  /** Extract a section by its heading text (case-insensitive match). */
  section?: string;

  /**
   * Heading depth to match for section extraction.
   * In Markdown: 1 = #, 2 = ##, 3 = ###, etc.
   * Defaults to any depth.
   */
  sectionDepth?: number;

  /** Start capturing after the first line containing this string. */
  startAfter?: string;

  /** Stop capturing before the first line containing this string (after start). */
  endBefore?: string;

  /** Start capturing at this line number (1-based, inclusive). */
  startLine?: number;

  /** Stop capturing at this line number (1-based, inclusive). */
  endLine?: number;

  /**
   * If set, renders extracted content inside a fenced code block
   * with this language for syntax highlighting.
   */
  lang?: string;

  /** Strip N leading spaces from every extracted line (like Sphinx :dedent:). */
  dedent?: number;

  /** Add N leading spaces to every extracted line. */
  indent?: number;

  /** When true, include the heading line itself in section output. Default: false. */
  includeHeading?: boolean;

  /** Optional title shown above code blocks. */
  title?: string;
}

// -- Helpers ----------------------------------------------------------------

function extractBySection(
  lines: string[],
  section: string,
  depth?: number,
  includeHeading = false,
): string[] {
  const sectionLower = section.toLowerCase().trim();
  let capturing = false;
  let matchedDepth = 0;
  const result: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);

    if (headingMatch) {
      const headingDepth = headingMatch[1].length;
      const headingText = headingMatch[2].trim().toLowerCase();

      if (capturing) {
        if (headingDepth <= matchedDepth) break;
      }

      if (
        !capturing &&
        headingText === sectionLower &&
        (depth == null || headingDepth === depth)
      ) {
        capturing = true;
        matchedDepth = headingDepth;
        if (includeHeading) result.push(line);
        continue;
      }
    }

    if (capturing) result.push(line);
  }

  return result;
}

function extractByMarkers(
  lines: string[],
  startAfter?: string,
  endBefore?: string,
): string[] {
  const result: string[] = [];
  let capturing = !startAfter;

  for (const line of lines) {
    if (!capturing && startAfter && line.includes(startAfter)) {
      capturing = true;
      continue;
    }
    if (capturing && endBefore && line.includes(endBefore)) {
      break;
    }
    if (capturing) result.push(line);
  }

  return result;
}

function extractByLineRange(
  lines: string[],
  startLine?: number,
  endLine?: number,
): string[] {
  const start = (startLine ?? 1) - 1;
  const end = endLine ?? lines.length;
  return lines.slice(Math.max(0, start), Math.min(lines.length, end));
}

function applyDedent(lines: string[], n: number): string[] {
  const pattern = new RegExp(`^ {0,${n}}`);
  return lines.map((line) => line.replace(pattern, ""));
}

function applyIndent(lines: string[], n: number): string[] {
  const prefix = " ".repeat(n);
  return lines.map((line) => (line.trim() === "" ? line : prefix + line));
}

function trimBlankEnds(lines: string[]): string[] {
  let start = 0;
  while (start < lines.length && lines[start].trim() === "") start++;
  let end = lines.length;
  while (end > start && lines[end - 1].trim() === "") end--;
  return lines.slice(start, end);
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// -- Component --------------------------------------------------------------

function IncludeError({ message }: { message: string }) {
  return (
    <div
      style={{
        padding: "0.75rem 1rem",
        margin: "1rem 0",
        background: "#fef2f2",
        border: "1px solid #fecaca",
        borderRadius: "8px",
        color: "#991b1b",
        fontFamily: "monospace",
        fontSize: "0.85rem",
      }}
    >
      <strong>Include error:</strong> {message}
    </div>
  );
}

export function Include({
  content,
  section,
  sectionDepth,
  startAfter,
  endBefore,
  startLine,
  endLine,
  lang,
  dedent,
  indent,
  includeHeading = false,
  title,
}: IncludeProps) {
  if (content == null || typeof content !== "string") {
    return <IncludeError message="content prop is missing or not a string" />;
  }

  let lines = content.split("\n");

  if (section) {
    lines = extractBySection(lines, section, sectionDepth, includeHeading);
  } else if (startAfter || endBefore) {
    lines = extractByMarkers(lines, startAfter, endBefore);
  } else if (startLine != null || endLine != null) {
    lines = extractByLineRange(lines, startLine, endLine);
  }

  if (lines.length === 0) {
    const target = section
      ? `section "${section}"`
      : startAfter
        ? `marker "${startAfter}"`
        : `lines ${startLine ?? "?"}–${endLine ?? "?"}`;
    return <IncludeError message={`No content found for ${target}`} />;
  }

  if (dedent != null && dedent > 0) lines = applyDedent(lines, dedent);
  if (indent != null && indent > 0) lines = applyIndent(lines, indent);
  lines = trimBlankEnds(lines);

  const extracted = lines.join("\n");

  if (lang) {
    return (
      <div className="include-block" data-source="include-component">
        {title && (
          <div
            style={{
              fontSize: "0.8rem",
              color: "var(--grayscale-a11, #666)",
              marginBottom: "0.25rem",
              fontFamily: "monospace",
            }}
          >
            {title}
          </div>
        )}
        <pre className="code-block-root not-prose" tabIndex={0}>
          <code className={`language-${lang}`}>
            {extracted}
          </code>
        </pre>
      </div>
    );
  }

  return (
    <div
      className="include-block fern-prose prose max-w-full"
      data-source="include-component"
      dangerouslySetInnerHTML={{ __html: markdownToHtml(extracted) }}
    />
  );
}

/**
 * Minimal markdown-to-HTML for prose includes.
 * Handles headings, bold, italic, inline code, links, lists, and paragraphs.
 * For full fidelity, rely on Fern's MDX pipeline instead (use lang prop for code).
 */
function markdownToHtml(md: string): string {
  const lines = md.split("\n");
  const html: string[] = [];
  let inList: "ul" | "ol" | null = null;

  for (const raw of lines) {
    const line = raw;

    if (inList === "ul" && !/^[-*] /.test(line.trim())) {
      html.push("</ul>");
      inList = null;
    }
    if (inList === "ol" && !/^\d+\. /.test(line.trim())) {
      html.push("</ol>");
      inList = null;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level}>${inlineFormat(heading[2])}</h${level}>`);
      continue;
    }

    if (/^[-*] /.test(line.trim())) {
      if (inList !== "ul") {
        html.push("<ul>");
        inList = "ul";
      }
      html.push(`<li>${inlineFormat(line.trim().slice(2))}</li>`);
      continue;
    }

    if (/^\d+\. /.test(line.trim())) {
      if (inList !== "ol") {
        html.push("<ol>");
        inList = "ol";
      }
      html.push(`<li>${inlineFormat(line.trim().replace(/^\d+\.\s*/, ""))}</li>`);
      continue;
    }

    if (line.trim() === "") {
      html.push("");
      continue;
    }

    html.push(`<p>${inlineFormat(line)}</p>`);
  }

  if (inList === "ul") html.push("</ul>");
  if (inList === "ol") html.push("</ol>");

  return html.join("\n");
}

function inlineFormat(text: string): string {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" class="fern-mdx-link">$1</a>',
    );
}
