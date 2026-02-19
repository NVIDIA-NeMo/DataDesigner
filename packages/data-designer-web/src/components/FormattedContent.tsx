/**
 * Smart content renderer that detects and formats code, JSON, YAML, and plain text.
 * Applies syntax highlighting for better readability.
 */

interface Props {
  value: string;
  columnName?: string;
}

type ContentType = "python" | "json" | "yaml" | "text";

function detectType(value: string, _columnName?: string): ContentType {
  const trimmed = value.trim();

  // Only detect code from explicit code fences -- no content sniffing
  if (/^```python/m.test(trimmed)) return "python";
  if (/^```json/m.test(trimmed)) return "json";
  if (/^```yaml/m.test(trimmed) || /^```yml/m.test(trimmed)) return "yaml";

  // Valid JSON (starts with { or [)
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    try {
      JSON.parse(trimmed);
      return "json";
    } catch {
      // not valid JSON, treat as text
    }
  }

  return "text";
}

function formatJson(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value.trim()), null, 2);
  } catch {
    return value;
  }
}

function stripCodeFences(value: string): string {
  let s = value.trim();
  if (s.startsWith("```")) {
    const firstNewline = s.indexOf("\n");
    if (firstNewline !== -1) s = s.slice(firstNewline + 1);
  }
  if (s.endsWith("```")) s = s.slice(0, -3);
  return s.trim();
}

// Simple syntax highlighting via spans
function highlightPython(code: string): React.ReactNode[] {
  const lines = code.split("\n");
  return lines.map((line, i) => (
    <div key={i} className="leading-relaxed">
      <span className="inline-block w-8 text-right text-gray-700 select-none mr-3 text-[10px]">
        {i + 1}
      </span>
      <PythonLine line={line} />
    </div>
  ));
}

function PythonLine({ line }: { line: string }) {
  const keywords =
    /\b(def|class|import|from|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|try|except|finally|with|as|raise|yield|lambda|pass|break|continue|async|await)\b/g;
  const strings = /(["'])(?:(?=(\\?))\2.)*?\1/g;
  const comments = /#.*/g;
  const decorators = /@\w+/g;
  const functions = /\b(\w+)(?=\()/g;
  const numbers = /\b\d+(\.\d+)?\b/g;

  // Simple approach: process comment first, then rest
  const commentMatch = line.match(comments);
  const commentIndex = line.indexOf("#");

  if (commentMatch && commentIndex >= 0) {
    const before = line.slice(0, commentIndex);
    const comment = line.slice(commentIndex);
    return (
      <>
        <HighlightSegment text={before} />
        <span className="text-gray-600 italic">{comment}</span>
      </>
    );
  }

  return <HighlightSegment text={line} />;
}

function HighlightSegment({ text }: { text: string }) {
  // Apply highlighting via regex replacement
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let key = 0;

  const regex =
    /\b(def|class|import|from|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|try|except|finally|with|as|raise|yield|lambda|pass|break|continue|async|await)\b|(["'])(?:(?=(\\?))\3.)*?\2|\b(\w+)(?=\()/g;

  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <span key={key++} className="text-gray-200">
          {text.slice(lastIndex, match.index)}
        </span>
      );
    }

    if (match[1]) {
      // Keyword
      parts.push(
        <span key={key++} className="text-purple-400">
          {match[0]}
        </span>
      );
    } else if (match[2]) {
      // String
      parts.push(
        <span key={key++} className="text-amber-300">
          {match[0]}
        </span>
      );
    } else if (match[4]) {
      // Function call
      parts.push(
        <span key={key++} className="text-blue-400">
          {match[0]}
        </span>
      );
    } else {
      parts.push(
        <span key={key++} className="text-gray-200">
          {match[0]}
        </span>
      );
    }

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(
      <span key={key++} className="text-gray-200">
        {text.slice(lastIndex)}
      </span>
    );
  }

  return <>{parts}</>;
}

function highlightJson(code: string): React.ReactNode[] {
  const lines = code.split("\n");
  return lines.map((line, i) => {
    const keyMatch = line.match(/^(\s*)"([^"]+)"(\s*:\s*)/);
    const rest = keyMatch ? line.slice(keyMatch[0].length) : null;

    return (
      <div key={i} className="leading-relaxed">
        <span className="inline-block w-8 text-right text-gray-700 select-none mr-3 text-[10px]">
          {i + 1}
        </span>
        {keyMatch ? (
          <>
            <span>{keyMatch[1]}</span>
            <span className="text-blue-400">"{keyMatch[2]}"</span>
            <span className="text-gray-500">{keyMatch[3]}</span>
            <JsonValue text={rest ?? ""} />
          </>
        ) : (
          <JsonValue text={line} />
        )}
      </div>
    );
  });
}

function JsonValue({ text }: { text: string }) {
  const trimmed = text.trim().replace(/,\s*$/, "");
  const trailing = text.endsWith(",") ? "," : "";

  if (trimmed.startsWith('"')) {
    return (
      <span>
        <span className="text-amber-300">{trimmed}</span>
        <span className="text-gray-500">{trailing}</span>
      </span>
    );
  }
  if (/^(true|false|null)$/.test(trimmed)) {
    return (
      <span>
        <span className="text-purple-400">{trimmed}</span>
        <span className="text-gray-500">{trailing}</span>
      </span>
    );
  }
  if (/^-?\d/.test(trimmed)) {
    return (
      <span>
        <span className="text-cyan-400">{trimmed}</span>
        <span className="text-gray-500">{trailing}</span>
      </span>
    );
  }
  return <span className="text-gray-300">{text}</span>;
}

function highlightYaml(code: string): React.ReactNode[] {
  const lines = code.split("\n");
  return lines.map((line, i) => {
    const kvMatch = line.match(/^(\s*)(- )?([a-zA-Z_][\w.-]*)(\s*:\s*)(.*)/);
    const commentMatch = line.trimStart().startsWith("#");

    return (
      <div key={i} className="leading-relaxed">
        <span className="inline-block w-8 text-right text-gray-700 select-none mr-3 text-[10px]">
          {i + 1}
        </span>
        {commentMatch ? (
          <span className="text-gray-600 italic">{line}</span>
        ) : kvMatch ? (
          <>
            <span>{kvMatch[1]}</span>
            {kvMatch[2] && <span className="text-gray-500">{kvMatch[2]}</span>}
            <span className="text-blue-400">{kvMatch[3]}</span>
            <span className="text-gray-500">{kvMatch[4]}</span>
            <span className="text-gray-300">{kvMatch[5]}</span>
          </>
        ) : (
          <span className="text-gray-300">{line}</span>
        )}
      </div>
    );
  });
}

export default function FormattedContent({ value, columnName }: Props) {
  if (!value) return <span className="text-gray-600 italic">null</span>;

  const cleaned = stripCodeFences(value);
  const type = detectType(cleaned, columnName);

  if (type === "text" && cleaned.length < 200 && !cleaned.includes("\n")) {
    return <p className="text-sm text-gray-200">{cleaned}</p>;
  }

  const highlighted =
    type === "python"
      ? highlightPython(cleaned)
      : type === "json"
        ? highlightJson(formatJson(cleaned))
        : type === "yaml"
          ? highlightYaml(cleaned)
          : null;

  if (highlighted) {
    return (
      <div className="bg-surface-0 rounded-lg border border-border overflow-hidden">
        <div className="flex items-center justify-between px-3 py-1 bg-surface-2 border-b border-border">
          <span className="text-[10px] uppercase tracking-wider font-medium text-gray-500">
            {type}
          </span>
        </div>
        <pre className="p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre">
          {highlighted}
        </pre>
      </div>
    );
  }

  // Plain long text
  return (
    <pre className="text-xs text-gray-200 bg-surface-1 rounded p-2.5 whitespace-pre-wrap leading-relaxed max-h-48 overflow-auto">
      {cleaned}
    </pre>
  );
}
