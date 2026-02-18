/**
 * Simple YAML syntax highlighter that renders colored spans.
 * Handles keys, strings, numbers, booleans, comments, and Jinja2 templates.
 */

interface Props {
  code: string;
}

function highlightLine(line: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let rest = line;
  let key = 0;

  // Comment lines
  if (rest.trimStart().startsWith("#")) {
    const indent = rest.length - rest.trimStart().length;
    return [
      <span key={0}>{rest.slice(0, indent)}</span>,
      <span key={1} className="text-gray-600 italic">
        {rest.slice(indent)}
      </span>,
    ];
  }

  // Leading whitespace
  const leadingMatch = rest.match(/^(\s+)/);
  if (leadingMatch) {
    parts.push(<span key={key++}>{leadingMatch[1]}</span>);
    rest = rest.slice(leadingMatch[1].length);
  }

  // List marker
  if (rest.startsWith("- ")) {
    parts.push(
      <span key={key++} className="text-gray-500">
        -{" "}
      </span>
    );
    rest = rest.slice(2);
  }

  // Key: value pattern
  const kvMatch = rest.match(/^([a-zA-Z_][\w.-]*)\s*:/);
  if (kvMatch) {
    parts.push(
      <span key={key++} className="text-blue-400">
        {kvMatch[1]}
      </span>
    );
    parts.push(
      <span key={key++} className="text-gray-500">
        :
      </span>
    );
    rest = rest.slice(kvMatch[0].length);
  }

  if (!rest) return parts;

  // Process the value portion
  const tokens = tokenizeValue(rest);
  for (const token of tokens) {
    parts.push(<span key={key++}>{token}</span>);
  }

  return parts;
}

function tokenizeValue(value: string): React.ReactNode[] {
  const result: React.ReactNode[] = [];
  let remaining = value;
  let idx = 0;

  while (remaining.length > 0) {
    // Jinja2 template: {{ ... }}
    const jinjaMatch = remaining.match(/^(.*?)(\{\{.*?\}\})/);
    if (jinjaMatch) {
      if (jinjaMatch[1]) {
        result.push(
          <span key={`v${idx++}`}>{colorizeScalar(jinjaMatch[1])}</span>
        );
      }
      result.push(
        <span key={`j${idx++}`} className="text-nvidia-green font-medium">
          {jinjaMatch[2]}
        </span>
      );
      remaining = remaining.slice(jinjaMatch[0].length);
      continue;
    }

    // No more Jinja, process rest as scalar
    result.push(
      <span key={`r${idx++}`}>{colorizeScalar(remaining)}</span>
    );
    break;
  }
  return result;
}

function colorizeScalar(text: string): React.ReactNode {
  const trimmed = text.trim();

  if (!trimmed) return <span>{text}</span>;

  // Quoted strings
  if (
    (trimmed.startsWith("'") && trimmed.endsWith("'")) ||
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
  ) {
    const ws = text.slice(0, text.indexOf(trimmed));
    return (
      <>
        {ws}
        <span className="text-amber-300">{trimmed}</span>
      </>
    );
  }

  // Booleans
  if (/^(true|false|yes|no|null|none)$/i.test(trimmed)) {
    const ws = text.slice(0, text.indexOf(trimmed));
    return (
      <>
        {ws}
        <span className="text-purple-400">{trimmed}</span>
      </>
    );
  }

  // Numbers
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    const ws = text.slice(0, text.indexOf(trimmed));
    return (
      <>
        {ws}
        <span className="text-cyan-400">{trimmed}</span>
      </>
    );
  }

  // Plain string values
  if (text.startsWith(" ")) {
    return (
      <>
        {" "}
        <span className="text-gray-300">{text.trimStart()}</span>
      </>
    );
  }

  return <span className="text-gray-300">{text}</span>;
}

export default function YamlHighlighter({ code }: Props) {
  const lines = code.split("\n");

  return (
    <pre className="text-xs font-mono leading-relaxed whitespace-pre overflow-auto">
      {lines.map((line, i) => (
        <div key={i} className="hover:bg-surface-2/50 px-1 -mx-1 rounded">
          <span className="inline-block w-8 text-right text-gray-700 select-none mr-3 text-[10px]">
            {i + 1}
          </span>
          {highlightLine(line)}
        </div>
      ))}
    </pre>
  );
}
