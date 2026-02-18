import { useState } from "react";
import {
  User,
  Bot,
  Monitor,
  Wrench,
  ChevronDown,
  ChevronRight,
  Loader2,
  Brain,
} from "lucide-react";

interface TraceMessage {
  role: string;
  content: unknown;
  reasoning_content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: string;
  function: { name: string; arguments: string };
}

interface Props {
  trace: Record<string, unknown>[];
  loading: boolean;
}

const ROLE_STYLES: Record<
  string,
  { icon: React.ReactNode; label: string; bg: string; border: string }
> = {
  system: {
    icon: <Monitor size={14} />,
    label: "System",
    bg: "bg-surface-3",
    border: "border-gray-700/50",
  },
  user: {
    icon: <User size={14} />,
    label: "User",
    bg: "bg-blue-900/20",
    border: "border-blue-700/30",
  },
  assistant: {
    icon: <Bot size={14} />,
    label: "Assistant",
    bg: "bg-purple-900/20",
    border: "border-purple-700/30",
  },
  tool: {
    icon: <Wrench size={14} />,
    label: "Tool",
    bg: "bg-amber-900/20",
    border: "border-amber-700/30",
  },
};

function formatContent(content: unknown): string {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block: any) => {
        if (typeof block === "string") return block;
        if (block.type === "text") return block.text;
        return JSON.stringify(block);
      })
      .join("\n");
  }
  return JSON.stringify(content, null, 2);
}

function ToolCallBlock({ call }: { call: ToolCall }) {
  const [expanded, setExpanded] = useState(false);

  let args: string;
  try {
    args = JSON.stringify(JSON.parse(call.function.arguments), null, 2);
  } catch {
    args = call.function.arguments;
  }

  return (
    <div className="border border-amber-700/30 rounded-md mt-2 overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-1.5 text-xs bg-amber-900/20 hover:bg-amber-900/30 transition-colors text-amber-300"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Wrench size={12} />
        <span className="font-mono font-medium">{call.function.name}</span>
        <span className="text-amber-500/60 ml-1">({call.id.slice(0, 8)})</span>
      </button>
      {expanded && (
        <pre className="px-3 py-2 text-xs font-mono text-amber-200/80 bg-amber-950/30 overflow-auto max-h-48 whitespace-pre-wrap">
          {args}
        </pre>
      )}
    </div>
  );
}

export default function TraceViewer({ trace, loading }: Props) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 py-4 text-sm text-gray-400">
        <Loader2 size={14} className="animate-spin" />
        Loading trace...
      </div>
    );
  }

  if (!trace || trace.length === 0) {
    return (
      <p className="text-xs text-gray-500 py-2">
        No trace data available for this column.
      </p>
    );
  }

  return (
    <div className="space-y-2 max-h-[500px] overflow-auto">
      {trace.map((msg, idx) => {
        const m = msg as unknown as TraceMessage;
        const style = ROLE_STYLES[m.role] ?? ROLE_STYLES.user;
        const content = formatContent(m.content);

        return (
          <div
            key={idx}
            className={`rounded-md border ${style.border} ${style.bg} overflow-hidden`}
          >
            <div className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-gray-300">
              {style.icon}
              {style.label}
              {m.tool_call_id && (
                <span className="text-gray-500 font-mono text-[10px]">
                  (call: {m.tool_call_id.slice(0, 8)})
                </span>
              )}
            </div>
            {content && (
              <pre className="px-3 py-2 text-xs font-mono text-gray-300 whitespace-pre-wrap leading-relaxed border-t border-white/5 max-h-64 overflow-auto">
                {content}
              </pre>
            )}
            {m.reasoning_content && (
              <div className="px-3 py-2 border-t border-white/5">
                <div className="flex items-center gap-1.5 text-xs text-violet-400 mb-1">
                  <Brain size={12} />
                  Reasoning
                </div>
                <pre className="text-xs font-mono text-violet-300/70 whitespace-pre-wrap max-h-48 overflow-auto">
                  {m.reasoning_content}
                </pre>
              </div>
            )}
            {m.tool_calls && m.tool_calls.length > 0 && (
              <div className="px-3 pb-2">
                {m.tool_calls.map((tc) => (
                  <ToolCallBlock key={tc.id} call={tc} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
