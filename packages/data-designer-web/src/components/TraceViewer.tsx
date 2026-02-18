import { useState } from "react";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  Loader2,
  Brain,
  Wrench,
  Sparkles,
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

function PromptBlock({
  label,
  content,
  defaultCollapsed = false,
}: {
  label: string;
  content: string;
  defaultCollapsed?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className="border border-gray-700/40 rounded-md overflow-hidden bg-surface-3/50">
      <button
        className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 hover:text-gray-400 transition-colors"
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
        <span className="uppercase tracking-wider font-medium text-[10px]">
          {label}
        </span>
        <span className="text-gray-600 font-normal text-[10px]">prompt</span>
      </button>
      {!collapsed && (
        <pre className="px-3 py-2 text-xs font-mono text-gray-500 whitespace-pre-wrap leading-relaxed border-t border-gray-700/30 max-h-48 overflow-auto">
          {content}
        </pre>
      )}
    </div>
  );
}

function ToolResponseBlock({ content }: { content: string }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="border border-amber-700/30 rounded-md overflow-hidden bg-amber-900/10">
      <button
        className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-amber-500/70 hover:text-amber-400 transition-colors"
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
        <Wrench size={12} />
        <span className="uppercase tracking-wider font-medium text-[10px]">
          Tool Response
        </span>
      </button>
      {!collapsed && (
        <pre className="px-3 py-2 text-xs font-mono text-amber-300/60 whitespace-pre-wrap leading-relaxed border-t border-amber-700/20 max-h-48 overflow-auto">
          {content}
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
        const content = formatContent(m.content);

        if (m.role === "system") {
          return (
            <PromptBlock
              key={idx}
              label="System"
              content={content}
              defaultCollapsed={true}
            />
          );
        }

        if (m.role === "user") {
          return (
            <PromptBlock
              key={idx}
              label="User"
              content={content}
            />
          );
        }

        if (m.role === "tool") {
          return (
            <ToolResponseBlock key={idx} content={content} />
          );
        }

        // Assistant = the actual generated output
        return (
          <div
            key={idx}
            className="rounded-md border border-nvidia-green/30 bg-nvidia-green/5 overflow-hidden"
          >
            <div className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-nvidia-green">
              <Sparkles size={14} />
              Generated Output
            </div>
            {content && (
              <pre className="px-3 py-2 text-sm font-mono text-gray-100 whitespace-pre-wrap leading-relaxed border-t border-nvidia-green/10 max-h-64 overflow-auto">
                {content}
              </pre>
            )}
            {m.reasoning_content && (
              <div className="px-3 py-2 border-t border-nvidia-green/10">
                <div className="flex items-center gap-1.5 text-xs text-violet-400 mb-1">
                  <Brain size={12} />
                  <span className="uppercase tracking-wider font-medium text-[10px]">
                    Model Reasoning (not in output)
                  </span>
                </div>
                <pre className="text-xs font-mono text-violet-300/60 whitespace-pre-wrap max-h-48 overflow-auto">
                  {m.reasoning_content}
                </pre>
              </div>
            )}
            {m.tool_calls && m.tool_calls.length > 0 && (
              <div className="px-3 pb-2 border-t border-nvidia-green/10">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-medium pt-2 pb-1">
                  Tool Calls
                </div>
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
