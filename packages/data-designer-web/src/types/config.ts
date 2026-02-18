export type ColumnType =
  | "sampler"
  | "llm-text"
  | "llm-code"
  | "llm-structured"
  | "llm-judge"
  | "expression"
  | "validation"
  | "seed-dataset"
  | "embedding"
  | "image";

export interface ColumnConfig {
  name: string;
  column_type: ColumnType;
  drop?: boolean;
  [key: string]: unknown;
}

export interface ModelConfig {
  alias: string;
  model: string;
  provider?: string;
  inference_parameters?: Record<string, unknown>;
  skip_health_check?: boolean;
}

export const COLUMN_TYPE_META: Record<
  ColumnType,
  { label: string; emoji: string; color: string }
> = {
  sampler: { label: "Sampler", emoji: "🎲", color: "bg-blue-900/40 text-blue-300 border-blue-700/50" },
  "llm-text": { label: "LLM Text", emoji: "💬", color: "bg-purple-900/40 text-purple-300 border-purple-700/50" },
  "llm-code": { label: "LLM Code", emoji: "💻", color: "bg-indigo-900/40 text-indigo-300 border-indigo-700/50" },
  "llm-structured": { label: "LLM Structured", emoji: "📋", color: "bg-violet-900/40 text-violet-300 border-violet-700/50" },
  "llm-judge": { label: "LLM Judge", emoji: "⚖️", color: "bg-amber-900/40 text-amber-300 border-amber-700/50" },
  expression: { label: "Expression", emoji: "🔢", color: "bg-teal-900/40 text-teal-300 border-teal-700/50" },
  validation: { label: "Validation", emoji: "✅", color: "bg-green-900/40 text-green-300 border-green-700/50" },
  "seed-dataset": { label: "Seed Dataset", emoji: "🌱", color: "bg-emerald-900/40 text-emerald-300 border-emerald-700/50" },
  embedding: { label: "Embedding", emoji: "📐", color: "bg-cyan-900/40 text-cyan-300 border-cyan-700/50" },
  image: { label: "Image", emoji: "🖼️", color: "bg-rose-900/40 text-rose-300 border-rose-700/50" },
};
