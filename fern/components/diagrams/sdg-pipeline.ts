/** SDG pipeline diagram from docs/devnotes/posts/design-principles.md */

export const sdgPipelineDiagram = `
      Seed Documents         Seed dataset column ingests documents
            │                 from local files or HuggingFace
            ▼
┌─────────────────────────┐
│  Artifact Extraction    │  LLM extracts key concepts, entities,
│                         │  relationships from each document
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  QA Generation          │  LLM generates questions & answers grounded
│                         │  in the extracted artifacts
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Quality Evaluation     │  LLM judge scores each QA pair
│                         │  on relevance, accuracy, clarity
└───────────┬─────────────┘
            │
            ▼
      Final Dataset
`;
