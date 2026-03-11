# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "mcp",
#     "bm25s",
#     "PyStemmer",
# ]
# ///
"""MCP + Tool Use Recipe: Deep Research Search Agent Trajectories

Generate multi-turn research trajectories where an LLM iteratively searches,
reads, and synthesizes evidence to answer questions -- the kind of data needed
to train deep research agents (OpenResearcher-style).

The pipeline:
  1. Runs a BM25S retriever as an MCP server with search/open/find tools
  2. Seeds with questions and reference answers
  3. Generates full research trajectories (every tool call captured via traces)
  4. Scores trajectories with an LLM judge for rejection sampling

Based on the "Deep Research Trajectories with NeMo Data Designer and MCP
Tool Use" dev note, which demonstrated that synthetic trajectories over local
retrieval can train small models to compete with much larger ones.

Prerequisites:
    - OPENAI_API_KEY environment variable for OpenAI provider model aliases.
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases (default model alias is "nvidia-text").

Run:
    # Basic usage (generates 2 trajectories with a built-in demo corpus)
    uv run search_agent.py

    # Use a custom corpus
    uv run search_agent.py --corpus-path /path/to/corpus.jsonl --questions-path /path/to/questions.jsonl

    # For help message and available options
    uv run search_agent.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import bm25s
from mcp.server.fastmcp import FastMCP

import data_designer.config as dd
from data_designer.interface import DataDesigner

MCP_SERVER_NAME = "corpus-retriever"

# Global state for the BM25 index (populated at server startup)
_bm25_retriever: bm25s.BM25 | None = None
_corpus: list[dict[str, str]] = []
_id_to_index: dict[str, int] = {}

mcp_server = FastMCP(MCP_SERVER_NAME)


# =============================================================================
# Corpus Loading & Indexing
# =============================================================================


def load_corpus(corpus_path: str) -> list[dict[str, str]]:
    """Load a JSONL corpus file into a list of document dicts.

    Expected format: {"id": "doc_001", "title": "Topic", "content": "Full text..."}
    """
    docs: list[dict[str, str]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            if "id" not in doc or "content" not in doc:
                print(f"Warning: skipping line {line_num}, missing 'id' or 'content'", file=sys.stderr)
                continue
            docs.append(
                {
                    "id": str(doc["id"]),
                    "title": str(doc.get("title", "")),
                    "content": str(doc["content"]),
                }
            )
    return docs


def build_index(docs: list[dict[str, str]]) -> bm25s.BM25:
    """Build a BM25S index over title + content for each document."""
    corpus_texts = [f"{d['title']} {d['content']}" for d in docs]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever


def initialize(corpus_path: str) -> None:
    """Load corpus and build index into global state."""
    global _bm25_retriever, _corpus, _id_to_index
    print(f"Loading corpus from {corpus_path}...", file=sys.stderr)
    _corpus = load_corpus(corpus_path)
    if not _corpus:
        print("Warning: corpus is empty", file=sys.stderr)
        return
    _id_to_index = {doc["id"]: idx for idx, doc in enumerate(_corpus)}
    print(f"Building BM25S index over {len(_corpus)} documents...", file=sys.stderr)
    _bm25_retriever = build_index(_corpus)
    print(f"Index ready. {len(_corpus)} documents indexed.", file=sys.stderr)


def _chunk_content(content: str) -> list[str]:
    """Split document content into cursor-addressable chunks."""
    paragraph_chunks = [c.strip() for c in re.split(r"\n\s*\n+", content) if c.strip()]
    if len(paragraph_chunks) > 1:
        return paragraph_chunks
    line_chunks = [line.strip() for line in content.splitlines() if line.strip()]
    if line_chunks:
        return line_chunks
    stripped = content.strip()
    return [stripped] if stripped else []


# =============================================================================
# MCP Server Tools
# =============================================================================


@mcp_server.tool()
def search(query: str, top_k: int = 10) -> dict:
    """Search for candidate documents to explore.

    Args:
        query: Search query string.
        top_k: Maximum number of ranked results (default: 10).
    """
    global _bm25_retriever, _corpus
    if _bm25_retriever is None or not _corpus:
        return {"error": "Search index not initialized", "results": []}
    query_tokens = bm25s.tokenize([query], stopwords="en")
    k = max(1, min(top_k, len(_corpus)))
    results, scores = _bm25_retriever.retrieve(query_tokens, k=k)
    search_results: list[dict] = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = float(scores[0, i])
        if score <= 0:
            continue
        doc = _corpus[doc_idx]
        snippet = doc["content"][:500]
        if len(doc["content"]) > 500:
            snippet += "..."
        search_results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "snippet": snippet,
                "score": round(score, 4),
            }
        )
    return {"results": search_results, "query": query, "total": len(search_results)}


@mcp_server.tool(name="open")
def open_document(doc_id: str) -> dict:
    """Open a document for detailed inspection with cursor-numbered chunks.

    Args:
        doc_id: The document ID (from search results).
    """
    global _corpus, _id_to_index
    if not _corpus:
        return {"error": "Corpus not loaded"}
    idx = _id_to_index.get(doc_id)
    if idx is None:
        return {"error": f"Document not found: {doc_id}"}
    doc = _corpus[idx]
    chunks = _chunk_content(doc["content"])
    numbered_chunks = [{"cursor": i + 1, "text": chunk} for i, chunk in enumerate(chunks)]
    formatted = "\n".join(f"[{e['cursor']}] {e['text']}" for e in numbered_chunks)
    return {
        "id": doc["id"],
        "title": doc["title"],
        "content": formatted,
        "chunks": numbered_chunks,
        "total_chunks": len(numbered_chunks),
    }


@mcp_server.tool()
def find(doc_id: str, query: str) -> dict:
    """Find matching passages inside a document by keyword.

    Args:
        doc_id: Document ID to search within.
        query: Text to find (case-insensitive substring and keyword matching).
    """
    global _corpus, _id_to_index
    if not _corpus:
        return {"error": "Corpus not loaded", "matches": []}
    idx = _id_to_index.get(doc_id)
    if idx is None:
        return {"error": f"Document not found: {doc_id}", "matches": []}
    query_text = query.strip().lower()
    if not query_text:
        return {"error": "Query must be non-empty", "matches": []}
    doc = _corpus[idx]
    chunks = _chunk_content(doc["content"])
    query_terms = [term for term in re.findall(r"\w+", query_text) if term]
    matches: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        haystack = chunk.lower()
        if query_text in haystack or (query_terms and all(t in haystack for t in query_terms)):
            matches.append({"cursor": i, "text": chunk})
    return {
        "doc_id": doc["id"],
        "title": doc["title"],
        "query": query,
        "matches": matches,
        "total_matches": len(matches),
    }


# =============================================================================
# Data Designer Configuration
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """\
You are a thorough research assistant. You have access to three tools \
for navigating a knowledge base:
- search(query, top_k): Find candidate documents relevant to your query
- open(doc_id): Open a document to read its full content in numbered chunks
- find(doc_id, query): Locate specific passages within a document by keyword

Your task is to research the given question by searching for relevant documents, \
reading their content, and synthesizing an answer from the evidence you find. \
Be systematic: formulate search queries, explore promising results, and gather \
evidence before answering. Cite specific passages when possible."""


def build_config(model_alias: str, provider_name: str) -> dd.DataDesignerConfigBuilder:
    """Build the Data Designer configuration for search agent trajectory generation."""
    tool_config = dd.ToolConfig(
        tool_alias="knowledge-base",
        providers=[provider_name],
        allow_tools=["search", "open", "find"],
        max_tool_call_turns=150,
        timeout_sec=60.0,
    )

    config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="research_question",
            expr="{{ question }}",
        )
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="research_answer",
            model_alias=model_alias,
            prompt="Research and answer thoroughly:\n\n{{ research_question }}",
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            tool_alias="knowledge-base",
            with_trace=dd.TraceType.ALL_MESSAGES,
            extract_reasoning_content=True,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="correctness",
            model_alias=model_alias,
            prompt=JUDGE_PROMPT,
            scores=[
                dd.Score(
                    name="correct",
                    description="Is the answer factually correct?",
                    options={
                        1: "Correct -- the generated answer matches the reference",
                        0: "Incorrect -- the generated answer contradicts or misses the reference",
                    },
                ),
            ],
        )
    )

    return config_builder


JUDGE_PROMPT = """\
Evaluate whether the generated answer correctly addresses the question.

Question: {{ research_question }}
Reference answer: {{ answer }}
Generated answer: {{ research_answer }}

Compare the generated answer against the reference. The generated answer does not
need to be word-for-word identical, but it must contain the same core factual content.
"""


# =============================================================================
# Demo Corpus & Questions
# =============================================================================

DEMO_CORPUS = [
    {
        "id": "doc_001",
        "title": "Python Programming Language",
        "content": (
            "Python is a high-level, general-purpose programming language created by "
            "Guido van Rossum and first released in 1991. Its design philosophy emphasizes "
            "code readability with the use of significant indentation. Python is dynamically "
            "typed and garbage-collected. It supports multiple programming paradigms, including "
            "structured, object-oriented, and functional programming.\n\n"
            "Python consistently ranks as one of the most popular programming languages. "
            "It is used extensively in web development, data science, artificial intelligence, "
            "scientific computing, and automation. Major organizations using Python include "
            "Google, NASA, CERN, and Instagram."
        ),
    },
    {
        "id": "doc_002",
        "title": "History of SQL",
        "content": (
            "SQL (Structured Query Language) was initially developed at IBM by Donald D. "
            "Chamberlin and Raymond F. Boyce in the early 1970s. Originally called SEQUEL "
            "(Structured English Query Language), it was designed to manipulate and retrieve "
            "data stored in IBM's original relational database management system, System R.\n\n"
            "The language was first standardized by ANSI in 1986 and by ISO in 1987. Since "
            "then, the standard has been revised multiple times. Major SQL dialects include "
            "PostgreSQL, MySQL, SQLite, Oracle SQL, and Microsoft SQL Server, each with their "
            "own extensions and syntax variations."
        ),
    },
    {
        "id": "doc_003",
        "title": "BM25 Information Retrieval",
        "content": (
            "BM25 (Best Matching 25) is a ranking function used by search engines to estimate "
            "the relevance of documents to a given search query. It is based on the probabilistic "
            "retrieval framework developed in the 1970s and 1980s by Stephen Robertson, Karen "
            "Sparck Jones, and others at the City University of London and Microsoft Research.\n\n"
            "BM25 considers term frequency, inverse document frequency, and document length "
            "normalization. Despite being decades old, BM25 remains competitive with modern neural "
            "retrieval methods for many tasks, especially when combined with re-ranking. It forms "
            "the backbone of search systems like Elasticsearch and Apache Lucene."
        ),
    },
    {
        "id": "doc_004",
        "title": "Deep Research Agents",
        "content": (
            "Deep research agents are AI systems that iteratively search, read, and synthesize "
            "information to answer complex questions. Unlike single-turn question answering, these "
            "agents formulate queries, retrieve documents, evaluate evidence, refine hypotheses, "
            "and eventually produce a comprehensive answer.\n\n"
            "OpenResearcher (Li, Jiang, Ma et al., 2026) demonstrated that synthetic trajectories "
            "generated against a local BM25 retriever are sufficient to train small models to "
            "outperform much larger ones on deep research benchmarks. Nemotron Nano 3, with only "
            "3B active parameters, beat GPT-4.1 on multi-hop research tasks when fine-tuned on "
            "such trajectories."
        ),
    },
    {
        "id": "doc_005",
        "title": "NVIDIA NeMo Framework",
        "content": (
            "NVIDIA NeMo is an open-source framework for building, training, and fine-tuning "
            "large language models (LLMs) and other generative AI models. The framework provides "
            "tools for data curation, model training, and deployment at scale.\n\n"
            "NeMo Data Designer is a component of the NeMo ecosystem focused on synthetic data "
            "generation. It provides a configuration-driven approach to creating training datasets "
            "using LLM-based generation, sampling, validation, and scoring. Data Designer supports "
            "MCP (Model Context Protocol) for tool-augmented generation."
        ),
    },
]

DEMO_QUESTIONS = [
    {
        "question": "Who created the Python programming language and when was it first released?",
        "answer": "Guido van Rossum created Python, first released in 1991",
    },
    {
        "question": "What is BM25 and why is it still relevant for modern search systems?",
        "answer": "BM25 is a ranking function for estimating document relevance, still competitive with neural methods",
    },
]


def write_demo_data(output_dir: Path) -> tuple[Path, Path]:
    """Write demo corpus and questions to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in DEMO_CORPUS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    questions_path = output_dir / "questions.jsonl"
    with open(questions_path, "w", encoding="utf-8") as f:
        for q in DEMO_QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return corpus_path, questions_path


# =============================================================================
# Main Entry Points
# =============================================================================


def serve() -> None:
    """Run the MCP server (called when launched as subprocess by Data Designer)."""
    corpus_path = os.environ.get("CORPUS_PATH", "corpus.jsonl")
    initialize(corpus_path)
    mcp_server.run()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate deep research trajectories using MCP tool calls with BM25S search."
    )
    subparsers = parser.add_subparsers(dest="command")

    # 'serve' subcommand for running the MCP server
    subparsers.add_parser("serve", help="Run the MCP server (used by Data Designer)")

    # Default command arguments (demo mode)
    parser.add_argument("--model-alias", type=str, default="nvidia-text", help="Model alias to use for generation")
    parser.add_argument("--num-records", type=int, default=2, help="Number of trajectories to generate")
    parser.add_argument("--corpus-path", type=str, default=None, help="Path to corpus JSONL file")
    parser.add_argument("--questions-path", type=str, default=None, help="Path to questions JSONL file")
    # For compatibility with Makefile test-run-recipes target (ignored in demo mode)
    parser.add_argument("--artifact-path", type=str, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> None:
    """Main entry point for the demo."""
    args = parse_args()

    if args.command == "serve":
        serve()
        return

    if os.environ.get("NVIDIA_API_KEY") is None and args.model_alias.startswith("nvidia"):
        raise RuntimeError("NVIDIA_API_KEY must be set when using NVIDIA model aliases.")

    # Use provided paths or generate demo data
    if args.corpus_path and args.questions_path:
        corpus_path = Path(args.corpus_path)
        questions_path = Path(args.questions_path)
    else:
        demo_dir = Path(tempfile.mkdtemp(prefix="search_agent_demo_"))
        corpus_path, questions_path = write_demo_data(demo_dir)
        print(f"Using demo data in: {demo_dir}")

    # Configure MCP provider to run via stdio transport (local subprocess)
    mcp_provider = dd.LocalStdioMCPProvider(
        name=MCP_SERVER_NAME,
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "serve"],
        env={"CORPUS_PATH": str(corpus_path)},
    )

    # Seed with questions
    config_builder = build_config(
        model_alias=args.model_alias,
        provider_name=MCP_SERVER_NAME,
    )
    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(questions_path)),
    )

    data_designer = DataDesigner(mcp_providers=[mcp_provider])
    preview_results = data_designer.preview(config_builder, num_records=args.num_records)

    # Display results
    print("\n" + "=" * 60)
    print("GENERATED RESEARCH TRAJECTORIES")
    print("=" * 60)
    preview_results.display_sample_record()


if __name__ == "__main__":
    main()
