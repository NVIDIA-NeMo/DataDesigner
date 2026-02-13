/** Full source for retriever_mcp.py - used in ExpandableCode on deep-research-trajectories page */

export const retriever_mcpCode = `# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp", "bm25s", "PyStemmer"]
# ///

"""MCP Server: BM25S Corpus Retriever for OpenResearcher-style Deep Research

A single-file MCP server that indexes a JSONL corpus and exposes BM25S
lexical search via three browser tools:

    - search(query, top_k): ranked document discovery
    - open(doc_id): full document inspection with cursor-numbered chunks
    - find(doc_id, query): in-document evidence lookup

Corpus format (JSONL, one document per line):
    {"id": "wiki_123", "title": "Christopher Nolan", "content": "Christopher Edward Nolan is a..."}

Server mode (used by Data Designer):
    CORPUS_PATH=corpus.jsonl uv run retriever_mcp.py serve
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import bm25s
from mcp.server.fastmcp import FastMCP

MCP_SERVER_NAME = "corpus-retriever"

# Global state — populated at server startup
_bm25_retriever: bm25s.BM25 | None = None
_corpus: list[dict[str, str]] = []
_id_to_index: dict[str, int] = {}

mcp_server = FastMCP(MCP_SERVER_NAME)


def load_corpus(corpus_path: str) -> list[dict[str, str]]:
    """Load a JSONL corpus file into a list of document dicts."""
    docs: list[dict[str, str]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
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
            docs.append({
                "id": str(doc["id"]),
                "title": str(doc.get("title", "")),
                "content": str(doc["content"]),
            })
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
        search_results.append({
            "id": doc["id"],
            "title": doc["title"],
            "snippet": snippet,
            "score": round(score, 4),
        })
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


def serve() -> None:
    """Run as MCP server subprocess (called by Data Designer)."""
    corpus_path = os.environ.get("CORPUS_PATH", "corpus.jsonl")
    initialize(corpus_path)
    mcp_server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25S corpus retriever MCP server")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the MCP server (reads CORPUS_PATH from env)")
    stats_parser = subparsers.add_parser("stats", help="Print corpus statistics")
    stats_parser.add_argument("--corpus-path", default="corpus.jsonl")
    args = parser.parse_args()
    if args.command == "serve":
        serve()
    elif args.command == "stats":
        docs = load_corpus(args.corpus_path)
        total_chars = sum(len(d["content"]) for d in docs)
        print(f"Corpus: {args.corpus_path}")
        print(f"Documents: {len(docs)}")
        print(f"Total content: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    else:
        parser.print_help()`;
