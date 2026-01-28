# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP + Tool Use Recipe: Document Q&A with BM25S Lexical Search

This recipe demonstrates an end-to-end MCP tool-calling workflow:

1) Load one or more PDF documents from URLs or local paths.
2) Index them with BM25S for fast lexical search.
3) Use Data Designer tool calls (`search_docs`) to generate grounded Q&A pairs.

Prerequisites:
- `NVIDIA_API_KEY` if using `--model-alias nvidia-text` (default)
- Recipe dependencies: Install with `make install-dev-recipes`

Run:
    # Install recipe dependencies (preserves workspace packages)
    make install-dev-recipes

    # Then run the recipe (uses default sample PDF)
    uv run docs/assets/recipes/mcp_and_tooluse/pdf_qa.py

    # With custom PDFs (can specify multiple)
    uv run docs/assets/recipes/mcp_and_tooluse/pdf_qa.py --pdf /path/to/doc.pdf
    uv run docs/assets/recipes/mcp_and_tooluse/pdf_qa.py --pdf https://example.com/doc.pdf --pdf ./local.pdf

    # Or run all recipes via Makefile
    make test-run-recipes

Common flags:
    # Generate a few Q&A pairs
    uv run docs/assets/recipes/mcp_and_tooluse/pdf_qa.py --num-records 3

    # Use a different model
    uv run docs/assets/recipes/mcp_and_tooluse/pdf_qa.py --model-alias gpt-4o

Server mode (used internally by Data Designer):
    python docs/assets/recipes/mcp_and_tooluse/pdf_qa.py serve

Notes:
- URLs are streamed directly into memory (no local download required).
- Local file paths are read directly from disk.
- The BM25S index is built at server startup from all provided PDFs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import bm25s
import fitz
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner

DEFAULT_PDF_URL = "https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf"
MCP_SERVER_NAME = "doc-bm25-search"

# Global state for the BM25 index (populated at server startup)
_bm25_retriever: bm25s.BM25 | None = None
_corpus: list[dict[str, str]] = []


class QAPair(BaseModel):
    question: str = Field(..., description="A question grounded in the document text.")
    answer: str = Field(..., description="A concise answer grounded in the supporting passage.")
    supporting_passage: str = Field(
        ..., description="A short excerpt (2-4 sentences) copied from the search result that supports the answer."
    )
    citation: str = Field(
        ..., description="The citation (e.g. source url, page number, etc) of the supporting passage."
    )


class TopicList(BaseModel):
    topics: list[str] = Field(
        ...,
        description="High-level topics covered by the document.",
    )


def _is_url(path_or_url: str) -> bool:
    """Check if the given string is a URL."""
    parsed = urlparse(path_or_url)
    return parsed.scheme in ("http", "https")


def _get_source_name(path_or_url: str) -> str:
    """Extract a human-readable source name from a path or URL."""
    if _is_url(path_or_url):
        parsed = urlparse(path_or_url)
        return Path(parsed.path).name or parsed.netloc
    return Path(path_or_url).name


def extract_pdf_text(path_or_url: str) -> list[dict[str, str]]:
    """Extract text from a PDF file or URL, returning a list of passages with metadata.

    Each passage corresponds to a page from the PDF.

    Args:
        path_or_url: Either a local file path or a URL to a PDF document.
            URLs are streamed directly into memory without saving to disk.

    Returns:
        List of passage dictionaries with 'text', 'page', and 'source' keys.
    """
    passages: list[dict[str, str]] = []
    source_name = _get_source_name(path_or_url)

    if _is_url(path_or_url):
        with urlopen(path_or_url) as response:
            pdf_bytes = response.read()
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    else:
        doc = fitz.open(path_or_url)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            passages.append(
                {
                    "text": text,
                    "page": str(page_num + 1),
                    "source": source_name,
                }
            )

    doc.close()
    return passages


def build_bm25_index(passages: list[dict[str, str]]) -> bm25s.BM25:
    """Build a BM25S index from the extracted passages."""
    corpus_texts = [p["text"] for p in passages]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    return retriever


def initialize_search_index(pdf_sources: list[str]) -> None:
    """Load PDFs from paths/URLs and build the BM25 index.

    Args:
        pdf_sources: List of PDF file paths or URLs to index.
    """
    global _bm25_retriever, _corpus

    _corpus = []
    for source in pdf_sources:
        passages = extract_pdf_text(source)
        _corpus.extend(passages)

    if _corpus:
        _bm25_retriever = build_bm25_index(_corpus)


# MCP Server Definition
mcp_server = FastMCP(MCP_SERVER_NAME)


@mcp_server.tool()
def search_docs(query: str, limit: int = 5, document: str = "", page: str = "") -> str:
    """Search through documents using BM25 lexical search.

    BM25 is a keyword-based retrieval algorithm that matches exact terms. For best results:

    - Use specific keywords, not full questions (e.g., "configuration parameters timeout" not "How do I set the timeout?")
    - Include domain-specific terms that would appear in the source text
    - Combine multiple relevant terms to narrow results (e.g., "installation requirements dependencies")
    - Try synonyms or alternative phrasings if initial searches return poor results
    - Avoid filler words and focus on content-bearing terms

    Examples:
        Good queries:
        - "error handling retry mechanism"
        - "authentication token expiration"
        - "memory allocation buffer size"

        Less effective queries:
        - "What are the error handling options?"
        - "Tell me about authentication"
        - "How does memory work?"

    Args:
        query: Search query string - use specific keywords for best results
        limit: Maximum number of results to return (default: 5)
        document: Optional document source name to restrict search to (use list_docs to see available documents)
        page: Optional page number to restrict search to (requires document to be specified)

    Returns:
        JSON string with search results including text excerpts and page numbers
    """
    global _bm25_retriever, _corpus

    if _bm25_retriever is None or not _corpus:
        return json.dumps({"error": "Search index not initialized"})

    # Validate that page requires document
    if page and not document:
        return json.dumps({"error": "The 'page' parameter requires 'document' to be specified"})

    query_tokens = bm25s.tokenize([query], stopwords="en")

    # When filtering, retrieve more results to ensure we have enough after filtering
    retrieve_limit = len(_corpus) if (document or page) else limit
    results, scores = _bm25_retriever.retrieve(query_tokens, k=min(retrieve_limit, len(_corpus)))

    search_results: list[dict[str, str | float]] = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = float(scores[0, i])

        if score <= 0:
            continue

        passage = _corpus[doc_idx]

        # Apply document filter
        if document and passage["source"] != document:
            continue

        # Apply page filter
        if page and passage["page"] != page:
            continue

        search_results.append(
            {
                "text": passage["text"][:2000],
                "page": passage["page"],
                "source": passage["source"],
                "score": round(score, 4),
                "url": f"file://{passage['source']}#page={passage['page']}",
            }
        )

        # Stop once we have enough results
        if len(search_results) >= limit:
            break

    return json.dumps({"results": search_results, "query": query, "total": len(search_results)})


@mcp_server.tool()
def list_docs() -> str:
    """List all documents in the search index with their page counts.

    Returns:
        JSON string with a list of documents, each containing the source name and page count.
    """
    global _corpus

    if not _corpus:
        return json.dumps({"error": "Search index not initialized", "documents": []})

    doc_pages: dict[str, set[str]] = {}
    for passage in _corpus:
        source = passage["source"]
        page = passage["page"]
        if source not in doc_pages:
            doc_pages[source] = set()
        doc_pages[source].add(page)

    documents = [{"source": source, "page_count": len(pages)} for source, pages in sorted(doc_pages.items())]

    return json.dumps({"documents": documents, "total_documents": len(documents)})


def build_config(model_alias: str, server_name: str) -> dd.DataDesignerConfigBuilder:
    """Build the Data Designer configuration for document Q&A generation."""
    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="seed_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(),
            drop=True,
        )
    )

    tool_config = dd.MCPToolConfig(
        server_name=server_name,
        tool_names=["list_docs", "search_docs"],
        max_tool_calls=100,
        timeout_sec=30.0,
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="topic_candidates",
            model_alias=model_alias,
            prompt="Extract a high-level list of all topics covered by documents our knowledge base.",
            system_prompt=(
                "You must call tools before answering. "
                "Do not use outside knowledge; only use tool results. "
                "You can use as many tool calls as required to answer the user query."
            ),
            output_format=TopicList,
            tool_config=tool_config,
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="topic",
            expr="{{ topic_candidates.topics | random }}",
        )
    )

    qa_prompt = """\
Create a question-answer pair on the topic "{{topic}}", with supporting text and citation.
The supporting_passage must be a 2-4 sentence excerpt copied from the tool result that demonstrates
why the answer is correct.
"""

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa_pair",
            model_alias=model_alias,
            prompt=qa_prompt,
            system_prompt=(
                "You must call tools before answering. "
                "Do not use outside knowledge; only use tool results. "
                "You can use as many tool calls as required to answer the user query."
            ),
            output_format=QAPair,
            tool_config=tool_config,
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="question",
            expr="{{ qa_pair.question }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="answer",
            expr="{{ qa_pair.answer }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="supporting_passage",
            expr="{{ qa_pair.supporting_passage }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="citation",
            expr="{{ qa_pair.citation }}",
        )
    )
    return config_builder


def generate_preview(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    mcp_server_config: dd.MCPServerConfig,
) -> PreviewResults:
    """Run Data Designer preview with the MCP server."""
    data_designer = DataDesigner(mcp_servers=[mcp_server_config])
    data_designer.set_run_config(dd.RunConfig(include_full_traces=True))
    return data_designer.preview(config_builder, num_records=num_records)


def _truncate(text: str, max_length: int = 100) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _format_trace_step(msg: dict[str, object]) -> str:
    """Format a single trace message as a concise one-liner."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content")
    tool_calls = msg.get("tool_calls")
    tool_call_id = msg.get("tool_call_id")

    if role == "system":
        return f"[bold cyan]system[/]({_truncate(str(content))})"

    if role == "user":
        return f"[bold green]user[/]({_truncate(str(content))})"

    if role == "assistant":
        parts: list[str] = []
        if reasoning:
            parts.append(f"[bold magenta]reasoning[/]({_truncate(str(reasoning))})")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    if isinstance(func, dict):
                        name = func.get("name", "?")
                        args = func.get("arguments", "")
                        parts.append(f"[bold yellow]tool_call[/]({name}: {_truncate(str(args), 60)})")
        if content:
            parts.append(f"[bold blue]content[/]({_truncate(str(content))})")
        return "\n".join(parts) if parts else "[bold blue]assistant[/](empty)"

    if role == "tool":
        tool_id = str(tool_call_id or "?")[:8]
        return f"[bold red]tool_response[/]([{tool_id}] {_truncate(str(content), 80)})"

    return f"[dim]{role}[/]({_truncate(str(content))})"


def _display_column_trace(column_name: str, trace: list[dict[str, object]]) -> None:
    """Display a trace for a single column using Rich Panel."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    lines: list[str] = []

    for msg in trace:
        if not isinstance(msg, dict):
            continue
        formatted = _format_trace_step(msg)
        for line in formatted.split("\n"):
            lines.append(f"  * {line}")

    trace_content = "\n".join(lines) if lines else "  (no trace messages)"
    panel = Panel(
        trace_content,
        title=f"[bold]Column Trace: {column_name}[/]",
        border_style="blue",
        padding=(0, 1),
    )
    console.print(panel)


def display_preview_record(preview_results: PreviewResults) -> None:
    """Display a sample record from the preview results with trace visualization."""
    from rich.console import Console

    console = Console()
    dataset = preview_results.dataset

    if dataset is None or dataset.empty:
        console.print("[red]No preview records generated.[/]")
        return

    record = dataset.iloc[0].to_dict()

    # Find trace columns and their base column names
    trace_columns = [col for col in dataset.columns if col.endswith("__trace")]

    # Display non-trace columns as summary
    non_trace_record = {k: v for k, v in record.items() if not k.endswith("__trace")}
    console.print("\n[bold]Sample Record (data columns):[/]")
    console.print(json.dumps(non_trace_record, indent=2, default=str))

    # Display each trace column in its own panel
    if trace_columns:
        console.print("\n[bold]Generation Traces:[/]")
        for trace_col in trace_columns:
            base_name = trace_col.replace("__trace", "")
            trace_data = record.get(trace_col)
            if isinstance(trace_data, list):
                _display_column_trace(base_name, trace_data)

    preview_results.display_sample_record()


def serve() -> None:
    """Run the MCP server (called when launched as subprocess by Data Designer)."""
    pdf_sources_json = os.environ.get("PDF_SOURCES", "[]")
    pdf_sources = json.loads(pdf_sources_json)
    if not pdf_sources:
        pdf_sources = [DEFAULT_PDF_URL]
    initialize_search_index(pdf_sources)
    mcp_server.run()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate document Q&A pairs using MCP tool calls with BM25S search.")
    subparsers = parser.add_subparsers(dest="command")

    # 'serve' subcommand for running the MCP server
    subparsers.add_parser("serve", help="Run the MCP server (used by Data Designer)")

    # Default command arguments (demo mode)
    parser.add_argument("--model-alias", type=str, default="nvidia-text", help="Model alias to use for generation")
    parser.add_argument("--num-records", type=int, default=4, help="Number of Q&A pairs to generate")
    parser.add_argument(
        "--pdf",
        type=str,
        action="append",
        dest="pdfs",
        metavar="PATH_OR_URL",
        help="PDF file path or URL to index (can be specified multiple times). Defaults to a sample PDF if not provided.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the demo."""
    args = parse_args()

    # Handle 'serve' subcommand
    if args.command == "serve":
        serve()
        return

    # Demo mode: run Data Designer with the BM25S MCP server
    if os.environ.get("NVIDIA_API_KEY") is None and args.model_alias.startswith("nvidia"):
        raise RuntimeError("NVIDIA_API_KEY must be set when using NVIDIA model aliases.")

    # Use provided PDFs or fall back to default
    pdf_sources = args.pdfs if args.pdfs else [DEFAULT_PDF_URL]

    # Configure MCP server to run via stdio transport
    mcp_server_config = dd.MCPServerConfig(
        name=MCP_SERVER_NAME,
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "serve"],
        env={"PDF_SOURCES": json.dumps(pdf_sources)},
    )

    config_builder = build_config(
        model_alias=args.model_alias,
        server_name=MCP_SERVER_NAME,
    )

    preview_results = generate_preview(
        config_builder=config_builder,
        num_records=args.num_records,
        mcp_server_config=mcp_server_config,
    )

    display_preview_record(preview_results)


if __name__ == "__main__":
    main()
