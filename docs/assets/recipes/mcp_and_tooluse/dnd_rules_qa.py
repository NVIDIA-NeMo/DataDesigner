# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP + Tool Use Recipe (D&D Q&A) with BM25S Lexical Search

This recipe demonstrates an end-to-end MCP tool-calling workflow:

1) Download the Dungeons & Dragons v1 rules PDF.
2) Index it with BM25S for fast lexical search.
3) Use Data Designer tool calls (`search_docs`) to generate grounded Q&A pairs.

Prerequisites:
- `NVIDIA_API_KEY` if using `--model-alias nvidia-text` (default)
- Recipe dependencies: Install with `make install-dev-recipes`

Run:
    # Install recipe dependencies (preserves workspace packages)
    make install-dev-recipes

    # Then run the recipe
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py

    # Or run all recipes via Makefile
    make test-run-recipes

Common flags:
    # Generate a few Q&A pairs
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py --num-records 3

    # Use a different model
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py --model-alias gpt-4o

Server mode (used internally by Data Designer):
    python docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py serve

Notes:
- Downloads are stored locally under this directory.
- The BM25S index is built at server startup from the PDF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import bm25s
import fitz
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner

PDF_URL = "https://idiscepolidellamanticora.wordpress.com/wp-content/uploads/2012/09/tsr2010-players-handbook.pdf"
PDF_FILENAME = "tsr2010-players-handbook.pdf"
MCP_SERVER_NAME = "dnd-bm25-search"

# Global state for the BM25 index (populated at server startup)
_bm25_retriever: bm25s.BM25 | None = None
_corpus: list[dict[str, str]] = []


class DndQAPair(BaseModel):
    question: str = Field(..., description="A question grounded in the D&D rules text.")
    answer: str = Field(..., description="A concise answer grounded in the supporting passage.")
    supporting_passage: str = Field(
        ..., description="A short excerpt (2-4 sentences) copied from the search result that supports the answer."
    )
    citation: str = Field(
        ..., description="The citation (e.g. source url, page number, etc) of the supporting passage."
    )


class DndTopicList(BaseModel):
    topics: list[str] = Field(
        ...,
        description="High-level topics from the D&D rulebook.",
    )


def download_pdf(pdf_url: str, destination_dir: Path) -> Path:
    """Download the PDF if not already cached."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = destination_dir / PDF_FILENAME
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        urlretrieve(pdf_url, pdf_path)
    return pdf_path


def extract_pdf_text(pdf_path: Path) -> list[dict[str, str]]:
    """Extract text from PDF, returning a list of passages with metadata.

    Each passage corresponds to a page from the PDF.
    """
    passages: list[dict[str, str]] = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            passages.append(
                {
                    "text": text,
                    "page": str(page_num + 1),
                    "source": pdf_path.name,
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


def initialize_search_index(downloads_dir: Path) -> None:
    """Download PDF and build the BM25 index."""
    global _bm25_retriever, _corpus

    pdf_path = download_pdf(PDF_URL, downloads_dir)
    _corpus = extract_pdf_text(pdf_path)
    _bm25_retriever = build_bm25_index(_corpus)


# MCP Server Definition
mcp_server = FastMCP(MCP_SERVER_NAME)


@mcp_server.tool()
def search_docs(query: str, limit: int = 5) -> str:
    """Search through documents using BM25 lexical search.

    BM25 is a keyword-based retrieval algorithm that matches exact terms. For best results:

    - Use specific keywords, not full questions (e.g., "fireball damage radius" not "How much damage does fireball do?")
    - Include domain-specific terms that would appear in the source text (e.g., "THAC0", "saving throw", "armor class")
    - Combine multiple relevant terms to narrow results (e.g., "cleric spell healing cure")
    - Try synonyms or alternative phrasings if initial searches return poor results
    - Avoid filler words and focus on content-bearing terms

    Examples:
        Good queries:
        - "ranger tracking wilderness survival"
        - "magic missile automatic hit"
        - "dwarf constitution bonus saving throw"

        Less effective queries:
        - "What are the rules for rangers?"
        - "Tell me about magic missile"
        - "How do dwarves work?"

    Args:
        query: Search query string - use specific keywords for best results
        limit: Maximum number of results to return (default: 5)

    Returns:
        JSON string with search results including text excerpts and page numbers
    """
    global _bm25_retriever, _corpus

    if _bm25_retriever is None or not _corpus:
        return json.dumps({"error": "Search index not initialized"})

    query_tokens = bm25s.tokenize([query], stopwords="en")
    results, scores = _bm25_retriever.retrieve(query_tokens, k=min(limit, len(_corpus)))

    search_results: list[dict[str, str | float]] = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = float(scores[0, i])

        if score <= 0:
            continue

        passage = _corpus[doc_idx]
        search_results.append(
            {
                "text": passage["text"][:2000],
                "page": passage["page"],
                "source": passage["source"],
                "score": round(score, 4),
                "url": f"file://{passage['source']}#page={passage['page']}",
            }
        )

    return json.dumps({"results": search_results, "query": query, "total": len(search_results)})


def build_config(model_alias: str, server_name: str) -> dd.DataDesignerConfigBuilder:
    """Build the Data Designer configuration for D&D Q&A generation."""
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
        tool_names=["search_docs"],
        max_tool_calls=100,
        timeout_sec=30.0,
    )

    topic_prompt = "Extract a high-level list of all topics covered by this document."

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="topic_candidates",
            model_alias=model_alias,
            prompt=topic_prompt,
            system_prompt=(
                "You must call the search_docs tool before answering. "
                "Do not use outside knowledge; only use tool results. "
                "You can use as many tool calls as required to answer the user query."
            ),
            output_format=DndTopicList,
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
                "You must call the search_docs tool before answering. "
                "Do not use outside knowledge; only use tool results. "
                "You can use as many tool calls as required to answer the user query."
            ),
            output_format=DndQAPair,
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
    downloads_dir = Path(os.environ.get("PDF_CACHE_DIR", Path(__file__).resolve().parent / "downloads"))
    initialize_search_index(downloads_dir)
    mcp_server.run()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate D&D Q&A pairs using MCP tool calls with BM25S search.")
    subparsers = parser.add_subparsers(dest="command")

    # 'serve' subcommand for running the MCP server
    subparsers.add_parser("serve", help="Run the MCP server (used by Data Designer)")

    # Default command arguments (demo mode)
    parser.add_argument("--model-alias", type=str, default="nvidia-text", help="Model alias to use for generation")
    parser.add_argument("--num-records", type=int, default=4, help="Number of Q&A pairs to generate")

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

    base_dir = Path(__file__).resolve().parent
    downloads_dir = base_dir / "downloads"

    # Ensure PDF is downloaded before starting server
    download_pdf(PDF_URL, downloads_dir)

    # Configure MCP server to run via stdio transport
    mcp_server_config = dd.MCPServerConfig(
        name=MCP_SERVER_NAME,
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "serve"],
        env={"PDF_CACHE_DIR": str(downloads_dir)},
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
