# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP + Tool Use Recipe (D&D Q&A)

This recipe demonstrates an end-to-end MCP tool-calling workflow:

1) Download the Dungeons & Dragons v1 rules PDF.
2) Index it with `docs-mcp-server` (MCP search).
3) Use Data Designer tool calls (`search_docs`) to generate grounded Q&A pairs.

Prerequisites:
- Node.js 20+ (for `npx @arabold/docs-mcp-server`)
- `OPENAI_API_KEY` for docs-mcp-server embeddings
- `NVIDIA_API_KEY` if using `--model-alias nvidia-text` (default)

Run:
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py

Common flags:
    # First run: scrape the PDF (default)
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py --num-records 3

    # Subsequent runs: reuse the indexed corpus
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py --skip-scrape --num-records 5

    # Customize embeddings
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py \
      --embedding-model text-embedding-3-small \
      --embedding-api-key-env OPENAI_API_KEY

    # Increase PDF size limit if needed (default: 40MB)
    uv run docs/assets/recipes/mcp_and_tooluse/dnd_rules_qa.py --document-max-size-mb 60

Notes:
- The script writes a docs-mcp config file to `docs_mcp_store/docs_mcp_config.yaml` to raise the PDF size limit.
- Downloads and artifacts are stored locally under this directory.
- If you want to use a different LLM provider for generation, set `--model-alias` and the
  corresponding API key for that provider.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import urlretrieve

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

PDF_URL = "https://idiscepolidellamanticora.wordpress.com/wp-content/uploads/2012/09/tsr2010-players-handbook.pdf"
PDF_FILENAME = "tsr2010-players-handbook.pdf"

DOCS_MCP_HOST = "127.0.0.1"
DOCS_MCP_PORT = 6280
DOCS_MCP_SERVER_NAME = "docs-mcp-server"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_LIBRARY = "dnd-basic-rules"
DEFAULT_LIBRARY_VERSION = "1"
DEFAULT_DOCS_MCP_MAX_SIZE_MB = 40


class DndQAPair(BaseModel):
    question: str = Field(..., description="A question grounded in the D&D rules text.")
    answer: str = Field(..., description="A concise answer grounded in the supporting passage.")
    supporting_passage: str = Field(..., description="A short excerpt (2-4 sentences) copied from the search result.")
    source_url: str = Field(..., description="The URL for the supporting passage.")


def resolve_embedding_provider(embedding_model: str) -> str:
    if ":" in embedding_model:
        return embedding_model.split(":", 1)[0]
    return "openai"


def build_docs_mcp_env(
    store_path: Path,
    embedding_model: str,
    embedding_api_base: str | None,
    embedding_api_key_env: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["DOCS_MCP_STORE_PATH"] = str(store_path)
    env["DOCS_MCP_TELEMETRY"] = "false"
    env["DOCS_MCP_EMBEDDING_MODEL"] = embedding_model

    provider = resolve_embedding_provider(embedding_model)
    if provider == "openai":
        api_key = os.environ.get(embedding_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{embedding_api_key_env} must be set to use embeddings with provider 'openai'."
            )
        env["OPENAI_API_KEY"] = api_key
        if embedding_api_base:
            env["OPENAI_API_BASE"] = embedding_api_base
    return env


def download_pdf(pdf_url: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = destination_dir / PDF_FILENAME
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        urlretrieve(pdf_url, pdf_path)
    return pdf_path


def scrape_pdf_with_docs_mcp(
    npx_path: str,
    env: dict[str, str],
    library: str,
    version: str | None,
    pdf_uri: str,
    config_path: Path,
) -> None:
    command = [
        npx_path,
        "--yes",
        "@arabold/docs-mcp-server@latest",
        "scrape",
        library,
        pdf_uri,
        "--max-pages",
        "1",
        "--max-depth",
        "1",
        "--scope",
        "subpages",
    ]
    if version:
        command.extend(["--version", version])
    command.extend(["--config", str(config_path)])
    subprocess.run(command, env=env, check=True)


def start_docs_mcp_server(
    npx_path: str,
    env: dict[str, str],
    host: str,
    port: int,
    config_path: Path,
) -> subprocess.Popen[str]:
    command = [
        npx_path,
        "--yes",
        "@arabold/docs-mcp-server@latest",
        "--protocol",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--config",
        str(config_path),
    ]
    return subprocess.Popen(command, env=env)


def wait_for_port(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"docs-mcp-server did not start on {host}:{port} within {timeout_sec} seconds.")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def write_docs_mcp_config(store_path: Path, max_document_size_mb: int) -> Path:
    config_path = store_path / "docs_mcp_config.yaml"
    max_size_bytes = max_document_size_mb * 1024 * 1024
    config_contents = f"document:\n  maxSize: {max_size_bytes}\n"
    config_path.write_text(config_contents, encoding="utf-8")
    return config_path


def build_config(
    model_alias: str,
    server_name: str,
    library: str,
    version: str | None,
) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "ability scores",
                    "saving throws",
                    "combat rounds",
                    "spell casting",
                    "equipment and encumbrance",
                    "hit points and healing",
                    "alignment",
                    "exploration turns",
                ]
            ),
        )
    )

    tool_config = dd.MCPToolConfig(server_name=server_name, tool_names=["search_docs"], max_tool_calls=5)

    prompt_lines = [
        "You are generating Q&A pairs grounded in the Dungeons & Dragons Basic Rules (v1).",
        "First, call the MCP tool `search_docs` to retrieve relevant rules text.",
        "Use the tool with:",
        f'- library: "{library}"',
    ]
    if version:
        prompt_lines.append(f'- version: "{version}"')
    prompt_lines.extend(
        [
            '- query: "Dungeons & Dragons basic rules {{ topic }}"',
            "- limit: 3",
            "",
            "If the tool returns no results, broaden the query and try again.",
            "Then choose one result and create a grounded Q&A pair.",
            "",
            "Return JSON with keys: question, answer, supporting_passage, source_url.",
            "The supporting_passage must be a 2-4 sentence excerpt copied from the tool result.",
        ]
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa_pair",
            model_alias=model_alias,
            prompt="\n".join(prompt_lines),
            system_prompt=(
                "You must call the search_docs tool before answering. "
                "Do not use outside knowledge; only use tool results."
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
            name="source_url",
            expr="{{ qa_pair.source_url }}",
        )
    )
    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None,
    mcp_server: dd.MCPServerConfig,
    dataset_name: str,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path, mcp_servers=[mcp_server])
    data_designer.set_run_config(dd.RunConfig(include_full_traces=True))
    return data_designer.create(config_builder, num_records=num_records, dataset_name=dataset_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate D&D Q&A pairs using MCP tool calls.")
    parser.add_argument("--model-alias", type=str, default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="dnd_rules_qa")
    parser.add_argument("--library", type=str, default=DEFAULT_LIBRARY)
    parser.add_argument("--version", type=str, default=DEFAULT_LIBRARY_VERSION)
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--port", type=int, default=DOCS_MCP_PORT)
    parser.add_argument(
        "--document-max-size-mb",
        type=int,
        default=DEFAULT_DOCS_MCP_MAX_SIZE_MB,
        help="Docs MCP max document size for PDFs (default: 40MB).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Docs MCP embedding model (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--embedding-api-base",
        type=str,
        default=None,
        help=f"Optional OpenAI-compatible base URL (omit to use {DEFAULT_OPENAI_BASE_URL}).",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Env var name holding the embeddings API key (default: OPENAI_API_KEY).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.environ.get("NVIDIA_API_KEY") is None and args.model_alias.startswith("nvidia"):
        raise RuntimeError("NVIDIA_API_KEY must be set when using NVIDIA model aliases.")

    npx_path = shutil.which("npx")
    if not npx_path:
        raise RuntimeError("npx was not found. Install Node.js 20+ and ensure npx is on PATH.")

    base_dir = Path(__file__).resolve().parent
    downloads_dir = base_dir / "downloads"
    store_path = base_dir / "docs_mcp_store"
    store_path.mkdir(parents=True, exist_ok=True)
    config_path = write_docs_mcp_config(store_path, args.document_max_size_mb)

    pdf_path = download_pdf(PDF_URL, downloads_dir)
    pdf_uri = pdf_path.resolve().as_uri()

    docs_mcp_env = build_docs_mcp_env(
        store_path=store_path,
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_api_base,
        embedding_api_key_env=args.embedding_api_key_env,
    )

    if not args.skip_scrape:
        scrape_pdf_with_docs_mcp(
            npx_path=npx_path,
            env=docs_mcp_env,
            library=args.library,
            version=args.version,
            pdf_uri=pdf_uri,
            config_path=config_path,
        )

    server_process = start_docs_mcp_server(
        npx_path=npx_path,
        env=docs_mcp_env,
        host=DOCS_MCP_HOST,
        port=args.port,
        config_path=config_path,
    )

    try:
        wait_for_port(DOCS_MCP_HOST, args.port, timeout_sec=60)
        mcp_server = dd.MCPServerConfig(
            name=DOCS_MCP_SERVER_NAME,
            url=f"http://{DOCS_MCP_HOST}:{args.port}/sse",
        )

        config_builder = build_config(
            model_alias=args.model_alias,
            server_name=DOCS_MCP_SERVER_NAME,
            library=args.library,
            version=args.version,
        )
        results = create_dataset(
            config_builder=config_builder,
            num_records=args.num_records,
            artifact_path=args.artifact_path or base_dir / "artifacts",
            mcp_server=mcp_server,
            dataset_name=args.dataset_name,
        )

        print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    finally:
        stop_process(server_process)


if __name__ == "__main__":
    main()
