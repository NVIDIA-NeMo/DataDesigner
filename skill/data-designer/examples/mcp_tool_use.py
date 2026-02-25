"""Fact-Checked Q&A Dataset — MCP tool calling with trace capture.

PATTERN REFERENCE ONLY — copy the structure, not the domain-specific values.

Demonstrates:
- Default model aliases (nvidia-text) — no manual ModelConfig needed
- FastMCP server with @mcp_server.tool() decorators
- LocalStdioMCPProvider launching the script as a subprocess
- ToolConfig with allow_tools whitelist
- tool_alias on LLMTextColumnConfig for grounded generation
- TraceType.ALL_MESSAGES for full tool-call history capture
- Self-contained server/client pattern (same script serves both roles)
"""

import sys

# --- MCP Server Definition ---
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("demo-tools")

KNOWLEDGE_BASE = {
    "python": "Python was created by Guido van Rossum and first released in 1991.",
    "rust": "Rust was first released in 2010 and is known for memory safety.",
    "javascript": "JavaScript was created by Brendan Eich in 1995 in 10 days.",
    "go": "Go was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson.",
    "java": "Java was developed by James Gosling at Sun Microsystems, released in 1995.",
}


@mcp_server.tool()
def lookup_language_facts(language: str) -> str:
    """Look up facts about a programming language.

    Args:
        language: Name of the programming language (e.g., 'python', 'rust').
    """
    key = language.lower().strip()
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]
    return f"No facts available for '{language}'. Available: {', '.join(KNOWLEDGE_BASE)}"


@mcp_server.tool()
def get_available_languages() -> str:
    """Get the list of programming languages available in the knowledge base."""
    return ", ".join(sorted(KNOWLEDGE_BASE.keys()))


# --- Main: Data Designer Client ---

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        mcp_server.run()
    else:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner

        # MCP provider: launch this script as subprocess server
        provider = dd.LocalStdioMCPProvider(
            name="demo-tools",
            command=sys.executable,
            args=[__file__, "serve"],
        )

        # Tool config: restrict to our two tools
        tool_config = dd.ToolConfig(
            tool_alias="lang-tools",
            providers=["demo-tools"],
            allow_tools=["lookup_language_facts", "get_available_languages"],
            max_tool_call_turns=5,
            timeout_sec=30.0,
        )

        data_designer = DataDesigner(mcp_providers=[provider])
        config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

        # --- Sampler columns ---

        config_builder.add_column(
            dd.SamplerColumnConfig(
                name="language",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=["Python", "Rust", "JavaScript", "Go", "Java"],
                ),
            )
        )

        # --- LLM column with tool access and trace capture ---

        config_builder.add_column(
            dd.LLMTextColumnConfig(
                name="fact_summary",
                prompt=(
                    "Use the available tools to look up facts about {{ language }}, "
                    "then write a 2-3 sentence summary about the language."
                ),
                model_alias="nvidia-text",
                system_prompt=(
                    "You are a helpful assistant with access to a programming language "
                    "knowledge base. Always use the lookup tool to get accurate facts "
                    "before writing your summary."
                ),
                tool_alias="lang-tools",
                with_trace=dd.TraceType.ALL_MESSAGES,
            )
        )

        # --- Preview ---

        preview = data_designer.preview(config_builder, num_records=3)
        preview.display_sample_record()

        results = data_designer.create(config_builder, num_records=10, dataset_name="lang-facts")
        dataset = results.load_dataset()
        analysis = results.load_analysis()
