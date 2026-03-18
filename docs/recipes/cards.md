# Use Case Recipes

Recipes are a collection of code examples that demonstrate how to leverage Data Designer in specific use cases.
Each recipe is a self-contained example that can be run independently.

!!! question "New to Data Designer?"
    Recipes provide working code for specific use cases without detailed explanations. If you're learning Data Designer for the first time, we recommend starting with our [tutorial notebooks](../../notebooks/), which offer step-by-step guidance and explain core concepts. Once you're familiar with the basics, return here for practical, ready-to-use implementations.

!!! tip Prerequisite
    These recipes use the Open AI model provider by default. Ensure your OpenAI model provider has been set up using the Data Designer CLI before running a recipe.

<div class="grid cards" markdown>

-   :material-snake:{ .lg .middle } **Text to Python**

    Generate a dataset of natural language instructions paired with Python code implementations, with varying complexity levels and industry focuses.

    ---

    **Demonstrates:**

    - Python code generation
    - Python code validation
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](code_generation/text_to_python.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/code_generation/text_to_python.py){ .md-button download="text_to_python.py" }

-   :material-database:{ .lg .middle } **Text to SQL**

    Generate a dataset of natural language instructions paired with SQL code implementations, with varying complexity levels and industry focuses.

    ---

    **Demonstrates:**

    - SQL code generation
    - SQL code validation
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](code_generation/text_to_sql.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/code_generation/text_to_sql.py){ .md-button download="text_to_sql.py" }

-   :material-database-search:{ .lg .middle } **Nemotron Super Text to SQL**

    Generate enterprise-grade text-to-SQL training data used for Nemotron Super v3 SFT -- dialect-specific SQL, distractor injection, dirty data, 5 LLM judges with 15 scoring dimensions.

    ---

    **Demonstrates:**

    - Dialect-specific SQL generation (SQLite, MySQL, PostgreSQL)
    - Distractor table/column and dirty data injection
    - Conditional sampling with SubcategorySamplerParams
    - 5 LLM judges with 15 score extraction columns

    ---

    [:material-book-open-page-variant: View Recipe](code_generation/enterprise_text_to_sql.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/code_generation/enterprise_text_to_sql.py){ .md-button download="enterprise_text_to_sql.py" }


-   :material-chat:{ .lg .middle } **Product Info QA**

    Generate a dataset that contains information about products and associated question/answer pairs.

    ---

    **Demonstrates:**

    - Structured outputs
    - Expression columns
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](qa_and_chat/product_info_qa.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/qa_and_chat/product_info_qa.py){ .md-button download="product_info_qa.py" }


-   :material-chat:{ .lg .middle } **Multi-Turn Chat**

    Generate a dataset of multi-turn chat conversations between a user and an AI assistant.

    ---

    **Demonstrates:**

    - Structured outputs
    - Expression columns
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](qa_and_chat/multi_turn_chat.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/qa_and_chat/multi_turn_chat.py){ .md-button download="multi_turn_chat.py" }


-   :material-tools:{ .lg .middle } **Basic MCP Tool Use**

    Minimal example of MCP tool calling with Data Designer. Defines a simple MCP server with basic tools and generates data that requires tool calls to complete.

    ---

    **Demonstrates:**

    - MCP tool calling with LocalStdioMCPProvider
    - Simple tool server definition
    - Tool-augmented text generation

    ---

    [:material-book-open-page-variant: View Recipe](mcp_and_tooluse/basic_mcp.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/mcp_and_tooluse/basic_mcp.py){ .md-button download="basic_mcp.py" }

-   :material-tools:{ .lg .middle } **PDF Document QA (MCP + Tool Use)**

    Generate grounded Q&A pairs from PDF documents using MCP tool calls and BM25 search.

    ---

    **Demonstrates:**

    - MCP tool calling with LocalStdioMCPProvider
    - BM25 lexical search for retrieval
    - Retrieval-grounded QA generation
    - Per-column trace capture

    ---

    [:material-book-open-page-variant: View Recipe](mcp_and_tooluse/pdf_qa.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/mcp_and_tooluse/pdf_qa.py){ .md-button download="pdf_qa.py" }

-   :material-magnify:{ .lg .middle } **Nemotron Super Search Agent (MCP + Tool Use)**

    Generate multi-turn search agent trajectories used for Nemotron Super post-training -- Tavily web search via MCP, Wikidata KG seeding, BrowseComp-style question generation.

    ---

    **Demonstrates:**

    - MCP tool calling with Tavily web search
    - Wikidata knowledge graph seeding
    - Two-stage question generation (draft + BrowseComp obfuscation)
    - Full trajectory capture with traces
    - Structured output formatting

    ---

    [:material-book-open-page-variant: View Recipe](mcp_and_tooluse/search_agent.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/mcp_and_tooluse/search_agent.py){ .md-button download="search_agent.py" }

-   :material-file-document-multiple:{ .lg .middle } **Markdown Section Seed Reader**

    Define a custom `FileSystemSeedReader` inline and turn Markdown files into one seed row per heading section.

    ---

    **Demonstrates:**

    - Single-file custom seed reader pattern
    - `hydrate_row()` fanout from `1 -> N`
    - Manifest-based file selection semantics
    - `DirectorySeedSource` customization without a new `seed_type`

    ---

    [:material-book-open-page-variant: View Recipe](plugin_development/markdown_seed_reader.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/plugin_development/markdown_seed_reader.py){ .md-button download="markdown_seed_reader.py" }

</div>
