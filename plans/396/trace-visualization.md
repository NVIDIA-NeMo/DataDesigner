# Plan: Trace Visualization in `display_sample_record`

Created: 2026-03-11
Status: Draft

Issue: [#396](https://github.com/NVIDIA-NeMo/DataDesigner/issues/396)

## Goal

Add first-class trace visualization to `display_sample_record()` so that
`__trace` columns (produced by LLM columns with `with_trace != TraceType.NONE`)
are rendered as readable conversation flows instead of raw dicts.

Two rendering backends:

1. **Rich terminal** — styled panels, works everywhere (terminal + notebook)
2. **Jupyter HTML** — colored block flow diagram with arrows, displayed after
   Rich output (same pattern as `_display_image_if_in_notebook`)

Traces are shown by default when trace data exists. Users opt out via
`include_traces=False`.

## Current State

### Where display lives

`display_sample_record` is defined in
`packages/data-designer-config/src/data_designer/config/utils/visualization.py`:

- **Mixin method** (`WithRecordSamplerMixin.display_sample_record`, line 168):
  resolves the record, delegates to the standalone function.
- **Standalone function** (`display_sample_record`, line 259): builds a
  `render_list` of Rich renderables, then prints via `Console`. Dedicated
  sections exist for seed columns, generated columns, images, code, validation,
  judge, and processor outputs.

### How trace columns are identified

- `TRACE_COLUMN_POSTFIX = "__trace"` in `config/utils/constants.py` (line 170).
- `LLMTextColumnConfig.side_effect_columns` returns
  `[f"{self.name}{TRACE_COLUMN_POSTFIX}"]` when `with_trace != TraceType.NONE`.
- `LLMCodeColumnConfig`, `LLMStructuredColumnConfig`, and
  `LLMJudgeColumnConfig` all inherit this from `LLMTextColumnConfig`.

### Trace data shape

Each `__trace` column value is a `list[dict]` of ChatML-style messages:

```python
{
    "role": "system" | "user" | "assistant" | "tool",
    "content": [{"type": "text", "text": "..."}],
    "reasoning_content": str | None,       # assistant only
    "tool_calls": [                        # assistant only
        {
            "id": "...",
            "type": "function",
            "function": {"name": "...", "arguments": "..."}
        }
    ] | None,
    "tool_call_id": str | None,            # tool role only
}
```

### Current gap

- Trace columns are **not rendered** in `display_sample_record`. The generated
  columns table shows the main column value (e.g., `answer`) but skips
  `answer__trace`.
- The side-effect column display loop (lines 313-316) only runs for
  `CUSTOM`/plugin column types, not LLM columns.

## Design

### New module: `trace_renderer.py`

Create `packages/data-designer-config/src/data_designer/config/utils/trace_renderer.py`.

Keep `visualization.py` focused on layout orchestration; trace rendering is
complex enough to warrant its own module.

#### Typed trace message

Trace data arrives as `list[dict]` (serialized via `ChatMessage.to_dict()` in
`llm_completion.py`). Define a `TypedDict` to give the renderer a strong
contract without coupling to the engine's `ChatMessage` dataclass:

```python
class TraceToolCallFunction(TypedDict):
    name: str
    arguments: str

class TraceToolCall(TypedDict):
    id: str
    type: str
    function: TraceToolCallFunction

class TraceContentBlock(TypedDict):
    type: str
    text: str

class TraceMessage(TypedDict, total=False):
    role: Required[Literal["system", "user", "assistant", "tool"]]
    content: Required[list[TraceContentBlock]]
    reasoning_content: str | None
    tool_calls: list[TraceToolCall] | None
    tool_call_id: str | None
```

This mirrors the dict shape produced by `ChatMessage.to_dict()` (in
`engine/models/utils.py`, line 36) without importing from the engine package.

#### `TraceRenderer` class

```python
class TraceRenderer:
    """Renders LLM conversation traces for display_sample_record."""

    def __init__(
        self,
        max_content_length: int = 300,
        max_args_length: int = 200,
    ) -> None: ...

    def render_rich(self, traces: list[TraceMessage], column_name: str) -> Panel:
        """Return a Rich Panel containing the formatted trace conversation."""
        ...

    def render_notebook_html(self, traces: list[TraceMessage], column_name: str) -> bool:
        """Display HTML trace in Jupyter. Returns True if displayed, False otherwise."""
        ...
```

Both methods accept `traces: list[TraceMessage]` — plural, since the parameter
is a list of message dicts representing the full conversation.

**Rich rendering** (`render_rich`):

Returns a `Panel` containing a `Group` of styled `Text` / `Pretty` blocks:

| Role | Style | Content |
|------|-------|---------|
| system | dim | Truncated system prompt |
| user | blue | Truncated user message |
| assistant (reasoning) | italic purple | `reasoning_content` field |
| assistant (tool call) | bold yellow | Function name + formatted JSON args |
| assistant (final) | green | Truncated final answer |
| tool result | magenta | Truncated tool response |

A summary line at the bottom: `"N tool call(s) across M turn(s)"`.

**Jupyter HTML rendering** (`render_notebook_html`):

Same guard pattern as `_display_image_if_in_notebook`:

```python
try:
    from IPython.display import HTML, display
    get_ipython()
except (ImportError, NameError):
    return False
```

Renders colored HTML blocks with role-based backgrounds and arrow connectors
between messages. Adapted from the prototype in
`.scratch/test-pr-373/04_mcp_tool_calling.ipynb` (cell 19).

### Integration into `visualization.py`

#### Parameter addition

Add `include_traces: bool = True` to both:

1. `WithRecordSamplerMixin.display_sample_record` (line 168)
2. Standalone `display_sample_record` (line 259)

#### Trace section placement

In the standalone `display_sample_record`, after the "Generated Columns" table
(~line 317) and before images (~line 320):

```python
# Trace sections
traces_to_display_later: list[tuple[str, list[TraceMessage]]] = []
if include_traces:
    trace_renderer = TraceRenderer()
    for col in config_builder.get_all_columns():
        for side_col in col.side_effect_columns:
            if side_col.endswith(TRACE_COLUMN_POSTFIX) and side_col in record:
                traces: list[TraceMessage] = record[side_col]
                if isinstance(traces, list) and len(traces) > 0:
                    render_list.append(
                        pad_console_element(trace_renderer.render_rich(traces, side_col))
                    )
                    traces_to_display_later.append((side_col, traces))
```

Then after the Rich console output (alongside images):

```python
if len(traces_to_display_later) > 0:
    trace_renderer = TraceRenderer()
    for col_name, traces in traces_to_display_later:
        trace_renderer.render_notebook_html(traces, col_name)
```

### What does NOT change

- The "Generated Columns" table continues to show the main column value
  (`answer`) as a row — no change there.
- Trace columns are NOT added as rows in the Generated Columns table — they get
  their own dedicated section.
- No changes to column configs, constants, engine code, or trace data shape.
- `_DEDICATED_DISPLAY_COL_TYPES` is not modified (trace is not a column type,
  it's a side-effect of LLM column types).

## Visual Preview

Below are mockups of both rendering backends using a real trace from the MCP
tool-calling notebook (question: "What is the current population of Tokyo?",
single `tavily_search` tool call).

### Rich terminal rendering

Appears inline in the `display_sample_record` output, after the "Generated
Columns" table:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                        Trace: answer__trace                                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                                                                                                           │
│  ⚙️ system                                                                                                │
│  You are a research assistant. Use the tavily_search tool to find current information. After searching,    │
│  provide a brief, factual answer with your sources.                                                       │
│                                                                                                           │
│  👤 user                                                                                                  │
│  Answer the following question using web search. Search for current information, then provide a concise   │
│  factual answer.                                                                                          │
│  Question: What is the current population of Tokyo?                                                       │
│                                                                                                           │
│  💭 reasoning                                                                                             │
│  We need to answer with current population of Tokyo. We must use tavily_search to find current info.      │
│  Then provide concise factual answer with sources. Let's perform a search…                                │
│                                                                                                           │
│  🔧 tool call #1 → tavily_search                                                                         │
│  {                                                                                                        │
│    "query": "current population of Tokyo",                                                                │
│    "max_results": 5,                                                                                      │
│    "search_depth": "basic",                                                                               │
│    "time_range": "year"                                                                                   │
│  }                                                                                                        │
│                                                                                                           │
│  📨 tool result                                                                                           │
│  {"query":"current population of Tokyo","results":[{"url":"https://www.nippon.com/en/japan-data/…",       │
│  "title":"Tokyo Third in UN Ranking of Global Megacities at 33.4 Million","content":"According to…"}…]}   │
│                                                                                                           │
│  🤖 assistant                                                                                             │
│  Current population of Tokyo (metropolitan area): ≈ 36.9 million people (2026 estimate).                  │
│  Sources: World Population Review, United Nations World Urbanization Prospects                             │
│                                                                                                           │
│  ─── 1 tool call in 1 turn ───                                                                            │
│                                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Jupyter HTML rendering

Displayed below the Rich output when in a notebook (same position as images).
Each message is a colored block; arrows connect them vertically:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Row 0 — Trace: answer__trace (1 tool call)                        │
│  Question: What is the current population of Tokyo?                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ⚙️ System                                               #e8e8e8│
│  │ You are a research assistant. Use the tavily_search tool…   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 👤 User                                              #dbeafe│   │
│  │ Answer the following question using web search…             │   │
│  │ Question: What is the current population of Tokyo?          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 💭 Reasoning                                         #f3e8ff│   │
│  │ We need to answer with current population of Tokyo. We      │   │
│  │ must use tavily_search to find current info…                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 🔧 Tool Call #1 — tavily_search                      #fef3c7│   │
│  │ {                                                           │   │
│  │   "query": "current population of Tokyo",                   │   │
│  │   "max_results": 5                                          │   │
│  │ }                                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 📨 Tool Result                                       #fce7f3│   │
│  │ {"query":"current population of Tokyo","results":[…]}       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 🤖 Assistant (final answer)                          #dcfce7│   │
│  │ Current population of Tokyo (metropolitan area):            │   │
│  │ ≈ 36.9 million people (2026 estimate).                      │   │
│  │ Sources: World Population Review, UN World Urbanization…    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The hex codes shown (e.g., `#dbeafe`) indicate the background color of each
block. In the actual HTML output these are CSS `background` properties on
`<div>` elements with `border-radius` for rounded corners.

## File Summary

| File | Change |
|------|--------|
| `config/utils/trace_renderer.py` | **New** — `TraceMessage` TypedDict, `TraceRenderer` class with `render_rich` and `render_notebook_html` |
| `config/utils/visualization.py` | Add `include_traces` param, wire in `TraceRenderer` at the right point in `display_sample_record` |

## Testing

- Unit tests for `TraceRenderer.render_rich` with various trace shapes (no
  tool calls, single tool call, multi-turn tool calls, empty trace, reasoning
  content present/absent).
- Integration test: build a config with `with_trace=TraceType.ALL_MESSAGES`,
  mock a trace column in the record, call `display_sample_record` and verify
  no exceptions.
- Verify existing `display_sample_record` tests still pass (no regressions
  from the new parameter defaulting to `True`).
