# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
import logging
from typing import Literal, Required, TypedDict

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


# --- Typed trace message (mirrors ChatMessage.to_dict() shape) ---


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


# --- Role display metadata ---

_ROLE_STYLES: dict[str, tuple[str, str]] = {
    "system": ("⚙️ system", "dim"),
    "user": ("👤 user", "blue"),
    "assistant": ("🤖 assistant", "green"),
    "tool": ("📨 tool result", "magenta"),
}

_ROLE_HTML_COLORS: dict[str, str] = {
    "system": "#e8e8e8",
    "user": "#dbeafe",
    "reasoning": "#f3e8ff",
    "tool_call": "#fef3c7",
    "tool_result": "#fce7f3",
    "assistant": "#dcfce7",
}


# --- Helpers ---


def _extract_text_content(message: TraceMessage) -> str:
    content = message.get("content")
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def _format_tool_call_args(raw_args: str) -> str:
    try:
        parsed = json.loads(raw_args)
        return json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw_args


class TraceRenderer:
    """Renders LLM conversation traces for display_sample_record."""

    def render_rich(self, traces: list[TraceMessage], column_name: str) -> Panel:
        """Return a Rich Panel containing the formatted trace conversation."""
        elements: list[Text] = []
        tool_call_count = 0
        turn_ids: set[str] = set()

        for msg in traces:
            role = msg.get("role", "unknown")

            # Reasoning content (assistant only, shown before tool calls / final answer)
            reasoning = msg.get("reasoning_content")
            if reasoning:
                label = Text("💭 reasoning", style="italic bold purple")
                body = Text(f"  {reasoning}\n", style="italic purple")
                elements.extend([Text(""), label, body])

            # Tool calls (assistant only)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    tool_call_count += 1
                    tc_id = tc.get("id", "")
                    turn_ids.add(tc_id)
                    func = tc.get("function", {})
                    func_name = func.get("name", "unknown")
                    formatted_args = _format_tool_call_args(func.get("arguments", ""))

                    label = Text(f"🔧 tool call #{tool_call_count} → {func_name}", style="bold yellow")
                    body = Text(f"  {formatted_args}\n", style="yellow")
                    elements.extend([Text(""), label, body])
                continue

            # Tool result
            if role == "tool":
                label_text, style = _ROLE_STYLES["tool"]
                label = Text(label_text, style=f"bold {style}")
                text_content = _extract_text_content(msg)
                body = Text(f"  {text_content}\n", style=style)
                elements.extend([Text(""), label, body])
                continue

            # Regular message (system / user / assistant final answer)
            label_text, style = _ROLE_STYLES.get(role, (f"❓ {role}", "white"))
            text_content = _extract_text_content(msg)
            if not text_content and not reasoning and not tool_calls:
                continue

            label = Text(label_text, style=f"bold {style}")
            body = Text(f"  {text_content}\n", style=style)
            elements.extend([Text(""), label, body])

        turn_count = max(len(turn_ids), 1) if tool_call_count > 0 else 0
        call_word = "call" if tool_call_count == 1 else "calls"
        turn_word = "turn" if turn_count == 1 else "turns"
        summary = f"{tool_call_count} tool {call_word} in {turn_count} {turn_word}" if tool_call_count > 0 else ""

        if summary:
            rule_text = Text(f"─── {summary} ───", style="dim", justify="center")
            elements.extend([Text(""), rule_text])

        return Panel(
            Group(*elements),
            title=f"Trace: {column_name}",
            expand=True,
        )

    def render_notebook_html(self, traces: list[TraceMessage], column_name: str) -> bool:
        """Display HTML trace in Jupyter. Returns True if displayed, False otherwise."""
        try:
            from IPython.display import HTML, display

            get_ipython()  # noqa: F821
        except (ImportError, NameError):
            return False

        blocks: list[str] = []
        tool_call_count = 0

        for msg in traces:
            role = msg.get("role", "unknown")

            reasoning = msg.get("reasoning_content")
            if reasoning:
                blocks.append(
                    _build_html_block(
                        "💭 Reasoning",
                        html.escape(reasoning),
                        _ROLE_HTML_COLORS["reasoning"],
                    )
                )

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    tool_call_count += 1
                    func = tc.get("function", {})
                    func_name = html.escape(func.get("name", "unknown"))
                    formatted_args = html.escape(_format_tool_call_args(func.get("arguments", "")))
                    blocks.append(
                        _build_html_block(
                            f"🔧 Tool Call #{tool_call_count} — {func_name}",
                            f"<pre style='margin:0;white-space:pre-wrap;'>{formatted_args}</pre>",
                            _ROLE_HTML_COLORS["tool_call"],
                            escape_body=False,
                        )
                    )
                continue

            if role == "tool":
                text_content = html.escape(_extract_text_content(msg))
                blocks.append(
                    _build_html_block(
                        "📨 Tool Result",
                        f"<pre style='margin:0;white-space:pre-wrap;'>{text_content}</pre>",
                        _ROLE_HTML_COLORS["tool_result"],
                        escape_body=False,
                    )
                )
                continue

            text_content = _extract_text_content(msg)
            if not text_content and not reasoning and not tool_calls:
                continue

            role_labels = {
                "system": "⚙️ System",
                "user": "👤 User",
                "assistant": "🤖 Assistant (final answer)",
            }
            label = role_labels.get(role, f"❓ {html.escape(role)}")
            color = _ROLE_HTML_COLORS.get(role, "#f0f0f0")
            blocks.append(_build_html_block(label, html.escape(text_content), color))

        escaped_col_name = html.escape(column_name)
        call_summary = f" ({tool_call_count} tool call{'s' if tool_call_count != 1 else ''})" if tool_call_count else ""
        arrow = "<div style='text-align:center;font-size:18px;color:#888;margin:2px 0;'>↓</div>"
        body_html = arrow.join(blocks)

        container = f"""
        <div style='border:1px solid #ccc;border-radius:8px;padding:16px;margin:16px 0;
                     max-width:800px;font-family:system-ui,sans-serif;font-size:14px;'>
            <div style='font-weight:bold;margin-bottom:12px;font-size:15px;'>
                Trace: {escaped_col_name}{html.escape(call_summary)}
            </div>
            {body_html}
        </div>
        """
        display(HTML(container))  # noqa: F821
        return True


def _build_html_block(
    title: str,
    body: str,
    bg_color: str,
    *,
    escape_body: bool = True,
) -> str:
    escaped_title = html.escape(title)
    body_content = html.escape(body) if escape_body else body
    return f"""
    <div style='background:{bg_color};border-radius:6px;padding:10px 14px;margin:4px 0;'>
        <div style='font-weight:600;margin-bottom:4px;color:#333;'>{escaped_title}</div>
        <div style='color:#444;white-space:pre-wrap;'>{body_content}</div>
    </div>
    """
