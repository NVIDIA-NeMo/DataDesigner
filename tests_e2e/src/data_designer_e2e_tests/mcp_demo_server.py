# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

LOG_ENV_VAR = "MCP_DEMO_LOG_PATH"

mcp_server = FastMCP("data-designer-e2e-mcp")


def _log_tool_call(tool_name: str, arguments: dict[str, object], result: object) -> None:
    log_path = os.environ.get(LOG_ENV_VAR)
    if not log_path:
        return
    payload = {"tool": tool_name, "arguments": arguments, "result": result}
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


@mcp_server.tool()
def get_fact(topic: str) -> str:
    facts = {
        "mcp": "MCP lets models call tools over standardized transports.",
        "data-designer": "Data Designer generates structured synthetic datasets.",
    }
    result = facts.get(topic.lower(), f"{topic} is interesting.")
    _log_tool_call("get_fact", {"topic": topic}, result)
    return result


@mcp_server.tool()
def add_numbers(a: int, b: int) -> int:
    result = a + b
    _log_tool_call("add_numbers", {"a": a, "b": b}, result)
    return result


def main() -> None:
    mcp_server.run()


if __name__ == "__main__":
    main()
