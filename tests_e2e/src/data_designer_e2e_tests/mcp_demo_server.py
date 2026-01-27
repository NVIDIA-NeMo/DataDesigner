# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("data-designer-e2e-mcp")


@mcp_server.tool()
def get_fact(topic: str) -> str:
    facts = {
        "mcp": "MCP lets models call tools over standardized transports.",
        "data-designer": "Data Designer generates structured synthetic datasets.",
    }
    return facts.get(topic.lower(), f"{topic} is interesting.")


@mcp_server.tool()
def add_numbers(a: int, b: int) -> int:
    return a + b


def main() -> None:
    mcp_server.run()


if __name__ == "__main__":
    main()
