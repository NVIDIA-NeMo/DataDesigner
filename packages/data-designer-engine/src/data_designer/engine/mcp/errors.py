# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError


class MCPError(DataDesignerError): ...


class MCPConfigurationError(MCPError): ...


class MCPClientUnavailableError(MCPError): ...


class MCPToolError(MCPError): ...


class DuplicateToolNameError(MCPConfigurationError):
    """Raised when the same tool name exists in multiple MCP providers or tool configs."""
