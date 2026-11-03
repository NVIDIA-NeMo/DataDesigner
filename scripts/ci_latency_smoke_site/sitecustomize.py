# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket

_original_getfqdn = socket.getfqdn


def numeric_loopback_name(name: str = "") -> str:
    if name in ("127.0.0.1", "::1"):
        return name
    return _original_getfqdn(name)


if os.environ.get("DD_SMOKE_NUMERIC_LOOPBACK") == "1":
    socket.getfqdn = numeric_loopback_name
