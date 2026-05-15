# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

import tiktoken


def count_text_tokens(text: str) -> int:
    return len(get_cl100k_base_tokenizer().encode(text, disallowed_special=()))


@lru_cache(maxsize=1)
def get_cl100k_base_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")
