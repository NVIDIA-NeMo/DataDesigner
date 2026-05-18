# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

import tiktoken

from data_designer.engine.utils.token_counting import count_text_tokens, get_cl100k_base_tokenizer


def test_count_text_tokens_counts_with_cl100k_base_tokenizer() -> None:
    """count_text_tokens delegates to the shared cl100k_base tokenizer."""
    text = "Hello, token counting."

    tokenizer = tiktoken.get_encoding("cl100k_base")
    assert count_text_tokens(text) == len(tokenizer.encode(text, disallowed_special=()))


def test_get_cl100k_base_tokenizer_returns_cached_instance() -> None:
    """get_cl100k_base_tokenizer returns the same cached tokenizer instance."""
    get_cl100k_base_tokenizer.cache_clear()
    tokenizer1 = get_cl100k_base_tokenizer()
    tokenizer2 = get_cl100k_base_tokenizer()

    assert tokenizer1 is tokenizer2


def test_importing_token_counting_does_not_eagerly_import_tiktoken() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import data_designer.engine.utils.token_counting; print(int('tiktoken' in sys.modules))",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"
