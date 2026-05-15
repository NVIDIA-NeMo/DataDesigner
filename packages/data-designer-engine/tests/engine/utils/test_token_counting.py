# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.utils.token_counting import count_text_tokens, get_cl100k_base_tokenizer


def test_count_text_tokens_counts_with_cl100k_base_tokenizer() -> None:
    """count_text_tokens delegates to the shared cl100k_base tokenizer."""
    text = "Hello, token counting."

    assert count_text_tokens(text) == len(get_cl100k_base_tokenizer().encode(text, disallowed_special=()))


def test_get_cl100k_base_tokenizer_returns_cached_instance() -> None:
    """get_cl100k_base_tokenizer returns the same cached tokenizer instance."""
    get_cl100k_base_tokenizer.cache_clear()
    tokenizer1 = get_cl100k_base_tokenizer()
    tokenizer2 = get_cl100k_base_tokenizer()

    assert tokenizer1 is tokenizer2
