# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.utils.code_lang import CodeLang, code_lang_to_syntax_lexer


def test_parse():
    assert CodeLang.parse("ruby") == ("ruby", None)
    assert CodeLang.parse("sql:sqlite") == ("sql", "sqlite")
    assert CodeLang.parse(CodeLang.RUBY) == ("ruby", None)
    assert CodeLang.parse(CodeLang.SQL_SQLITE) == ("sql", "sqlite")


def test_parse_lang():
    assert CodeLang.parse_lang("ruby") == "ruby"
    assert CodeLang.parse_lang(CodeLang.RUBY) == "ruby"
    assert CodeLang.parse_lang("sql:sqlite") == "sql"
    assert CodeLang.parse_lang(CodeLang.SQL_SQLITE) == "sql"


def test_parse_dialect():
    assert CodeLang.parse_dialect("ruby") is None
    assert CodeLang.parse_dialect(CodeLang.RUBY) is None
    assert CodeLang.parse_dialect("sql:sqlite") == "sqlite"
    assert CodeLang.parse_dialect(CodeLang.SQL_SQLITE) == "sqlite"


def test_supported_values():
    assert len(CodeLang.supported_values()) == 21


def test_code_lang_to_syntax_lexer():
    assert code_lang_to_syntax_lexer("ruby") == "ruby"
    assert code_lang_to_syntax_lexer(CodeLang.RUBY) == "ruby"
    assert code_lang_to_syntax_lexer("sql:sqlite") == "sql"
    assert code_lang_to_syntax_lexer(CodeLang.SQL_SQLITE) == "sql"
    assert code_lang_to_syntax_lexer("something-not-in-mapping") == "something-not-in-mapping"
