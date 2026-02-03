# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class CodeLang(str, Enum):
    BASH = "bash"
    C = "c"
    COBOL = "cobol"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    KOTLIN = "kotlin"
    PYTHON = "python"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    TYPESCRIPT = "typescript"
    SQL_SQLITE = "sql:sqlite"
    SQL_TSQL = "sql:tsql"
    SQL_BIGQUERY = "sql:bigquery"
    SQL_MYSQL = "sql:mysql"
    SQL_POSTGRES = "sql:postgres"
    SQL_ANSI = "sql:ansi"

    @staticmethod
    def parse(value: str | CodeLang) -> tuple[str, str | None]:
        value = value.value if isinstance(value, CodeLang) else value
        split_vals = value.split(":")
        return (split_vals[0], split_vals[1] if len(split_vals) > 1 else None)

    @staticmethod
    def parse_lang(value: str | CodeLang) -> str:
        return CodeLang.parse(value)[0]

    @staticmethod
    def parse_dialect(value: str | CodeLang) -> str | None:
        return CodeLang.parse(value)[1]

    @staticmethod
    def supported_values() -> set[str]:
        return {lang.value for lang in CodeLang}


SQL_DIALECTS: set[CodeLang] = {
    CodeLang.SQL_SQLITE,
    CodeLang.SQL_TSQL,
    CodeLang.SQL_BIGQUERY,
    CodeLang.SQL_MYSQL,
    CodeLang.SQL_POSTGRES,
    CodeLang.SQL_ANSI,
}

##########################################################
# Helper functions
##########################################################


def code_lang_to_syntax_lexer(code_lang: CodeLang | str) -> str:
    """Convert the code language to a syntax lexer for Pygments.

    Reference: https://pygments.org/docs/lexers/
    """
    code_lang_to_lexer = {
        CodeLang.BASH: "bash",
        CodeLang.C: "c",
        CodeLang.COBOL: "cobol",
        CodeLang.CPP: "cpp",
        CodeLang.CSHARP: "csharp",
        CodeLang.GO: "golang",
        CodeLang.JAVA: "java",
        CodeLang.JAVASCRIPT: "javascript",
        CodeLang.KOTLIN: "kotlin",
        CodeLang.PYTHON: "python",
        CodeLang.RUBY: "ruby",
        CodeLang.RUST: "rust",
        CodeLang.SCALA: "scala",
        CodeLang.SWIFT: "swift",
        CodeLang.TYPESCRIPT: "typescript",
        CodeLang.SQL_SQLITE: "sql",
        CodeLang.SQL_ANSI: "sql",
        CodeLang.SQL_TSQL: "tsql",
        CodeLang.SQL_BIGQUERY: "sql",
        CodeLang.SQL_MYSQL: "mysql",
        CodeLang.SQL_POSTGRES: "postgres",
    }
    return code_lang_to_lexer.get(code_lang, code_lang)
