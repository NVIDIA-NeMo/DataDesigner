# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class ParserException(Exception):
    """Identifies errors resulting from generic parser errors.

    Attributes:
        source (str | None): The source string that the parser
            attempted to parse.
    """

    source: str | None

    @staticmethod
    def _log_format(source: str) -> str:
        ## NOTE: The point of this was to be able to report offending
        ##  failure cases to the logs. This might not be what we want
        ##  to do in all cases. In the meantime, this note is left
        ##  for later review.
        #
        # return f"<source>{source}</source>"
        return ""

    def __init__(self, msg: str | None = None, source: str | None = None):
        msg = "" if msg is None else msg.strip()

        if source is not None:
            msg += self._log_format(source)

        super().__init__(msg)
        self.source = source
