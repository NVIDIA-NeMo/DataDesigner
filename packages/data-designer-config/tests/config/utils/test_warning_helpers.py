# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

from data_designer.config.utils.warning_helpers import _module_in_prefixes, warn_at_caller


def test_module_in_prefixes_exact_match():
    assert _module_in_prefixes("pydantic", ("pydantic",)) is True


def test_module_in_prefixes_dotted_submodule():
    assert _module_in_prefixes("pydantic.fields", ("pydantic",)) is True
    assert _module_in_prefixes("data_designer.config.models", ("data_designer",)) is True


def test_module_in_prefixes_rejects_prefix_collision():
    """Regression for PR #594 review (johnnygreco): ``startswith`` matching
    naively on the prefix would silently treat ``pydantic_helpers`` as part of
    the ``pydantic`` package. Anchor on exact-or-dotted-prefix instead.
    """
    assert _module_in_prefixes("pydantic_helpers", ("pydantic",)) is False
    assert _module_in_prefixes("pydanticfoo", ("pydantic",)) is False
    assert _module_in_prefixes("data_designer_other", ("data_designer",)) is False


def test_warn_at_caller_attributes_to_direct_caller():
    """When called from a non-skipped module, the warning attributes to the
    caller's frame.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_at_caller("hello", DeprecationWarning)  # line anchored below

    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert "hello" in str(caught[0].message)


def test_warn_at_caller_skips_skip_prefix_frames():
    """The walk should escape any frame whose module is listed in
    ``skip_prefixes`` and attribute to the first frame outside them. We
    simulate a library frame by ``exec``-ing a helper with a fake module name
    in its globals; calling that helper produces a frame whose
    ``f_globals["__name__"]`` is the fake name, mirroring how a real library
    frame would appear during the walk.
    """
    library_globals: dict[str, object] = {
        "__name__": "fake_library.dispatch",
        "warn_at_caller": warn_at_caller,
        "DeprecationWarning": DeprecationWarning,
    }
    exec(
        "def emit():\n    warn_at_caller('from-library', DeprecationWarning, skip_prefixes=('fake_library',))\n",
        library_globals,
    )
    emit = library_globals["emit"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit()

    assert len(caught) == 1
    assert caught[0].filename == __file__, f"Expected attribution at {__file__!r}, got {caught[0].filename!r}"


def test_warn_at_caller_default_skips_pydantic_and_data_designer():
    """Default ``skip_prefixes`` should cover both pydantic and data_designer
    so warnings emitted from validators inside DataDesigner internals attribute
    to the user, not to either library.
    """
    from data_designer.config.utils.warning_helpers import DEFAULT_INTERNAL_PREFIXES

    assert "pydantic" in DEFAULT_INTERNAL_PREFIXES
    assert "data_designer" in DEFAULT_INTERNAL_PREFIXES


def test_warn_at_caller_dedup_keys_on_user_call_site():
    """Python's once-per-location dedup keys on (text, category, lineno) inside
    the *attributing* frame's ``__warningregistry__``. With proper user
    attribution, two distinct call sites in the user's file each emit a
    warning under ``default`` filtering, while a repeated call at the same
    site emits only the first.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", DeprecationWarning)
        warn_at_caller("dedup-test", DeprecationWarning)  # site A
        warn_at_caller("dedup-test", DeprecationWarning)  # site B

    linenos = {w.lineno for w in caught}
    assert len(caught) == 2, [str(w.message) for w in caught]
    assert len(linenos) == 2, "Each call site should produce a distinct dedup key"
