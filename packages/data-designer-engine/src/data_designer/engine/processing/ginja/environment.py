# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial, wraps
from typing import Any

from jinja2 import meta
from jinja2 import nodes as j_nodes
from jinja2.exceptions import SecurityError, TemplateSyntaxError, UndefinedError
from jinja2.nodes import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jsonpath_rust_bindings import Finder

from data_designer.config.run_config import JinjaRenderingEngine
from data_designer.engine.processing.ginja.ast import (
    AccessChain,
    ast_count_name_references,
    ast_descendant_count,
    ast_extract_access_chains,
    ast_max_depth,
    resolve_access_chain,
)
from data_designer.engine.processing.ginja.exceptions import (
    EmptyTemplateRenderError,
    UserTemplateError,
    UserTemplateUnsupportedFiltersError,
    maybe_handle_missing_filter_exception,
)
from data_designer.engine.processing.ginja.record import sanitize_record

MAX_RENDERED_LEN = 512_000
MAX_AST_NODE_COUNT = 600
MAX_AST_DEPTH = 10
ALLOWED_JINJA_FILTERS = [
    ## Jinja2 Builtin Filters
    "abs",
    "capitalize",
    "escape",
    "first",
    "float",
    "forceescape",
    "int",
    "items",
    "last",
    "length",
    "list",
    "lower",
    "max",
    "min",
    "random",
    "replace",
    "reverse",
    "round",
    "sort",
    "string",
    "title",
    "trim",
    "truncate",
    "unique",
    "upper",
    "urlencode",
    ## Custom Filters
    "jsonpath",
]

USER_PROMPT_TEMPLATE_ERROR_MESSAGE = """\
User provided prompt generation template is invalid.\
"""
UNSUPPORTED_AST_NODES = [
    j_nodes.Import,  # No {% include ... %}
    j_nodes.Macro,  # No {% macro ... %}
    j_nodes.Assign,  # No {% set ... %}
    j_nodes.Extends,  # No {% extends ... %}
    j_nodes.Block,  # No {% block ... %}
]


def jsonpath_jinja_filter(data: dict, expression: str) -> list[Any]:
    """Defines JSONPath-based operations on variables.

    Args:
        data (dict): data object to filter.
        expression (str): a valid JSONPath string.

    Returns:
        list[Any]: A list of JSONPath match values.
    """
    if not isinstance(data, dict):
        raise ValueError("Cannot perform JSONPath filter on non-structured data.")

    return [result.data for result in Finder(data).find(expression)]


def is_jinja_template(user_template: str) -> bool:
    """Determine if a prompt template is a Jinja2 template from heuristics.

    This function is intended to help migration from format strings->Jinja.
    If we only support Jinja2, then this function is not needed.

    Args:
        user_template (str): A user-provided template string to test.

    Returns:
        True if the heuristic believes it is a Jinja2 template.
    """
    jinja_pattern_pairs = [("{{", "}}"), ("{%", "%}"), ("{#", "#}")]
    for open_pattern, close_pattern in jinja_pattern_pairs:
        if open_pattern in user_template and close_pattern in user_template:
            return True

    return False


class UserTemplateSandboxEnvironment(ImmutableSandboxedEnvironment):
    """Defines a robust environment for rendering Gretel's Jinja2 subset.

    The use of Jinja2 sandboxing is critical. We are taking Jinja2
    templates from users -- we need to take steps to ensure that users
    are not able to break containment or exfiltrate server-side secrets.

    This Environment definition attempts to lock down as much as we can
    for a pure python implementation by extending restrictions past
    that of the `ImmutableSandboxedEnviornment.` While that environment
    provides a base layer of protections, including:

        - No references to private attributes
        - Restrictions on loop iterations (OverflowError)

    We enforce further precautions:

        - Forced auto-escaping templates (preventing some injection attacks).
        - Prevents access to the template's `self` attribute.
        - Prevents reference to variables except for a provided white-list.
        - Removes support for: include, extend, macro, set, block, and nested loops
        - Errors on too-long rendered templates (e.g. >128k chars).
        - Remove all default Jinja filter operations except for JSONPath (negotiable).
        - Uses AST static analysis to threshold the complexity of allowed templates.

    """

    max_rendered_len: int
    max_ast_node_count: int
    max_ast_depth: int
    allowed_references: list[str]
    _prefer_dict_key_access: bool

    def __init__(
        self,
        allowed_references: list[str] | None = None,
        max_rendered_len: int = MAX_RENDERED_LEN,
        max_ast_node_count: int = MAX_AST_NODE_COUNT,
        max_ast_depth: int = MAX_AST_DEPTH,
        prefer_dict_key_access: bool = False,
        **kwargs,
    ):
        """Args:
        max_rendered_len (int): The maximum allowable character count for
            rendered templates.

        allowed_references (optional, list[str]): If set, indicates which variables
            are allowed to be referenced by the Jinja2 template. If not specified,
            defaults to [], which indicates that the Jinja2 template is not
            allowed to refer to _any_ variables outside of itself.

        max_ast_node_count (optional, int): Parameter for static analysis of
            Jinja2 template complexity -- counts the number of distinct nodes
            in the parsed Jinja2 AST. A large number of nodes indicates many
            distinct operations within the provided user template, which can
            cause long compute times, or may be malicious in nature. If not
            specified, defaults to MAX_AST_NODE_COUNT set by this module.

        max_ast_depth (optional, int): Parameter for static analysis of
            Jinja2 template complexity -- measures the maximum depth of the
            parsed Jinja2 AST. A high depth indicates a high degree of nesting
            within the user template. This may can cause long compute times,
            or may be malicious in nature. If not specified, defaults to
            MAX_AST_DEPTH set by this module.

        **kwargs: Additional kwargs passed to ImmutableSandboxedEnvironment.
        """
        super().__init__(autoescape=False, **kwargs)
        self.max_rendered_len = max_rendered_len
        self.max_ast_node_count = max_ast_node_count
        self.max_ast_depth = max_ast_depth
        self.allowed_references = allowed_references if allowed_references else []
        self._prefer_dict_key_access = prefer_dict_key_access

        ## Add on our supported filters
        self.filters["jsonpath"] = jsonpath_jinja_filter

        ## Cut out all but approved Jinja filters
        self.filters = {k: v for k, v in self.filters.items() if k in ALLOWED_JINJA_FILTERS}

    def getattr(self, obj: Any, attribute: str) -> Any:
        # When enabled, prefer dict key lookup over attribute access so that
        # keys like "items" resolve to dict["items"] instead of dict.items.
        if self._prefer_dict_key_access and isinstance(obj, dict) and attribute in obj:
            return obj[attribute]
        return super().getattr(obj, attribute)

    def _assert_template_has_valid_references(self, ast: Template) -> None:
        """Assert that all named variable references are allowed.

        Checks against the environment's allowed reference list created
        at initialization.
        """
        template_vars = meta.find_undeclared_variables(ast)
        unallowed_vars = set(template_vars) - set(self.allowed_references)
        if len(unallowed_vars) > 0:
            raise UserTemplateError(f"Unknown variable references in Jinja template: {unallowed_vars}")

    def _assert_template_has_valid_ast_nodes(self, ast: Template) -> None:
        """Assert that un-allowed operations aren't in the template."""
        black_list_node_count = sum(ast_descendant_count(ast, node_type) for node_type in UNSUPPORTED_AST_NODES)

        if black_list_node_count != 0:
            raise UserTemplateError("Non-permitted operations in Jinja template.")

    def _assert_template_has_no_recursive_for(self, ast: Template) -> None:
        """Assert that the template does not use {% for ... recursive %}"""
        if any(node.recursive for node in ast.find_all(j_nodes.For)):
            raise UserTemplateError("Non-permitted operations in Jinja template.")

    def _assert_template_has_no_nested_for(self, ast: Template) -> None:
        """Assert that the template does not contain nested loops.

        This assertion is made to ensure that templates cannot combinatorially
        explode. High-range values are controlled by the `MAX_RANGE` setting
        on `SandboxedEnvironment`.
        """
        # Check each For node in the AST to see if it has For descendants
        for node in ast.find_all(j_nodes.For):
            if ast_descendant_count(node, only_type=j_nodes.For):
                raise UserTemplateError("Non-permitted operations in Jinja template (nested-for).")

    def _assert_template_ast_complexity(self, ast: Template) -> None:
        """Assert that the AST tree parsed from the template is not overly complex.

        Complexity is measured by the depth of the tree (measure of nesting),
        as well as the number of nodes it contains (how many distinct operations).
        If either is over a fixed limit specified at initialization, the assert fails.
        """
        node_count = ast_descendant_count(ast)
        max_depth = ast_max_depth(ast)

        if node_count > self.max_ast_node_count or max_depth > self.max_ast_depth:
            raise UserTemplateError("Jinja template too complex, simplify your template.")

    def _assert_template_has_no_self_reference(self, ast: Template) -> None:
        """Assert that the template cannot refer to its own settings.

        Templates may attempt to use {{ self }} references to gain
        access to properties of the template object itself. This
        is disallowed.
        """
        if ast_count_name_references(ast, "self") != 0:
            raise UserTemplateError("Non-permitted operations in Jinja template.")

    def validate_template(self, user_template: str) -> None:
        """Template validations are run against the template object itself.
        First-layer injection attacks are (on the parse operation) are
        prevented by using `autoescape=True` on environment creation.

        Afterwards, we can analyze the AST of the parsed template to detect
        and mitigate a wide range of attacks.

        Args:
            user_template (str): A submitted user Jinja2 template.

        Raises:
            TemplateSyntaxError: If the provided template is malformed or
                not parseable as a Jinja2 template.
            UserTemplateError: If any of the assertions fail.
        """
        try:
            ast = self.parse(user_template)
            self._assert_template_has_valid_ast_nodes(ast)
            self._assert_template_has_no_recursive_for(ast)
            self._assert_template_has_no_nested_for(ast)
            self._assert_template_ast_complexity(ast)
            self._assert_template_has_no_self_reference(ast)
            self._assert_template_has_valid_references(ast)
        except UserTemplateError:
            raise
        except Exception as exception:
            maybe_handle_missing_filter_exception(exception, available_jinja_filters=list(self.filters.keys()))
            raise

    def _assert_rendered_text_length(self, rendered_text: str) -> None:
        """Check against the length of the rendered string."""
        rendered_len = len(rendered_text)
        if rendered_len > self.max_rendered_len:
            raise UserTemplateError(f"Rendered Jinja template too large ({rendered_len} > {self.max_rendered_len}).")

    def _assert_rendered_text_has_no_builtin_descriptions(self, rendered_text: str) -> None:
        """Check to make sure that the outputs aren't descriptions of methods.

        In the event that the user types the name of a __builtin__
        object method, but doesn't call it, we don't want to report
        information about the system's memory contents.

        Further, if the user made a mistake, we'd rather error out
        rather than continue task processing, for instance.
        """
        patterns = [
            r"<built-in method (.*?) of (.*?) object at 0x(.*?)>",
            r"<function (.*?) at (.*?)>",
        ]
        for pattern in patterns:
            matches = re.search(pattern, rendered_text)
            if bool(matches):
                raise UserTemplateError("User template has uncalled __builtin__ method.")

    def _assert_rendered_text_not_empty(
        self,
        rendered_text: str,
        user_template: str | None = None,
        record: dict | None = None,
    ) -> None:
        """Check to make sure the resulting text isn't an empty string.

        When ``user_template`` and ``record`` are provided, raises an
        ``EmptyTemplateRenderError`` with a row-level actionable message
        identifying likely culprit fields. Otherwise falls back to a
        plain ``UserTemplateError`` (the legacy behavior).
        """
        if len(rendered_text) != 0:
            return
        if user_template is not None and record is not None:
            raise EmptyTemplateRenderError(_build_empty_render_message(user_template, record, self.parse))
        raise UserTemplateError("User template renders to empty text.")

    def validate_rendered_text(
        self,
        rendered_text: str,
        user_template: str | None = None,
        record: dict | None = None,
    ) -> None:
        """Raises UserTemplateError on invalid renders.

        This is used as a post-processing step for capturing and
        acting on strings before they go out the door. When
        ``user_template`` and ``record`` are provided, empty-render
        failures get a row-level diagnostic message.
        """
        self._assert_rendered_text_not_empty(rendered_text, user_template=user_template, record=record)
        self._assert_rendered_text_length(rendered_text)
        self._assert_rendered_text_has_no_builtin_descriptions(rendered_text)

    def safe_render(
        self,
        user_template: str,
        record: dict,
        skip_template_validation: bool = False,
    ) -> str:
        """Attempt to safely render a user's template.

        Args:
            user_template (str): The user submitted Jinja2 template string.
            record (dict): a record of fields which are able to be referenced by the template.
            skip_template_validation (optional, bool): If true, then AST checks against the
                template itself will not be performed. WARNING: this should ONLY be set to true
                if the template has already been validated.

        Raises:
            UserTemplateError: If the template cannot be rendered because the
                user template does not conform to Gretel's Jinja2 subset,
                is too long, or contains some attempted malicious payload.
                If skip_template_validation is False, this error may also indicate
                that the template itself has failed static analysis. See the error
                message for more details.

            EmptyTemplateRenderError: If the template references a field
                that is missing/None/empty in this row, either rendering
                to empty text or triggering a Jinja ``UndefinedError``.

            RecordContentsError: If there is a system-internal error with
                the supplied record data. This error is raised to prevent Jinja2
                processing of potentially insecure data objects.
        """
        if not skip_template_validation:
            self.validate_template(user_template)

        record = sanitize_record(record)

        try:
            template = self.from_string(user_template)
            rendered_text = template.render(record)
        except SecurityError:
            raise UserTemplateError("Non-permitted operations in Jinja template.")
        except OverflowError:
            raise UserTemplateError("Template too large.")
        except UndefinedError as exception:
            # Raised when a chain like ``{{ a.b.c }}`` hits a missing
            # intermediate. Convert into the actionable EmptyTemplateRenderError
            # so the user sees the culprit field rather than a raw
            # "'dict object' has no attribute 'b'" string.
            raise EmptyTemplateRenderError(
                _build_empty_render_message(user_template, record, self.parse)
            ) from exception
        except Exception as exception:
            maybe_handle_missing_filter_exception(exception, available_jinja_filters=list(self.filters.keys()))
            raise exception

        self.validate_rendered_text(rendered_text, user_template=user_template, record=record)

        return rendered_text

    def render_template(
        self,
        user_template: str,
        record: dict,
        skip_template_validation: bool = False,
    ) -> str:
        return self.safe_render(
            user_template,
            record,
            skip_template_validation=skip_template_validation,
        )

    def get_references(self, user_template: str) -> set[str]:
        """Get all referenced variables from the provided template.

        Args:
            user_template (str): A user-provided Jinja template.

        Returns:
            set[str]: A set of all variable names referenced in
                the supplied Jinja template. If no variables are
                referenced, then this will be an empty list.
        """
        ast = self.parse(user_template)
        return meta.find_undeclared_variables(ast)


class NativeJinjaSandboxEnvironment(ImmutableSandboxedEnvironment):
    """Jinja2's built-in sandbox with Data Designer's reference whitelist."""

    allowed_references: list[str]
    _prefer_dict_key_access: bool

    def __init__(
        self,
        allowed_references: list[str] | None = None,
        prefer_dict_key_access: bool = False,
        **kwargs,
    ):
        super().__init__(autoescape=False, **kwargs)
        self.allowed_references = allowed_references if allowed_references else []
        self._prefer_dict_key_access = prefer_dict_key_access
        self.filters["jsonpath"] = jsonpath_jinja_filter

    def getattr(self, obj: Any, attribute: str) -> Any:
        if self._prefer_dict_key_access and isinstance(obj, dict) and attribute in obj:
            return obj[attribute]
        return super().getattr(obj, attribute)

    def validate_template(self, user_template: str) -> None:
        try:
            ast = self.parse(user_template)
            template_vars = meta.find_undeclared_variables(ast)
            unallowed_vars = set(template_vars) - set(self.allowed_references)
            if len(unallowed_vars) > 0:
                raise UserTemplateError(f"Unknown variable references in Jinja template: {unallowed_vars}")
        except UserTemplateError:
            raise
        except Exception as exception:
            maybe_handle_missing_filter_exception(exception, available_jinja_filters=list(self.filters.keys()))
            raise

    def render_template(
        self,
        user_template: str,
        record: dict,
        skip_template_validation: bool = False,
    ) -> str:
        if not skip_template_validation:
            self.validate_template(user_template)

        try:
            template = self.from_string(user_template)
            return template.render(record)
        except SecurityError as exception:
            raise UserTemplateError("Non-permitted operations in Jinja template.") from exception
        except UndefinedError as exception:
            raise EmptyTemplateRenderError(
                _build_empty_render_message(user_template, record, self.parse)
            ) from exception
        except Exception as exception:
            maybe_handle_missing_filter_exception(exception, available_jinja_filters=list(self.filters.keys()))
            raise UserTemplateError(str(exception)) from exception


def sanitize_user_exceptions(func):
    """Sanitize returned user-space exceptions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (UserTemplateUnsupportedFiltersError, EmptyTemplateRenderError) as exception:
            ## Informative messaging is already handled in these
            ## specific cases.
            ## NOTE: ordering matters -- both of these are subclasses of
            ## UserTemplateError, so they must be caught before the generic
            ## ``UserTemplateError`` clause below.
            raise exception
        except (UserTemplateError, TemplateSyntaxError):
            ## All other details are wrapped in a generic error message
            raise UserTemplateError(USER_PROMPT_TEMPLATE_ERROR_MESSAGE)

    return wrapper


class WithJinja2UserTemplateRendering:
    """Mixin class to support user-supplied Jinja2 rendering.

    Provides `self.render_template(record: dict)` to the receiving
    class, which can be used to safely render user-provided Jinja2
    templates using `UserTemplateSandboxedEnvironment`.

    This mixin also provides error message sanitization for exceptions
    raised by the rendering environment.

    Usage:

        class Foo(WithJinja2UserTemplateRendering):
            def my_func(self, user_template: str, records: list[dict]):

                ## Call once per template -- must be prepared before
                ## being able to call self.render_template
                self.prepare_jinja2_template_renderer(user_template)

                ## Can call many times after
                for record in records:
                    self.render_template(record)
    """

    _template_render_fn: Callable

    def _get_jinja_rendering_engine(self) -> JinjaRenderingEngine:
        engine = getattr(self, "_jinja_rendering_engine", None)
        if engine is not None:
            return JinjaRenderingEngine(engine)

        resource_provider = getattr(self, "_resource_provider", None)
        if resource_provider is not None:
            return JinjaRenderingEngine(resource_provider.run_config.jinja_rendering_engine)

        # The mixin predates the RunConfig toggle, so preserve the historical
        # secure-by-default behavior when no explicit engine is wired in.
        return JinjaRenderingEngine.SECURE

    def _create_render_environment(
        self,
        *,
        dataset_variables: list[str],
        record_str_fn: Callable[[Any], str] | None = None,
    ) -> UserTemplateSandboxEnvironment | NativeJinjaSandboxEnvironment:
        env_kwargs: dict[str, Any] = {}
        if record_str_fn is not None:
            env_kwargs["finalize"] = record_str_fn
            env_kwargs["prefer_dict_key_access"] = True
        if self._get_jinja_rendering_engine() == JinjaRenderingEngine.SECURE:
            return UserTemplateSandboxEnvironment(allowed_references=dataset_variables, **env_kwargs)
        return NativeJinjaSandboxEnvironment(allowed_references=dataset_variables, **env_kwargs)

    @sanitize_user_exceptions
    def prepare_jinja2_template_renderer(
        self,
        prompt_template: str,
        dataset_variables: list[str],
        record_str_fn: Callable[[Any], str] | None = None,
    ) -> None:
        """Build Jinja2 template render function.

        Args:
            prompt_template: A user-provided Jinja2 template string.
            dataset_variables: Column names allowed as template references.
            record_str_fn: When set, the environment uses Jinja2's finalize hook
                to apply this callable to every interpolated value at render time,
                and enables dict-key-priority attribute lookup for nested dot access
                ({{ col.sub.field }}).
        """
        jinja_render_env = self._create_render_environment(
            dataset_variables=dataset_variables,
            record_str_fn=record_str_fn,
        )
        jinja_render_env.validate_template(prompt_template)
        self._template_render_fn = partial(
            jinja_render_env.render_template,
            prompt_template,
            skip_template_validation=True,
        )

    @sanitize_user_exceptions
    def render_template(self, record: dict) -> str:
        return self._template_render_fn(record)

    @sanitize_user_exceptions
    def prepare_jinja2_multi_template_renderer(
        self,
        template_name: str,
        prompt_template: str,
        dataset_variables: list[str],
    ) -> None:
        if not self._template_prepared_in_multi_template_renderer(template_name):
            self._create_render_func_registry()
            jinja_render_env = self._create_render_environment(dataset_variables=dataset_variables)
            jinja_render_env.validate_template(prompt_template)
            self._render_func_registry[template_name] = partial(
                jinja_render_env.render_template,
                prompt_template,
                skip_template_validation=True,
            )

    @sanitize_user_exceptions
    def render_multi_template(self, template_name: str, record: dict) -> str:
        if not hasattr(self, "_render_func_registry"):
            raise UserTemplateError("Multi-template renderer not prepared.")
        if template_name not in self._render_func_registry:
            raise UserTemplateError(f"Template {template_name} not prepared.")
        return self._render_func_registry[template_name](record)

    def _template_prepared_in_multi_template_renderer(self, template_name: str) -> bool:
        if not hasattr(self, "_render_func_registry"):
            return False
        return template_name in self._render_func_registry

    def _create_render_func_registry(self) -> None:
        if not hasattr(self, "_render_func_registry"):
            self._render_func_registry = {}


# Sentinel describing why a chain showed up as a likely culprit. The values
# are user-facing labels that get embedded in the error message.
_CULPRIT_MISSING = "missing from record"
_CULPRIT_NONE = "resolved to None"
_CULPRIT_EMPTY_STRING = "resolved to empty string"
_CULPRIT_EMPTY_COLLECTION = "resolved to empty collection"


def _format_access_chain(name: str, accessors: list[str | int]) -> str:
    """Render an access chain in user-friendly dotted/bracket notation."""
    parts: list[str] = [name]
    for acc in accessors:
        if isinstance(acc, str) and acc.isidentifier():
            parts.append(f".{acc}")
        elif isinstance(acc, str):
            parts.append(f"[{acc!r}]")
        else:
            parts.append(f"[{acc}]")
    return "".join(parts)


def _classify_chain(record: dict, name: str, accessors: list[str | int]) -> tuple[str, list[str | int]] | None:
    """Return ``(reason, prefix)`` if a chain is a likely empty-render culprit.

    The returned ``prefix`` is the longest accessor list that did resolve,
    so callers can suggest fallback patterns that gate on it.
    """
    resolved, value, prefix = resolve_access_chain(record, name, accessors)
    if not resolved:
        return (_CULPRIT_MISSING, prefix)
    if value is None:
        return (_CULPRIT_NONE, prefix)
    if isinstance(value, str) and value == "":
        return (_CULPRIT_EMPTY_STRING, prefix)
    # Empty collections (lists/dicts/tuples/sets) render to "[]"/"{}"/etc.
    # via Jinja2 by default, but they're a common source of empty output when
    # used as a loop iterable, e.g. ``{% for x in items %}...{% endfor %}``
    # with ``items=[]``. Surface them as culprits so the user sees which
    # field to gate on.
    if isinstance(value, (list, dict, tuple, set, frozenset)) and len(value) == 0:
        return (_CULPRIT_EMPTY_COLLECTION, prefix)
    return None


def _collect_culprit_chains(
    chains: list[AccessChain], record: dict
) -> list[tuple[str, list[str | int], str, list[str | int]]]:
    """Identify chains in ``chains`` that resolve to empty-ish values in ``record``.

    Returns a list of ``(name, accessors, reason, resolvable_prefix)`` tuples,
    deduped on the chain identity to avoid repeating the same culprit when a
    template references it multiple times.
    """
    seen: set[tuple[str, tuple[str | int, ...]]] = set()
    culprits: list[tuple[str, list[str | int], str, list[str | int]]] = []
    for name, accessors in chains:
        key = (name, tuple(accessors))
        if key in seen:
            continue
        seen.add(key)
        classification = _classify_chain(record, name, accessors)
        if classification is None:
            continue
        reason, prefix = classification
        culprits.append((name, accessors, reason, prefix))
    return culprits


def _build_empty_render_message(user_template: str, record: dict, parser: Callable[[str], j_nodes.Template]) -> str:
    """Compose an actionable error message for empty/undefined-access renders.

    ``parser`` is the environment's own ``parse`` method; we accept it as a
    parameter so this helper stays decoupled from any specific environment.
    """
    header = (
        "Template rendered to empty text. This usually happens when one or "
        "more referenced fields are missing, None, or empty in this row."
    )
    culprits: list[tuple[str, list[str | int], str, list[str | int]]] = []
    try:
        ast = parser(user_template)
        chains = ast_extract_access_chains(ast)
        # Filter out chains rooted in Jinja-scoped names (e.g. ``person`` in
        # ``{% for person in people %}{{ person.name }}{% endfor %}``). The
        # canonical way to identify true external references is
        # ``meta.find_undeclared_variables``, which already understands loop
        # targets and other scoping constructs.
        undeclared = meta.find_undeclared_variables(ast)
        chains = [(name, accessors) for name, accessors in chains if name in undeclared]
        culprits = _collect_culprit_chains(chains, record)
    except Exception:
        # If anything in the culprit-finding path fails, fall back to a
        # message without the per-row diagnostic. We never want this helper
        # to mask the original render failure.
        culprits = []

    if not culprits:
        return (
            f"{header}\n"
            "\nTo handle missing values, provide a fallback in your template "
            "using a Jinja conditional, e.g. "
            "`{{ field if field else 'N/A' }}`, or gate generation with a "
            "SkipConfig."
        )

    culprit_lines = []
    for name, accessors, reason, _prefix in culprits:
        culprit_lines.append(f"  - {_format_access_chain(name, accessors)} ({reason})")
    culprit_block = "\n".join(culprit_lines)

    sample_name, sample_accessors, _sample_reason, sample_prefix = culprits[0]
    full_chain = _format_access_chain(sample_name, sample_accessors)
    # Pick a "gate" expression that is safe to evaluate in Jinja. Going one
    # accessor past the resolvable prefix gives us the first Undefined value,
    # which Jinja happily stringifies as "" and treats as falsy. Going any
    # further would attempt another lookup on Undefined and re-raise.
    # For chains that fully resolved (None/empty leaf), prefix == accessors,
    # so the slice collapses to the full chain.
    #
    # Special case: when the root variable itself is absent from the record,
    # ``resolve_access_chain`` returns ``prefix=[]``. Slicing one past that
    # would yield ``person.address`` for a missing ``person`` root, but
    # Jinja's ``Undefined.__getattr__`` raises ``UndefinedError`` -- the very
    # error we're trying to help the user fix. Fall back to gating on the
    # root name alone, which Jinja's ``Undefined`` happily treats as falsy.
    if sample_name not in record:
        gate_accessors: list[str | int] = []
    else:
        gate_accessors = sample_accessors[: len(sample_prefix) + 1]
    gate_chain = _format_access_chain(sample_name, gate_accessors)

    return (
        f"{header}\n"
        "\nLikely culprits in this row:\n"
        f"{culprit_block}\n"
        "\nTo handle missing values, you can:\n"
        "\n  1. Provide a fallback in your template using a Jinja conditional:\n"
        f"       {{{{ {full_chain} if {gate_chain} else 'N/A' }}}}\n"
        "\n  2. Skip rows where required fields are missing using SkipConfig:\n"
        f'       skip=SkipConfig(when="{{{{ not {gate_chain} }}}}")'
    )
