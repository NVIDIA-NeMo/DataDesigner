# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial, wraps
from typing import Any

from jinja2 import meta
from jinja2 import nodes as j_nodes
from jinja2.exceptions import SecurityError, TemplateSyntaxError
from jinja2.nodes import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jsonpath_rust_bindings import Finder

from data_designer.engine.processing.ginja.ast import (
    ast_count_name_references,
    ast_descendant_count,
    ast_max_depth,
)
from data_designer.engine.processing.ginja.exceptions import (
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

    def __init__(
        self,
        allowed_references: list[str] | None = None,
        max_rendered_len: int = MAX_RENDERED_LEN,
        max_ast_node_count: int = MAX_AST_NODE_COUNT,
        max_ast_depth: int = MAX_AST_DEPTH,
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

        ## Add on our supported filters
        self.filters["jsonpath"] = jsonpath_jinja_filter

        ## Cut out all but approved Jinja filters
        self.filters = {k: v for k, v in self.filters.items() if k in ALLOWED_JINJA_FILTERS}

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
        except Exception as exception:
            maybe_handle_missing_filter_exception(exception, available_jinja_filters=list(self.filters.keys()))
            raise exception

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

    def _assert_rendered_text_not_empty(self, rendered_text: str) -> None:
        """Check to make sure the resulting text isn't an empty string"""
        if len(rendered_text) == 0:
            raise UserTemplateError("User template renders to empty text.")

    def validate_rendered_text(self, rendered_text: str) -> None:
        """Raises UserTemplateError on invalid renders.

        This is used as a post-processing step for capturing and
        acting on strings before they go out the door.
        """
        self._assert_rendered_text_not_empty(rendered_text)
        self._assert_rendered_text_length(rendered_text)
        self._assert_rendered_text_has_no_builtin_descriptions(rendered_text)

    def safe_render(self, user_template: str, record: dict, skip_template_validation: bool = False) -> str:
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
        except Exception as exception:
            maybe_handle_missing_filter_exception(exception, available_jinja_filters=list(self.filters.keys()))
            raise exception

        self.validate_rendered_text(rendered_text)

        return rendered_text

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


def sanitize_user_exceptions(func):
    """Sanitize returned user-space exceptions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UserTemplateUnsupportedFiltersError as exception:
            ## Informative messaging is already handled in this
            ## specific case.
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

    @sanitize_user_exceptions
    def prepare_jinja2_template_renderer(self, prompt_template: str, dataset_variables: list[str]) -> None:
        """Build Jinja2 template render function."""
        jinja_render_env = UserTemplateSandboxEnvironment(allowed_references=dataset_variables)
        jinja_render_env.validate_template(prompt_template)
        self._template_render_fn = partial(
            jinja_render_env.safe_render,
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
            jinja_render_env = UserTemplateSandboxEnvironment(allowed_references=dataset_variables)
            jinja_render_env.validate_template(prompt_template)
            self._render_func_registry[template_name] = partial(
                jinja_render_env.safe_render,
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
