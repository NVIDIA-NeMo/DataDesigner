# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.processing.ginja.environment import (
    ALLOWED_JINJA_FILTERS,
    UserTemplateSandboxEnvironment,
    WithJinja2UserTemplateRendering,
    is_jinja_template,
    jsonpath_jinja_filter,
)
from data_designer.engine.processing.ginja.exceptions import UserTemplateError, UserTemplateUnsupportedFiltersError

SECURITY_EXCEPTIONS = [
    "{{ self.__init__ }}",
    "{{ self._TemplateReference__context.cycler.__init__.__globals__.os }}",
    "{{ self._TemplateReference__context.joiner.__init__.__globals__.os }}",
    "{{ self._TemplateReference__context.namespace.__init__.__globals__.os }}",
    "{{ field_c.__class__.__mro__ }}",
    "{{ field_c.__class__.__init__.__globals__ }}",
    "{{ field_c.__init__.__globals__ }}",
    "{{ field_c.sub_a.foo.__class__.__mro__ }}",
    "{{ ''.__class__.__mro__ }}",
    "{% for var in range(100000000) %}\n{var}\n{% endfor %}{{ config.items() }}{{ __class__ }}",
]

OOB_EXCEPTIONS = [
    "{{ field_z }}",
    "{{ record }}",
    "{{ self }}",
    "{{ field_a }} {{ self }}",
]

UNSUPPORTED_FEATURES = [
    "{% for i in range(2) %}{% for j in range(2) %}hi{% endfor %}{% endfor %}",
    "{% for i in range(2) %}{% if diversion %}{% for j in range(2) %}hi{% endfor %}{% endif %}{% endfor %}",
    "{% macro foo() %}...{% endmacro %}",
    "{% set x = foo %}",
    "{% block foo %}...{% endblock %}",
    "{% for i in foobar recursive %}...{% endfor %}",
]

TEST_RECORD = {
    "field_a": 1,
    "field_y": "foo",
    "field_b": [1, 2, 3],
    "field_c": {"sub_a": {"foo": [1, 2, 3]}},
    "field_d": [
        {"type": "foo", "name": "house"},
        {"type": "foo", "name": "cat"},
        {"type": "bar", "name": "outside"},
        {"type": "bar", "name": "dog"},
    ],
}


@pytest.fixture
def stub_sandbox_env():
    return UserTemplateSandboxEnvironment(allowed_references=list(TEST_RECORD.keys()))


@pytest.mark.parametrize("user_template", SECURITY_EXCEPTIONS + OOB_EXCEPTIONS + UNSUPPORTED_FEATURES)
def test_user_template_sandbox_environment_exceptions(stub_sandbox_env, user_template):
    with pytest.raises(UserTemplateError):
        stub_sandbox_env.safe_render(user_template, TEST_RECORD)


def test_user_template_sandbox_environment_filters():
    env = UserTemplateSandboxEnvironment()
    assert "eval" not in env.filters
    assert all(name in ALLOWED_JINJA_FILTERS for name in env.filters.keys())


@pytest.mark.parametrize(
    "template_string,expected_result",
    [
        ("This is an {fstring} template.", False),
        ("This is an {fstring} template, This is how I escape braces {{", False),
        ("This is an {fstring} template, This is how I escape braces }}", False),
        ("This is a {{ jinja }} template.{% for i in range(10) %} This is a jinja template. {% endfor %}", True),
    ],
)
def test_is_jinja_template(template_string, expected_result):
    assert is_jinja_template(template_string) == expected_result


@pytest.mark.parametrize(
    "jsonpath_query,expected_result",
    [
        ("$.field_b[:2]", [1, 2]),
        ("$.field_d[?(@.type=='bar')].name", ["outside", "dog"]),
    ],
)
def test_jsonpath_jinja_filter(jsonpath_query, expected_result):
    assert jsonpath_jinja_filter(TEST_RECORD, jsonpath_query) == expected_result


@pytest.mark.parametrize(
    "jinja_template,expected_result",
    [
        ("No placeholders here", "No placeholders here"),
        ("{{ field_a }}...{{ field_b }}", "1...[1, 2, 3]"),
        ("{% for item in field_b %}{{ item }}{% endfor %}", "123"),
    ],
)
def test_safe_render(stub_sandbox_env, jinja_template, expected_result):
    assert stub_sandbox_env.safe_render(jinja_template, TEST_RECORD) == expected_result

    # Test depth restriction
    restricted_env = UserTemplateSandboxEnvironment(
        allowed_references=list(TEST_RECORD.keys()),
        max_ast_depth=0,
        max_ast_node_count=1_000,
    )
    with pytest.raises(UserTemplateError, match=r"complex"):
        restricted_env.validate_template(jinja_template)

    # Test node count restriction
    node_count_restricted_env = UserTemplateSandboxEnvironment(
        allowed_references=list(TEST_RECORD.keys()),
        max_ast_depth=1_000,
        max_ast_node_count=0,
    )
    with pytest.raises(UserTemplateError, match=r"complex"):
        node_count_restricted_env.validate_template(jinja_template)


def test_safe_render_with_uncalled_methods(stub_sandbox_env):
    """If a user doesn't call a method, should raise a UserTemplateError"""

    def all_nonprivate_method_templates(var, var_name):
        return [
            f"{{{{ {var_name}.{name} }}}}"
            for name in dir(var)
            if not name.startswith("_") and callable(getattr(var, name))
        ]

    for key, value in TEST_RECORD.items():
        for template in all_nonprivate_method_templates(value, key):
            with pytest.raises(UserTemplateError):
                stub_sandbox_env.safe_render(template, TEST_RECORD)


@pytest.mark.parametrize(
    "test_case,template_1,template_2,expected_result",
    [
        ("valid_single", "Safe template {{ safe }}", None, ["Safe template 42", "Safe template 42"]),
        ("invalid_single", "Safe template {{ notsafe }}", None, UserTemplateError),
        (
            "complex_single",
            "{% for i in range(10) %}{% for j in range(10) %}Safe template {{ safe }}{% endfor %}{% endfor %}",
            None,
            UserTemplateError,
        ),
        ("unsupported_filter_single", "I am a template {{ foo | asdf }}", None, UserTemplateUnsupportedFiltersError),
        (
            "valid_multi",
            "Safe template {{ safe }}",
            "Super safe template {{ safe }}",
            ["Safe template 42", "Super safe template 42"],
        ),
        ("invalid_multi", "Safe template {{ notsafe }}", "Super safe template {{ notsafe }}", UserTemplateError),
        (
            "complex_multi",
            "{% for i in range(10) %}{% for j in range(10) %}Safe template {{ safe }}{% endfor %}{% endfor %}",
            "{% for i in range(10) %}{% for j in range(10) %}Super safe template {{ safe }}{% endfor %}{% endfor %}",
            UserTemplateError,
        ),
        (
            "unsupported_filter_multi",
            "I am template 1 {{ foo | asdf }}",
            "I am template 2 {{ foo | asdf }}",
            UserTemplateUnsupportedFiltersError,
        ),
    ],
)
def test_with_jinja2_user_template_rendering_mixin(test_case, template_1, template_2, expected_result):
    n = 2

    class Foo(WithJinja2UserTemplateRendering):
        def __init__(self, template_1: str, template_2: str = None):
            if template_2 is None:
                # Single template
                self.prepare_jinja2_template_renderer(template_1, dataset_variables=["safe"])
            else:
                # Multi template
                self.prepare_jinja2_multi_template_renderer(
                    template_name="template_1",
                    prompt_template=template_1,
                    dataset_variables=["safe"],
                )
                self.prepare_jinja2_multi_template_renderer(
                    template_name="template_2",
                    prompt_template=template_2,
                    dataset_variables=["safe"],
                )

        def bar(self, record):
            if template_2 is None:
                return [self.render_template(record) for _ in range(n)]
            else:
                return [
                    self.render_multi_template("template_1", record),
                    self.render_multi_template("template_2", record),
                ]

    if test_case.startswith("valid"):
        f = Foo(template_1, template_2)
        assert f.bar({"safe": 42}) == expected_result
    else:
        with pytest.raises(expected_result):
            f = Foo(template_1, template_2)
