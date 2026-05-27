# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.run_config import JinjaRenderingEngine
from data_designer.engine.processing.ginja.environment import (
    ALLOWED_JINJA_FILTERS,
    NativeJinjaSandboxEnvironment,
    UserTemplateSandboxEnvironment,
    WithJinja2UserTemplateRendering,
    is_jinja_template,
    jsonpath_jinja_filter,
)
from data_designer.engine.processing.ginja.exceptions import (
    EmptyTemplateRenderError,
    UserTemplateError,
    UserTemplateUnsupportedFiltersError,
)

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


def test_native_jinja_sandbox_environment_supports_jsonpath_filter() -> None:
    env = NativeJinjaSandboxEnvironment(allowed_references=list(TEST_RECORD.keys()))

    assert env.render_template('{{ field_c | jsonpath("$.sub_a.foo[:2]") }}', TEST_RECORD) == str([1, 2])


def test_user_template_sandbox_environment_supports_upper_filter(stub_sandbox_env) -> None:
    assert stub_sandbox_env.safe_render("{{ field_y | upper }}", TEST_RECORD) == "FOO"


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
            self._jinja_rendering_engine = JinjaRenderingEngine.SECURE
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


def test_with_jinja2_user_template_rendering_defaults_to_secure_mode() -> None:
    class Foo(WithJinja2UserTemplateRendering):
        pass

    renderer = Foo()

    with pytest.raises(UserTemplateUnsupportedFiltersError):
        renderer.prepare_jinja2_template_renderer("{{ items | join('-') }}", dataset_variables=["items"])


# Regression tests for https://github.com/NVIDIA-NeMo/DataDesigner/issues/629
# (Empty/undefined-access render failures used to surface a generic, unactionable
# "User provided prompt generation template is invalid." message.)


def _make_secure_renderer(template: str, dataset_variables: list[str]) -> WithJinja2UserTemplateRendering:
    class Demo(WithJinja2UserTemplateRendering):
        def __init__(self):
            self._jinja_rendering_engine = JinjaRenderingEngine.SECURE

    renderer = Demo()
    renderer.prepare_jinja2_template_renderer(template, dataset_variables=dataset_variables)
    return renderer


def _make_native_renderer(template: str, dataset_variables: list[str]) -> WithJinja2UserTemplateRendering:
    class Demo(WithJinja2UserTemplateRendering):
        def __init__(self):
            self._jinja_rendering_engine = JinjaRenderingEngine.NATIVE

    renderer = Demo()
    renderer.prepare_jinja2_template_renderer(template, dataset_variables=dataset_variables)
    return renderer


def test_empty_render_raises_empty_template_render_error_with_culprit_chain():
    # Bug 1: ``{{ x }}`` renders to empty when x is missing from the record.
    renderer = _make_secure_renderer("{{ person.preferred_english_name }}", dataset_variables=["person"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"person": {"first_name": "John", "last_name": "Doe"}})

    msg = str(exc_info.value)
    # Names the offending chain so the user can find it in their data.
    assert "person.preferred_english_name" in msg
    assert "missing from record" in msg
    # Includes both remediation suggestions verbatim enough that copy-paste works.
    assert "{{ person.preferred_english_name if person.preferred_english_name else 'N/A' }}" in msg
    assert 'skip=SkipConfig(when="{{ not person.preferred_english_name }}")' in msg


def test_undefined_nested_attr_raises_empty_template_render_error_with_safe_gate():
    # Bug 3: nested missing-attr lookups used to leak raw Jinja UndefinedError.
    renderer = _make_secure_renderer("Hi {{ person.address.street }}", dataset_variables=["person"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"person": {}})

    msg = str(exc_info.value)
    assert "person.address.street" in msg
    assert "missing from record" in msg
    # The "gate" expression in the suggestions stops one step short of the
    # broken accessor so it stays safe to evaluate in Jinja.
    assert "{{ person.address.street if person.address else 'N/A' }}" in msg
    assert 'skip=SkipConfig(when="{{ not person.address }}")' in msg


def test_empty_render_message_reports_resolved_to_none():
    # When a finalize callable converts None -> "", the chain resolves but the
    # render is still empty; the diagnostic should call out the None value.
    class Demo(WithJinja2UserTemplateRendering):
        def __init__(self):
            self._jinja_rendering_engine = JinjaRenderingEngine.SECURE

    demo = Demo()
    demo.prepare_jinja2_template_renderer(
        "{{ x }}",
        dataset_variables=["x"],
        record_str_fn=lambda v: "" if v is None else str(v),
    )

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        demo.render_template({"x": None})

    msg = str(exc_info.value)
    assert "x (resolved to None)" in msg


def test_empty_render_message_reports_resolved_to_empty_string():
    renderer = _make_secure_renderer("{{ x }}", dataset_variables=["x"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"x": ""})

    assert "x (resolved to empty string)" in str(exc_info.value)


def test_empty_render_message_lists_all_culprits():
    renderer = _make_secure_renderer(
        "{{ a.b }}{{ c.d.e }}{{ z }}",
        dataset_variables=["a", "c", "z"],
    )

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"a": {}, "c": {}, "z": ""})

    msg = str(exc_info.value)
    assert "a.b" in msg
    assert "c.d.e" in msg
    assert "- z (resolved to empty string)" in msg


def test_empty_template_render_error_bypasses_generic_sanitizer():
    # ``sanitize_user_exceptions`` wraps every UserTemplateError into the same
    # generic message; EmptyTemplateRenderError must escape that wrapper so the
    # actionable detail survives end-to-end.
    renderer = _make_secure_renderer("{{ x }}", dataset_variables=["x"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"x": ""})

    # The generic sanitized text is precisely what we DO NOT want here.
    assert "User provided prompt generation template is invalid." not in str(exc_info.value)


def test_native_engine_converts_undefined_error_to_empty_template_render_error():
    renderer = _make_native_renderer("Hi {{ person.address.street }}", dataset_variables=["person"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"person": {}})

    msg = str(exc_info.value)
    assert "person.address.street" in msg


def test_happy_path_still_renders():
    renderer = _make_secure_renderer("{{ person.first_name }}", dataset_variables=["person"])
    assert renderer.render_template({"person": {"first_name": "John"}}) == "John"


def test_safe_render_with_skip_template_validation_still_attaches_diagnostic():
    # safe_render is called via the prepared renderer with skip_template_validation=True;
    # the error path needs to reparse the template for AST walking, which is what we
    # exercise here.
    renderer = _make_secure_renderer("{{ person.foo }}", dataset_variables=["person"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"person": {"bar": 1}})

    assert "person.foo" in str(exc_info.value)


def test_undefined_root_variable_produces_safe_remediation_template():
    # Regression for the Greptile P1 review on PR #633: when the root variable
    # is entirely absent from the record, the gate expression used to be one
    # accessor too deep (e.g. ``person.address`` for a missing ``person``),
    # which itself raised ``UndefinedError`` in Jinja. The fix falls back to
    # gating on just the root name.
    renderer = _make_secure_renderer("Hi {{ person.address.street }}", dataset_variables=["person"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({})

    msg = str(exc_info.value)
    assert "person.address.street" in msg
    assert "missing from record" in msg
    # Gate on the root name only, not a nested attribute -- the latter would
    # be unsafe to evaluate when ``person`` is Undefined.
    assert "{{ person.address.street if person else 'N/A' }}" in msg
    assert 'skip=SkipConfig(when="{{ not person }}")' in msg
    # Make sure the suggested gate does NOT include the broken attribute.
    assert "if person.address" not in msg
    assert "not person.address" not in msg


def test_undefined_root_variable_remediation_template_is_renderable():
    # The remediation suggestion must itself be safe to render against the
    # same broken record. Previously the suggested template re-raised
    # UndefinedError, defeating the purpose of the diagnostic.
    renderer = _make_secure_renderer("Hi {{ person.address.street }}", dataset_variables=["person"])

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({})

    msg = str(exc_info.value)
    # Pull the suggested template out of the message and render it ourselves.
    suggestion = "{{ person.address.street if person else 'N/A' }}"
    assert suggestion in msg

    rerender = _make_secure_renderer(suggestion, dataset_variables=["person"])
    assert rerender.render_template({}) == "N/A"


def test_loop_variable_is_not_reported_as_missing_culprit():
    # Regression for the andreatgretel review on PR #633: the AST walker
    # previously reported loop-local names (e.g. ``person`` in
    # ``{% for person in people %}...{% endfor %}``) as missing from the
    # record. The chain extractor should now defer to
    # ``meta.find_undeclared_variables`` for scoping.
    renderer = _make_secure_renderer(
        "{% for person in people %}{{ person.name }}{% endfor %}",
        dataset_variables=["people"],
    )

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"people": []})

    msg = str(exc_info.value)
    # ``person`` is a loop-local variable; only ``people`` is a real culprit.
    assert "people" in msg
    assert "- person " not in msg
    assert "person.name" not in msg


def test_empty_collection_iterable_reported_as_culprit():
    # Regression for the andreatgretel follow-up on PR #633: ``items=[]``
    # used to fall through to the no-culprit fallback message because the
    # classifier only checked for None / empty-string leaves. Empty
    # collections are now surfaced explicitly.
    renderer = _make_secure_renderer(
        "{% for item in items %}{{ item }}{% endfor %}",
        dataset_variables=["items"],
    )

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"items": []})

    msg = str(exc_info.value)
    assert "items (resolved to empty collection)" in msg


def test_empty_dict_iterable_reported_as_culprit():
    renderer = _make_secure_renderer(
        "{% for k in data %}{{ k }}{% endfor %}",
        dataset_variables=["data"],
    )

    with pytest.raises(EmptyTemplateRenderError) as exc_info:
        renderer.render_template({"data": {}})

    msg = str(exc_info.value)
    assert "data (resolved to empty collection)" in msg
