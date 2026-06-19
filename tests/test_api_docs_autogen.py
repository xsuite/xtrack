import inspect

import pytest

import xtrack as xt
from xtrack.environment import (
    ENVIRONMENT_DOC_GROUP_ORDER,
    EnvElements,
    EnvLines,
    EnvParticleRef,
    EnvParticles,
    EnvRef,
    EnvVars,
    EnvXfields,
    VarsTable,
)
from xtrack.survey import SurveyTable
from xtrack.line import LINE_DOC_GROUP_ORDER, LineTable


def _iter_public_members(cls):
    for name, member in cls.__dict__.items():
        if name.startswith("_"):
            continue

        if isinstance(member, property):
            yield "property", name, member, member.fget
            continue

        if isinstance(member, (classmethod, staticmethod)):
            func = member.__func__
            yield "method", name, member, func
            continue

        if callable(member):
            yield "method", name, member, member


@pytest.mark.parametrize(
    "cls",
    [
        xt.Line,
        xt.Environment,
        xt.Table,
        xt.TwissTable,
        SurveyTable,
        LineTable,
        EnvVars,
        EnvElements,
        EnvParticles,
        EnvLines,
        EnvRef,
        EnvParticleRef,
        EnvXfields,
        VarsTable,
    ],
    ids=[
        "Line",
        "Environment",
        "Table",
        "TwissTable",
        "SurveyTable",
        "LineTable",
        "EnvVars",
        "EnvElements",
        "EnvParticles",
        "EnvLines",
        "EnvRef",
        "EnvParticleRef",
        "EnvXfields",
        "VarsTable",
    ],
)
def test_public_api_members_have_docstrings(cls):
    missing_methods = []
    missing_properties = []

    for kind, name, _member, doc_obj in _iter_public_members(cls):
        if inspect.getdoc(doc_obj):
            continue
        if kind == "method":
            missing_methods.append(name)
        else:
            missing_properties.append(name)

    errors = []
    if missing_methods:
        errors.append(
            f"{cls.__name__} methods missing docstring: {sorted(missing_methods)}"
        )
    if missing_properties:
        errors.append(
            f"{cls.__name__} properties missing docstring: {sorted(missing_properties)}"
        )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "cls,valid_groups",
    [
        (xt.Line, set(LINE_DOC_GROUP_ORDER)),
        (xt.Environment, set(ENVIRONMENT_DOC_GROUP_ORDER)),
    ],
    ids=["Line", "Environment"],
)
def test_public_api_members_have_valid_doc_groups(cls, valid_groups):
    missing_group = []
    invalid_group = []

    for _kind, name, member, group_obj in _iter_public_members(cls):
        if isinstance(member, property):
            group = getattr(group_obj, "__doc_group__", None)
        elif isinstance(member, (classmethod, staticmethod)):
            group = getattr(member, "__doc_group__", None)
            if group is None:
                group = getattr(group_obj, "__doc_group__", None)
        else:
            group = getattr(member, "__doc_group__", None)

        if group is None:
            missing_group.append(name)
        elif group not in valid_groups:
            invalid_group.append((name, group))

    ungrouped = sorted(getattr(cls, "__doc_groups_ungrouped__", []))
    if ungrouped:
        missing_group.extend(ungrouped)
        missing_group = sorted(set(missing_group))

    errors = []
    if missing_group:
        errors.append(
            f"{cls.__name__} members missing doc group: {sorted(missing_group)}"
        )
    if invalid_group:
        errors.append(
            f"{cls.__name__} members with invalid doc group: {sorted(invalid_group)}"
        )
    assert not errors, "\n".join(errors)


def test_generate_doc_rst_methods_run_without_errors():
    line_rst = xt.Line._generate_doc_rst()
    env_rst = xt.Environment._generate_doc_rst()

    assert isinstance(line_rst, str)
    assert isinstance(env_rst, str)
    assert line_rst.strip()
    assert env_rst.strip()
