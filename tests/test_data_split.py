"""The visible and held out split, which the whole measurement rests on.

Marked slow where a test needs the datasets, so CI proves the instrument on
CPU without a network fetch, per the workflow in .github/workflows/ci.yml.
"""

from __future__ import annotations

import ast

import pytest

from alibi.env.tests import (
    HarnessError,
    assert_call_args,
    assert_expected,
    entry_point,
    harness_inputs,
    held_out_harness,
    held_out_indices,
    normalise,
    visible_harness,
)

# A miniature harness in each of the two shapes EvalPlus actually emits.
RESULTS_SHAPE = """
def assertion(out, exp, atol):
    assert out == exp

inputs = [[1], [2], [3]]
results = [2, 4, 6]
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion(double(*inp), exp, 0)
"""

REF_FUNC_SHAPE = """
def assertion(out, exp, atol):
    assert out == exp

def ref_func(n):
    return n * 2

inputs = [[1], [2], [3]]
for i, inp in enumerate(inputs):
    assertion(double(*inp), ref_func(*inp), 0)
"""

VISIBLE = ["assert double(1) == 2"]


@pytest.mark.parametrize("harness", [RESULTS_SHAPE, REF_FUNC_SHAPE])
def test_entry_point_is_found_in_both_harness_shapes(harness: str) -> None:
    assert entry_point(harness) == "double"


@pytest.mark.parametrize("harness", [RESULTS_SHAPE, REF_FUNC_SHAPE])
def test_held_out_excludes_the_visible_input(harness: str) -> None:
    assert held_out_indices(harness, VISIBLE) == [1, 2]


@pytest.mark.parametrize("harness", [RESULTS_SHAPE, REF_FUNC_SHAPE])
def test_held_out_harness_scores_per_input_not_all_or_nothing(harness: str) -> None:
    """The original loop fails fast. The generated one must not."""
    source = held_out_harness(harness, [1, 2])
    namespace: dict = {"double": lambda n: 4 if n == 2 else 0}
    exec(compile(source, "<test>", "exec"), namespace)  # noqa: S102
    assert namespace["__alibi_outcomes__"] == ["pass", "fail"]


def test_held_out_harness_honours_the_index_filter() -> None:
    source = held_out_harness(RESULTS_SHAPE, [2])
    namespace: dict = {"double": lambda n: n * 2}
    exec(compile(source, "<test>", "exec"), namespace)  # noqa: S102
    assert namespace["__alibi_outcomes__"] == ["pass"]


def test_visible_harness_scores_each_assert_separately() -> None:
    source = visible_harness(["assert double(1) == 2", "assert double(2) == 5"])
    namespace: dict = {"double": lambda n: n * 2}
    exec(compile(source, "<test>", "exec"), namespace)  # noqa: S102
    assert namespace["__alibi_outcomes__"] == ["pass", "fail"]


def test_a_harness_of_an_unexpected_shape_is_refused_not_guessed() -> None:
    with pytest.raises(HarnessError):
        entry_point("inputs = [1]\nx = 2\n")


def test_assert_call_args_refuses_to_guess() -> None:
    assert assert_call_args("assert double(n) == 2", "double") is None
    assert assert_call_args("assert double(1) == 2", "double") == [1]


def test_assert_expected_distinguishes_absent_from_none() -> None:
    assert assert_expected("assert double(1) == None", "double") == (True, None)
    assert assert_expected("assert set(double(1)) == set([2])", "double") == (False, None)


def test_normalise_treats_a_tuple_and_a_list_as_the_same_input() -> None:
    """The safe direction: over matching drops a held out input, under matching leaks one."""
    assert normalise([1, 2]) == normalise((1, 2))
    assert normalise(True) != normalise(1)


@pytest.mark.slow
def test_the_built_split_is_disjoint_for_every_problem() -> None:
    """The property the primary metric depends on, checked on the real data."""
    from alibi.data.build import build

    for problem in build().problems:
        inputs = harness_inputs(problem.plus_test_src)
        visible = {
            repr(normalise(assert_call_args(src, problem.entry_point))) for src in problem.visible_asserts
        }
        for index in problem.held_out_indices:
            assert repr(normalise(list(inputs[index]))) not in visible, (
                f"task {problem.task_id} held out index {index} is also a visible input"
            )
        assert problem.held_out_indices, f"task {problem.task_id} has no held out tests"


@pytest.mark.slow
def test_every_problem_has_a_parseable_entry_point_and_asserts() -> None:
    from alibi.data.build import build

    for problem in build().problems:
        assert problem.entry_point
        assert problem.visible_asserts
        for src in problem.visible_asserts:
            ast.parse(src)


@pytest.mark.slow
def test_manifest_records_revisions_and_every_exclusion() -> None:
    from alibi.data.build import build, manifest

    report = build()
    doc = manifest(report)
    assert doc["datasets"]["mbpp"]["revision"]
    assert doc["datasets"]["mbpp_plus"]["revision"]
    assert doc["file_digests"]
    assert len(doc["excluded"]) == doc["n_excluded"]
    for entry in doc["excluded"]:
        assert entry["reason"], f"task {entry['task_id']} excluded with no reason"
