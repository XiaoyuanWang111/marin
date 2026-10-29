# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source symbolic comparison remains bounded in grading workers."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from verifyit.grade import GradingInfraError

from taskcompendium.importers.evalchemy.scorers.harness_generation import grade_harness_generation_candidate
from taskcompendium.importers.evalchemy.scorers.math import grade_boxed_math
from taskcompendium.importers.evalchemy.scorers.minerva import is_equiv
from taskcompendium.importers.evalchemy.scorers.olympiad import grade_olympiad


def test_harness_math_symbolic_fallback_grades_correctly_in_a_worker_thread():
    answer = "1.2345678901234567"
    with ThreadPoolExecutor(max_workers=1) as workers:
        result = workers.submit(grade_harness_generation_candidate, "gsm8k", ((answer,),), "#### " + answer).result(
            timeout=15
        )
    assert result.reward == 1
    assert result.detail["metrics"] == {
        "exact_match,strict-match": 1.0,
        "exact_match,flexible-extract": 1.0,
    }


@pytest.mark.parametrize("threaded", [False, True])
def test_minerva_symbolic_timeout_is_a_nonmatch(threaded):
    if threaded:
        with ThreadPoolExecutor(max_workers=1) as workers:
            result = workers.submit(is_equiv, r"9^{9^{9}}", "2").result(timeout=45)
    else:
        result = is_equiv(r"9^{9^{9}}", "2")
    assert result is False


@pytest.mark.slow
@pytest.mark.timeout(90)
def test_minerva_concurrent_workers_complete_symbolic_matches():
    with ThreadPoolExecutor(max_workers=48) as workers:
        results = list(workers.map(is_equiv, [r"\sqrt{4}"] * 48, ["2"] * 48))
    assert results == [True] * 48


def test_minerva_worker_crash_remains_an_infrastructure_failure(tmp_path, monkeypatch):
    (tmp_path / "sitecustomize.py").write_text("import os; os._exit(23)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with ThreadPoolExecutor(max_workers=1) as workers:
        with pytest.raises(GradingInfraError):
            workers.submit(is_equiv, r"\sqrt{4}", "2").result(timeout=45)


# OlympiadBench document 4's reference: a relation without free symbols.
RELATION = r"I_{e}(0)=\gamma I_{i}(0)"
GRADERS = {
    "olympiad": lambda reference, response: grade_olympiad((reference,), response),
    "boxed_math": grade_boxed_math,
}


@pytest.mark.parametrize(
    ("reference", "candidate", "correct"),
    [
        # Equations that solve to no value, and other relations without free symbols, do not match.
        (RELATION, "x = x", False),
        (RELATION, "x = x + 1", False),
        (RELATION, r"I_{i}(0) = I_{e}(0)", False),
        ("3 = 4", "1 = 2", False),
        # The reference's own relation matches, either way round.
        (RELATION, r"I_{e}(0)=\gamma I_{i}(0)", True),
        (RELATION, r"\gamma I_{i}(0) = I_{e}(0)", True),
        # Other references still go through math-verify.
        ("x=1", "x = 1", True),
        ("42", "42", True),
        ("0.5", r"\frac{1}{2}", True),
    ],
)
@pytest.mark.parametrize("grade", GRADERS.values(), ids=GRADERS.keys())
def test_symbol_free_relation_reference_matches_only_its_own_relations(grade, reference, candidate, correct):
    assert grade(reference, rf"The answer is \boxed{{{candidate}}}.").reward == float(correct)
