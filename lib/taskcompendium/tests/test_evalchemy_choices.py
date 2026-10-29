# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.importers.evalchemy.scorers.choices import grade_mmlu_answer


@pytest.mark.parametrize(
    "response,answer",
    [
        ("Earlier A was plausible. Answer: C", "C"),
        ("B. This is a valid explanation.", "B"),
        ("A was plausible.\nB. This is a valid explanation.", "B"),
        ("b. This is a valid explanation.", None),
        ("The ANSWER IS (c). Later mention B.", "B"),
        ("aNsWeR: d. Later mention B.", "B"),
        ("", None),
        # TIGER-Lab's reference extractor matches "answer is" case-sensitively.
        ("The answer is about 56%, so the answer is (C).", "C"),
        ("the answer is (b). Later mention D.", "D"),
        # Its "Answer:" fallback takes the last match on the first line that has one.
        ("Answer: B at first, but the final answer: C", "C"),
        ("answer: c", None),
        # The same reply reads C with its think markers kept and with them removed.
        ("<|start_think|>The answer is about 56%, not (B).<|end_think|>The answer is (C).", "C"),
        ("The answer is about 56%, not (B).The answer is (C).", "C"),
    ],
)
def test_mmlu_answer_matches_the_reference_extractor(response, answer):
    result = grade_mmlu_answer(answer or "B", response)

    assert result.detail["extracted"] == answer
    assert result.reward == float(answer is not None)
    assert grade_mmlu_answer("J", response).reward == 0
