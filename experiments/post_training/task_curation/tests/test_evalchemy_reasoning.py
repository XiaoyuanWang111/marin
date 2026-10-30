# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the auxiliary Parquet split and lay MMLU-Pro's worked examples out as conversation turns."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.evaluation.selection import prompt_variant
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.evaluation import EvaluationCase

from experiments.post_training.task_curation.datasets.evalchemy.reasoning import (
    MMLU_VALIDATION_FILE,
    mmlu_support,
    sources,
)
from experiments.post_training.task_curation.tests.conversion import converted_task

INSTRUCTION = (
    "The following are multiple choice questions (with answers) about math. Think step by step and then finish your "
    'answer with "the answer is (X)" where X is the correct letter choice.\n\n\n\n'
)
# Each worked example as the source's one-message prompt writes it: the question block, then the answer.
WORKED = tuple(
    (
        f"Question:\nWhat is {n} + {n}?\nOptions:\nA. {2 * n}\nB. {n}\n",
        f"Answer: Let's think step by step. {n} + {n} = {2 * n}. The answer is (A).\n\n",
    )
    for n in range(1, 6)
)
QUESTION = "Question:\nWhat is 7 + 7?\nOptions:\nA. 14\nB. 7\nAnswer: Let's think step by step."


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


def source_prompt(shots: int) -> str:
    return INSTRUCTION + "".join(question + answer for question, answer in WORKED[:shots]) + QUESTION


def mmlu_pro_case(tmp_path) -> EvaluationCase:
    support = [
        {
            "question": f"What is {n} + {n}?",
            "options": [str(2 * n), str(n)],
            "answer": "A",
            "cot_content": f"A: Let's think step by step. {n} + {n} = {2 * n}. The answer is (A).",
            "category": "math",
        }
        for n in range(1, 6)
    ]
    path = tmp_path / MMLU_VALIDATION_FILE
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(support), path)
    dataset = next(source.config for source in sources() if source.config.protocol.benchmark == "MMLUPro")
    row = {"question": "What is 7 + 7?", "options": ["14", "7", "N/A"], "answer": "A", "category": "math"}
    task = converted_task(dataset.recipe, row, inputs={"fewshot": StoragePath(str(tmp_path))})
    return EvaluationCase(task, dataset.item_builder(task))


@pytest.mark.parametrize("as_url", [False, True])
def test_mmlu_support_reads_staged_parquet_from_local_or_url_prefix(tmp_path, as_url):
    rows = [{"question": "Choose an even integer.", "options": ["1", "2"], "answer": "B", "category": "math"}]
    path = tmp_path / MMLU_VALIDATION_FILE
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path)
    root = tmp_path.as_uri() + "/" if as_url else str(tmp_path)
    assert mmlu_support(root) == tuple(rows)


@pytest.mark.parametrize("shots", range(6))
def test_mmlu_pro_variant_sends_worked_examples_as_turns_of_the_source_prompt(tmp_path, shots):
    case = mmlu_pro_case(tmp_path)
    events = case.item.operation.variants[shots].context.events
    texts = [event.content for event in events]
    assert [event.role for event in events] == ["user", "assistant"] * shots + ["user"]
    assert texts[1::2] == [answer for _, answer in WORKED[:shots]]
    assert "".join(texts) == source_prompt(shots)
    assert case.task.context == case.item.operation.variants[-1].context


@pytest.mark.parametrize("shots", range(6))
def test_mmlu_pro_context_fitting_keeps_the_most_shots_whose_source_prompt_fits(tmp_path, shots):
    case = mmlu_pro_case(tmp_path)
    # Room for one character more than the source prompt with `shots` examples, so no larger variant fits.
    selected = prompt_variant(case, len(source_prompt(shots)) + 1 + 100, 100, CharacterTokenizer())
    assert selected.context == case.item.operation.variants[shots].context
