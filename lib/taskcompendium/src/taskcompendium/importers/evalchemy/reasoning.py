# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA and MMLU-Pro source prompts, staged fewshots, and evaluation items."""

import hashlib
import random
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    ChatGeneration,
    ChatPromptVariant,
    EvaluationItem,
)
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow
from taskcompendium.submission import conversation_messages

MMLU_VALIDATION_FILE = "data/validation-00000-of-00001.parquet"
GPQA_PROMPT = (
    "Return your final response within \\boxed{{}} and only include the letter choice "
    "(A, B, C, or D) as your final response.\nProblem: {problem}\nOptions: {options}\nAnswer:"
)
# The source package file ends with three newlines; generate_cot_prompt adds another.
MMLU_INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {$}. Think step by step "
    'and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n'
)
MMLU_FEWSHOT_COUNT = 5


def gpqa_choices(data: Mapping[str, Any]) -> tuple[str, str]:
    answers = [data[key] for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
    rng = random.Random(42 + int(hashlib.md5(data["Question"].encode()).hexdigest(), 16))
    rng.shuffle(answers)
    options = ", ".join(f"{letter}) {answer}" for letter, answer in zip("ABCD", answers, strict=True))
    expected = next(letter for letter, answer in zip("ABCD", answers, strict=True) if answer == data["Correct Answer"])
    return options, expected


def convert_gpqa(row: RawRow, context: ConversionContext) -> TaskSpec:
    options, expected = gpqa_choices(row.data)
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=GPQA_PROMPT.format(problem=row.data["Question"], options=options)),),
        package=source_package("gpqa", {"expected": expected}, environment=required_grader_environment(context)),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:GPQADiamond")})


def mmlu_question(example: Mapping[str, Any]) -> str:
    options = [option for option in example["options"] if option != "N/A"]
    text = "Question:\n" + example["question"] + "\nOptions:\n"
    return text + "".join(f"{chr(65 + index)}. {option}\n" for index, option in enumerate(options))


def mmlu_messages(support: tuple[dict, ...], example: Mapping[str, Any], *, num_fewshot: int) -> tuple[TextMessage, ...]:
    """The source's chain-of-thought prompt with each worked example as a user question and an assistant answer.

    The message texts concatenate to the source's one-message prompt.
    """
    category = example["category"]
    selected = [shot for shot in support if shot["category"] == category][:num_fewshot]
    prefix = MMLU_INITIAL_PROMPT.replace("{$}", category) + "\n"
    messages: list[TextMessage] = []
    for shot in selected:
        answer = shot["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        messages += [
            TextMessage(role="user", content=prefix + mmlu_question(shot)),
            TextMessage(role="assistant", content=answer + "\n\n"),
        ]
        prefix = ""
    question = mmlu_question(example) + "Answer: Let's think step by step."
    return (*messages, TextMessage(role="user", content=prefix + question))


@lru_cache
def mmlu_support(root: str) -> tuple[dict, ...]:
    """Read the small, immutable validation split once per conversion worker."""
    return tuple(load_parquet(str(StoragePath(root) / MMLU_VALIDATION_FILE)))


def convert_mmlu(row: RawRow, context: ConversionContext) -> TaskSpec:
    support = mmlu_support(str(context.inputs["fewshot"]))
    variants = tuple(mmlu_messages(support, row.data, num_fewshot=shots) for shots in range(MMLU_FEWSHOT_COUNT + 1))
    task = conversation_task(
        row,
        events=variants[-1],
        package=source_package(
            "mmlu_pro", {"expected": row.data["answer"]}, environment=required_grader_environment(context)
        ),
        evidence={**row.data, "prompt_variants": [conversation_messages(events) for events in variants]},
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:MMLUPro")})


def reasoning_item(task: TaskSpec, *, benchmark: str) -> EvaluationItem:
    evidence = source_evidence(task)
    groups = (
        {"category": evidence["category"]}
        if benchmark == "MMLUPro"
        else {key: evidence[key] for key in ("High-level domain", "Subdomain") if key in evidence}
    )
    operation = (
        ChatGeneration(
            prompt_selection="fit_context",
            response_selection="final_or_complete_reasoning",
            variants=tuple(
                ChatPromptVariant(
                    context=ConversationInput(
                        events=tuple(TextMessage(role=turn["role"], content=turn["content"]) for turn in messages)
                    ),
                    fewshot=shots,
                )
                for shots, messages in enumerate(evidence["prompt_variants"])
            ),
        )
        if benchmark == "MMLUPro"
        else ChatGeneration()
    )
    return chat_item(task).model_copy(update={"groups": groups, "operation": operation})
