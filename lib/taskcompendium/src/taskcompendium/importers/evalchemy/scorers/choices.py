# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA and MMLU-Pro extraction from Marin's Evalchemy fork.

GPQA's extraction was adapted there from QwenLM/Qwen2.5-Math (Apache-2.0).
These protocols deliberately differ: GPQA takes the last A-D anywhere, while
MMLU-Pro prioritizes the first answer marker, matched case-sensitively like TIGER-Lab's reference extractor.
"""

import re

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored


def extract_gpqa_answer(response: str) -> str:
    matches = re.findall(r"\b(A|B|C|D)\b", response.upper())
    answer = matches[-1] if matches else response.strip().strip(".")
    return answer.rstrip(".").rstrip("/")


def extract_mmlu_answer(response: str) -> str | None:
    text = final_text(response)
    # The patterns and their case follow TIGER-AI-Lab/MMLU-Pro evaluate_from_api.py, so "the answer is about 56%" does
    # not read as A, and the "Answer:" fallback takes the last match on the first line that has one.
    for pattern in (r"answer is \(?([A-J])\)?", r".*[aA]nswer:\s*([A-J])"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def grade_gpqa_answer(expected: str, response: str) -> Reward:
    answer = extract_gpqa_answer(response)
    return scored(float(answer == expected), extracted=answer, expected=expected)


def grade_mmlu_answer(expected: str, response: str) -> Reward:
    answer = extract_mmlu_answer(response)
    return scored(float(answer == expected), extracted=answer, expected=expected)
