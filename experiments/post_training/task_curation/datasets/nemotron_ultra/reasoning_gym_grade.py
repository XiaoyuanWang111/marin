# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Nemotron Ultra Reasoning Gym reply with the puzzle task's own scorer and print the reward.

The candidate is the final answer's last ``<answer>`` block, else its last boxed answer, else the whole
final answer. The grader packages' ``reasoning_gym`` scores it against the row's question, answer and
metadata; the metadata's ``source_dataset`` names the task.
"""

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import reasoning_gym
from verifyit.modes.extract import extract_markdown_answer

sys.path.insert(0, "/tests")

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer

SCORER_ERRORS = (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError)
"""What task scorers raise on a reply they cannot read. Any other exception stays a grader failure."""


def candidate(reply: str) -> str:
    """Prefer explicit final-answer wrappers, then preserve the source scorer's text fallback."""
    text = final_answer_text(reply)
    blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    boxed = last_boxed_answer(text)
    if boxed is not None:
        return boxed
    section = extract_markdown_answer(text)
    if section is not None:
        return section
    return text.strip()


def score(scorer: Callable[[str, Any], float], reply: str, entry: Any) -> float:
    """A reply that makes the scorer raise scores 0."""
    try:
        return float(scorer(reply, entry))
    except SCORER_ERRORS as error:
        print(f"The scorer raised {error!r}", file=sys.stderr)
        return 0.0


def main() -> None:
    contract = json.loads(Path("/tests/config.json").read_text())["contract"]
    entry = {"question": contract["question"], "answer": contract.get("answer"), "metadata": contract["metadata"]}
    scorer = reasoning_gym.get_score_answer_fn(contract["metadata"]["source_dataset"])
    print(score(scorer, candidate(Path("/app/answer.txt").read_text()), entry))


if __name__ == "__main__":
    main()
