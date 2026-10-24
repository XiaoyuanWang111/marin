"""Reusable item graders selected by sources through GraderSpec."""

from __future__ import annotations

from collections.abc import Sequence
import re
import tempfile
from typing import Any

from eval.contracts.eval_spec import EvalSpec, Grade
from eval.graders.livecodebench import run, run_lcb_tests


def live_code(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    blocks = re.findall(r"```(?:[a-zA-Z]*)\n(.*?)```", str(responses[-1]), flags=re.DOTALL)
    if not blocks:
        return Grade({"accuracy": 0.0}, filtered_resps=[[]])
    code = blocks[-1].split("</code>")[0].replace("```python", "").split("```")[0].replace("<code>", "")
    outcomes = run_lcb_tests(dict(spec.doc), code, float(spec.grader.reference.get("timeout", 6)))
    passed = bool(outcomes) and all(outcome[0] for outcome in outcomes)
    return Grade({"accuracy": float(passed)}, filtered_resps=[blocks])

def program(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Grade one fenced program against normalized test cases in an owned sandbox."""
    blocks = re.findall(r"```(?:[a-zA-Z]*)\n(.*?)```", str(responses[-1]), flags=re.DOTALL)
    if not blocks:
        return Grade({"accuracy": 0.0}, filtered_resps=[[]])
    reference = spec.grader.reference
    tests = reference["tests"]
    timeout = float(reference["timeout"])
    deadline_grace = float(reference.get("deadline_grace", 5))
    code = blocks[-1].split("</code>")[0].replace("<code>", "")
    if '__name__ == "__main__"' in code:
        code = '__name__ = "__main__"\n' + code
    result = run(
        tests, code, stdin_mode=reference["stdin"], fn_name=reference.get("entry_point"),
        # Competition benchmarks use only the original aggregate watchdog.
        test_timeout=0, deadline=(timeout + 1) * len(tests) + deadline_grace,
        comparison=reference["comparison"],
        maximum_memory_bytes=None,
    )
    return Grade({"accuracy": float(bool(tests) and result["passed"])}, filtered_resps=[blocks])

def crux(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    from eval.chat_benchmarks.CruxEval.execution import check_correctness

    answer = str(responses[-1])
    if "[ANSWER]" in answer and "[/ANSWER]" in answer:
        answer = answer.split("[ANSWER]", 1)[1].split("[/ANSWER]", 1)[0].strip()
    # Split at the first "==" as CRUXEval's harness does (inference/tasks/*_prediction.py, evaluation/utils_general.py):
    # an output answer is the text after it and an input answer is the call before it. An output answer containing
    # f(<input>) scores 0, and so does an input answer without "f(".
    direction = spec.grader.reference["direction"]
    if "==" in answer:
        answer = answer.split("==", 1)[0 if direction == "input" else 1].strip()
    if direction == "input" and "assert f" in answer:
        answer = "f" + answer.split("assert f", 1)[1].strip()
    if (direction == "output" and f"f({spec.doc['input']})" in answer) or (direction == "input" and "f(" not in answer):
        return Grade({"accuracy": 0.0}, filtered_resps=[answer])
    doc = dict(spec.doc)
    doc["test_code"] = f"{doc['code']}\nassert {doc['output']} == {answer}"
    with tempfile.TemporaryDirectory(prefix="evalchemy-crux-") as directory:
        outcome = check_correctness(str(spec.source_id), doc, "python", 10.0, directory, 0)
    return Grade({"accuracy": float(outcome["passed"])}, filtered_resps=[answer])
