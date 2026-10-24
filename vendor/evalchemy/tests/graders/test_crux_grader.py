"""The CRUXEval grader reads an answer by CRUXEval's harness rule."""

import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.programs import crux

CONCAT = "def f(text, value):\n    return text + value"


def _spec(direction: str, arguments: str, output: str) -> EvalSpec:
    return EvalSpec(
        task="CruxEval", source=SourceRef("fixture"), source_id="sample", ordinal=0, requests=(),
        grader=GraderSpec("crux", reference={"direction": direction}),
        doc={"code": CONCAT, "input": arguments, "output": output}, namespace=direction,
    )


@pytest.mark.parametrize(
    ("direction", "arguments", "output", "reply", "accuracy"),
    [
        # A ")" inside the input does not end the call.
        ("output", "'(ac', ')'", "'(ac)'", "[ANSWER]\nassert f('(ac', ')') == '(ac)'\n[/ANSWER]", 1.0),
        ("input", "'(ac', ')'", "'(ac)'", "[ANSWER]\nassert f('(ac', ')') == '(ac)'\n[/ANSWER]", 1.0),
        # An "=" inside the output stays.
        ("output", "'h', '=l'", "'h=l'", "[ANSWER]\nassert f('h', '=l') == 'h=l'\n[/ANSWER]", 1.0),
        # A wrong answer scores 0.
        ("output", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('a', 'b') == 'ba'\n[/ANSWER]", 0.0),
        ("input", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('b', 'a') == 'ab'\n[/ANSWER]", 0.0),
        # An output answer containing f(<input>), or an input answer without a call, is not an answer.
        ("output", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('a', 'b') == f('a', 'b')\n[/ANSWER]", 0.0),
        ("input", "'a', 'b'", "'ab'", "[ANSWER]\n'ab'\n[/ANSWER]", 0.0),
    ],
)
def test_crux_reads_answers_by_the_harness_rule(direction, arguments, output, reply, accuracy):
    assert crux(_spec(direction, arguments, output), [reply]).metrics == {"accuracy": accuracy}
