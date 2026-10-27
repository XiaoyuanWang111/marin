import pytest

from eval.chat_benchmarks.MMLUPro.eval_instruct import MMLUProBenchmark
from eval.completion_response import CompletionContentPolicy, CompletionResponse, CompletionText
from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import mmlu_choice
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def test_mmlu_choice_uses_the_final_answer_marker():
    spec = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("mmlu_choice"), target="C")

    grade = mmlu_choice(spec, ["Earlier A was plausible. Answer: C"])

    assert grade.metrics["accuracy"] == 1.0
    assert grade.filtered_resps == ["C"]


@pytest.mark.parametrize("response,answer", [
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
])
def test_mmlu_choice_matches_the_reference_extractor(response, answer):
    spec = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0,
                    requests=(), grader=GraderSpec("mmlu_choice"), target=answer or "B")

    grade = mmlu_choice(spec, [response])

    assert grade.filtered_resps == [answer]
    assert grade.metrics["accuracy"] == float(answer is not None)
    wrong = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0,
                     requests=(), grader=GraderSpec("mmlu_choice"), target="J")
    assert mmlu_choice(wrong, [response]).metrics["accuracy"] == 0.0


@pytest.mark.parametrize("finish_reason,answer", [("stop", "B"), ("length", None)])
def test_mmlu_choice_preserves_original_reasoning_only_policy(finish_reason, answer):
    reasoning = "B. This is a valid explanation."
    response = CompletionResponse(None, reasoning, finish_reason, None, {}, {})
    output = CompletionText(reasoning, response, CompletionContentPolicy.COMBINE)
    spec = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0,
                    requests=(), grader=GraderSpec("mmlu_choice"), target="B")

    grade = mmlu_choice(spec, [output])

    assert grade.filtered_resps == [answer]
    assert grade.metrics["accuracy"] == float(answer is not None)
    wrong = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0,
                     requests=(), grader=GraderSpec("mmlu_choice"), target="J")
    assert mmlu_choice(wrong, [output]).metrics["accuracy"] == 0.0


def test_mmlu_source_keeps_category_counts_and_micro_accuracy(monkeypatch):
    rows = [
        {"question_id": index, "category": category, "question": "Choose", "options": ["first", "second"],
         "answer": answer, "cot_content": ""}
        for index, category, answer in [(1, "math", "B"), (2, "law", "A"), (3, "math", "B")]
    ]
    source = MMLUProBenchmark(num_fewshot=0)
    monkeypatch.setattr(source, "load_questions", lambda: (rows, []))
    responses = {1: "B. This is a valid explanation.", 2: "Answer: A", 3: "b. This is a valid explanation."}

    metrics, samples = evaluate_source(
        source, SourceOptions("MMLUPro", source.SOURCE),
        FixedEndpoint(lambda request: responses[request.metadata["source_id"]]),
    )

    assert metrics["accuracy_avg"] == pytest.approx(2 / 3)
    assert metrics["accuracy_math"] == 0.5
    assert metrics["accuracy_law"] == 1.0
    assert metrics["count_math"] == 2
    assert metrics["count_law"] == 1
    assert metrics["num_total"] == len(samples) == 3
    assert sum(sample["accuracy"] for sample in samples) == 2
