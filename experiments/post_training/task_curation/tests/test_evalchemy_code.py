# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish code rows while keeping executable tests in verifier-only resources."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from shellbox.machine import PackageSpec
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.assets import code_protocol
from experiments.post_training.task_curation.datasets.evalchemy.code import sources
from experiments.post_training.task_curation.pipeline import source_recipe

INPUTS = [
    {
        "suite": "humaneval",
        "rows": [
            {
                "task_id": "HumanEval/private",
                "prompt": "def identity(value):\n    ",
                "canonical_solution": "return value\n",
                "test": "def check(candidate):\n    assert candidate(7) == 7\n",
                "entry_point": "identity",
            }
        ],
    },
    {
        "suite": "cruxeval",
        "rows": [{"id": "identity", "code": "def f(x): return x", "input": "7", "output": "7"}],
    },
]


@pytest.mark.parametrize("case", INPUTS, ids=lambda case: case["suite"])
def test_python_publication_keeps_tests_private_and_expands_crux_directions(case, tmp_path):
    source = next(source for source in sources() if source.info.id == "Evalchemy:" + case["suite"])
    dataset = source.config
    root = tmp_path / "source"
    path = root / dataset.files[0]
    path.parent.mkdir(parents=True)
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(case["rows"]), path)
    else:
        path.write_text("".join(json.dumps(row) + "\n" for row in case["rows"]))
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        package_spec=PackageSpec(pypi_packages=("numpy==2.3.5",)),
    )
    protocol = dataset.protocol.model_copy(
        update={"expected_population": len(case["rows"]) * (2 if case["suite"] == "cruxeval" else 1)}
    )
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=1,
                chunk_storage_prefix=str(tmp_path / "chunks"),
            ) as context,
        ):
            publish_evaluation_corpus(
                source_recipe(dataset.recipe, {}, environment),
                protocol,
                dataset.item_builder,
                context,
                str(root),
                output,
            )
        published = list(evaluation_cases(output))
    finally:
        client.shutdown()
    assert len(published) == protocol.expected_population
    if case["suite"] == "cruxeval":
        assert [item.item.groups["direction"] for item in published] == ["input", "output"]
        assert len({item.task.id for item in published}) == 2
    for item in published:
        assert not item.task.resources.worker
        config = next(resource for resource in item.task.resources.verifier if resource.path == "config.json")
        private = json.loads(resource_bytes(config))
        assert private["problem"]["code" if case["suite"] == "cruxeval" else "test"]
        if case["suite"] == "humaneval":
            assert private["problem"]["test"] not in item.task.context.events[0].content


CONCAT = "def f(text, value):\n    return text + value"


def _passes(program: str) -> bool:
    """Whether a test program runs without raising, the grading wrapper's pass rule."""
    try:
        exec(compile(program, "<candidate>", "exec"), {})
    except Exception:
        return False
    return True


@pytest.mark.parametrize(
    ("direction", "arguments", "output", "reply", "passed"),
    [
        # A ")" inside the input does not end the call.
        ("output", "'(ac', ')'", "'(ac)'", "[ANSWER]\nassert f('(ac', ')') == '(ac)'\n[/ANSWER]", True),
        ("input", "'(ac', ')'", "'(ac)'", "[ANSWER]\nassert f('(ac', ')') == '(ac)'\n[/ANSWER]", True),
        # An "=" inside the output stays.
        ("output", "'h', '=l'", "'h=l'", "[ANSWER]\nassert f('h', '=l') == 'h=l'\n[/ANSWER]", True),
        # A wrong answer fails.
        ("output", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('a', 'b') == 'ba'\n[/ANSWER]", False),
        ("input", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('b', 'a') == 'ab'\n[/ANSWER]", False),
        # An output answer containing f(<input>), or an input answer without a call, is not an answer.
        ("output", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('a', 'b') == f('a', 'b')\n[/ANSWER]", False),
        ("input", "'a', 'b'", "'ab'", "[ANSWER]\n'ab'\n[/ANSWER]", False),
    ],
)
def test_crux_reads_answers_by_the_harness_rule(direction, arguments, output, reply, passed):
    problem = {"code": CONCAT, "input": arguments, "output": output}

    assert _passes(code_protocol.test_program(problem, reply, "cruxeval", direction)) is passed
