# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comparing the scores of two evaluation archives, such as a run and a regrade of its stored responses."""

import json

from evalrecords.records import (
    CuratedRef,
    CuratedSettings,
    EvalRef,
    EvalRunRecord,
    HardwareRef,
    ModelRef,
    Provenance,
)
from evalrecords.rollout_attempts import ATTEMPTS_TABLE, RUN_OBJECT, Attempt
from evalrecords.rollout_results import TrialChanges, compare_scores, score_curated_record
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind, sample_to_archive_row
from finestore.reader import ReadView
from finestore.schema import arrow_schema
from finestore.store import DataStore
from taskcompendium.evaluation.selection import SELECTIONS_TABLE, Selection
from taskcompendium.evaluation.spec import EvaluationProtocol, Metric, Sampling, TrialPolicy
from taskcompendium.identity import canonical_sha256

STRICT = "exact_match,strict-match"
FLEXIBLE = "exact_match,flexible-extract"
GSM8K = EvaluationProtocol(
    benchmark="gsm8k",
    version="1",
    operation="chat_generation",
    sampling=Sampling(temperature=0, top_p=None, top_k=None, max_tokens=16, stop=(), chat_template_kwargs={}),
    trials=TrialPolicy(count=1, seed=0),
    metrics=(
        Metric(name=STRICT, kind="binary", higher_is_better=True),
        Metric(name=FLEXIBLE, kind="binary", higher_is_better=True),
    ),
    primary_metric=FLEXIBLE,
    summary="mean",
    expected_population=4,
)


def write_archive(root, corpus, observations):
    """Store a GSM8K run as the evaluation driver does: one trial per source row, with retries.

    Each observation is ``(row, retry, phase, outcome, metrics)``. The corpus name enters the dataset
    snapshot identity and the task ids, as a republished corpus changes both.
    """
    manifest = {"rows": 4, "source_dataset": "openai/gsm8k", "source_revision": "pin", "conversion": corpus}
    dataset = canonical_sha256(manifest)
    with DataStore.open(root) as store:
        EvaluationStore(store)
        store.table(
            ATTEMPTS_TABLE,
            schema=arrow_schema(Attempt),
            primary_key=("dataset", "source_row", "trial", "retry", "phase"),
            schema_version=1,
        )
        store.table(
            SELECTIONS_TABLE, schema=arrow_schema(Selection), primary_key=("dataset", "source_row"), schema_version=1
        )
        with store.transaction() as transaction:
            run = {"datasets": [{"dataset": dataset, "protocol": GSM8K.model_dump(mode="json"), "manifest": manifest}]}
            transaction.write_object(RUN_OBJECT, json.dumps(run).encode())
            source_rows = {row: f"main/test-00000-of-00001.parquet:{row}" for row, *_ in observations}
            for row, source_row in sorted(source_rows.items()):
                transaction.table(SELECTIONS_TABLE).add(
                    Selection(
                        dataset=dataset,
                        source_row=source_row,
                        source_ordinal=row,
                        status="selected",
                        reason=None,
                        rank=row,
                        context_window=1024,
                        max_tokens=16,
                        prompt_tokens=1,
                        reference_tokens=None,
                    ).model_dump(mode="json")
                )
            for row, retry, phase, outcome, metrics in observations:
                attempt = Attempt(
                    dataset=dataset,
                    task_id=f"{corpus}-{row}",
                    source_row=source_rows[row],
                    trial=0,
                    retry=retry,
                    phase=phase,
                    seed=0,
                    outcome=outcome,
                    error=None,
                    result_blob=f"evaluation/attempts/{corpus}-{row}-{retry}.json" if phase == "finished" else None,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
                transaction.table(ATTEMPTS_TABLE).add(attempt.model_dump(mode="json"))
                if phase != "finished":
                    continue
                sample = EvalSample(
                    task="gsm8k",
                    doc_id=attempt.task_id,
                    kind=SampleKind.GENERATION,
                    output="72",
                    metrics=metrics,
                    grading=Grading(method="fixture", metric=FLEXIBLE, score=metrics.get(FLEXIBLE)),
                    doc=json.dumps({"source": attempt.source_row, "groups": {}}),
                )
                transaction.table("samples").add(sample_to_archive_row(sample, trial_id=f"{dataset}/0/{retry}"))
    return ReadView(root)


def graded(values):
    """One graded first attempt per source row, from ``{row: (strict, flexible)}``."""
    return [
        (row, 0, "finished", "graded", {STRICT: strict, FLEXIBLE: flexible})
        for row, (strict, flexible) in values.items()
    ]


def test_compare_scores_pairs_a_republished_corpus_by_source_row_and_counts_primary_changes(tmp_path):
    before = write_archive(str(tmp_path / "before"), "v1", graded({0: (1, 1), 1: (0, 0), 2: (0, 0), 3: (1, 1)}))
    after = write_archive(str(tmp_path / "after"), "v2", graded({0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (0, 0)}))

    gsm8k = compare_scores(before, after)["gsm8k"]

    assert gsm8k.before is not None and gsm8k.after is not None
    assert (gsm8k.before.metrics[FLEXIBLE], gsm8k.after.metrics[FLEXIBLE]) == (0.5, 0.75)
    assert gsm8k.changes[FLEXIBLE] == TrialChanges(paired=4, increased=2, decreased=1, before_only=0, after_only=0)


def test_compare_scores_reports_a_strict_match_change_under_an_unchanged_primary_metric(tmp_path):
    before = write_archive(str(tmp_path / "before"), "v1", graded({0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 1)}))
    after = write_archive(str(tmp_path / "after"), "v2", graded({0: (1, 1), 1: (0, 0), 2: (1, 1), 3: (1, 1)}))

    gsm8k = compare_scores(before, after)["gsm8k"]

    assert gsm8k.before is not None and gsm8k.after is not None
    assert gsm8k.changes[FLEXIBLE] == TrialChanges(paired=4, increased=0, decreased=0, before_only=0, after_only=0)
    assert gsm8k.changes[STRICT] == TrialChanges(paired=4, increased=1, decreased=0, before_only=0, after_only=0)
    assert (gsm8k.before.metrics[STRICT], gsm8k.after.metrics[STRICT]) == (0.5, 0.75)


def test_compare_scores_leaves_rows_present_on_one_side_unpaired(tmp_path):
    before = write_archive(str(tmp_path / "before"), "v1", graded({0: (1, 1), 1: (0, 0), 2: (0, 0), 3: (1, 1)}))
    after = write_archive(str(tmp_path / "after"), "v1", graded({1: (0, 1), 2: (0, 0), 3: (1, 1), 4: (0, 0)}))

    gsm8k = compare_scores(before, after)["gsm8k"]

    assert gsm8k.after is not None
    assert gsm8k.after.metrics[FLEXIBLE] == 0.5
    assert gsm8k.changes[FLEXIBLE] == TrialChanges(paired=3, increased=1, decreased=0, before_only=1, after_only=1)


def test_compare_scores_pairs_the_terminal_attempt_of_each_trial(tmp_path):
    before = write_archive(str(tmp_path / "before"), "v1", graded({0: (1, 1), 1: (0, 0), 2: (0, 0), 3: (1, 1)}))
    after = write_archive(
        str(tmp_path / "after"),
        "v2",
        [
            # Grading failed, and the retry never finished: the trial has no score.
            (0, 0, "finished", "infra_error", {}),
            (0, 1, "started", None, {}),
            # Grading failed, and the retry was graded.
            (1, 0, "finished", "infra_error", {}),
            (1, 1, "finished", "graded", {STRICT: 1, FLEXIBLE: 1}),
            *graded({2: (0, 0), 3: (1, 1)}),
        ],
    )

    gsm8k = compare_scores(before, after)["gsm8k"]

    assert gsm8k.after is not None
    assert gsm8k.after.coverage.n_scored == 3
    assert gsm8k.changes[FLEXIBLE] == TrialChanges(paired=3, increased=1, decreased=0, before_only=1, after_only=0)


def test_compare_scores_of_an_archive_with_itself_gives_the_scores_evaldash_ingests(tmp_path):
    root = str(tmp_path / "run")
    archive = write_archive(
        root,
        "v1",
        [
            (0, 0, "finished", "infra_error", {}),
            (1, 0, "finished", "infra_error", {}),
            (1, 1, "finished", "graded", {STRICT: 0, FLEXIBLE: 1}),
            *graded({2: (0, 0), 3: (1, 1)}),
        ],
    )
    record = EvalRunRecord(
        run_id="run",
        group_id="group",
        created_at="2026-11-19T00:00:00Z",
        user="test",
        model=ModelRef(name="solver", location="test/solver", backend="vllm"),
        eval=EvalRef(
            name="gsm8k", mechanism="curated", curated=CuratedRef(datasets=("gsm8k",), settings=CuratedSettings())
        ),
        hardware=HardwareRef(platform="gpu", accelerator="H100x1", region_or_cluster="test"),
        provenance=Provenance(git_sha="revision", launch_host="test"),
        status="succeeded",
        results_path=root,
        error=None,
        metrics={},
        jobs={},
        log_tails={},
    )

    gsm8k = compare_scores(archive, archive)["gsm8k"]

    assert gsm8k.before is not None and gsm8k.after is not None
    assert gsm8k.before.metrics == gsm8k.after.metrics == score_curated_record(record).canonical_metrics["gsm8k"]
    assert (gsm8k.before.metrics[FLEXIBLE], gsm8k.before.metrics[STRICT]) == (2 / 3, 1 / 3)
    assert gsm8k.changes == {
        metric: TrialChanges(paired=3, increased=0, decreased=0, before_only=0, after_only=0)
        for metric in (STRICT, FLEXIBLE)
    }
