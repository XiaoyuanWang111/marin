# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for corrected trial-level score artifacts."""

import importlib.util
import math
import sys
from pathlib import Path


def load_plot():
    spec = importlib.util.spec_from_file_location(
        "campaign_critical_difference_plot", Path(__file__).with_name("plot.py")
    )
    assert spec is not None and spec.loader is not None
    plot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plot
    spec.loader.exec_module(plot)
    return plot


def test_corrected_replay_cell_uses_trial_rewards_for_uncertainty(monkeypatch) -> None:
    plot = load_plot()
    replay = {
        "model": "open-athena/Snowball-Test",
        "benchmark": "olympiadbench",
        "num_trials": 3,
        "num_judge_failed": 0,
        "trials": [{"corrected_correct": value} for value in (True, False, True)],
    }
    monkeypatch.setattr(plot, "read_json", lambda _: replay)
    cell = plot.Cell(replay["model"], replay["benchmark"], 0.667, "s3://bucket/regrade/model.json")

    statistics = plot.cell_statistics(cell, {})

    assert statistics["trial_count"] == 3
    assert math.isclose(statistics["score"], 2 / 3)
    assert statistics["tracker_score"] == 0.667
    assert math.isclose(statistics["raw_reward_mean"], 2 / 3)
    assert math.isclose(statistics["sem"], 1 / 3)
    assert statistics["sem_basis"] == "corrected_replay_trial_rewards"


def test_corrected_replay_sem_counts_documents_not_replies(monkeypatch) -> None:
    plot = load_plot()
    replay = {
        "model": "open-athena/Snowball-Test",
        "benchmark": "aime24",
        "num_trials": 6,
        "num_judge_failed": 0,
        "trials": [
            {"doc_id": document, "trial_id": trial, "corrected_correct": value}
            for document, values in enumerate(((True, True, True), (False, False, True)))
            for trial, value in enumerate(values)
        ],
    }
    monkeypatch.setattr(plot, "read_json", lambda _: replay)
    cell = plot.Cell(replay["model"], replay["benchmark"], 0.667, "s3://bucket/regrade/model.json")

    statistics = plot.cell_statistics(cell, {})
    rewards = plot.item_rewards(cell, {})

    # Document means 1 and 1/3 give a standard error of 1/3 over two documents; six independent replies would give
    # 0.211.
    assert statistics["trial_count"] == 6
    assert math.isclose(statistics["score"], 2 / 3)
    assert math.isclose(statistics["sem"], 1 / 3)
    assert statistics["sem_basis"] == "corrected_replay_item_means"
    assert rewards == {"0": [1.0, 1.0, 1.0], "1": [0.0, 0.0, 1.0]}


def test_corrected_replay_with_one_reply_per_document_keeps_the_binary_sem(monkeypatch) -> None:
    plot = load_plot()
    replay = {
        "model": "open-athena/Snowball-Test",
        "benchmark": "math500",
        "num_trials": 4,
        "num_judge_failed": 0,
        "trials": [
            {"doc_id": document, "corrected_correct": value} for document, value in enumerate((True, False, True, True))
        ],
    }
    monkeypatch.setattr(plot, "read_json", lambda _: replay)
    cell = plot.Cell(replay["model"], replay["benchmark"], 0.75, "s3://bucket/regrade/model.json")

    statistics = plot.cell_statistics(cell, {})

    assert statistics["sem_basis"] == "corrected_replay_item_means"
    assert math.isclose(statistics["sem"], math.sqrt(0.75 * 0.25 / 3))


def test_binary_archive_cell_uses_full_precision_sample_rewards_that_reproduce_the_tracker(monkeypatch) -> None:
    plot = load_plot()
    rows = [
        {"task": "AIME24", "doc_id": document, "metrics": {"accuracy": value, "loose": 1.0}, "filter": None}
        for document, values in (("0", (1.0, 1.0)), ("1", (0.0, 1.0)), ("2", (1.0, 0.0)))
        for value in values
    ]

    class SampleTable:
        def to_pylist(self, **_kwargs):
            return rows

    class SampleView:
        def __init__(self, _url):
            pass

        def scan(self, _table, **_kwargs):
            return SampleTable()

    class Filesystem:
        def exists(self, _path):
            return True

    monkeypatch.setattr(plot, "ReadView", SampleView)
    monkeypatch.setattr(plot, "filesystem_for", lambda url: (Filesystem(), url))
    monkeypatch.setattr(plot, "read_json", lambda _: {"status": "succeeded", "coverage": {"AIME24": {"n_scored": 3}}})
    matching = plot.Cell("model", "aime24", 0.667, "s3://bucket/model-aime24/results")
    unmatched = plot.Cell("model", "aime24", 0.700, "s3://bucket/model-aime24/results")

    statistics = plot.cell_statistics(matching, {})

    assert plot.item_rewards(matching, {}) == {"0": [1.0, 1.0], "1": [0.0, 1.0], "2": [1.0, 0.0]}
    assert math.isclose(statistics["score"], 2 / 3)
    assert statistics["tracker_score"] == 0.667
    assert statistics["sem_basis"] == "binary_sample_mean_from_record_coverage"
    # No sample metric reproduces a binary tracker score of 0.700: the cell keeps the tracker score and stays unpaired.
    assert plot.item_rewards(unmatched, {}) is None
    assert plot.cell_statistics(unmatched, {})["score"] == 0.700


def test_cruxeval_counts_only_output_tasks(monkeypatch) -> None:
    plot = load_plot()

    class SampleTable:
        def to_pylist(self, **_kwargs):
            return [
                {"doc": '{"cruxeval_task": "input"}', "metrics": {"pass_rate": 0}, "filter": None},
                {"doc": '{"cruxeval_task": "output"}', "metrics": {"pass_rate": 1}, "filter": None},
                {"doc": '{"cruxeval_task": "output"}', "metrics": {"pass_rate": 0}, "filter": None},
            ]

    class SampleView:
        def __init__(self, _url):
            pass

        def scan(self, _table, **_kwargs):
            return SampleTable()

    monkeypatch.setattr(plot, "ReadView", SampleView)
    cell = plot.Cell("model", "cruxeval", 0.5, "s3://bucket/results")

    rewards, selector = plot.continuous_rewards(cell)
    count, basis = plot.scored_count(cell)

    assert rewards.tolist() == [1.0, 0.0]
    assert selector == "metrics:pass_rate:None"
    assert count == 2
    assert basis == "cruxeval_output_sample_rewards"


def test_tblite_fractional_rewards_use_observed_variance(monkeypatch) -> None:
    plot = load_plot()

    class SampleTable:
        def to_pylist(self, **_kwargs):
            return [{"metrics": {"reward": value}, "filter": None} for value in (0.0, 0.5, 1.0)]

    class SampleView:
        def __init__(self, _url):
            pass

        def scan(self, _table, **_kwargs):
            return SampleTable()

    class Filesystem:
        def exists(self, _path):
            return True

    monkeypatch.setattr(plot, "ReadView", SampleView)
    monkeypatch.setattr(plot, "filesystem_for", lambda url: (Filesystem(), url))
    monkeypatch.setattr(plot, "read_json", lambda _: {"status": "succeeded", "coverage": {"tblite": {"n_scored": 3}}})
    cell = plot.Cell("model", "ot-tblite-recovery", 0.5, "s3://bucket/results")
    result = plot.cell_statistics(cell, {})
    assert result["trial_count"] == 3
    assert result["score"] == 0.5
    assert result["tracker_score"] == 0.5
    assert result["raw_reward_mean"] == 0.5
    assert math.isclose(result["sem"], 1 / math.sqrt(12), abs_tol=1e-12)
    assert math.isclose(result["adjusted_sem"], 1 / math.sqrt(12 * 0.9), abs_tol=1e-12)


def test_tblite_early_seal_requires_ninety_percent_of_trials(monkeypatch) -> None:
    plot = load_plot()

    class SampleView:
        def __init__(self, _url):
            pass

        def is_sealed(self):
            return False

    class Filesystem:
        def exists(self, _path):
            return True

    monkeypatch.setattr(plot, "ReadView", SampleView)
    monkeypatch.setattr(plot, "filesystem_for", lambda url: (Filesystem(), url))
    monkeypatch.setattr(plot, "read_json", lambda _: {"status": "failed", "coverage": {}})
    cell = plot.Cell("model", "ot-tblite-recovery", 0.5, "s3://bucket/results")
    monkeypatch.setattr(plot, "continuous_rewards", lambda _: (plot.np.full(270, 0.5), "grading:reward:None"))

    count, basis = plot.scored_count(cell)

    assert count == 270
    assert basis == "early_sealed_sample_rewards"

    monkeypatch.setattr(plot, "continuous_rewards", lambda _: (plot.np.full(269, 0.5), "grading:reward:None"))
    try:
        plot.scored_count(cell)
    except ValueError as error:
        assert "unsealed results" in str(error)
    else:
        raise AssertionError("an early seal below 90% must not be accepted")


def test_comparison_benchmarks_uses_selection_panel() -> None:
    plot = load_plot()
    models = ["snowball", "baseline"]
    cells = {
        model: {
            benchmark: plot.Cell(model, benchmark, 0.5, f"s3://bucket/{model}/{benchmark}/results")
            for benchmark in ("included", "excluded")
        }
        for model in models
    }

    assert plot.comparison_benchmarks(["included", "excluded"], cells, models, {"benchmarks": ["included"]}) == [
        "included"
    ]
