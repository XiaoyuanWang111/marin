# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Snowball/Grug tournament selection."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

PAIRWISE_PATH = Path(__file__).with_name("pairwise.py")
SPEC = importlib.util.spec_from_file_location("campaign_pairwise", PAIRWISE_PATH)
assert SPEC is not None and SPEC.loader is not None
pairwise = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pairwise
SPEC.loader.exec_module(pairwise)


def statistic(
    model: str,
    benchmark: str,
    score: float,
    adjusted_sem: float,
    tracker_score: float | None = None,
) -> dict[str, object]:
    return {
        "model": model,
        "benchmark": benchmark,
        "score": score,
        "tracker_score": score if tracker_score is None else tracker_score,
        "adjusted_sem": adjusted_sem,
    }


def reply_statistic(model: str, benchmark: str, rewards: dict[str, list[float]]) -> dict[str, object]:
    """The published cell for these 0/1 rewards: reply mean and the per-reply binary SEM with the policy factor."""
    replies = [reward for values in rewards.values() for reward in values]
    score = sum(replies) / len(replies)
    return statistic(model, benchmark, score, math.sqrt(score * (1 - score) / (len(replies) - 1) / 0.9))


def shared_item_benchmarks(
    seed: int, draws: int, items: int, replies: int, beta: float, loss: float
) -> tuple[list[str], dict, dict]:
    """Benchmarks on which two checkpoints answer the same items; B solves each item `loss` less often, relatively.

    Item solve rates are Beta(beta, beta), so item difficulty dominates and the checkpoints' item scores correlate.
    """
    rng = np.random.default_rng(seed)
    benchmarks = [f"draw-{index}" for index in range(draws)]
    rewards: dict[tuple[str, str], dict[str, list[float]]] = {}
    for benchmark in benchmarks:
        rates = rng.beta(beta, beta, size=items)
        for model, model_rates in (("Grug-A", rates), ("Grug-B", rates * (1 - loss))):
            solved = rng.random((items, replies)) < model_rates[:, None]
            rewards[(model, benchmark)] = {f"item-{item}": row.astype(float).tolist() for item, row in enumerate(solved)}
    statistics = {key: reply_statistic(*key, values) for key, values in rewards.items()}
    return benchmarks, statistics, rewards


def test_pairwise_outcomes_use_adjusted_difference_confidence_interval() -> None:
    candidates = ["Grug-A", "Grug-B", "Snowball-C"]
    benchmarks = ["clear", "uncertain"]
    statistics = {
        ("Grug-A", "clear"): statistic("Grug-A", "clear", 0.8, 0.02),
        ("Grug-B", "clear"): statistic("Grug-B", "clear", 0.7, 0.02),
        ("Snowball-C", "clear"): statistic("Snowball-C", "clear", 0.6, 0.02),
        ("Grug-A", "uncertain"): statistic("Grug-A", "uncertain", 0.55, 0.04),
        ("Grug-B", "uncertain"): statistic("Grug-B", "uncertain", 0.5, 0.04),
        ("Snowball-C", "uncertain"): statistic("Snowball-C", "uncertain", 0.45, 0.04),
    }

    outcomes, pairs, summary = pairwise.pairwise_rows(benchmarks, candidates, statistics, {})

    assert [row["outcome"] for row in outcomes] == ["left", "tie", "left", "tie", "left", "tie"]
    assert [(row["model"], row["mean_pairwise_win_rate"]) for row in summary] == [
        ("Grug-A", 0.75),
        ("Grug-B", 0.5),
        ("Snowball-C", 0.25),
    ]
    assert next(row for row in pairs if row["model"] == "Grug-A" and row["opponent"] == "Grug-B") == {
        "model": "Grug-A",
        "opponent": "Grug-B",
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "benchmarks": 2,
        "win_rate": 0.75,
    }


def test_pairwise_outcomes_use_full_precision_score_not_tracker_display() -> None:
    candidates = ["Grug-Linear", "Grug-Recovery"]
    benchmarks = ["tau3-pi"]
    statistics = {
        ("Grug-Linear", "tau3-pi"): statistic("Grug-Linear", "tau3-pi", 0.05, 0.012477446057244202, tracker_score=0.05),
        ("Grug-Recovery", "tau3-pi"): statistic(
            "Grug-Recovery", "tau3-pi", 0.020710059171597635, 0.00817730339839839, tracker_score=0.021
        ),
    }

    outcomes, _, summary = pairwise.pairwise_rows(benchmarks, candidates, statistics, {})

    rounded_difference = 0.05 - 0.021
    threshold = 1.96 * (0.012477446057244202**2 + 0.00817730339839839**2) ** 0.5
    assert rounded_difference < threshold < float(outcomes[0]["score_difference"])
    assert outcomes[0]["outcome"] == "left"
    assert summary[0]["model"] == "Grug-Linear"


def test_candidate_panel_excludes_nonselective_benchmarks_and_requires_complete_scores(tmp_path: Path) -> None:
    tracker = tmp_path / "TRACKER.md"
    tracker.write_text(
        "| Model | math500 | financebench | piqa | triviaqa | tau3-pi |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Metric | accuracy | accuracy | accuracy | accuracy | accuracy |\n"
        "| open-athena/Snowball-A | 0.900 (s3://bucket/a-math/results) | CRASHED | CRASHED | CRASHED | "
        "0.500 (s3://bucket/a-tau/results) |\n"
        "| open-athena/Grug-B | 0.800 (s3://bucket/b-math/results) | CRASHED | CRASHED | CRASHED | RUNNING |\n"
        "| Qwen/baseline | 0.990 (s3://bucket/qwen-math/results) | CRASHED | CRASHED | CRASHED | "
        "0.990 (s3://bucket/qwen-tau/results) |\n"
    )

    try:
        pairwise.candidate_panel(tracker)
    except ValueError as error:
        assert "open-athena/Grug-B / tau3-pi" in str(error)
        assert "financebench" not in str(error)
    else:
        raise AssertionError("an incomplete included benchmark must fail the tournament")


def test_top_two_model_card_reports_direct_result_and_exclusions(tmp_path: Path) -> None:
    candidates = ["open-athena/Grug-Champion", "open-athena/Grug-Runner-Up"]
    benchmarks = ["math500"]
    statistics = {
        (candidates[0], "math500"): statistic(candidates[0], "math500", 0.8, 0.02),
        (candidates[1], "math500"): statistic(candidates[1], "math500", 0.7, 0.02),
    }
    outcomes, pairs, summary = pairwise.pairwise_rows(benchmarks, candidates, statistics, {})
    cells = {
        model: {
            benchmark: pairwise.critical_difference.Cell(
                model, benchmark, score, f"s3://bucket/{model}/{benchmark}/results"
            )
            for benchmark, score in {
                "math500": 0.8 if model == candidates[0] else 0.7,
                "financebench": 0.5,
                "piqa": 0.6,
                "triviaqa": 0.4,
            }.items()
        }
        for model in candidates
    }

    pairwise.write_top_two_model_card(
        tmp_path,
        benchmarks,
        cells,
        outcomes,
        pairs,
        summary,
        statistics,
        "tracker-hash",
        "recovered-hash",
    )

    report = (tmp_path / "TOP_TWO_MODEL_CARD.md").read_text()
    assert "**1-0-0**" in report
    assert "| math500 | 0.800 [0.761, 0.839] | 0.700 [0.661, 0.739] | +10.0 pp | Champion |" in report
    assert "## Benchmarks excluded from selection" in report
    assert (tmp_path / "top_two_head_to_head.csv").exists()


def test_paired_outcome_counts_repeated_replies_once_per_item_at_full_precision() -> None:
    candidates = ["Grug-A", "Grug-B"]
    rewards = {
        ("Grug-A", "aime24"): {"p1": [1.0, 1.0], "p2": [1.0, 0.0], "p3": [0.0, 0.0]},
        ("Grug-B", "aime24"): {"p1": [1.0, 0.0], "p2": [0.0, 0.0], "p3": [0.0, 0.0]},
    }
    # Cells carry the three-decimal display scores; the paired path reads the rewards themselves.
    statistics = {
        ("Grug-A", "aime24"): statistic("Grug-A", "aime24", 0.5, 0.2),
        ("Grug-B", "aime24"): statistic("Grug-B", "aime24", 0.167, 0.2),
    }

    outcomes, _, _ = pairwise.pairwise_rows(["aime24"], candidates, statistics, rewards)

    # Item differences 1/2, 1/2, 0 have mean 1/3 and standard error 1/6 over three items; the six reply pairs taken
    # as independent would give 0.211 over five degrees of freedom.
    assert outcomes[0]["difference_basis"] == "paired"
    assert outcomes[0]["paired_items"] == 3
    assert math.isclose(outcomes[0]["score_difference"], 1 / 3)
    assert math.isclose(outcomes[0]["difference_95_threshold"], student_t.ppf(0.975, 2) / 6 / math.sqrt(0.9))
    assert outcomes[0]["outcome"] == "tie"


def test_paired_rule_on_same_distribution_checkpoints_decides_at_its_nominal_rate() -> None:
    draws = 2000
    benchmarks, statistics, rewards = shared_item_benchmarks(0, draws, items=100, replies=4, beta=0.5, loss=0.0)

    outcomes, _, _ = pairwise.pairwise_rows(benchmarks, ["Grug-A", "Grug-B"], statistics, rewards)

    false_decisions = sum(row["outcome"] != "tie" for row in outcomes) / draws
    # A two-sided 95% t interval widened by the policy factor 1 / sqrt(0.9) errs 3.9% of the time at 99 degrees of
    # freedom (5% without the factor).
    nominal = 2 * student_t.sf(student_t.ppf(0.975, 99) / math.sqrt(0.9), 99)
    assert all(row["difference_basis"] == "paired" for row in outcomes)
    assert abs(false_decisions - nominal) < 4 * math.sqrt(nominal * (1 - nominal) / draws)


def test_paired_rule_detects_planted_difference_more_often_than_unpaired_rule() -> None:
    draws = 500
    benchmarks, statistics, rewards = shared_item_benchmarks(1, draws, items=200, replies=1, beta=0.3, loss=0.25)

    paired, _, _ = pairwise.pairwise_rows(benchmarks, ["Grug-A", "Grug-B"], statistics, rewards)
    unpaired, _, _ = pairwise.pairwise_rows(benchmarks, ["Grug-A", "Grug-B"], statistics, {})

    # B solves each item 25% less often: 12.5 points on average, about 3.5 paired standard errors.
    paired_power = sum(row["outcome"] == "left" for row in paired) / draws
    unpaired_power = sum(row["outcome"] == "left" for row in unpaired) / draws
    assert all(row["difference_basis"] == "unpaired" for row in unpaired)
    assert paired_power > 0.9
    assert paired_power > unpaired_power + 0.1


def test_outcomes_without_rewards_on_the_same_items_use_unpaired_rule_and_say_so() -> None:
    candidates = ["Grug-A", "Grug-B", "Grug-C"]
    rewards = {
        ("Grug-A", "math500"): {"q1": [1.0], "q2": [1.0], "q3": [0.0]},
        ("Grug-B", "math500"): {"q1": [1.0], "q2": [0.0], "q4": [0.0]},
    }
    statistics = {
        ("Grug-A", "math500"): statistic("Grug-A", "math500", 2 / 3, 0.3),
        ("Grug-B", "math500"): statistic("Grug-B", "math500", 1 / 3, 0.3),
        ("Grug-C", "math500"): statistic("Grug-C", "math500", 0.0, 0.1),
    }

    outcomes, _, _ = pairwise.pairwise_rows(["math500"], candidates, statistics, rewards)

    # A and B scored different items; C has no per-item rewards.
    assert [(row["difference_basis"], row["paired_items"]) for row in outcomes] == [("unpaired", 0)] * 3
    assert math.isclose(outcomes[0]["score_difference"], 1 / 3)
    assert math.isclose(outcomes[0]["difference_95_threshold"], 1.96 * math.hypot(0.3, 0.3))
