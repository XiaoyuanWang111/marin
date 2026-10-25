# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skyrl_gym.envs.nemotron_ultra.calendar import grade_calendar

EXPECTED = {"1": {"duration": 30, "min_time": "9am", "max_time": "5pm", "constraint": None}}


@pytest.mark.parametrize(
    "start_time,grade",
    [("10am", (1.0, "pass")), (None, (0.0, "invalid_response")), (600, (0.0, "invalid_response"))],
    ids=["text", "null", "number"],
)
def test_calendar_scores_a_start_time_by_whether_it_is_text(start_time, grade):
    response = json.dumps([{"event_id": 1, "start_time": start_time, "duration": 30}])

    assert grade_calendar(response, EXPECTED) == grade
