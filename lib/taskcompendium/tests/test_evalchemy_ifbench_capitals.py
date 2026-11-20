# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IFBench's European capitals checker accepts the complete list north of 45 degrees, which includes Kyiv.

The IFBench checkers import nltk, emoji and syllapy, which IFBench's grading environment provides, so the cases skip
without them.
"""

import pytest

from taskcompendium.importers.evalchemy.scorers.instruction_spec import InstructionProfile
from taskcompendium.importers.evalchemy.scorers.instructions import source_registry

CAPITALS = (
    "Reykjavík, Helsinki, Oslo, Tallinn, Stockholm, Riga, Moscow, Copenhagen, Vilnius, Minsk, Dublin, Berlin, "
    "Amsterdam, Warsaw, London, Brussels, Kyiv, Prague, Luxembourg, Paris, Vienna, Bratislava, Budapest, Vaduz, "
    "Chișinău, Bern, Ljubljana, Zagreb"
)


@pytest.mark.parametrize(
    ("response", "passes"),
    [(CAPITALS, True), (CAPITALS.replace("Kyiv, ", ""), False)],
    ids=["all-28", "without-kyiv"],
)
def test_european_capitals_checker_requires_kyiv(response, passes):
    for module in ("emoji", "syllapy", "nltk"):
        pytest.importorskip(module, reason="Requires the IFBench grading environment")
    instruction_id = "custom:european_capitals_sort"
    checker = source_registry(InstructionProfile.IFBENCH)[instruction_id](instruction_id)
    checker.build_description()
    assert checker.check_following(response) is passes
