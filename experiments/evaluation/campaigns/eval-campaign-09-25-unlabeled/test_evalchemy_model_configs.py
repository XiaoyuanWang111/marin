# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy requests built from the campaign's Grug and Snowball model configs."""

import json
import re
from pathlib import Path

import pytest
import yaml
from marin.evaluation.evalchemy.client import build_command
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _run_config_json
from marin.evaluation.model_config import load_model_config
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.evaluation.evals import EvalchemyDefinition

CAMPAIGN_DIR = Path(__file__).parent
EVALCHEMY_CONFIG_DIR = CAMPAIGN_DIR / "evalchemy-configs"
GRUG_LOCATION = re.compile(r"grug|snowball", re.IGNORECASE)


def _evalchemy_model_configs() -> list[Path]:
    """Model configs the launcher pairs with Evalchemy configs, including per-benchmark overrides."""
    campaign = yaml.safe_load((CAMPAIGN_DIR / "campaign.yaml").read_text())
    evalchemy_names = {path.stem for path in EVALCHEMY_CONFIG_DIR.glob("*.yaml")}
    overrides = [
        CAMPAIGN_DIR / override
        for by_eval in campaign["model_config_overrides"].values()
        for eval_name, override in by_eval.items()
        if eval_name in evalchemy_names
    ]
    return [*sorted((CAMPAIGN_DIR / "model-configs").glob("*.yaml")), *overrides]


GRUG_MODEL_CONFIGS = [
    path for path in _evalchemy_model_configs() if GRUG_LOCATION.search(load_model_config(path).location)
]


@pytest.mark.parametrize("model_path", GRUG_MODEL_CONFIGS, ids=lambda path: str(path.relative_to(CAMPAIGN_DIR)))
def test_grug_evalchemy_generation_requests_keep_think_delimiters(model_path, monkeypatch):
    # vLLM's default skip_special_tokens=true removes <|start_think|> and <|end_think|>, so a grader that cuts at
    # the last <|end_think|> would read the reasoning in front of the answer.
    # The evaluator package is not installed beside Marin's tests; the generation budget is not under test.
    monkeypatch.setattr("marin.evaluation.evalchemy.client.is_evalchemy_benchmark", lambda name: True)
    model = load_model_config(model_path)
    served = RunningModel(
        endpoint=OpenAIEndpoint(base_url="http://127.0.0.1:8000/v1", model=model.location), tokenizer=model.location
    )
    for config_path in sorted(EVALCHEMY_CONFIG_DIR.glob("*.yaml")):
        definition = EvalchemyDefinition(name=config_path.stem, config_path=config_path)
        run = definition.config_for(load_evalchemy_config(config_path), model, None)
        client_config = json.loads(_run_config_json(served, run, "gs://bucket/evals/campaign"))
        for task in client_config["tasks"]:
            if not task["generation"]:
                continue
            command = build_command(client_config, task, "/opt/py", model.serve.max_model_len)
            gen_kwargs = command[command.index("--gen_kwargs") + 1].split(",")
            assert "skip_special_tokens=false" in gen_kwargs, (config_path.stem, task["name"])
