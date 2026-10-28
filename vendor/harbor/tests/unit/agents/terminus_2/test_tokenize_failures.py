import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from tenacity import wait_none

from harbor.agents.terminus_2 import terminus_2
from harbor.llms.base import ContextManagementInfrastructureError


@pytest.mark.parametrize("failure", ["timeout", 429, 503])
async def test_token_count_recovers_without_local_tokenizer(monkeypatch, failure):
    requests = []

    def tokenize(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json={"count": 17})

    agent = object.__new__(terminus_2.Terminus2)
    agent._llm = SimpleNamespace(_api_base="http://model/v1")
    agent.model_alias = "hosted_vllm/model"
    agent._token_encoding = None
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hello"}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(tokenize)) as client:
        monkeypatch.setattr(terminus_2, "shared_async_tokenize_client", lambda: client)
        monkeypatch.setattr(terminus_2._server_token_count.retry, "wait", wait_none())
        assert await agent._acount_total_tokens(chat) == 17

    assert len(requests) == 2
    assert requests[0].content == requests[1].content


@pytest.mark.parametrize(("status", "attempts"), [(503, 3), (400, 1), (404, 1)])
async def test_tokenizer_failure_preserves_http_cause(monkeypatch, status, attempts):
    requests = []

    def tokenize(request):
        requests.append(request)
        return httpx.Response(status)

    agent = object.__new__(terminus_2.Terminus2)
    agent._llm = SimpleNamespace(_api_base="http://model/v1")
    agent.model_alias = "hosted_vllm/model"
    agent._token_encoding = None
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hello"}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(tokenize)) as client:
        monkeypatch.setattr(terminus_2, "shared_async_tokenize_client", lambda: client)
        monkeypatch.setattr(terminus_2._server_token_count.retry, "wait", wait_none())
        with pytest.raises(ContextManagementInfrastructureError) as error:
            await agent._acount_total_tokens(chat)

    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)
    assert error.value.__cause__.response.status_code == status
    assert len(requests) == attempts


async def test_context_unwind_pending_count_lets_other_trials_run(monkeypatch):
    events = []
    other_trial_ran = asyncio.Event()

    async def tokenize(request):
        events.append("count sent")
        await other_trial_ran.wait()
        events.append("count answered")
        return httpx.Response(200, json={"count": 1})

    async def other_trial():
        events.append("other trial ran")
        other_trial_ran.set()

    agent = object.__new__(terminus_2.Terminus2)
    agent._llm = SimpleNamespace(
        _api_base="http://model/v1", get_model_context_limit=lambda: 10
    )
    agent.model_alias = "hosted_vllm/model"
    agent.logger = logging.getLogger("test-tokenize-failures")
    chat = SimpleNamespace(
        messages=[{"role": "user", "content": "hello"}],
        reset_response_chain=lambda: None,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(tokenize)) as client:
        monkeypatch.setattr(terminus_2, "shared_async_tokenize_client", lambda: client)
        await asyncio.wait_for(
            asyncio.gather(
                agent._unwind_messages_to_free_tokens(chat, target_free_tokens=1),
                other_trial(),
            ),
            timeout=5,
        )

    assert events[:3] == ["count sent", "other trial ran", "count answered"]
