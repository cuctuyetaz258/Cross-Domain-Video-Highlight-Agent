import json
from types import SimpleNamespace

import pytest

from highlight_agent.llm.client import (
    LLMClientConfig,
    LLMProviderError,
    OpenAICompatibleAssessmentClient,
)
from highlight_agent.schemas import CandidateTranscriptContext


def _context() -> CandidateTranscriptContext:
    return CandidateTranscriptContext(
        candidate_id="c1",
        start_time=0,
        end_time=30,
        ltr_score=8,
        before="",
        core="[0.000-30.000] Main idea.",
        after="",
    )


def _response_json() -> str:
    return json.dumps(
        {
            "assessments": [
                {
                    "candidate_id": "c1",
                    "semantic_relevance": 0.9,
                    "standalone_value": 0.8,
                    "completeness": 0.9,
                    "hook_strength": 0.7,
                    "shareability": 0.8,
                    "title": "Main idea",
                    "summary": "A complete explanation.",
                    "evidence": "Main idea.",
                    "suggested_start_time": None,
                    "suggested_end_time": None,
                    "risk_flags": [],
                }
            ]
        }
    )


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_response_json()))]
        )


def test_openai_provider_uses_strict_json_schema() -> None:
    client = OpenAICompatibleAssessmentClient(
        LLMClientConfig(provider="openai", model="test-model", api_key="test-key")
    )
    completions = _FakeCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.assess([_context()], domain="lecture")

    assert result.assessments[0].candidate_id == "c1"
    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_groq_provider_uses_compatible_json_mode() -> None:
    client = OpenAICompatibleAssessmentClient(
        LLMClientConfig(
            provider="groq",
            model="test-model",
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
        )
    )
    completions = _FakeCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    client.assess([_context()], domain="lecture")

    assert completions.kwargs["response_format"] == {"type": "json_object"}


def test_config_does_not_accept_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        LLMClientConfig.from_env(provider="openai")


def test_malformed_provider_response_becomes_fallback_safe_error() -> None:
    client = OpenAICompatibleAssessmentClient(
        LLMClientConfig(provider="openai", model="test-model", api_key="test-key")
    )

    class MalformedCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
            )

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=MalformedCompletions())
    )

    with pytest.raises(LLMProviderError, match="assessment failed"):
        client.assess([_context()], domain="lecture")
