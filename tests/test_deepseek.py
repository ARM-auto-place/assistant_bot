import json

import httpx
import pytest
from pydantic import ValidationError

import deepseek_client


def install_transport(monkeypatch, answer=None, *, status=200, finish="stop", content=None):
    requests = []
    client_class = httpx.Client

    def handler(request):
        requests.append(request)
        body = content if content is not None else json.dumps(answer, ensure_ascii=False)
        return httpx.Response(
            status, json={"choices": [{"finish_reason": finish, "message": {"content": body}}]}
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        deepseek_client.httpx,
        "Client",
        lambda **kwargs: client_class(transport=transport, **kwargs),
    )
    return requests


def test_request_has_grounding_auth_and_no_conversation_history(monkeypatch, faqs):
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    requests = install_transport(
        monkeypatch, {"status": "answer", "answer": "250 AMD", "source_ids": ["promotion"]}
    )
    first = deepseek_client.ask_deepseek("Первый вопрос", faqs)
    deepseek_client.ask_deepseek("Второй вопрос", faqs)
    assert first.answer == "250 AMD" and first.source_ids == ["promotion"]
    assert len(requests) == 2
    for request, question in zip(requests, ["Первый вопрос", "Второй вопрос"], strict=True):
        assert request.headers["Authorization"] == "Bearer ci-test-key-not-a-real-secret"
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model" and payload["stream"] is False
        assert payload["response_format"] == {"type": "json_object"}
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]
        assert json.loads(payload["messages"][1]["content"]) == {"question": question, "faqs": faqs}


@pytest.mark.parametrize("key", ["", "  ", "сюда_вставь_свой_ключ"])
def test_missing_api_key_fails_before_network(monkeypatch, faqs, key):
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        deepseek_client.ask_deepseek("hello", faqs)


@pytest.mark.parametrize(
    "answer",
    [
        {"status": "answer", "answer": "250 AMD", "source_ids": []},
        {"status": "answer", "answer": "250 AMD", "source_ids": ["invented-source"]},
        {"status": "answer", "answer": " ", "source_ids": ["promotion"]},
        {"status": "answer", "answer": "x" * 3001, "source_ids": ["promotion"]},
        {"status": "clarification", "answer": "hello", "source_ids": []},
        {"status": "answer", "answer": "250 AMD", "source_ids": ["promotion"], "extra": "injected"},
    ],
)
def test_unverified_or_invalid_model_answers_are_rejected(monkeypatch, faqs, answer):
    install_transport(monkeypatch, answer)
    with pytest.raises((ValueError, ValidationError)):
        deepseek_client.ask_deepseek("hello", faqs)


def test_handoff_clears_sources(monkeypatch, faqs):
    install_transport(
        monkeypatch,
        {"status": "handoff", "answer": "Предлагаю оператора", "source_ids": ["promotion"]},
    )
    result = deepseek_client.ask_deepseek("hello", faqs)
    assert result.status == "handoff" and result.source_ids == []


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_truncated_or_filtered_answers_are_rejected(monkeypatch, faqs, finish):
    install_transport(
        monkeypatch,
        {"status": "answer", "answer": "250 AMD", "source_ids": ["promotion"]},
        finish=finish,
    )
    with pytest.raises(ValueError, match="не завершил"):
        deepseek_client.ask_deepseek("hello", faqs)


def test_malformed_json_is_rejected(monkeypatch, faqs):
    install_transport(monkeypatch, content="not JSON")
    with pytest.raises(ValidationError):
        deepseek_client.ask_deepseek("hello", faqs)


def test_http_error_is_propagated_to_api_fallback(monkeypatch, faqs):
    install_transport(monkeypatch, {}, status=503)
    with pytest.raises(httpx.HTTPStatusError):
        deepseek_client.ask_deepseek("hello", faqs)
