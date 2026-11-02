import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

import main
from deepseek_client import AIAnswer


@pytest.fixture(autouse=True)
def block_external_services(monkeypatch):
    """An accidentally unmocked dependency must fail, never spend API credits."""

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not contact external services")

    async def blocked_async(*args, **kwargs):
        raise AssertionError("Tests must not contact external services")

    monkeypatch.setattr(psycopg, "connect", blocked)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ci-test-key-not-a-real-secret")


@pytest.fixture
def faqs():
    return [
        {"id": "promotion", "question": "Сколько стоит продвижение?", "answer": "250 AMD в месяц."}
    ]


@pytest.fixture
def client(monkeypatch, faqs):
    monkeypatch.setattr(main, "get_all_faqs", lambda: faqs)
    monkeypatch.setattr(
        main,
        "ask_deepseek",
        lambda message, knowledge: AIAnswer(
            status="answer", answer="250 AMD в месяц.", source_ids=["promotion"]
        ),
    )
    with TestClient(main.app) as test_client:
        yield test_client
