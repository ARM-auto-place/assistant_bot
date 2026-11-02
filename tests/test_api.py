import httpx
import psycopg
import pytest

import main
from deepseek_client import AIAnswer
from faq import FAQS


def test_health_does_not_need_database_or_deepseek(client, monkeypatch):
    def unexpected():
        raise AssertionError("Health endpoint must not query the database")

    monkeypatch.setattr(main, "get_all_faqs", unexpected)
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_chat_returns_answer_sources_and_session(client, monkeypatch, faqs):
    calls = []

    def ask(message, knowledge):
        calls.append((message, knowledge))
        return AIAnswer(status="answer", answer="250 AMD в месяц.", source_ids=["promotion"])

    monkeypatch.setattr(main, "ask_deepseek", ask)
    response = client.post(
        "/chat", json={"message": "  Сколько стоит продвижение?  ", "session_id": "session-1"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "answer",
        "answer": "250 AMD в месяц.",
        "offer_operator": False,
        "reason": None,
        "source_ids": ["promotion"],
        "session_id": "session-1",
    }
    assert calls == [("Сколько стоит продвижение?", faqs)]
    assert "charset=utf-8" in response.headers["content-type"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": "hello", "session_id": ""},
        {"message": "hello", "session_id": "s" * 129},
    ],
)
def test_invalid_chat_input_is_rejected_before_dependencies(client, monkeypatch, body):
    def unexpected():
        raise AssertionError("Invalid input reached the database")

    monkeypatch.setattr(main, "get_all_faqs", unexpected)
    assert client.post("/chat", json=body).status_code == 422


def test_empty_knowledge_base_offers_operator_without_llm(client, monkeypatch):
    monkeypatch.setattr(main, "get_all_faqs", list)

    def unexpected(*args):
        raise AssertionError("Empty knowledge base reached the LLM")

    monkeypatch.setattr(main, "ask_deepseek", unexpected)
    result = client.post("/chat", json={"message": "hello"}).json()
    assert result["status"] == "handoff" and result["offer_operator"]
    assert result["reason"] == "knowledge_base_empty" and result["source_ids"] == []


def test_deepseek_handoff_never_claims_operator_connected(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "ask_deepseek",
        lambda *args: AIAnswer(
            status="handoff", answer="Хотите обратиться к оператору?", source_ids=[]
        ),
    )
    result = client.post("/chat", json={"message": "нужен оператор", "session_id": "abc"}).json()
    assert result["status"] == "handoff" and result["offer_operator"]
    assert result["source_ids"] == [] and result["session_id"] == "abc"


def test_database_outage_uses_embedded_faq(client, monkeypatch):
    def unavailable():
        raise psycopg.OperationalError("Database unavailable")

    monkeypatch.setattr(main, "get_all_faqs", unavailable)
    result = client.post("/chat", json={"message": FAQS[0]["question"]}).json()
    assert result["status"] == "answer" and result["answer"] == FAQS[0]["answer"]
    assert result["reason"] == "database_unavailable" and not result["offer_operator"]


@pytest.mark.parametrize(
    "error,reason",
    [
        (httpx.ConnectError("Offline"), "llm_unavailable"),
        (httpx.ReadTimeout("Timed out"), "llm_unavailable"),
        (
            httpx.HTTPStatusError(
                "Rate limit",
                request=httpx.Request("POST", "https://example.invalid"),
                response=httpx.Response(429),
            ),
            "llm_unavailable",
        ),
        (ValueError("Invalid response"), "llm_invalid_response_or_config"),
        (KeyError("choices"), "llm_invalid_response_or_config"),
        (IndexError("Empty choices"), "llm_invalid_response_or_config"),
        (TypeError("Wrong response type"), "llm_invalid_response_or_config"),
    ],
)
def test_llm_failures_fall_back_to_faq_or_operator(client, monkeypatch, faqs, error, reason):
    def fail(*args):
        raise error

    monkeypatch.setattr(main, "ask_deepseek", fail)
    found = client.post("/chat", json={"message": faqs[0]["question"]}).json()
    assert found["answer"] == faqs[0]["answer"] and found["source_ids"] == ["promotion"]
    assert found["status"] == "answer" and found["reason"] == reason
    unknown = client.post(
        "/chat", json={"message": "Неизвестный вопрос", "session_id": "abc"}
    ).json()
    assert unknown["status"] == "handoff" and unknown["offer_operator"]
    assert unknown["reason"] == reason and unknown["session_id"] == "abc"


@pytest.mark.parametrize("request_id", [0, 123, "req-1", None])
def test_rpc_echoes_id_and_returns_ai_type(client, request_id):
    result = client.post(
        "/rpc",
        json={"jsonrpc": "2.0", "id": request_id, "method": "chat", "params": {"message": "hello"}},
    ).json()
    assert result == {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"answer": "250 AMD в месяц.", "type_id": 1},
    }


@pytest.mark.parametrize(
    "payload,code",
    [
        (None, -32600),
        ([], -32600),
        (42, -32600),
        ({"jsonrpc": "1.0", "method": "chat", "id": 1}, -32600),
        ({"jsonrpc": "2.0", "method": 2, "id": 1}, -32600),
        ({"jsonrpc": "2.0", "method": "chat", "id": True}, -32600),
        ({"jsonrpc": "2.0", "method": "chat", "params": None, "id": 1}, -32600),
        ({"jsonrpc": "2.0", "method": "unknown", "id": 1}, -32601),
        ({"jsonrpc": "2.0", "method": "chat", "params": {}, "id": 1}, -32602),
        ({"jsonrpc": "2.0", "method": "chat", "params": ["hello"], "id": 1}, -32602),
        ({"jsonrpc": "2.0", "method": "chat", "params": {"message": " "}, "id": 1}, -32602),
        (
            {
                "jsonrpc": "2.0",
                "method": "chat",
                "params": {"message": "hello", "extra": "x"},
                "id": 1,
            },
            -32602,
        ),
    ],
)
def test_rpc_protocol_and_parameter_errors(client, payload, code):
    # Explicit encoding also tests JSON null, rather than an empty HTTP body.
    import json

    response = client.post(
        "/rpc", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize("body", [b"{", b"\xff", b'{"id": NaN}', b'{"id": Infinity}'])
def test_rpc_rejects_invalid_json(client, body):
    result = client.post("/rpc", content=body, headers={"Content-Type": "application/json"}).json()
    assert result["id"] is None and result["error"]["code"] == -32700


def test_rpc_notifications_return_no_content_and_batches_exclude_them(client):
    notification = {"jsonrpc": "2.0", "method": "chat", "params": {"message": "hello"}}
    response = client.post("/rpc", json=notification)
    assert response.status_code == 204 and response.content == b""
    response = client.post("/rpc", json=[notification, {**notification, "id": "a"}, 17])
    assert [item["id"] for item in response.json()] == ["a", None]
    assert response.json()[1]["error"]["code"] == -32600
    assert client.post("/rpc", json=[notification, notification]).status_code == 204


def test_rpc_internal_error_does_not_expose_exception_or_secret(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("private-database-password")

    monkeypatch.setattr(main, "chat", fail)
    response = client.post(
        "/rpc", json={"jsonrpc": "2.0", "id": 7, "method": "chat", "params": {"message": "hello"}}
    )
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32603, "message": "Internal error"},
    }
    assert "private-database-password" not in response.text


@pytest.mark.parametrize(
    "status,reason,type_id",
    [
        ("answer", None, 1),
        ("answer", "llm_unavailable", 2),
        ("handoff", "unknown", 3),
        ("clarification", None, 4),
    ],
)
def test_backend_answer_type_contract(status, reason, type_id):
    result = main.ChatResponse(
        status=status, answer="Ответ", offer_operator=status == "handoff", reason=reason
    )
    assert int(main.get_answer_type(result)) == type_id
