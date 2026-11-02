import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import db
import embeddings
import faq_repository
from faq import FAQS, find_faq, normalize_question


@pytest.mark.parametrize(
    "question", ["  КАК РАЗМЕСТИТЬ ОБЪЯВЛЕНИЕ?!  ", "Как  разместить   объявление."]
)
def test_faq_match_normalizes_case_spaces_and_trailing_punctuation(question):
    assert find_faq(question)["id"] == "create_listing"


def test_unknown_question_does_not_invent_an_answer():
    assert find_faq("Сколько стоит доставка автомобиля на Марс?") is None
    assert normalize_question("  Привет   МИР! ") == "привет мир"
    assert len({item["id"] for item in FAQS}) == len(FAQS)


def test_database_configuration_uses_environment_and_timeout(monkeypatch):
    values = {
        "POSTGRES_HOST": "db.test",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "faq",
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "not-a-real-password",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    connect = MagicMock(return_value=object())
    monkeypatch.setattr(db.psycopg, "connect", connect)
    connection = db.get_connection()
    assert connection is connect.return_value
    connect.assert_called_once_with(
        host="db.test",
        port=5432,
        dbname="faq",
        user="test",
        password="not-a-real-password",
        connect_timeout=5,
    )


def test_repository_returns_ordered_rows_and_matches_normalized_question(monkeypatch, faqs):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = faqs
    monkeypatch.setattr(faq_repository, "get_connection", lambda: connection)
    connection.__enter__.return_value = connection
    assert faq_repository.get_all_faqs() == faqs
    sql = cursor.execute.call_args.args[0]
    assert "ORDER BY id" in sql
    assert faq_repository.find_faq_in_db("  СКОЛЬКО СТОИТ ПРОДВИЖЕНИЕ?! ") == faqs[0]
    assert faq_repository.find_faq_in_db("unknown") is None
    assert connection.__exit__.call_count == 3


def test_embedding_prefixes_normalization_and_model_cache_without_download(monkeypatch):
    model = MagicMock()
    model.encode.return_value = [0.1, 0.2]
    factory = MagicMock(return_value=model)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory)
    )
    embeddings.get_model.cache_clear()
    try:
        embeddings.embed_passages(["hello", "world"])
        model.encode.assert_called_with(
            ["passage: hello", "passage: world"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embeddings.embed_query("question")
        model.encode.assert_called_with(
            "query: question",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        factory.assert_called_once_with("intfloat/multilingual-e5-small", device="cpu")
        assert embeddings.VECTOR_SIZE == 384
    finally:
        embeddings.get_model.cache_clear()
