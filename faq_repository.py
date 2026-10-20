from psycopg.rows import dict_row

from db import get_connection
from faq import normalize_question


def find_faq_in_db(message: str) -> dict[str, str] | None:
    normalized = normalize_question(message)

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT id, question, answer FROM faq;")
            faqs = cursor.fetchall()

    for faq in faqs:
        if normalize_question(faq["question"]) == normalized:
            return faq

    return None

def get_all_faqs() -> list[dict]:
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT id, question, answer FROM faq ORDER BY id;"
            )
            return cursor.fetchall()