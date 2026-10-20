from psycopg.rows import dict_row

from db import get_connection
from deepseek_client import ask_deepseek


def main():
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT id, question, answer FROM faq ORDER BY id;"
            )
            faqs = cursor.fetchall()

    if not faqs:
        print("Сначала загрузи FAQ в базу.")
        return

    question = input("Твой вопрос: ").strip()

    if not question:
        print("Введи непустой вопрос.")
        return

    result = ask_deepseek(question, faqs)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()