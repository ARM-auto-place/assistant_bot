from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from db import get_connection
from embeddings import embed_query


def search(message: str):
    vector = embed_query(message)

    with get_connection() as connection:
        register_vector(connection)

        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    question,
                    1 - (embedding <=> %s) AS similarity
                FROM faq
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT 3;
                """,
                (vector, vector),
            )
            return cursor.fetchall()


def main():
    questions = [
        "Где разместить информацию о машине, которую я продаю?",
        "С меня возьмут деньги за публикацию?",
        "Машину уже продал, как снять публикацию?",
        "Хочу обсудить покупку с владельцем, где его найти?",
        "Сколько стоит доставка машины?",
        "Какая завтра погода?",
        "Сколько стоит платное продвижение?",
    ]

    for question in questions:
        print(f"\nВопрос: {question}")

        for result in search(question):
            print(
                f"  {result['similarity']:.4f}"
                f" | {result['id']}"
                f" | {result['question']}"
            )


if __name__ == "__main__":
    main()