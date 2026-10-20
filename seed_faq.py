from db import get_connection
from faq import FAQS


def main():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faq (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                );
            """)

            for item in FAQS:
                cursor.execute(
                    """
                    INSERT INTO faq (id, question, answer)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        question = EXCLUDED.question,
                        answer = EXCLUDED.answer;
                    """,
                    (
                        item["id"],
                        item["question"],
                        item["answer"],
                    ),
                )

            cursor.execute("SELECT COUNT(*) FROM faq;")
            count = cursor.fetchone()[0]

    print(f"Готово! FAQ в базе: {count}")


if __name__ == "__main__":
    main()