from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from db import get_connection
from embeddings import embed_passages, VECTOR_SIZE


def main():
    # Сначала читаем FAQ и закрываем подключение.
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT id, question, answer FROM faq ORDER BY id;"
            )
            faqs = cursor.fetchall()

    if not faqs:
        print("В базе нет FAQ. Сначала запусти seed_faq.py.")
        return

    question_variants = {
        "create_listing": [
            "Как разместить объявление?",
            "Хочу выставить машину на продажу",
            "Как добавить автомобиль на сайт?",
            "Как опубликовать объявление о продаже машины?",
        ],
        "listing_price": [
            "Сколько стоит размещение?",
            "Надо платить за объявление?",
            "Можно разместить объявление бесплатно?",
            "Платное продвижение объявления",
        ],
        "edit_delete_listing": [
            "Как отредактировать или удалить объявление?",
            "Как убрать моё объявление?",
            "Как изменить данные в объявлении?",
            "Где удалить объявление о продаже машины?",
        ],
        "contact_seller": [
            "Как связаться с продавцом?",
            "Где найти контакты продавца?",
            "Как написать владельцу машины?",
            "Как позвонить продавцу автомобиля?",
        ],
        "contact_support": [
            "Как обратиться в поддержку?",
            "Как связаться со службой поддержки?",
            "Мне нужна помощь оператора",
        ],
    }

    texts = [
        "\n".join(
            question_variants.get(faq["id"], [faq["question"]])
        )
        for faq in faqs
    ]

    print("Загружаем модель и вычисляем векторы...")
    vectors = embed_passages(texts)

    if vectors.shape != (len(faqs), VECTOR_SIZE):
        raise ValueError(
            f"Неожиданный размер векторов: {vectors.shape}"
        )

    # Записываем результаты одной транзакцией.
    with get_connection() as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute("""
                ALTER TABLE faq
                ADD COLUMN IF NOT EXISTS embedding vector(384);
            """)

            for faq, vector in zip(faqs, vectors, strict=True):
                cursor.execute(
                    """
                    UPDATE faq
                    SET embedding = %s
                    WHERE id = %s;
                    """,
                    (vector, faq["id"]),
                )

    print(f"Готово! Записано векторов: {len(faqs)}")


if __name__ == "__main__":
    main()