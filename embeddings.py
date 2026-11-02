from functools import lru_cache

MODEL_NAME = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_model():
    # Vector search is optional; importing the API must not load ML libraries.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def embed_passages(texts: list[str]):
    return get_model().encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def embed_query(text: str):
    return get_model().encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
