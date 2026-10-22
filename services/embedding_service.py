from langchain_openai import OpenAIEmbeddings

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


def get_embedding_model() -> OpenAIEmbeddings:

    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_actual_openai_api_key":
        raise ValueError("Configure a valid OPENAI_API_KEY in your .env file")

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    return embeddings