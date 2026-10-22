from langchain_chroma import Chroma
from langchain_core.documents import Document

from services.embedding_service import get_embedding_model


CHROMA_DB_PATH = "./chroma_db"

COLLECTION_NAME = "pdf_documents"


def get_vector_store() -> Chroma:

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    return vector_store


def store_documents(
    chunks: list[Document],
    document_id: str,
    filename: str
) -> int:

    if not chunks:
        return 0

    vector_store = get_vector_store()

    ids = []

    for index, chunk in enumerate(chunks):

        chunk.metadata["document_id"] = document_id
        chunk.metadata["filename"] = filename

        chunk_id = f"{document_id}_chunk_{index}"

        chunk.metadata["chunk_id"] = chunk_id

        ids.append(chunk_id)

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return len(chunks)


def search_similar_chunks(
    question: str,
    document_id: str,
    k: int = 3
) -> list[Document]:

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=question,
        k=k,
        filter={
            "document_id": document_id
        }
    )

    return results