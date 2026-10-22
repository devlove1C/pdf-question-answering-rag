from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_documents(file_path: str) -> list[Document]:
    """
    Load PDF pages as LangChain Document objects.
    """

    loader = PyPDFLoader(file_path)

    return loader.load()   