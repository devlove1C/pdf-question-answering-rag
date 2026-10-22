from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from uuid import uuid4
import shutil

from models.schemas import QuestionResponse
from services.pdf_service import load_pdf_documents
from services.chunk_service import split_documents
from services.rag_service import generate_answer
from services.vector_service import (
    store_documents,
    search_similar_chunks
)


from models.schemas import (
    QuestionRequest,
    QuestionResponse
)

app = FastAPI(
    title="PDF Question Answering API",
    description="RAG API using FastAPI, LangChain and OpenAI",
    version="1.0.0"
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Welcome to PDF Question Answering API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF QA API"
    }


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    document_id = str(uuid4())

    file_path = UPLOAD_DIR / f"{document_id}.pdf"

    try:

        # Step 1: Save PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Extract PDF text
        documents = load_pdf_documents(str(file_path))

        if not any(doc.page_content.strip() for doc in documents):
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF"
            )

        # Step 3: Split PDF into chunks
        chunks = split_documents(documents)

        # Step 4: Generate embeddings and store in ChromaDB
        stored_chunks = store_documents(
            chunks=chunks,
            document_id=document_id,
            filename=file.filename
        )

        return {
            "message": "PDF uploaded and indexed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "stored_chunks": stored_chunks
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception as error:

        print(f"PDF processing error: {error}")

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Unable to process and index PDF"
        )

    finally:
        file.file.close()


@app.get("/search")
def search_pdf(
    document_id: str,
    question: str
):

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        results = search_similar_chunks(
            question=question,
            document_id=document_id,
            k=3
        )

        return {
            "question": question,
            "document_id": document_id,
            "total_results": len(results),
            "results": [
                {
                    "text": doc.page_content,
                    "page_number": doc.metadata.get("page", 0) + 1,
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "filename": doc.metadata.get("filename")
                }
                for doc in results
            ]
        }

    except Exception as error:

        print(f"Search error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to search PDF"
        )

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = generate_answer(
            question=request.question,
            document_id=request.document_id
        )

        return result

    except Exception as error:
        print(f"RAG error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate answer"
        )