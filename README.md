# PDF Question-Answering API using RAG, LangChain & OpenAI

A Retrieval-Augmented Generation (RAG) application built with Python, FastAPI, LangChain, OpenAI, and ChromaDB.

The application allows users to upload PDF documents, extract and split their content, generate vector embeddings, and ask natural-language questions about the uploaded documents.

It retrieves relevant document chunks using semantic similarity search and generates context-aware answers using OpenAI GPT, along with references to the original PDF pages.

## Features

- Upload PDF documents using FastAPI.
- Extract PDF text using LangChain's `PyPDFLoader`.
- Split documents into chunks using `RecursiveCharacterTextSplitter`.
- Generate embeddings using OpenAI's `text-embedding-3-small`.
- Store document embeddings in ChromaDB.
- Perform semantic similarity searches.
- Generate answers using OpenAI GPT-4.1-mini.
- Return PDF source references, including filename, page number, and chunk ID.
- Persist vector embeddings locally using ChromaDB.
- Test REST APIs through FastAPI's Swagger UI.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| FastAPI | REST API development |
| LangChain | RAG orchestration |
| OpenAI GPT-4.1-mini | Answer generation |
| OpenAI Embeddings | Text vectorization |
| ChromaDB | Vector storage and retrieval |
| PyPDFLoader | PDF text extraction |
| Pydantic | Request and response validation |
| Uvicorn | ASGI server |

## System Architecture

```text
                 PDF Upload
                     |
                     v
                 FastAPI
                     |
                     v
                PyPDFLoader
                     |
                     v
                Extract Text
                     |
                     v
          RecursiveCharacterTextSplitter
                     |
                     v
                Text Chunks
                     |
                     v
             OpenAI Embeddings
                     |
                     v
                  ChromaDB
                     |
                     v
             Vector Persistence


              User Question
                     |
                     v
                FastAPI /ask
                     |
                     v
             Query Embedding
                     |
                     v
           ChromaDB Similarity Search
                     |
                     v
            Relevant PDF Chunks
                     |
                     v
             LangChain Prompt
                     |
                     v
              OpenAI GPT Model
                     |
                     v
          Answer + Source References
```

## Project Structure

```text
pdf-question-answering-rag/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── models/
│   ├── __init__.py
│   └── schemas.py
├── services/
│   ├── __init__.py
│   ├── pdf_service.py
│   ├── chunk_service.py
│   ├── embedding_service.py
│   ├── vector_service.py
│   └── rag_service.py
├── uploads/
└── chroma_db/
```

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd pdf-question-answering-rag
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

An OpenAI API key with available API billing is required.

**Never commit your actual API key to GitHub.**

### 5. Start the Application

```bash
python -m uvicorn main:app --reload
```

The application runs locally at:

`http://127.0.0.1:8000`

Interactive Swagger API documentation is available at:

`http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Application health check |
| POST | `/upload` | Upload, process, and index a PDF |
| GET | `/search` | Retrieve relevant PDF chunks |
| POST | `/ask` | Generate an answer with source references |

## API Usage

### 1. Upload a PDF

**Endpoint:**

`POST /upload`

Upload a PDF using the `file` multipart form field.

Example response:

```json
{
  "message": "PDF uploaded and indexed successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "machine_learning.pdf",
  "total_pages": 10,
  "total_chunks": 25,
  "stored_chunks": 25
}
```

Save the returned `document_id` to use when asking questions.

### 2. Search PDF Content

**Endpoint:**

`GET /search`

Query parameters:

- `document_id`: Identifier returned by the upload endpoint.
- `question`: Search query.

The endpoint retrieves semantically relevant PDF chunks from ChromaDB.

### 3. Ask a Question

**Endpoint:**

`POST /ask`

Example request:

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What is machine learning?"
}
```

Example response:

```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data.",
  "sources": [
    {
      "filename": "machine_learning.pdf",
      "page_number": 2,
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000_chunk_3"
    }
  ]
}
```

Responses depend on the content of the uploaded PDF.

Source references identify retrieved document chunks; they are not independently verified claim-level citations.

## How RAG Works

The application uses Retrieval-Augmented Generation to answer questions using uploaded documents.

**Document indexing:**

1. The user uploads a PDF.
2. LangChain extracts text from the PDF.
3. The text is divided into smaller chunks.
4. OpenAI converts the chunks into embeddings.
5. ChromaDB stores the embeddings and metadata.

**Question answering:**

1. The user submits a question and document ID.
2. OpenAI generates an embedding for the question.
3. ChromaDB retrieves the most relevant chunks.
4. LangChain constructs a prompt using the retrieved context.
5. OpenAI generates an answer based on that context.
6. The API returns the answer and source references.

## Limitations

- OpenAI API usage incurs charges.
- Scanned or image-only PDFs may require OCR.
- Answers depend on the quality of extracted and retrieved content.
- The current implementation handles questions independently without conversation memory.
- Local ChromaDB storage is intended for development and learning.
- Authentication, file-size limits, and advanced document access controls are not yet implemented.

## Future Improvements

- Conversation history and follow-up questions.
- Streaming LLM responses.
- Multiple-document question answering.
- Document listing and deletion.
- Hybrid keyword and semantic search.
- RAG evaluation and answer-quality metrics.
- Authentication and authorization.
- Improved error handling and upload validation.
- Docker deployment.

## Learning Objectives

This project demonstrates practical experience with:

- Building REST APIs using FastAPI.
- Integrating LangChain with OpenAI.
- Implementing Retrieval-Augmented Generation.
- Generating and storing text embeddings.
- Performing semantic similarity searches.
- Working with vector databases.
- Designing modular Python backend services.

## License

Add a license file if you choose to distribute this project under an open-source license.