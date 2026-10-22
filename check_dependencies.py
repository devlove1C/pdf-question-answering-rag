from fastapi import FastAPI, UploadFile

from langchain_community.document_loaders import PyPDFLoader


from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_chroma import Chroma

import chromadb

print("All required packages imported successfully!")