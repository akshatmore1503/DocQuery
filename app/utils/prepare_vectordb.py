from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

def extract_pdf_text(pdfs):
    docs = []
    for pdf in pdfs:
        pdf_path = os.path.join("docs", pdf)
        docs.extend(PyPDFLoader(pdf_path).load())
    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def get_vectorstore(pdfs, from_session_state=False):
    load_dotenv()
    embedding = OpenAIEmbeddings()
    if from_session_state and os.path.exists("Vector_DB - Documents"):
        vectordb = Chroma(persist_directory="Vector_DB - Documents", embedding_function=embedding)
        return vectordb
    elif not from_session_state:
        docs = extract_pdf_text(pdfs)
        chunks = get_text_chunks(docs)
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="Vector_DB - Documents")
        return vectordb
    return None