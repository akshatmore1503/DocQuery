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

def get_vectorstore(pdfs=None, from_session_state=False):
    """
    Build a fresh Chroma DB when `pdfs` are provided; otherwise load existing.
    Compatible with chromadb <0.5 and >=0.5 (tenant-aware via client_settings).
    """
    load_dotenv()
    embedding = OpenAIEmbeddings()
    persist_dir = "Vector_DB - Documents"

    # Try to prepare client_settings for chromadb>=0.5; stay no-op for older versions.
    client_settings = None
    try:
        from chromadb.config import Settings  # available in newer chromadb
        client_settings = Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )
    except Exception:
        client_settings = None  # older chromadb; Chroma will ignore this

    # If new PDFs are provided, rebuild a fresh DB
    if pdfs:
        # Clear old DB so it never mixes with previous docs
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir, ignore_errors=True)

        docs = extract_pdf_text(pdfs)
        chunks = get_text_chunks(docs)
        if not chunks:
            return None

        # Embed in batches to avoid hitting OpenAI's per-request token cap
        batch_size = 80  # safe default
        vectordb = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if vectordb is None:
                # create new collection with the first batch
                if client_settings:
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embedding,
                        persist_directory=persist_dir,
                        client_settings=client_settings,
                    )
                else:
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embedding,
                        persist_directory=persist_dir,
                    )
            else:
                vectordb.add_documents(batch)

        if vectordb:
            vectordb.persist()
        return vectordb

    # If no new PDFs but a DB exists and caller wants to load it
    if from_session_state and os.path.exists(persist_dir):
        if client_settings:
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding,
                client_settings=client_settings,
            )
        else:
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding,
            )

    return None


