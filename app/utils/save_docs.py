import streamlit as st
import os
from utils.prepare_vectordb import get_vectorstore

def save_docs_to_vectordb(pdf_docs, upload_docs):
    new_files = [pdf for pdf in pdf_docs if pdf.name not in upload_docs]
    if new_files and st.button("Process"):
        for pdf in new_files:
            pdf_path = os.path.join("docs", pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf.getvalue())
            st.session_state.uploaded_pdfs.extend(pdf_docs)
        with st.spinner("Processing"):
            all_docs = os.listdir("docs")
            get_vectorstore(all_docs)
            st.session_state.vectordb = get_vectorstore(all_docs, from_session_state=True)
            st.success("All documents processed and indexed.")