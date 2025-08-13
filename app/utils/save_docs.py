import streamlit as st
import os
from utils.prepare_vectordb import get_vectorstore

def save_docs_to_vectordb(pdf_docs, upload_docs):
    new_files = [pdf for pdf in pdf_docs if pdf.name not in upload_docs]
    
    if not new_files:
        return

    # Save uploaded PDFs
    for pdf in new_files:
        pdf_path = os.path.join("docs", pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.getvalue())

    st.session_state.uploaded_pdfs = new_files  # overwrite for fresh start

    with st.spinner("Processing documents..."):
        all_docs = os.listdir("docs")

        # Create a fresh vectordb from the combined docs
        st.session_state.vectordb = get_vectorstore(all_docs, from_session_state=True)

    st.success("Documents processed and indexed successfully!")
