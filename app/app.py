import streamlit as st
import os
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat

class ChatApp:
    def __init__(self):
        if not os.path.exists("docs"):
            os.makedirs("docs")
        st.set_page_config(page_title="Chat with PDFS :books:")
        st.title("Chat with PDFS :books:")
        initialize_session_state_variables(st)
        self.docs_files = st.session_state.processed_documents

    def run(self):
        upload_docs = os.listdir("docs")
    
        with st.sidebar:
            st.subheader("Your documents")
            if upload_docs:
                st.write("Uploaded Documents:")
                st.text(", ".join(upload_docs))
            else:
                st.info("No documents uploaded yet.")

            st.subheader("Upload PDF documents")
            pdf_docs = st.file_uploader(
                "Select a PDF document and click on 'Process'",
                type=['pdf'],
                accept_multiple_files=True
            )

            if pdf_docs:
                save_docs_to_vectordb(pdf_docs, upload_docs)
                # Build vectordb immediately after upload
                st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
                st.session_state.previous_upload_docs_length = len(upload_docs)

        # Only allow chat if vectordb exists
        if st.session_state.vectordb:
            st.session_state.chat_history = chat(
            st.session_state.chat_history,
            st.session_state.vectordb
        )
        else:
            st.info("Upload a PDF file to start chatting.")

if __name__ == "__main__":
    app = ChatApp()
    app.run()