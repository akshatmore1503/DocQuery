import os
from utils.prepare_vectordb import get_vectorstore

def initialize_session_state_variables(st):
    upload_docs = os.listdir("docs")
    variables_to_initialize = ["chat_history", "uploaded_pdfs", "processed_documents", "vectordb", "previous_upload_docs_length"]
    for variable in variables_to_initialize:
        if variable not in st.session_state:
            if variable == "processed_documents":
                st.session_state.processed_documents = upload_docs
            elif variable == "vectordb":
                st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
            elif variable == "previous_upload_docs_length":
                st.session_state.previous_upload_docs_length = len(upload_docs)
            else:
                st.session_state[variable] = []