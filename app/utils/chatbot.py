import streamlit as st
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

def get_context_retriever_chain(vectordb):
    load_dotenv()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    retriever = vectordb.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a chatbot. You'll receive a prompt that includes a chat history and retrieved content from the vectorDB based on the user's question. Your task is to respond to the user's question using the information from the vectordb, relying as little as possible on your own knowledge. If for some reason you don't know the answer for the question, or the question cannot be answered because there's no context, ask the user for more details. Do not invent an answer. Answer the questions from this context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain

def get_response(question, chat_history, vectordb):
    chain = get_context_retriever_chain(vectordb)
    response = chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"], response["context"]

def chat(chat_history, vectordb):
    user_query = st.chat_input("Ask a question:")
    if user_query is not None and user_query != "":
        response, context = get_response(user_query, chat_history, vectordb)
        chat_history = chat_history + [HumanMessage(content=user_query), AIMessage(content=response)]
        with st.sidebar:
            metadata_dict = defaultdict(list)
            for metadata in [doc.metadata for doc in context]:
                metadata_dict[metadata['source']].append(metadata['page'])
            for source, pages in metadata_dict.items():
                st.write(f"Source: {source}")
                st.write(f"Pages: {', '.join(map(str, pages))}")
    for message in chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)
    return chat_history