import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os

from langchain.document_loaders import UnstructuredPDFLoader

from config import MODELS, TEMPERATURE, MAX_TOKENS, APP_NAME, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR
from app_utils import embedding_function, get_retriever, generate_response, initialize_session_state, create_knowledge_base, generate_kb_response


def run_upload_and_settings():
    """This function runs the upload and settings container"""

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # copy the file to "raw" folder
        with open(os.path.join("../data/raw/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["uploaded_file"] = uploaded_file.name


def run_chatbot():
    template = ""
    model = st.session_state["generation_model"]

    # Start button
    start_button = st.button("Click to build Vector Database") #click to build vector database / knowledge base

    if start_button:
        st.session_state["messages"] = []
        # read and load 10k pdf file
        loader = UnstructuredPDFLoader(os.path.join("../data/raw/", st.session_state["uploaded_file"]))
        docs = loader.load()

        # process time series data to save to knowledge base
        create_knowledge_base(docs)
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input("What is in the context?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = get_retriever()        
                response = generate_kb_response(prompt, model, retriever, system_prompt="",template=None, temperature=0)
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)