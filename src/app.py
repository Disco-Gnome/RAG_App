import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, APP_NAME, EMBEDDING_MODELS
from app_utils import initialize_session_state
from app_sections import run_upload_and_settings,  run_chatbot

# default session state variables
initialize_session_state()

# App layout
st.title(APP_NAME)

#general context for prompts
with st.sidebar:
    model = st.selectbox(f"Select Model ", MODELS)

    temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=1.0*TEMPERATURE, step=.1)
    st.session_state["temperature"] = temperature

    st.session_state["generation_model"]=model

    emb_model = st.selectbox(f"Select Retrieval Approach", EMBEDDING_MODELS)
    st.session_state["embedding_model"]=emb_model



#don't have it appear until responses are generated
clear_button = None

# start with the upload and settings 
run_upload_and_settings()

run_chatbot()

