import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


LOCAL_LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")


@st.cache_resource(show_spinner=False)
def load_reranker_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_LLM_NAME,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_qa_model():
    return load_local_llm()


@st.cache_resource(show_spinner=False)
def load_summary_model():
    return load_local_llm()