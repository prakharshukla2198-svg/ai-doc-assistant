# model_manager.py

import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


PRIMARY_LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"

EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def get_gpu_info():
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "CPU",
            "vram_gb": 0,
        }

    props = torch.cuda.get_device_properties(0)

    return {
        "available": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    Keep embeddings on CPU.

    RTX 4050 has 6GB VRAM.
    The LLM needs most GPU memory.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )


@st.cache_resource(show_spinner=False)
def get_reranker_model():
    """
    Reranker is CPU-only and used only for heavier extraction tasks.
    Normal QA does not load it.
    """

    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cpu",
    )


def load_qwen_7b_4bit():
    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_LLM_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        PRIMARY_LLM_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.eval()
    return tokenizer, model


def load_qwen_3b():
    tokenizer = AutoTokenizer.from_pretrained(
        FALLBACK_LLM_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_LLM_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Best practical loader for RTX 4050 6GB.

    First:
        Qwen2.5-7B-Instruct 4-bit

    Fallback:
        Qwen2.5-3B-Instruct
    """

    if not torch.cuda.is_available():
        return load_qwen_3b()

    try:
        return load_qwen_7b_4bit()

    except Exception as error:
        print("Qwen 7B 4-bit failed. Falling back to Qwen 3B.")
        print("Reason:", str(error))

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return load_qwen_3b()


def get_models_for_task(task_type):
    """
    Lazy model router.

    Normal QA:
    - app.py separately loads embedding model for FAISS index
    - this loads only LLM
    - no reranker for speed

    Table/list/extraction:
    - can use reranker
    """

    if task_type in {"index", "search"}:
        return {
            "embedding_model": get_embedding_model(),
        }

    if task_type in {"qa", "section_qa"}:
        return {
            "llm": get_llm(),
        }

    if task_type in {"table_list", "extract_list"}:
        return {
            "reranker_model": get_reranker_model(),
            "llm": get_llm(),
        }

    if task_type in {"summary", "detailed_summary", "bullet_summary"}:
        return {
            "llm": get_llm(),
        }

    if task_type == "translation":
        return {
            "llm": get_llm(),
        }

    return {}


def get_model_status():
    return {
        "gpu": get_gpu_info(),
        "primary_llm": PRIMARY_LLM_NAME,
        "fallback_llm": FALLBACK_LLM_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "embedding_device": "cpu",
        "reranker_device": "cpu",
    }