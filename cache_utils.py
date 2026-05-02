# cache_utils.py

import json
import hashlib
from pathlib import Path

import faiss
import numpy as np
import streamlit as st

from vector_store import build_faiss_index


CACHE_ROOT = Path(".aidoc_cache")
INDEX_DIR = CACHE_ROOT / "indexes"


def ensure_index_dir():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def stable_json_hash(data):
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_file_signature(uploaded_files):
    return tuple((file.name, file.size) for file in uploaded_files)


def get_embedding_model_name(embedding_model):
    """
    Tries to get a stable model name for cache signature.
    This prevents reusing the wrong FAISS index if you change embedding models.
    """

    if embedding_model is None:
        return "unknown_embedding_model"

    if hasattr(embedding_model, "_model_card_text"):
        return str(getattr(embedding_model, "_model_card_text"))[:120]

    if hasattr(embedding_model, "model_name"):
        return str(embedding_model.model_name)

    if hasattr(embedding_model, "_first_module"):
        try:
            module = embedding_model._first_module()
            if hasattr(module, "auto_model"):
                return str(module.auto_model.name_or_path)
        except Exception:
            pass

    return embedding_model.__class__.__name__


def make_index_signature(
    file_signature,
    chunks,
    embedding_model_name="multi-qa-mpnet-base-dot-v1",
):
    """
    Creates a stable signature for the FAISS index cache.
    """

    chunk_fingerprint = {
        "chunk_count": len(chunks),
        "first_chunk": chunks[0].get("text", "")[:300] if chunks else "",
        "last_chunk": chunks[-1].get("text", "")[:300] if chunks else "",
        "sources": sorted(list({chunk.get("source", "") for chunk in chunks if isinstance(chunk, dict)})),
    }

    payload = {
        "file_signature": file_signature,
        "embedding_model": embedding_model_name,
        "chunk_fingerprint": chunk_fingerprint,
    }

    return stable_json_hash(payload)


def get_index_paths(index_signature):
    ensure_index_dir()

    folder = INDEX_DIR / index_signature
    folder.mkdir(parents=True, exist_ok=True)

    return {
        "folder": folder,
        "index": folder / "faiss.index",
        "embeddings": folder / "embeddings.npy",
        "metadata": folder / "metadata.json",
    }


def save_index(index_signature, faiss_index, embeddings, metadata):
    paths = get_index_paths(index_signature)

    faiss.write_index(faiss_index, str(paths["index"]))

    if embeddings is not None:
        np.save(paths["embeddings"], embeddings)

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_index(index_signature):
    paths = get_index_paths(index_signature)

    if not paths["index"].exists():
        return None, None

    try:
        faiss_index = faiss.read_index(str(paths["index"]))
    except Exception:
        return None, None

    embeddings = None

    if paths["embeddings"].exists():
        try:
            embeddings = np.load(paths["embeddings"])
        except Exception:
            embeddings = None

    return faiss_index, embeddings


@st.cache_resource(show_spinner=False)
def get_or_build_faiss_index(index_signature, chunks, _embedding_model):
    """
    Loads FAISS index from disk cache or builds it.
    """

    faiss_index, embeddings = load_index(index_signature)

    if faiss_index is not None:
        return faiss_index, embeddings, True

    faiss_index, embeddings = build_faiss_index(
        chunks=chunks,
        embedding_model=_embedding_model,
    )

    metadata = {
        "index_signature": index_signature,
        "chunk_count": len(chunks),
        "embedding_model": get_embedding_model_name(_embedding_model),
    }

    save_index(
        index_signature=index_signature,
        faiss_index=faiss_index,
        embeddings=embeddings,
        metadata=metadata,
    )

    return faiss_index, embeddings, False


@st.cache_resource(show_spinner=False)
def cached_build_index(file_signature, chunks, _embedding_model):
    embedding_model_name = get_embedding_model_name(_embedding_model)

    index_signature = make_index_signature(
        file_signature=file_signature,
        chunks=chunks,
        embedding_model_name=embedding_model_name,
    )

    faiss_index, embeddings, _ = get_or_build_faiss_index(
        index_signature=index_signature,
        chunks=chunks,
        _embedding_model=_embedding_model,
    )

    return faiss_index, embeddings