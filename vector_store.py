# vector_store.py

import re
import faiss
import numpy as np


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_chunk(chunk, index):
    """
    Converts any chunk format into the format needed by FAISS and QA.

    Your app's chunks usually look like:
    {
        "id": ...,
        "text": "...",
        "source": "...",
        "page": ...,
        "type": "...",
        "section": ...
    }
    """

    if isinstance(chunk, str):
        return {
            "id": index,
            "text": clean_text(chunk),
            "source": "Uploaded document",
            "page": "Unknown",
            "type": "text",
            "section": "Unknown",
        }

    if not isinstance(chunk, dict):
        return None

    metadata = chunk.get("metadata", {}) or {}

    text = clean_text(
        chunk.get("text")
        or chunk.get("content")
        or metadata.get("text")
        or ""
    )

    if len(text) < 20:
        return None

    return {
        "id": chunk.get("id", metadata.get("id", index)),
        "text": text,
        "source": (
            chunk.get("source")
            or metadata.get("source")
            or chunk.get("file_name")
            or metadata.get("file_name")
            or chunk.get("filename")
            or metadata.get("filename")
            or "Uploaded document"
        ),
        "page": (
            chunk.get("page")
            or metadata.get("page")
            or chunk.get("page_number")
            or metadata.get("page_number")
            or "Unknown"
        ),
        "type": (
            chunk.get("type")
            or metadata.get("type")
            or "text"
        ),
        "section": (
            chunk.get("section")
            or metadata.get("section")
            or "Unknown"
        ),
        "metadata": metadata,
    }


def normalize_chunks(chunks):
    normalized = []

    for index, chunk in enumerate(chunks):
        item = normalize_chunk(chunk, index)

        if item is not None:
            normalized.append(item)

    return normalized


def build_faiss_index(chunks, embedding_model):
    """
    This is the function your cache_utils.py expects.

    Returns:
        faiss_index, embeddings

    Important:
    - The FAISS index order must match the chunks order.
    - So we do NOT remove/reorder chunks here.
    - qa_engine.py normalizes chunks separately using the same original order.
    """

    if embedding_model is None:
        raise ValueError("embedding_model is required to build FAISS index.")

    if not chunks:
        raise ValueError("No chunks provided to build FAISS index.")

    texts = []

    for chunk in chunks:
        if isinstance(chunk, str):
            text = clean_text(chunk)
        elif isinstance(chunk, dict):
            metadata = chunk.get("metadata", {}) or {}
            text = clean_text(
                chunk.get("text")
                or chunk.get("content")
                or metadata.get("text")
                or ""
            )
        else:
            text = ""

        # Keep list length aligned with chunks.
        # If text is empty, use placeholder to avoid index mismatch.
        if not text:
            text = "empty document chunk"

        texts.append(text)

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    return faiss_index, embeddings


class VectorStore:
    """
    Optional class-based vector store.

    Your current app.py does not directly need this class,
    but keeping it is useful for testing or future refactoring.
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks, embedding_model=None):
        if embedding_model is not None:
            self.embedding_model = embedding_model

        if self.embedding_model is None:
            raise ValueError("embedding_model is required.")

        self.chunks = normalize_chunks(chunks)

        if not self.chunks:
            self.index = None
            self.embeddings = None
            return

        self.index, self.embeddings = build_faiss_index(
            chunks=self.chunks,
            embedding_model=self.embedding_model,
        )

    def search(self, query, top_k=10):
        if self.index is None or not self.chunks:
            return []

        if self.embedding_model is None:
            raise ValueError("embedding_model is required for search.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if idx >= len(self.chunks):
                continue

            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def has_data(self):
        return self.index is not None and len(self.chunks) > 0

    def get_all_chunks(self):
        return self.chunks