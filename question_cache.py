import json
import os
import numpy as np

CACHE_FILE = "question_cache.json"
MAX_CACHE_SIZE = 500

_memory_cache = None


def load_cache():
    global _memory_cache

    if _memory_cache is not None:
        return _memory_cache

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _memory_cache = json.load(f)
        except Exception:
            _memory_cache = []
    else:
        _memory_cache = []

    return _memory_cache


def save_cache(cache):
    global _memory_cache
    _memory_cache = cache

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def save_to_cache(question, answer, sources, file_signature, embedding_model):
    answer_text = str(answer or "").strip()

    # Do not cache failures or low-value fallback messages.
    if not answer_text or answer_text.lower() == "not found in document." or answer_text.startswith("Error:"):
        return

    cache = load_cache()

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].tolist()

    cache.append({
        "file_signature": list(file_signature),
        "question": question,
        "answer": answer,
        "sources": sources,
        "embedding": question_embedding
    })

    if len(cache) > MAX_CACHE_SIZE:
        cache = cache[-MAX_CACHE_SIZE:]

    save_cache(cache)


def find_cached_answer(question, embedding_model, file_signature, threshold=0.90):
    cache = load_cache()

    if not cache:
        return None

    sig = list(file_signature)

    relevant = [
        item for item in cache
        if item.get("file_signature") == sig and item.get("embedding") is not None
    ]

    if not relevant:
        return None

    q_emb = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    past_embs = np.array(
        [item["embedding"] for item in relevant],
        dtype="float32"
    )

    scores = past_embs @ q_emb
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return relevant[best_idx]

    return None


def clear_cache():
    global _memory_cache
    _memory_cache = None

    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
