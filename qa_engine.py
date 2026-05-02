# qa_engine.py

import re
import numpy as np


NOT_FOUND = "Not found in document."


def answer_question(
    question,
    chunks,
    embedding_model=None,
    faiss_index=None,
    reranker_model=None,
    tokenizer=None,
    model=None,
    allow_general_knowledge=False,
    task_type="qa",
):
    """
    Generic RAG document QA.

    Flow:
    1. Normalize chunks
    2. Retrieve relevant chunks using FAISS + keyword search
    3. Build compact context
    4. Ask the LLM to answer only from context
    5. Return answer + sources

    No PDF-specific rules.
    No document-specific keywords.
    No hard-coded answers.
    """

    question = clean_text(question)

    if not question:
        return "Please ask a valid question.", []

    if not chunks:
        return NOT_FOUND, []

    normalized_chunks = normalize_chunks(chunks)

    if not normalized_chunks:
        return NOT_FOUND, []

    page_number = extract_page_number(question)

    if page_number is not None:
        page_chunks = [
            chunk for chunk in normalized_chunks
            if str(chunk.get("page", "")).strip() == str(page_number)
        ]

        if page_chunks:
            normalized_chunks = page_chunks
            faiss_index = None

    retrieved_chunks = retrieve_chunks(
        question=question,
        chunks=normalized_chunks,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        reranker_model=reranker_model,
        task_type=task_type,
        top_k=6,
    )

    if not retrieved_chunks:
        return NOT_FOUND, []

    sources = build_sources(retrieved_chunks[:4])

    if tokenizer is None or model is None:
        fallback_answer = extractive_answer(question, retrieved_chunks)

        if is_bad_answer(fallback_answer):
            return NOT_FOUND, sources

        return fallback_answer, sources

    context = build_context(retrieved_chunks[:4])

    prompt = build_prompt(
        question=question,
        context=context,
        task_type=task_type,
        allow_general_knowledge=allow_general_knowledge,
    )

    raw_answer = generate_answer(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        task_type=task_type,
    )

    answer = clean_answer(raw_answer)

    if is_bad_answer(answer):
        fallback_answer = extractive_answer(question, retrieved_chunks)

        if not is_bad_answer(fallback_answer):
            answer = fallback_answer

    if is_bad_answer(answer):
        answer = NOT_FOUND

    return answer, sources


# -------------------------------------------------------
# Chunk normalization
# -------------------------------------------------------

def normalize_chunks(chunks):
    """
    Normalize chunks while preserving order.

    FAISS index positions depend on original chunk order,
    so this function must not drop or reorder chunks.
    """

    normalized = []

    for index, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            metadata = chunk.get("metadata", {}) or {}

            text = clean_text(
                chunk.get("text")
                or chunk.get("content")
                or metadata.get("text")
                or ""
            )

            normalized.append({
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
            })

        else:
            normalized.append({
                "id": index,
                "text": clean_text(chunk),
                "source": "Uploaded document",
                "page": "Unknown",
                "type": "text",
                "section": "Unknown",
            })

    return normalized


# -------------------------------------------------------
# Retrieval
# -------------------------------------------------------

def retrieve_chunks(
    question,
    chunks,
    embedding_model=None,
    faiss_index=None,
    reranker_model=None,
    task_type="qa",
    top_k=6,
):
    dense_results = []

    if embedding_model is not None and faiss_index is not None:
        dense_results = dense_search(
            question=question,
            chunks=chunks,
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            top_k=max(top_k * 3, 12),
        )

    keyword_results = keyword_search(
        question=question,
        chunks=chunks,
        top_k=max(top_k * 3, 12),
    )

    merged_results = merge_results(
        dense_results=dense_results,
        keyword_results=keyword_results,
    )

    if not merged_results:
        return []

    if reranker_model is not None and task_type in {"table_list", "extract_list"}:
        merged_results = rerank_results(
            question=question,
            results=merged_results,
            reranker_model=reranker_model,
        )

    merged_results = deduplicate_results(merged_results)

    return merged_results[:top_k]


def dense_search(question, chunks, embedding_model, faiss_index, top_k=12):
    try:
        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indexes = faiss_index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indexes[0]):
            if index == -1:
                continue

            if index >= len(chunks):
                continue

            chunk = dict(chunks[index])

            if not chunk.get("text"):
                continue

            chunk["dense_score"] = float(score)
            chunk["final_score"] = float(score)

            results.append(chunk)

        return results

    except Exception:
        return []


def keyword_search(question, chunks, top_k=12):
    question_terms = tokenize(question)

    if not question_terms:
        return []

    results = []

    for chunk in chunks:
        text = chunk.get("text", "")

        if not text:
            continue

        text_terms = tokenize(text)

        if not text_terms:
            continue

        overlap = question_terms.intersection(text_terms)

        if not overlap:
            continue

        keyword_score = len(overlap) / max(len(question_terms), 1)

        item = dict(chunk)
        item["keyword_score"] = float(keyword_score)
        item["final_score"] = float(keyword_score)

        results.append(item)

    results.sort(key=lambda item: item.get("keyword_score", 0.0), reverse=True)

    return results[:top_k]


def merge_results(dense_results, keyword_results):
    merged = {}

    for item in dense_results:
        key = make_result_key(item)
        merged[key] = dict(item)

    for item in keyword_results:
        key = make_result_key(item)

        if key not in merged:
            merged[key] = dict(item)
        else:
            merged[key]["keyword_score"] = max(
                merged[key].get("keyword_score", 0.0),
                item.get("keyword_score", 0.0),
            )

    results = []

    for item in merged.values():
        dense_score = item.get("dense_score", 0.0)
        keyword_score = item.get("keyword_score", 0.0)

        if dense_score > 0 and keyword_score > 0:
            final_score = (0.75 * dense_score) + (0.25 * keyword_score)
        elif dense_score > 0:
            final_score = dense_score
        else:
            final_score = keyword_score

        item["final_score"] = float(final_score)
        results.append(item)

    results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)

    return results


def rerank_results(question, results, reranker_model):
    try:
        pairs = [
            (question, item.get("text", ""))
            for item in results
        ]

        scores = reranker_model.predict(pairs)

        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)
            item["final_score"] = float(score)

        results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)

        return results

    except Exception:
        results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        return results


def deduplicate_results(results):
    seen = set()
    unique = []

    for item in results:
        signature = clean_text(item.get("text", "")).lower()[:250]

        if not signature:
            continue

        if signature in seen:
            continue

        seen.add(signature)
        unique.append(item)

    return unique


# -------------------------------------------------------
# Prompting
# -------------------------------------------------------

def build_context(chunks):
    parts = []

    for index, chunk in enumerate(chunks, start=1):
        text = clean_text(chunk.get("text", ""))

        if not text:
            continue

        text = text[:1000]

        source = chunk.get("source", "Uploaded document")
        page = chunk.get("page", "Unknown")
        section = chunk.get("section", "Unknown")

        parts.append(
            f"[Source {index} | File: {source} | Page: {page} | Section: {section}]\n{text}"
        )

    return "\n\n".join(parts)


def build_prompt(question, context, task_type="qa", allow_general_knowledge=False):
    if task_type in {"table_list", "extract_list"}:
        style_rule = "If the answer contains multiple items, use bullet points."
        length_rule = "Be complete but only include items supported by the context."
    else:
        style_rule = "Answer clearly and directly."
        length_rule = "Keep the answer concise unless the question asks for detail."

    outside_rule = (
        "You may use simple language to explain the context, but do not add facts that are not present in it."
        if allow_general_knowledge
        else "Do not use outside knowledge."
    )

    return f"""
You are a document question-answering assistant.

Use only the document context below.

Rules:
- {outside_rule}
- Do not guess.
- Do not invent names, numbers, dates, features, claims, or conclusions.
- If the answer is not present in the context, say exactly: {NOT_FOUND}
- {style_rule}
- {length_rule}
- When the question asks for names with numbers, IDs, dates, or values, preserve the exact pairs from the context.
- Do not repeat the full context.

Document context:
{context}

Question:
{question}

Answer:
""".strip()


def generate_answer(tokenizer, model, prompt, task_type="qa"):
    input_text = prompt

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You answer questions using only the provided document context."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = prompt

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=3072,
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    max_new_tokens = 220 if task_type in {"table_list", "extract_list"} else 160

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[1]

    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    return answer.strip()


# -------------------------------------------------------
# Extractive fallback
# -------------------------------------------------------

def extractive_answer(question, chunks):
    question_terms = tokenize(question)

    if not question_terms:
        return ""

    scored_sentences = []

    for chunk in chunks[:4]:
        text = chunk.get("text", "")

        for sentence in split_sentences(text):
            sentence = clean_text(sentence)

            if len(sentence) < 25:
                continue

            sentence_terms = tokenize(sentence)

            if not sentence_terms:
                continue

            overlap = question_terms.intersection(sentence_terms)

            if not overlap:
                continue

            score = len(overlap) / max(len(question_terms), 1)

            scored_sentences.append((score, sentence))

    if not scored_sentences:
        return ""

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    selected = []
    seen = set()

    for _, sentence in scored_sentences:
        signature = sentence.lower()[:120]

        if signature in seen:
            continue

        seen.add(signature)
        selected.append(sentence)

        if len(selected) >= 3:
            break

    if not selected:
        return ""

    if len(selected) == 1:
        return selected[0]

    return "\n".join(f"- {sentence}" for sentence in selected)


# -------------------------------------------------------
# Sources
# -------------------------------------------------------

def build_sources(chunks):
    sources = []

    for chunk in chunks:
        sources.append({
            "source": chunk.get("source", "Uploaded document"),
            "page": chunk.get("page", "Unknown"),
            "type": chunk.get("type", "text"),
            "section": chunk.get("section", "Unknown"),
            "text": chunk.get("text", ""),
            "score": chunk.get("final_score", 0.0),
            "dense_score": chunk.get("dense_score", 0.0),
            "keyword_score": chunk.get("keyword_score", 0.0),
            "rerank_score": chunk.get("rerank_score", 0.0),
            "final_score": chunk.get("final_score", 0.0),
        })

    return sources


# -------------------------------------------------------
# Cleanup
# -------------------------------------------------------

def clean_answer(answer):
    answer = clean_text(answer)

    if not answer:
        return ""

    prefixes = [
        "Answer:",
        "Final answer:",
        "The answer is:",
        "According to the document,",
        "According to the context,",
        "Based on the document,",
        "Based on the context,",
    ]

    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    stop_markers = [
        "Document context:",
        "Question:",
        "Rules:",
        "[Source",
    ]

    for marker in stop_markers:
        position = answer.lower().find(marker.lower())

        if position != -1:
            answer = answer[:position].strip()

    return remove_duplicate_lines(answer)


def is_bad_answer(answer):
    if not answer:
        return True

    answer = clean_text(answer)
    lower = answer.lower()

    if lower == NOT_FOUND.lower():
        return True

    weak_phrases = [
        "i don't know",
        "i do not know",
        "not enough information",
        "cannot determine",
        "can't determine",
        "not mentioned",
        "not provided",
        "no information",
        "unknown",
        "n/a",
    ]

    if any(phrase in lower for phrase in weak_phrases):
        return True

    if len(answer.split()) < 3:
        return True

    return False


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def tokenize(text):
    text = clean_text(text).lower()

    words = re.findall(r"[a-zA-Z0-9]+", text)

    stopwords = {
        "a", "an", "the", "is", "are", "was", "were",
        "am", "be", "been", "being",
        "to", "of", "in", "on", "for", "with", "and",
        "or", "by", "as", "at", "from", "into", "about",
        "this", "that", "these", "those", "it", "its",
        "what", "who", "when", "where", "why", "how", "which",
        "give", "tell", "show", "explain", "list",
        "document", "file", "pdf", "image",
        "me", "my", "your", "you", "we", "our",
    }

    return {
        word
        for word in words
        if word not in stopwords and len(word) > 1
    }


def split_sentences(text):
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) > 1:
        return sentences

    words = text.split()

    return [
        " ".join(words[start:start + 35])
        for start in range(0, len(words), 35)
    ]


def extract_page_number(question):
    match = re.search(r"\bpage\s+(\d+)\b", question.lower())

    if match:
        return int(match.group(1))

    return None


def remove_duplicate_lines(text):
    lines = text.splitlines()
    cleaned = []
    seen = set()

    for line in lines:
        key = line.strip().lower()

        if not key:
            cleaned.append(line)
            continue

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def make_result_key(item):
    source = item.get("source", "Uploaded document")
    page = item.get("page", "Unknown")
    chunk_id = item.get("id", "")
    text_signature = clean_text(item.get("text", ""))[:100]

    return f"{source}|{page}|{chunk_id}|{text_signature}"