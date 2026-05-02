from vector_store import search_faiss


def retrieve_chunks(
    question,
    chunks,
    embedding_model,
    faiss_index,
    top_k=80
):
    return search_faiss(
        question=question,
        chunks=chunks,
        embedding_model=embedding_model,
        index=faiss_index,
        top_k=top_k
    )


def rerank_chunks(question, results, reranker_model, top_n=12):
    if not results:
        return []

    pairs = [(question, result["chunk"]["text"]) for result in results]

    try:
        rerank_scores = reranker_model.predict(pairs)
    except Exception:
        return results[:top_n]

    reranked = []

    for result, rerank_score in zip(results, rerank_scores):
        result["rerank_score"] = float(rerank_score)
        result["final_score"] = (
            result.get("score", 0.0) * 0.68 +
            float(rerank_score) * 0.32
        )
        reranked.append(result)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)

    return reranked[:top_n]


def expand_with_neighbors(best_results, chunks, window=1, max_chunks=16):
    by_id = {}

    for chunk in chunks:
        if "id" in chunk:
            by_id[chunk["id"]] = chunk

    expanded = []
    seen = set()

    for result in best_results:
        chunk = result["chunk"]
        chunk_id = chunk.get("id")

        if chunk_id is not None:
            neighbor_ids = range(chunk_id - window, chunk_id + window + 1)
        else:
            neighbor_ids = [None]

        for neighbor_id in neighbor_ids:
            candidate = chunk if neighbor_id is None else by_id.get(neighbor_id)

            if not candidate:
                continue

            key = candidate.get(
                "id",
                (
                    candidate.get("source"),
                    candidate.get("page"),
                    candidate.get("text", "")[:120]
                )
            )

            if key in seen:
                continue

            seen.add(key)
            expanded.append(candidate)

            if len(expanded) >= max_chunks:
                return expanded

    return expanded