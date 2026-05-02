import os
import re
import sys

from pdf_loader import extract_text_from_pdf
from chunker import create_chunks
from models import load_embedding_model, load_reranker_model, load_qa_model
from vector_store import (
    build_faiss_index,
    search_faiss,
    correct_query_with_document_vocabulary
)
from qa_engine import answer_question


DEFAULT_QUESTION = "name the models in the list"


def extract_candidate_names(text):
    candidates = set()

    cleaned = text.replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    citation_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9\-]{1,40}|[A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*)\s*\[\d+\]"
    )

    for match in citation_pattern.finditer(cleaned):
        item = match.group(1).strip()
        if 2 <= len(item) <= 50:
            candidates.add(item)

    technical_pattern = re.compile(
        r"\b([A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*|[A-Za-z]+(?:Text|Net|BERT|GPT|CLIP|Lip|Former|VQA|T5|VL|LM|MoE)\b[A-Za-z0-9\-]*)\b"
    )

    bad = {
        "Table", "Figure", "Page", "Chapter", "References", "Abstract",
        "Related", "Works", "International", "Conference", "Artificial",
        "Intelligence", "Journal", "Survey", "Question", "Answering",
        "Introduction", "Conclusion"
    }

    for match in technical_pattern.finditer(cleaned):
        item = match.group(1).strip()

        if len(item) < 2 or len(item) > 50:
            continue

        if item in bad:
            continue

        candidates.add(item)

    return sorted(candidates)


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python debug_rag_test.py "path_to_pdf.pdf"')
        print('  python debug_rag_test.py "path_to_pdf.pdf" "your question here"')
        print()
        print("Example:")
        print('  python debug_rag_test.py "C:\\Users\\basel\\Downloads\\3728635.pdf" "name the models in the list"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_QUESTION

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        print("Use the full PDF path, or place the PDF in this project folder.")
        sys.exit(1)

    print_header("0. DEBUG INPUT")
    print(f"PDF path : {pdf_path}")
    print(f"Question : {question}")

    print_header("1. LOADING PDF")
    with open(pdf_path, "rb") as f:
        pages = extract_text_from_pdf(f)

    print(f"Pages/items extracted: {len(pages)}")

    type_counts = {}
    for page in pages:
        page_type = page.get("type", "text")
        type_counts[page_type] = type_counts.get(page_type, 0) + 1

    print(f"Page item types: {type_counts}")

    print_header("2. FIRST FEW EXTRACTED ITEMS")
    for item in pages[:8]:
        print(f"\nSOURCE={item.get('source')} PAGE={item.get('page')} TYPE={item.get('type', 'text')}")
        print(item.get("text", "")[:1000])

    print_header("3. CHUNKING")
    chunks = create_chunks(pages)
    print(f"Chunks created: {len(chunks)}")

    chunk_type_counts = {}
    for chunk in chunks:
        chunk_type = chunk.get("type", "text")
        chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

    print(f"Chunk types: {chunk_type_counts}")

    table_chunks = [chunk for chunk in chunks if chunk.get("type") == "table"]
    print(f"Table chunks: {len(table_chunks)}")

    print_header("4. FIRST TABLE CHUNKS")
    if table_chunks:
        for chunk in table_chunks[:5]:
            print(f"\nCHUNK_ID={chunk.get('id')} PAGE={chunk.get('page')} SECTION={chunk.get('section')}")
            print(chunk.get("text", "")[:1600])
    else:
        print("No table chunks found.")

    print_header("5. QUERY CORRECTION")
    corrected_question = correct_query_with_document_vocabulary(question, chunks)
    print(f"Original question : {question}")
    print(f"Corrected question: {corrected_question}")

    print_header("6. BUILDING MODELS / INDEX")
    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()
    faiss_index, _ = build_faiss_index(chunks, embedding_model)

    print_header("7. RAW RETRIEVAL RESULTS")
    retrieved = search_faiss(
        question=question,
        chunks=chunks,
        embedding_model=embedding_model,
        index=faiss_index,
        top_k=20
    )

    for i, result in enumerate(retrieved[:20], start=1):
        chunk = result["chunk"]

        print(f"\n#{i}")
        print(f"PAGE={chunk.get('page')} TYPE={chunk.get('type')} SECTION={chunk.get('section')}")
        print(
            f"SCORE={result.get('score', 0):.4f} "
            f"DENSE={result.get('dense_score', 0):.4f} "
            f"BM25={result.get('bm25_score', 0):.4f} "
            f"STRUCT={result.get('structure_score', result.get('position_score', 0)):.4f}"
        )
        print(chunk.get("text", "")[:1200])

    print_header("8. CANDIDATE NAMES EXTRACTED FROM TOP RETRIEVAL")
    combined_top_text = "\n".join(result["chunk"].get("text", "") for result in retrieved[:10])
    candidates = extract_candidate_names(combined_top_text)

    if candidates:
        for item in candidates:
            print(f"- {item}")
    else:
        print("No candidates extracted from top retrieved chunks.")

    print_header("9. FINAL QA MODEL ANSWER")
    tokenizer, model = load_qa_model()

    answer, sources = answer_question(
        question=question,
        chunks=chunks,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        reranker_model=reranker_model,
        tokenizer=tokenizer,
        model=model,
        allow_general_knowledge=False
    )

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for i, source in enumerate(sources[:8], start=1):
        print(f"\nSOURCE #{i}: PAGE={source.get('page')}")
        print(source.get("text", "")[:700])


if __name__ == "__main__":
    main()