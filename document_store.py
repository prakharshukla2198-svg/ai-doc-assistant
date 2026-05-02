import os
import io
import json
import hashlib
from pathlib import Path

from pdf_loader import extract_text_from_multiple_pdfs
from image_loader import extract_text_from_multiple_images
from chunker import create_chunks


CACHE_ROOT = Path(".aidoc_cache")
DOCUMENTS_DIR = CACHE_ROOT / "documents"


class NamedBytesIO(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def ensure_dirs():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(data):
    return hashlib.sha256(data).hexdigest()


def get_uploaded_file_records(uploaded_files):
    records = []

    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        file_hash = compute_sha256(data)

        records.append({
            "name": uploaded_file.name,
            "size": len(data),
            "hash": file_hash,
            "bytes": data
        })

    return records


def remove_duplicate_files(file_records):
    seen = set()
    unique = []

    for record in file_records:
        if record["hash"] in seen:
            continue

        seen.add(record["hash"])
        unique.append(record)

    return unique


def get_file_signature(file_records):
    return tuple(
        (record["name"], record["size"], record["hash"])
        for record in file_records
    )


def get_document_cache_dir(file_hash):
    ensure_dirs()
    return DOCUMENTS_DIR / file_hash


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cache_exists(file_hash):
    cache_dir = get_document_cache_dir(file_hash)

    return (
        (cache_dir / "metadata.json").exists()
        and (cache_dir / "pages.json").exists()
        and (cache_dir / "chunks.json").exists()
    )


def extract_pages_from_record(record):
    file_obj = NamedBytesIO(record["bytes"], record["name"])
    lower_name = record["name"].lower()

    if lower_name.endswith(".pdf"):
        return extract_text_from_multiple_pdfs([file_obj])

    if lower_name.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_multiple_images([file_obj])

    return []


def process_and_cache_file(record):
    cache_dir = get_document_cache_dir(record["hash"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_exists(record["hash"]):
        metadata = read_json(cache_dir / "metadata.json")
        pages = read_json(cache_dir / "pages.json")
        chunks = read_json(cache_dir / "chunks.json")

        return {
            "metadata": metadata,
            "pages": pages,
            "chunks": chunks,
            "from_cache": True
        }

    pages = extract_pages_from_record(record)
    chunks = create_chunks(pages)

    table_count = sum(1 for page in pages if page.get("type") == "table")
    page_count = len(pages)
    chunk_count = len(chunks)

    sections = []
    seen_sections = set()

    for chunk in chunks:
        section = chunk.get("section", "Unknown")

        if section and section != "Unknown" and section not in seen_sections:
            seen_sections.add(section)
            sections.append(section)

    metadata = {
        "file_name": record["name"],
        "file_size": record["size"],
        "file_hash": record["hash"],
        "pages": page_count,
        "chunks": chunk_count,
        "tables": table_count,
        "sections": sections[:80]
    }

    write_json(cache_dir / "metadata.json", metadata)
    write_json(cache_dir / "pages.json", pages)
    write_json(cache_dir / "chunks.json", chunks)

    return {
        "metadata": metadata,
        "pages": pages,
        "chunks": chunks,
        "from_cache": False
    }


def reassign_global_chunk_ids(chunks):
    fixed = []

    for idx, chunk in enumerate(chunks):
        new_chunk = dict(chunk)
        new_chunk["id"] = idx
        fixed.append(new_chunk)

    return fixed


def load_documents(uploaded_files):
    records = get_uploaded_file_records(uploaded_files)
    unique_records = remove_duplicate_files(records)

    all_pages = []
    all_chunks = []
    metadatas = []
    cache_hits = 0
    cache_misses = 0

    for record in unique_records:
        result = process_and_cache_file(record)

        all_pages.extend(result["pages"])
        all_chunks.extend(result["chunks"])
        metadatas.append(result["metadata"])

        if result["from_cache"]:
            cache_hits += 1
        else:
            cache_misses += 1

    all_chunks = reassign_global_chunk_ids(all_chunks)

    file_signature = get_file_signature(unique_records)

    return {
        "file_records": unique_records,
        "file_signature": file_signature,
        "pages": all_pages,
        "chunks": all_chunks,
        "metadata": metadatas,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses
    }


def clear_document_cache():
    if not CACHE_ROOT.exists():
        return

    for root, dirs, files in os.walk(CACHE_ROOT, topdown=False):
        for file in files:
            Path(root, file).unlink(missing_ok=True)

        for directory in dirs:
            Path(root, directory).rmdir()

    CACHE_ROOT.rmdir()