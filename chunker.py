import re


def clean_text(text):
    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading_line(line):
    line = line.strip()

    if not line:
        return False

    if len(line.split()) > 14:
        return False

    if re.match(r"^(chapter|section|unit)\s+\d+", line, re.IGNORECASE):
        return True

    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
        return True

    if line.isupper() and len(line) >= 4:
        return True

    return False


def detect_section(text, current_section=None):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]

    for line in lines[:10]:
        if is_heading_line(line):
            return line

    return current_section


def split_into_paragraphs(text):
    text = clean_text(text)

    parts = re.split(r"\n\s*\n", text)
    paragraphs = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        lines = [line.strip() for line in part.splitlines() if line.strip()]

        if not lines:
            continue

        paragraph = " ".join(lines)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def chunk_table(page, chunk_id):
    text = page.get("text", "").strip()

    if not text:
        return [], chunk_id

    rows = [row for row in text.splitlines() if row.strip()]
    chunks = []

    if len(rows) <= 20:
        chunks.append({
            "id": chunk_id,
            "source": page.get("source", "Unknown"),
            "page": page.get("page", "Unknown"),
            "type": "table",
            "section": page.get("section", "Table"),
            "text": text,
            "search_text": f"TABLE\nSECTION {page.get('section', 'Table')}\nPAGE {page.get('page', '')}\n{text}"
        })
        chunk_id += 1
        return chunks, chunk_id

    header = rows[:2]
    body = rows[2:]
    batch_size = 16

    for start in range(0, len(body), batch_size):
        batch = body[start:start + batch_size]
        table_text = "\n".join(header + batch)

        chunks.append({
            "id": chunk_id,
            "source": page.get("source", "Unknown"),
            "page": page.get("page", "Unknown"),
            "type": "table",
            "section": page.get("section", "Table"),
            "text": table_text,
            "search_text": f"TABLE\nSECTION {page.get('section', 'Table')}\nPAGE {page.get('page', '')}\n{table_text}"
        })
        chunk_id += 1

    return chunks, chunk_id


def create_chunks(pages, chunk_size=240, overlap=80):
    chunks = []
    chunk_id = 0
    current_section = None

    for page in pages:
        page_type = page.get("type", "text")
        text = page.get("text", "")

        if not str(text).strip():
            continue

        current_section = detect_section(text, current_section)

        if page_type == "table":
            page["section"] = current_section or "Table"
            table_chunks, chunk_id = chunk_table(page, chunk_id)
            chunks.extend(table_chunks)
            continue

        paragraphs = split_into_paragraphs(text)
        current_words = []

        for paragraph in paragraphs:
            if is_heading_line(paragraph):
                current_section = paragraph

            paragraph_words = paragraph.split()

            if not paragraph_words:
                continue

            if len(current_words) + len(paragraph_words) <= chunk_size:
                current_words.extend(paragraph_words)
            else:
                if current_words:
                    chunk_text = " ".join(current_words).strip()

                    chunks.append({
                        "id": chunk_id,
                        "source": page.get("source", "Unknown"),
                        "page": page.get("page", "Unknown"),
                        "type": "text",
                        "section": current_section or "Unknown",
                        "text": chunk_text,
                        "search_text": f"SECTION {current_section or 'Unknown'}\nPAGE {page.get('page', '')}\n{chunk_text}"
                    })
                    chunk_id += 1

                    current_words = current_words[-overlap:] if overlap > 0 else []

                if len(paragraph_words) > chunk_size:
                    start = 0

                    while start < len(paragraph_words):
                        end = start + chunk_size
                        chunk_text = " ".join(paragraph_words[start:end]).strip()

                        if chunk_text:
                            chunks.append({
                                "id": chunk_id,
                                "source": page.get("source", "Unknown"),
                                "page": page.get("page", "Unknown"),
                                "type": "text",
                                "section": current_section or "Unknown",
                                "text": chunk_text,
                                "search_text": f"SECTION {current_section or 'Unknown'}\nPAGE {page.get('page', '')}\n{chunk_text}"
                            })
                            chunk_id += 1

                        start += chunk_size - overlap
                else:
                    current_words.extend(paragraph_words)

        if current_words:
            chunk_text = " ".join(current_words).strip()

            chunks.append({
                "id": chunk_id,
                "source": page.get("source", "Unknown"),
                "page": page.get("page", "Unknown"),
                "type": "text",
                "section": current_section or "Unknown",
                "text": chunk_text,
                "search_text": f"SECTION {current_section or 'Unknown'}\nPAGE {page.get('page', '')}\n{chunk_text}"
            })
            chunk_id += 1

    return chunks