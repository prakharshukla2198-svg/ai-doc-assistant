import re


GENERIC_BAD_CANDIDATES = {
    "table", "figure", "page", "chapter", "references", "abstract", "related",
    "works", "introduction", "conclusion", "survey", "question", "answering",
    "international", "conference", "journal", "paper", "study", "method",
    "methods", "model", "models", "dataset", "datasets", "approach", "system",
    "assistant", "evidence", "source", "section", "unknown", "result", "results",
    "accuracy", "score", "year", "author", "authors", "place", "text", "image",
    "visual", "language", "task", "tasks", "feature", "features", "degree",
    "department", "technology", "computer", "science", "engineering", "university",
    "institute", "kanpur", "lucknow", "signature", "urgent", "care", "india",
    "medical", "healthcare", "patient", "patients", "hospital", "hospitals"
}


def split_camel_name(text):
    text = str(text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_candidate(item):
    item = split_camel_name(item)
    item = re.sub(r"\[[0-9,\s\-]+\]", "", item)
    item = re.sub(r"\([0-9,\s\-]+\)", "", item)
    item = item.replace("✓", " ")
    item = item.replace("✗", " ")
    item = item.replace("–", "-")
    item = item.replace("_", " ")
    item = re.sub(r"\s+", " ", item)
    return item.strip(" :-|,.;")


def is_author_like(item):
    if re.search(r"\bet\s+al\b", item, flags=re.IGNORECASE):
        return True

    if re.match(r"^[A-Z]\.?\s?[A-Z]?\.\s?[A-Z][a-z]+", item):
        return True

    if re.match(r"^[A-Z][a-z]+,\s?[A-Z]", item):
        return True

    return False


def is_valid_candidate(item):
    item = normalize_candidate(item)

    if not item:
        return False

    if len(item) < 2 or len(item) > 70:
        return False

    lowered = item.lower()

    if lowered in GENERIC_BAD_CANDIDATES:
        return False

    if re.fullmatch(r"[\d\W]+", item):
        return False

    if re.fullmatch(r"\d{4}", item):
        return False

    if is_author_like(item):
        return False

    words = item.split()

    if len(words) > 6:
        return False

    digit_count = len(re.findall(r"\d", item))
    alpha_count = len(re.findall(r"[A-Za-z]", item))

    if alpha_count < 2:
        return False

    if digit_count > alpha_count:
        return False

    if re.search(
        r"\b(page|chapter|section|figure|table|journal|conference|proceedings|signature|roll)\b",
        lowered
    ):
        return False

    return True


def dedupe_candidates(candidates):
    final = []
    seen = set()

    for item in candidates:
        item = normalize_candidate(item)

        if not is_valid_candidate(item):
            continue

        key = re.sub(r"[^a-z0-9]+", "", item.lower())

        if not key or key in seen:
            continue

        seen.add(key)
        final.append(item)

    return final


def parse_table_rows(text):
    rows = []

    for line in str(text).splitlines():
        line = line.strip()

        if "|" not in line:
            continue

        if re.fullmatch(r"[\|\-\s:]+", line):
            continue

        cells = [normalize_candidate(cell) for cell in line.split("|")]
        cells = [cell for cell in cells if cell]

        if cells:
            rows.append(cells)

    return rows


def extract_table_items(chunks):
    candidates = []

    for chunk in chunks:
        text = chunk.get("text", "")

        if chunk.get("type") != "table" and "|" not in text:
            continue

        rows = parse_table_rows(text)

        for row in rows:
            for cell in row:
                pieces = re.split(r"\s{2,}|;|,", cell)

                for piece in pieces:
                    piece = normalize_candidate(piece)

                    if is_valid_candidate(piece):
                        candidates.append(piece)

    return dedupe_candidates(candidates)


def extract_technical_items(chunks):
    candidates = []

    combined = "\n".join(chunk.get("text", "") for chunk in chunks)
    cleaned = combined.replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    citation_pattern = re.compile(
        r"\b([A-Za-z][A-Za-z0-9\-]{1,45}(?:\s+[A-Za-z][A-Za-z0-9\-]{1,30}){0,3})\s*\[\d+\]"
    )

    for match in citation_pattern.finditer(cleaned):
        item = normalize_candidate(match.group(1))

        if is_valid_candidate(item):
            candidates.append(item)

    technical_pattern = re.compile(
        r"\b([A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*|[A-Za-z]+(?:Text|Net|BERT|GPT|CLIP|Former|VQA|T5|VL|LM|MoE|Lip|TTS|FSDP)\b[A-Za-z0-9\-]*)\b"
    )

    for match in technical_pattern.finditer(cleaned):
        item = normalize_candidate(match.group(1))

        if is_valid_candidate(item):
            candidates.append(item)

    return dedupe_candidates(candidates)


# ─────────────────────────────────────────────
# People extraction
# ─────────────────────────────────────────────

def clean_person_name(name):
    name = split_camel_name(name)

    name = re.sub(
        r"\broll\s*(no|number)?\.?\s*[:\-]?\s*\d*.*$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(
        r"\benrollment\s*(no|number)?\.?\s*[:\-]?\s*\d*.*$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(
        r"\bregistration\s*(no|number)?\.?\s*[:\-]?\s*\d*.*$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(r"\b\d{4,}\b", "", name)
    name = re.sub(r"[^A-Za-z\s.]", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" :-,.;")
    name = re.sub(r"^(mr|mrs|ms|dr|prof)\.?\s+", "", name, flags=re.IGNORECASE).strip()

    return name


def title_case_name(name):
    parts = clean_person_name(name).split()
    return " ".join(part.capitalize() for part in parts)


def looks_like_person_name(name):
    name = clean_person_name(name)

    if not name:
        return False

    if len(name) < 3 or len(name) > 60:
        return False

    lowered = name.lower()

    bad_words = {
        "degree", "department", "technology", "computer", "science", "engineering",
        "university", "institute", "kanpur", "lucknow", "signature", "submitted",
        "fulfillment", "requirements", "supervision", "head", "professor", "doctor",
        "project", "report", "certificate", "declaration", "acknowledgements",
        "urgent", "care", "india", "medical", "healthcare", "patient", "patients",
        "hospital", "hospitals", "platform", "services", "availability", "critical",
        "access", "routine", "emergency", "public", "system", "information",
        "under", "supervision"
    }

    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in bad_words):
        return False

    if re.search(r"\d", name):
        return False

    if "." in name:
        return False

    parts = name.split()

    if len(parts) < 1 or len(parts) > 4:
        return False

    for part in parts:
        clean = re.sub(r"[^A-Za-z]", "", part)

        if not clean:
            return False

        if not ((clean[0].isupper() and clean[1:].islower()) or clean.isupper()):
            return False

        if clean.lower() in bad_words:
            return False

    return True


def person_key(name):
    parts = clean_person_name(name).lower().split()

    if not parts:
        return ""

    if len(parts) >= 2:
        return re.sub(r"[^a-z]", "", parts[-1] + parts[0][:3])

    return re.sub(r"[^a-z]", "", parts[0])


def supervisor_key(name):
    key = re.sub(r"[^a-z]", "", clean_person_name(name).lower())

    if key in {"nyancy", "nayncy"}:
        return "nyancy"

    return key


def choose_best_name(existing, new):
    existing_clean = title_case_name(existing)
    new_clean = title_case_name(new)

    if len(new_clean) > len(existing_clean):
        return new_clean

    return existing_clean


def get_front_text(chunks, max_front_chunks=24):
    front_chunks = chunks[:max_front_chunks]
    return "\n".join(chunk.get("text", "") for chunk in front_chunks)


def dedupe_student_records(records):
    grouped = {}

    for record in records:
        name = title_case_name(record.get("name", ""))
        roll = str(record.get("roll_no") or "").strip()
        confidence = float(record.get("confidence", 0.0))

        if not looks_like_person_name(name):
            continue

        key = roll if roll else person_key(name)

        if not key:
            continue

        if key not in grouped:
            grouped[key] = {
                "name": name,
                "roll_no": roll or None,
                "confidence": confidence
            }
        else:
            old = grouped[key]

            if confidence >= old.get("confidence", 0.0):
                grouped[key] = {
                    "name": name,
                    "roll_no": roll or old.get("roll_no"),
                    "confidence": confidence
                }

    return list(grouped.values())


def extract_student_records(chunks, max_front_chunks=24):
    text = get_front_text(chunks, max_front_chunks=max_front_chunks)
    records = []

    # Normalize joined names like RitikChaurasia before regex matching.
    normalized_text = split_camel_name(text)

    roll_pattern = re.compile(
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s*"
        r"\(\s*Roll\s*no\.?\s*[:\-]?\s*([0-9]{6,})\s*\)",
        flags=re.IGNORECASE
    )

    for match in roll_pattern.finditer(normalized_text):
        name = title_case_name(match.group(1))
        roll = match.group(2).strip()

        if looks_like_person_name(name):
            records.append({
                "name": name,
                "roll_no": roll,
                "confidence": 1.0
            })

    # Backup: lines with name and roll number but no clean parentheses.
    loose_roll_pattern = re.compile(
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}).{0,60}?"
        r"Roll\s*no\.?\s*[:\-]?\s*([0-9]{6,})",
        flags=re.IGNORECASE
    )

    for match in loose_roll_pattern.finditer(normalized_text):
        name = title_case_name(match.group(1))
        roll = match.group(2).strip()

        if looks_like_person_name(name):
            records.append({
                "name": name,
                "roll_no": roll,
                "confidence": 0.9
            })

    name_roll_pattern = re.compile(
        r"\bName\s*:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}).{0,120}?"
        r"\bRoll\s*No\.?\s*[:\-]?\s*([0-9]{6,})",
        flags=re.IGNORECASE | re.DOTALL
    )

    for match in name_roll_pattern.finditer(normalized_text):
        name = title_case_name(match.group(1))
        roll = match.group(2).strip()

        if looks_like_person_name(name):
            records.append({
                "name": name,
                "roll_no": roll,
                "confidence": 0.95
            })

    if records:
        return dedupe_student_records(records)

    names = extract_people_from_front_matter(chunks, max_front_chunks=max_front_chunks)

    return [
        {
            "name": name,
            "roll_no": None,
            "confidence": 0.6
        }
        for name in names
    ]


def dedupe_people(names):
    grouped = {}

    for name in names:
        name = title_case_name(name)

        if not looks_like_person_name(name):
            continue

        key = person_key(name)

        if not key:
            continue

        if key not in grouped:
            grouped[key] = name
        else:
            grouped[key] = choose_best_name(grouped[key], name)

    return list(grouped.values())


def extract_people_from_front_matter(chunks, max_front_chunks=24):
    candidates = []

    for record in extract_student_records(chunks, max_front_chunks=max_front_chunks):
        if record.get("name"):
            candidates.append(record["name"])

    return dedupe_people(candidates)


def extract_supervisor_records(chunks, max_front_chunks=24):
    text = split_camel_name(get_front_text(chunks, max_front_chunks=max_front_chunks))
    records = []

    patterns = [
        r"\bUnder\s+the\s+Supervision\s+of\s+(?:Dr\.?|Prof\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
        r"\bproject\s+supervisor\s+(?:Dr\.?|Prof\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
        r"\bsupervisor\s+(?:Dr\.?|Prof\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
        r"\bguide\s+(?:Dr\.?|Prof\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = match.group(1)
            raw = re.sub(r"\bHead\b.*$", "", raw, flags=re.IGNORECASE).strip()
            name = title_case_name(raw)

            if looks_like_person_name(name):
                records.append(name)

    grouped = {}

    for name in records:
        key = supervisor_key(name)

        if not key:
            continue

        if key not in grouped:
            grouped[key] = title_case_name(name)
        else:
            grouped[key] = choose_best_name(grouped[key], name)

    return list(grouped.values())


def answer_people_question(question, chunks):
    q = question.lower()

    wants_roll = bool(re.search(r"\broll|roll\s*number|roll\s*no|enrollment|registration\b", q))
    wants_supervisor = bool(re.search(r"\bsupervisor|guide|under supervision|mentor\b", q))
    wants_students = bool(re.search(r"\bstudent|students|submitted by|prepared by|team|members|group|contributors\b", q))

    if wants_supervisor:
        supervisors = extract_supervisor_records(chunks)

        if not supervisors:
            return "Not found in document."

        return "\n".join(f"- {name}" for name in supervisors)

    if wants_students or wants_roll:
        records = extract_student_records(chunks)

        if not records:
            return "Not found in document."

        lines = []

        for record in records:
            name = record.get("name", "").strip()
            roll = record.get("roll_no")

            if wants_roll and roll:
                lines.append(f"- {name} — Roll No. {roll}")
            else:
                lines.append(f"- {name}")

        return "\n".join(lines)

    people = extract_people_from_front_matter(chunks)

    if not people:
        return "Not found in document."

    return "\n".join(f"- {name}" for name in people)


def format_people_answer(names):
    if not names:
        return "Not found in document."

    return "\n".join(f"- {name}" for name in names)


def format_extracted_items(items, max_items=35):
    items = dedupe_candidates(items)

    if not items:
        return "Not found in document."

    return "\n".join(f"- {item}" for item in items[:max_items])


def extract_items_from_chunks(chunks, prefer_tables=False):
    if prefer_tables:
        table_items = extract_table_items(chunks)

        if table_items:
            return table_items

    table_items = extract_table_items(chunks)
    technical_items = extract_technical_items(chunks)

    return dedupe_candidates(table_items + technical_items)