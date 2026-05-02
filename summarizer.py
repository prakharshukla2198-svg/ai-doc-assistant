import re
from collections import defaultdict, Counter


GENERIC_IMPORTANT_SECTION_TERMS = [
    "abstract",
    "executive summary",
    "overview",
    "introduction",
    "background",
    "motivation",
    "problem statement",
    "objectives",
    "objective",
    "scope",
    "literature review",
    "related work",
    "methodology",
    "method",
    "approach",
    "proposed system",
    "system design",
    "architecture",
    "high level design",
    "low level design",
    "implementation",
    "experiment",
    "experiments",
    "evaluation",
    "results",
    "result",
    "discussion",
    "analysis",
    "findings",
    "limitations",
    "limitation",
    "future scope",
    "future work",
    "conclusion",
    "recommendations"
]


NOISE_SECTION_TERMS = {
    "certificate",
    "declaration",
    "acknowledgement",
    "acknowledgements",
    "table of contents",
    "contents",
    "list of figures",
    "list of tables",
    "references",
    "bibliography",
    "appendix"
}


GENERIC_IMPORTANT_WORDS = [
    "aim",
    "aims",
    "goal",
    "goals",
    "objective",
    "objectives",
    "purpose",
    "problem",
    "challenge",
    "solution",
    "proposed",
    "method",
    "methodology",
    "approach",
    "framework",
    "system",
    "model",
    "architecture",
    "design",
    "implementation",
    "process",
    "workflow",
    "data",
    "analysis",
    "result",
    "results",
    "finding",
    "findings",
    "performance",
    "evaluation",
    "benefit",
    "benefits",
    "limitation",
    "limitations",
    "future",
    "conclusion",
    "recommendation",
    "recommendations"
]


def clean_text(text):
    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text):
    text = clean_text(text)

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []

    for part in parts:
        sentence = clean_text(part)

        if not sentence:
            continue

        words = sentence.split()

        if len(words) < 8:
            continue

        if len(words) > 65:
            sentence = " ".join(words[:65]).strip() + "..."

        sentences.append(sentence)

    return sentences


def get_section_name(chunk):
    section = clean_text(chunk.get("section", ""))

    if not section or section.lower() == "unknown":
        return f"Page {chunk.get('page', 'Unknown')}"

    return section


def is_noise_chunk(chunk):
    section = normalize(chunk.get("section", ""))
    text_start = normalize(chunk.get("text", "")[:600])
    combined = f"{section} {text_start}"

    return any(term in combined for term in NOISE_SECTION_TERMS)


def section_priority(section):
    section_norm = normalize(section)

    for index, term in enumerate(GENERIC_IMPORTANT_SECTION_TERMS):
        if term in section_norm:
            return index

    return 999


def collect_document_keywords(chunks, max_keywords=40):
    words = []

    for chunk in chunks:
        if is_noise_chunk(chunk):
            continue

        text = normalize(chunk.get("text", ""))

        for word in text.split():
            if len(word) < 5:
                continue

            if word in {
                "there", "their", "which", "would", "could", "should",
                "about", "after", "before", "because", "through", "between",
                "using", "based", "these", "those", "where", "while"
            }:
                continue

            words.append(word)

    counts = Counter(words)

    return {
        word for word, _ in counts.most_common(max_keywords)
    }


def score_sentence(sentence, section="", document_keywords=None):
    sentence_norm = normalize(sentence)
    section_norm = normalize(section)

    score = 0

    if document_keywords is None:
        document_keywords = set()

    for term in GENERIC_IMPORTANT_SECTION_TERMS:
        if term in section_norm:
            score += 4

        if term in sentence_norm[:180]:
            score += 3

    for word in GENERIC_IMPORTANT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", sentence_norm):
            score += 2

    sentence_words = set(sentence_norm.split())
    keyword_overlap = len(sentence_words.intersection(document_keywords))
    score += min(keyword_overlap, 5)

    word_count = len(sentence.split())

    if 12 <= word_count <= 35:
        score += 3
    elif 36 <= word_count <= 55:
        score += 1
    elif word_count > 55:
        score -= 2

    if re.search(r"\b(first|second|third|finally|therefore|however|moreover|overall|in conclusion)\b", sentence_norm):
        score += 1

    if re.search(r"\b\d+(\.\d+)?\b", sentence_norm):
        score += 1

    if sentence_norm.count(",") > 5:
        score -= 1

    return score


def group_chunks_by_section(chunks):
    grouped = defaultdict(list)

    for chunk in chunks:
        if is_noise_chunk(chunk):
            continue

        section = get_section_name(chunk)
        grouped[section].append(chunk)

    return grouped


def extract_best_sentences_from_chunks(chunks, max_sentences=5, document_keywords=None):
    scored = []

    for chunk in chunks:
        section = get_section_name(chunk)
        text = chunk.get("text", "")

        for sentence in split_sentences(text):
            score = score_sentence(
                sentence=sentence,
                section=section,
                document_keywords=document_keywords
            )

            if score > 0:
                scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)

    final = []
    seen = set()

    for _, sentence in scored:
        key = normalize(sentence)[:140]

        if key in seen:
            continue

        seen.add(key)
        final.append(sentence)

        if len(final) >= max_sentences:
            break

    return final


def extract_document_title(chunks):
    for chunk in chunks[:10]:
        text = clean_text(chunk.get("text", ""))
        lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]

        for line in lines[:12]:
            word_count = len(line.split())
            line_norm = normalize(line)

            if 2 <= word_count <= 14:
                if not any(noise in line_norm for noise in NOISE_SECTION_TERMS):
                    if not re.fullmatch(r"\d+", line_norm):
                        return line

    return "Uploaded document"


def select_relevant_chunks(chunks, max_chunks=30):
    selected = []
    seen = set()

    first_chunks = [
        chunk for chunk in chunks[:8]
        if not is_noise_chunk(chunk)
    ]

    important_chunks = []

    for chunk in chunks:
        if is_noise_chunk(chunk):
            continue

        section = normalize(get_section_name(chunk))
        text_start = normalize(chunk.get("text", "")[:600])

        if any(term in section or term in text_start for term in GENERIC_IMPORTANT_SECTION_TERMS):
            important_chunks.append(chunk)

    last_chunks = [
        chunk for chunk in chunks[-6:]
        if not is_noise_chunk(chunk)
    ]

    for chunk in first_chunks + important_chunks + last_chunks:
        key = chunk.get(
            "id",
            (
                chunk.get("source"),
                chunk.get("page"),
                chunk.get("text", "")[:100]
            )
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(chunk)

        if len(selected) >= max_chunks:
            break

    if not selected:
        selected = [chunk for chunk in chunks[:max_chunks] if not is_noise_chunk(chunk)]

    return selected


def summarize_short(chunks):
    if not chunks:
        return "No content to summarize."

    title = extract_document_title(chunks)
    selected_chunks = select_relevant_chunks(chunks, max_chunks=20)
    document_keywords = collect_document_keywords(selected_chunks)

    best_sentences = extract_best_sentences_from_chunks(
        selected_chunks,
        max_sentences=6,
        document_keywords=document_keywords
    )

    if not best_sentences:
        return "No useful summary content found."

    output = [f"**{title} — Short Summary**", ""]

    for sentence in best_sentences:
        output.append(f"- {sentence}")

    return "\n".join(output)


def summarize_bullet(chunks):
    if not chunks:
        return "No content to summarize."

    selected_chunks = select_relevant_chunks(chunks, max_chunks=35)
    grouped = group_chunks_by_section(selected_chunks)
    document_keywords = collect_document_keywords(selected_chunks)

    if not grouped:
        return "No useful summary content found."

    ordered_sections = sorted(grouped.keys(), key=section_priority)

    output = ["**Bullet Summary**", ""]

    used_sections = 0

    for section in ordered_sections:
        if used_sections >= 9:
            break

        section_chunks = grouped[section]

        best_sentences = extract_best_sentences_from_chunks(
            section_chunks,
            max_sentences=3,
            document_keywords=document_keywords
        )

        if not best_sentences:
            continue

        output.append(f"### {section}")

        for sentence in best_sentences:
            output.append(f"- {sentence}")

        output.append("")
        used_sections += 1

    return "\n".join(output).strip()


def summarize_detailed(chunks):
    if not chunks:
        return "No content to summarize."

    selected_chunks = select_relevant_chunks(chunks, max_chunks=40)
    grouped = group_chunks_by_section(selected_chunks)
    document_keywords = collect_document_keywords(selected_chunks)

    if not grouped:
        return "No useful summary content found."

    title = extract_document_title(chunks)

    output = [
        f"**{title} — Detailed Summary**",
        ""
    ]

    section_groups = [
        ("Overview", ["abstract", "executive summary", "overview", "introduction"]),
        ("Problem / Motivation", ["problem", "motivation", "background", "challenge"]),
        ("Objectives / Scope", ["objective", "objectives", "scope", "goal", "goals"]),
        ("Related Work / Literature", ["literature review", "related work"]),
        ("Methodology / Approach", ["methodology", "method", "approach", "proposed system"]),
        ("Architecture / Design", ["architecture", "design", "high level design", "low level design"]),
        ("Implementation / Technology", ["implementation", "technology", "tools", "framework"]),
        ("Results / Findings", ["result", "results", "finding", "findings", "evaluation", "analysis"]),
        ("Limitations / Future Scope / Conclusion", ["limitation", "limitations", "future scope", "future work", "conclusion", "recommendation"])
    ]

    used_sections = set()

    for heading, terms in section_groups:
        matching_chunks = []

        for section, section_chunks in grouped.items():
            section_norm = normalize(section)

            if section in used_sections:
                continue

            if any(term in section_norm for term in terms):
                matching_chunks.extend(section_chunks)
                used_sections.add(section)

        if not matching_chunks:
            continue

        best_sentences = extract_best_sentences_from_chunks(
            matching_chunks,
            max_sentences=4,
            document_keywords=document_keywords
        )

        if not best_sentences:
            continue

        output.append(f"### {heading}")

        for sentence in best_sentences:
            output.append(f"- {sentence}")

        output.append("")

    if len(output) <= 2:
        best_sentences = extract_best_sentences_from_chunks(
            selected_chunks,
            max_sentences=12,
            document_keywords=document_keywords
        )

        output.append("### Key Points")

        for sentence in best_sentences:
            output.append(f"- {sentence}")

    return "\n".join(output).strip()


def summarize_document(chunks, tokenizer=None, model=None, summary_mode="Detailed Summary"):
    if not chunks:
        return "No content to summarize."

    if summary_mode == "Short Summary":
        return summarize_short(chunks)

    if summary_mode == "Bullet Summary":
        return summarize_bullet(chunks)

    return summarize_detailed(chunks)