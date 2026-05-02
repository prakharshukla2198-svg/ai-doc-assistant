import re
from pypdf import PdfReader
import pdfplumber


def _get_file_name(uploaded_file):
    return getattr(uploaded_file, "name", "uploaded_document.pdf")


def clean_text(text):
    if not text:
        return ""

    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def table_to_markdown(table):
    if not table:
        return ""

    rows = []

    for row in table:
        cleaned_row = []

        for cell in row:
            cell_text = clean_text(cell if cell is not None else "")
            cell_text = cell_text.replace("\n", " ").strip()
            cleaned_row.append(cell_text)

        if any(cell.strip() for cell in cleaned_row):
            rows.append(cleaned_row)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    header = rows[0]
    body = rows[1:]

    markdown_rows = []
    markdown_rows.append("| " + " | ".join(header) + " |")
    markdown_rows.append("| " + " | ".join(["---"] * max_cols) + " |")

    for row in body:
        markdown_rows.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_rows)


def extract_text_from_pdf(uploaded_file):
    file_name = _get_file_name(uploaded_file)
    pages = []

    uploaded_file.seek(0)

    try:
        pdf_reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = clean_text(page.extract_text())

            if page_text:
                pages.append({
                    "source": file_name,
                    "page": page_number,
                    "type": "text",
                    "text": page_text
                })

    except Exception:
        pass

    uploaded_file.seek(0)

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []

                for table_index, table in enumerate(tables, start=1):
                    table_text = table_to_markdown(table)

                    if table_text:
                        pages.append({
                            "source": file_name,
                            "page": page_number,
                            "type": "table",
                            "table_index": table_index,
                            "text": table_text
                        })

    except Exception:
        pass

    return pages


def extract_text_from_multiple_pdfs(uploaded_files):
    all_pages = []

    for uploaded_file in uploaded_files:
        pages = extract_text_from_pdf(uploaded_file)
        all_pages.extend(pages)

    return all_pages