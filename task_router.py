# task_router.py

import re


def normalize_question(question):
    if question is None:
        return ""

    question = str(question).lower().strip()
    question = re.sub(r"\s+", " ", question)

    return question


def classify_task(question):
    """
    Simple generic router.

    Most questions go to normal QA.
    Avoid over-routing because that causes noisy behavior.
    """

    q = normalize_question(question)

    if not q:
        return "qa"

    if any(term in q for term in [
        "translate",
        "translation",
        "convert to hindi",
        "convert to english",
        "in hindi",
        "in english",
    ]):
        return "translation"

    if any(term in q for term in [
        "summarize",
        "summary",
        "short summary",
        "detailed summary",
        "brief summary",
        "overview",
        "gist",
        "main idea",
    ]):
        return "summary"

    if re.search(r"\bpage\s+\d+\b", q):
        return "section_qa"

    if any(term in q for term in [
        "table",
        "tables",
        "tabular",
        "columns",
        "rows",
    ]):
        return "table_list"

    if any(term in q for term in [
        "list all",
        "give all",
        "extract all",
        "all points",
        "all features",
        "all advantages",
        "all disadvantages",
        "all limitations",
        "all requirements",
    ]):
        return "extract_list"

    return "qa"


def needs_embedding(task_type):
    return task_type in {
        "qa",
        "section_qa",
        "table_list",
        "extract_list",
        "summary",
    }