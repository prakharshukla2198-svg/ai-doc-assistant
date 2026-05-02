# online_doc_ai.py

import os
import re
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-flash"

SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def get_api_key() -> Optional[str]:
    """
    Reads Gemini API key from Streamlit secrets or environment variables.
    """

    try:
        key_from_secrets = st.secrets.get("GEMINI_API_KEY", None)
        if key_from_secrets:
            return str(key_from_secrets).strip()
    except Exception:
        pass

    key_from_env = os.getenv("GEMINI_API_KEY")

    if key_from_env:
        return key_from_env.strip()

    return None


def has_api_key() -> bool:
    return bool(get_api_key())


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def get_file_signature(uploaded_files) -> tuple:
    return tuple(
        (
            uploaded_file.name,
            uploaded_file.size,
            uploaded_file.type,
        )
        for uploaded_file in uploaded_files
    )


def guess_mime_type(file_name: str, uploaded_type: Optional[str] = None) -> str:
    suffix = Path(file_name).suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[suffix]

    if uploaded_type:
        return uploaded_type

    return "application/octet-stream"


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def upload_files_to_gemini(uploaded_files) -> List[Any]:
    """
    Uploads PDF/image files to Gemini Files API.
    """

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit secrets or environment variables."
        )

    client = get_gemini_client(api_key)

    gemini_files = []
    temp_paths = []

    try:
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix.lower()

            if suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type: {uploaded_file.name}. "
                    "Use PDF, PNG, JPG, or JPEG."
                )

            temp_path = save_uploaded_file_to_temp(uploaded_file)
            temp_paths.append(temp_path)

            mime_type = guess_mime_type(uploaded_file.name, uploaded_file.type)

            uploaded = client.files.upload(
                file=temp_path,
                config={
                    "mime_type": mime_type,
                    "display_name": uploaded_file.name,
                },
            )

            uploaded = wait_for_file_ready(client, uploaded)
            gemini_files.append(uploaded)

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception:
                pass

    return gemini_files


def wait_for_file_ready(client, uploaded_file, timeout_seconds: int = 120):
    start_time = time.time()
    file_obj = uploaded_file

    while True:
        state_name = get_file_state_name(file_obj)

        if state_name in {"ACTIVE", "READY", "SUCCEEDED", ""}:
            return file_obj

        if state_name in {"FAILED", "ERROR"}:
            raise RuntimeError(f"Gemini file processing failed for: {file_obj.name}")

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(
                f"Timed out while processing file: {getattr(file_obj, 'name', 'unknown')}"
            )

        time.sleep(2)

        try:
            file_obj = client.files.get(name=file_obj.name)
        except Exception:
            return uploaded_file


def get_file_state_name(file_obj) -> str:
    state = getattr(file_obj, "state", None)

    if state is None:
        return ""

    name = getattr(state, "name", None)

    if name:
        return str(name).upper()

    return str(state).upper()


def build_history_text(chat_history: List[Dict[str, str]], max_turns: int = 6) -> str:
    if not chat_history:
        return "No previous conversation."

    recent = chat_history[-max_turns:]
    lines = []

    for item in recent:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if question:
            lines.append(f"User: {question}")

        if answer:
            lines.append(f"Assistant: {answer}")

    return "\n".join(lines).strip() or "No previous conversation."


def answer_from_documents(
    question: str,
    gemini_files: List[Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Answers a question from uploaded documents.

    Returns:
    {
        "answer": "...",
        "sources": [...]
    }
    """

    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please ask a valid question.",
            "sources": [],
        }

    if not gemini_files:
        return {
            "answer": "No document is uploaded.",
            "sources": [],
        }

    api_key = get_api_key()

    if not api_key:
        return {
            "answer": "Gemini API key is missing.",
            "sources": [],
        }

    client = get_gemini_client(api_key)
    history_text = build_history_text(chat_history or [])

    prompt = f"""
You are an accurate document question-answering assistant.

Use only the uploaded document files and conversation history.

Conversation history is only for understanding follow-up questions.
The factual answer must come from the uploaded documents.

Rules:
- Do not use outside knowledge.
- Do not invent names, numbers, dates, IDs, fees, addresses, claims, or conclusions.
- If the answer is not present in the uploaded documents, answer exactly:
  "Not found in document."
- Preserve exact values from the documents.
- If multiple files are uploaded, compare across all relevant files.
- Keep the answer complete but concise.
- For comparisons, use a clean markdown table when useful.
- Do not return JSON as visible prose.
- Do not wrap the response in code fences.

Return ONLY valid JSON in this exact format:
{{
  "answer": "final user-facing answer in markdown, without JSON braces",
  "sources": [
    {{
      "source": "file name if known",
      "page": "page number if known, otherwise Unknown",
      "section": "section or heading if known, otherwise Unknown",
      "text": "short supporting evidence from the document"
    }}
  ]
}}

Conversation history:
{history_text}

Current question:
{question}
""".strip()

    contents = []
    contents.extend(gemini_files)
    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.9,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )

        raw_text = extract_response_text(response)
        parsed = parse_json_response(raw_text)

        if parsed:
            answer = str(parsed.get("answer", "")).strip()
            sources = parsed.get("sources", [])

            return {
                "answer": clean_user_answer(answer),
                "sources": clean_sources(sources),
            }

        fallback_answer = extract_answer_from_broken_json(raw_text)

        return {
            "answer": clean_user_answer(fallback_answer or raw_text or "No response text returned."),
            "sources": [],
        }

    except Exception as error:
        return {
            "answer": f"Error from Gemini: {str(error)}",
            "sources": [],
        }


def summarize_documents(
    gemini_files: List[Any],
    summary_mode: str,
    model_name: str = DEFAULT_MODEL,
) -> str:
    if not gemini_files:
        return "No document is uploaded."

    api_key = get_api_key()

    if not api_key:
        return "Gemini API key is missing."

    client = get_gemini_client(api_key)

    if summary_mode == "Short Summary":
        instruction = """
Create a short summary of the uploaded document files.
Include the main topic, purpose, and most important points.
Keep it concise.
""".strip()

    elif summary_mode == "Detailed Summary":
        instruction = """
Create a detailed summary of the uploaded document files.
Cover the objective, background, important sections, methodology, features, findings, and conclusion where available.
Do not invent missing information.
""".strip()

    else:
        instruction = """
Create a bullet-point summary of the uploaded document files.
Use clear bullets.
Each bullet should contain one important document-supported idea.
Do not invent missing information.
""".strip()

    prompt = f"""
You are a document summarization assistant.

Use only the uploaded document files.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If multiple files are uploaded, combine their content logically.
- Preserve important names, numbers, dates, headings, and conclusions when relevant.
- Return normal markdown text only, not JSON.

Task:
{instruction}
""".strip()

    contents = []
    contents.extend(gemini_files)
    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.9,
                max_output_tokens=2500,
            ),
        )

        return clean_user_answer(extract_response_text(response))

    except Exception as error:
        return f"Error from Gemini: {str(error)}"


def clean_sources(sources) -> List[Dict[str, str]]:
    if not isinstance(sources, list):
        return []

    clean = []

    for source in sources[:8]:
        if not isinstance(source, dict):
            continue

        text = str(source.get("text", "")).strip()

        clean.append({
            "source": str(source.get("source", "Uploaded document")).strip() or "Uploaded document",
            "page": str(source.get("page", "Unknown")).strip() or "Unknown",
            "section": str(source.get("section", "Unknown")).strip() or "Unknown",
            "text": text,
            "type": "Evidence",
        })

    return clean


def extract_response_text(response) -> str:
    text = getattr(response, "text", None)

    if text:
        return str(text).strip()

    try:
        candidates = getattr(response, "candidates", [])
        parts = candidates[0].content.parts
        return "\n".join(
            getattr(part, "text", "")
            for part in parts
            if getattr(part, "text", "")
        ).strip()
    except Exception:
        return "No response text returned."


def parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None

    text = raw_text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def extract_answer_from_broken_json(raw_text: str) -> str:
    """
    If Gemini output gets truncated or malformed as JSON,
    extract just the answer value so the UI never shows raw JSON.
    """

    if not raw_text:
        return ""

    text = raw_text.strip()

    match = re.search(
        r'"answer"\s*:\s*"(.*?)(?:"\s*,\s*"sources"|"sources"\s*:|\}\s*$)',
        text,
        flags=re.DOTALL,
    )

    if match:
        value = match.group(1)
        value = value.replace("\\n", "\n")
        value = value.replace('\\"', '"')
        value = value.replace("\\/", "/")
        return value.strip()

    if text.startswith("{"):
        text = re.sub(r'^\s*\{\s*"?answer"?\s*:\s*"?', "", text, flags=re.IGNORECASE)
        text = re.sub(r'"?\s*,\s*"?sources"?\s*:.*$', "", text, flags=re.DOTALL)
        text = text.strip().strip('"').strip()
        text = text.replace("\\n", "\n")
        return text

    return text


def clean_user_answer(answer: str) -> str:
    """
    Final safety layer so raw JSON is never displayed.
    """

    if not answer:
        return ""

    text = str(answer).strip()

    if text.startswith("{") and '"answer"' in text:
        extracted = extract_answer_from_broken_json(text)
        if extracted:
            text = extracted

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = text.replace("\\n", "\n")
    text = text.replace('\\"', '"')
    text = text.strip()

    return text


def delete_gemini_files(gemini_files: List[Any]) -> None:
    api_key = get_api_key()

    if not api_key:
        return

    client = get_gemini_client(api_key)

    for file_obj in gemini_files:
        try:
            client.files.delete(name=file_obj.name)
        except Exception:
            pass