# AI Document Assistant

[Live Demo](https://ai-doc-assistant-prakhar.streamlit.app)

AI Document Assistant is a Streamlit-based document intelligence application for PDFs and images. It supports document-grounded question answering, chat history recall, short/detailed/bullet summaries, and evidence-backed responses.

## Features

- Upload PDF, PNG, JPG, and JPEG files
- Multi-file document workspace
- Document-grounded Q&A
- Chat history recall for follow-up questions
- Short, detailed, and bullet summary modes
- Source/evidence display
- Summary download
- Dark themed Streamlit interface
- Deployment-ready API key handling using Streamlit secrets

## Tech Stack

- Python
- Streamlit
- Gemini Document Understanding
- Google GenAI SDK
- pypdf
- pdfplumber
- Pillow
- pytesseract
- NumPy
- Git/GitHub
- Streamlit Cloud

## Architecture

```text
User uploads PDF/Image
        ↓
Local document preview
        ↓
Gemini Files API upload
        ↓
Gemini document understanding
        ↓
Question Answering / Summarization
        ↓
Answer + Evidence display