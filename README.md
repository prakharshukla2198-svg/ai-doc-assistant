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
Streamlit file manager
        - Validates supported file types
        - Handles multi-file workspace
        - Maintains selected document state
        ↓
Document preprocessing layer
        - Extracts basic text/metadata where possible
        - Supports PDF and image inputs
        - Uses OCR fallback for image-based content
        ↓
AI orchestration layer
        - Builds task-specific prompts
        - Handles Q&A, summaries, and follow-up questions
        - Sends document context to Gemini Document Understanding
        ↓
Response generation
        - Produces document-grounded answers
        - Generates short, detailed, and bullet summaries
        - Maintains chat history for contextual follow-ups
        ↓
Evidence and UI layer
        - Displays answer with supporting evidence
        - Shows uploaded document workspace
        - Supports summary download
