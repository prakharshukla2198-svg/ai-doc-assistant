from PIL import Image
import pytesseract


# Change this path only if Tesseract is installed somewhere else.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)

    if not text.strip():
        return []

    return [{
        "source": uploaded_file.name,
        "page": "Image",
        "text": text.strip()
    }]


def extract_text_from_multiple_images(uploaded_files):
    all_pages = []

    for uploaded_file in uploaded_files:
        pages = extract_text_from_image(uploaded_file)
        all_pages.extend(pages)

    return all_pages
