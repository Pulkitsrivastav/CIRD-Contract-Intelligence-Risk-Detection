"""
document_utils.py
Handles PDF validation, text extraction via PyMuPDF (fitz), and local report archiving.
Designed for clear structure, beginner readability, and safe error handling.
"""

import os
import re
from datetime import datetime
import fitz  # PyMuPDF

# Minimum character threshold to distinguish valid text-based PDFs from scanned/empty PDFs
MIN_VALID_TEXT_LENGTH = 200
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200


def extract_text_from_pdf(file_bytes: bytes, filename: str = "uploaded_contract.pdf") -> dict:
    """
    Extracts text page-by-page from a PDF byte stream using PyMuPDF (fitz).

    Args:
        file_bytes: Raw binary content of the uploaded PDF file.
        filename: Original name of the uploaded file.

    Returns:
        A dictionary containing extraction status, page-level text, combined text,
        word/character statistics, and descriptive error messages if extraction fails.
    """
    if not file_bytes:
        return {
            "success": False,
            "error": "The uploaded file is empty (0 bytes). Please upload a valid PDF document.",
            "filename": filename,
        }
    if len(file_bytes) > MAX_PDF_BYTES:
        return {
            "success": False,
            "error": f"The PDF is larger than the {MAX_PDF_BYTES // (1024 * 1024)} MB upload limit. Please use a smaller document.",
            "filename": filename,
        }

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        return {
            "success": False,
            "error": f"Unable to open PDF file. The file may be corrupted or invalid ({exc}).",
            "filename": filename,
        }

    if doc.is_encrypted or getattr(doc, "needs_pass", False):
        doc.close()
        return {
            "success": False,
            "error": "This PDF is password-protected/encrypted. Please upload an unlocked text-based PDF.",
            "filename": filename,
        }

    total_pages = len(doc)
    if total_pages > MAX_PDF_PAGES:
        doc.close()
        return {
            "success": False,
            "error": f"The PDF has {total_pages} pages. The current academic demo limit is {MAX_PDF_PAGES} pages.",
            "filename": filename,
            "page_count": total_pages,
        }
    if total_pages == 0:
        doc.close()
        return {
            "success": False,
            "error": "The uploaded PDF contains zero pages.",
            "filename": filename,
        }

    pages_data = []
    formatted_chunks = []
    empty_pages_count = 0

    try:
        for page_index in range(total_pages):
            page_number = page_index + 1
            page = doc.load_page(page_index)
            raw_text = page.get_text("text") or ""
            cleaned_text = raw_text.strip()

            if not cleaned_text:
                empty_pages_count += 1
                pages_data.append({
                    "page_number": page_number,
                    "text": "",
                    "is_empty": True,
                    "word_count": 0,
                })
            else:
                words_in_page = len(cleaned_text.split())
                pages_data.append({
                    "page_number": page_number,
                    "text": cleaned_text,
                    "is_empty": False,
                    "word_count": words_in_page,
                })
                formatted_chunks.append(f"--- Page {page_number} ---\n{cleaned_text}")
        metadata = doc.metadata or {}
    except Exception as exc:
        doc.close()
        return {
            "success": False,
            "error": f"Unable to extract text from the PDF pages: {exc}",
            "filename": filename,
            "page_count": total_pages,
        }
    finally:
        if not doc.is_closed:
            doc.close()

    full_text = "\n\n".join(formatted_chunks)
    plain_text_only = " ".join(p["text"] for p in pages_data if p["text"]).strip()
    char_count = len(plain_text_only)
    word_count = len(plain_text_only.split()) if plain_text_only else 0

    if char_count < MIN_VALID_TEXT_LENGTH:
        return {
            "success": False,
            "error": (
                f"Insufficient readable text extracted ({char_count} characters across {total_pages} page(s)). "
                "This PDF may be scanned/image-only or nearly blank. The current version uses PyMuPDF text extraction "
                "and does not include OCR; upload a text-based PDF for analysis."
            ),
            "filename": filename,
            "page_count": total_pages,
            "empty_pages": empty_pages_count,
            "char_count": char_count,
            "word_count": word_count,
        }

    return {
        "success": True,
        "error": None,
        "filename": filename,
        "page_count": total_pages,
        "empty_pages": empty_pages_count,
        "char_count": char_count,
        "word_count": word_count,
        "metadata": {
            "title": metadata.get("title") or "Not specified",
            "author": metadata.get("author") or "Not specified",
        },
        "pages": pages_data,
        "full_text": full_text,
    }


def save_report_files(
    markdown_content: str,
    json_content: str,
    source_filename: str,
    reports_dir: str = "reports",
) -> tuple[str, str]:
    """
    Saves generated Markdown and JSON analysis reports into the local reports/ folder.

    Returns:
        Tuple of (markdown_filepath, json_filepath).
    """
    os.makedirs(reports_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source_filename))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "contract"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    md_path = os.path.join(reports_dir, f"{safe_stem}_analysis_{timestamp}.md")
    json_path = os.path.join(reports_dir, f"{safe_stem}_analysis_{timestamp}.json")

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_content)

    return md_path, json_path
