import io
import tempfile
import os

import pymupdf4llm
import mammoth
from markdownify import markdownify


def convert_pdf_to_markdown(file_bytes: bytes) -> str:
    """Convert PDF bytes to Markdown using pymupdf4llm.

    Faithful 1:1 conversion — no content modification or summarization.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    return md_text


def convert_docx_to_markdown(file_bytes: bytes) -> str:
    """Convert DOCX bytes to Markdown via mammoth (DOCX->HTML) + markdownify (HTML->MD).

    Faithful 1:1 conversion — no content modification or summarization.
    """
    buf = io.BytesIO(file_bytes)
    result = mammoth.convert_to_html(buf)
    html = result.value
    md_text = markdownify(html, heading_style="ATX", strip=["img"])
    return md_text
