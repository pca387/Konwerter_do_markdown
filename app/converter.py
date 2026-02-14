import io
import re
import tempfile
import os
from collections import Counter

import pymupdf
import pymupdf4llm
import mammoth
from markdownify import markdownify

# Margin zone in points (1 point = 1/72 inch). Text spans falling inside
# this zone AND matching the short-text heuristic are treated as paraphs.
_MARGIN_LEFT = 50
_MARGIN_RIGHT = 50
_MARGIN_TOP = 36
_MARGIN_BOTTOM = 36

# Handwriting-style font name fragments (case-insensitive).
_HANDWRITING_KEYWORDS = [
    "script", "hand", "signature", "cursive", "brush",
    "marker", "pen", "ink", "writing", "freestyle",
]

# Pattern: meaningful handwritten content (dates, amounts) we want to KEEP.
_MEANINGFUL_RE = re.compile(
    r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}"  # dates
    r"|[\d\s,.]+\s*(zl|PLN|USD|EUR|pln|zł)"  # amounts with currency
    r"|\d+[.,]\d{2}"  # decimal amounts
)


def _get_dominant_font(doc: pymupdf.Document) -> str:
    """Return the single most frequently used font name across the document."""
    font_counter: Counter[str] = Counter()
    for page in doc:
        blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        font_counter[span["font"]] += len(text)
    if font_counter:
        return font_counter.most_common(1)[0][0]
    return ""


def _is_handwriting_font(font_name: str) -> bool:
    low = font_name.lower()
    return any(kw in low for kw in _HANDWRITING_KEYWORDS)


def _span_in_margin(bbox: tuple, page_rect: pymupdf.Rect) -> bool:
    """Check if a span's bounding box falls within the page margin zone."""
    x0, y0, x1, y1 = bbox
    return (
        x0 < page_rect.x0 + _MARGIN_LEFT
        or x1 > page_rect.x1 - _MARGIN_RIGHT
        or y0 < page_rect.y0 + _MARGIN_TOP
        or y1 > page_rect.y1 - _MARGIN_BOTTOM
    )


def _is_margin_paraph(span: dict, page_rect: pymupdf.Rect,
                      dominant_font: str) -> bool:
    """Decide whether a text span is a handwritten margin paraph to remove.

    Keep the span if it contains meaningful content (dates, amounts).
    """
    text = span["text"].strip()
    if not text:
        return False

    # Always keep meaningful content (dates, amounts)
    if _MEANINGFUL_RE.search(text):
        return False

    in_margin = _span_in_margin(span["bbox"], page_rect)
    is_hw_font = _is_handwriting_font(span["font"])
    is_foreign_font = span["font"] != dominant_font
    is_short = len(text) <= 4

    # Definite paraph: handwriting font in margin
    if is_hw_font and in_margin:
        return True

    # Short text in margin with a non-dominant font — likely initials/paraph
    if in_margin and is_short and is_foreign_font:
        return True

    return False


def _clean_pdf(file_bytes: bytes) -> bytes:
    """Remove handwritten margin paraphs from a PDF, return cleaned PDF bytes.

    Keeps dates, amounts, and all main-body text intact.
    """
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    dominant_font = _get_dominant_font(doc)

    for page in doc:
        page_rect = page.rect
        blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]

        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if _is_margin_paraph(span, page_rect, dominant_font):
                        rect = pymupdf.Rect(span["bbox"])
                        page.add_redact_annot(rect)

        page.apply_redactions()

    cleaned = doc.tobytes()
    doc.close()
    return cleaned


def _is_structural_line(line: str) -> bool:
    """Check if a line is a Markdown structural element that should not be merged."""
    s = line.strip()
    return bool(
        s.startswith("#")
        or s.startswith("|")
        or s.startswith("```")
        or s.startswith("---")
        or s.startswith("***")
        or s.startswith("___")
        or s.startswith("![")
        or re.match(r"^(\d+\.|[-*+])\s", s)
    )


def _merge_broken_lines(md: str) -> str:
    """Merge lines that were broken by PDF layout into continuous paragraphs.

    pymupdf4llm outputs each PDF text line separated by \\n\\n (double newline).
    Real paragraph breaks appear as \\n\\n\\n (triple). This function:
    1. Splits on real paragraph breaks (3+ newlines)
    2. Within each paragraph block, joins continuation lines with spaces
    3. Preserves headings, lists, tables, code blocks, and other structure
    """
    # Normalize: 3+ newlines = real paragraph break, mark them
    PARA_BREAK = "\n\x00PARA\x00\n"
    text = re.sub(r"\n{3,}", PARA_BREAK, md)

    # Split into paragraph blocks
    blocks = text.split(PARA_BREAK)

    result_blocks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        # Remove empty lines within the block (artifacts of double-newline splitting)
        lines = [l for l in lines if l.strip()]

        if not lines:
            continue

        # If the block is a code fence, table, or other structure — keep as-is
        if lines[0].strip().startswith("```"):
            result_blocks.append("\n".join(lines))
            continue

        # Merge continuation lines within the block
        merged_lines = []
        current = ""
        for line in lines:
            if _is_structural_line(line):
                if current:
                    merged_lines.append(current)
                    current = ""
                merged_lines.append(line.strip())
            else:
                if current:
                    current += " " + line.strip()
                else:
                    current = line.strip()
        if current:
            merged_lines.append(current)

        result_blocks.append("\n\n".join(merged_lines))

    return "\n\n".join(result_blocks).strip() + "\n"


def convert_pdf_to_markdown(file_bytes: bytes) -> str:
    """Convert PDF bytes to Markdown using pymupdf4llm.

    Pre-processes the PDF to remove handwritten margin paraphs/initials
    while keeping dates, amounts, and all typed content.
    """
    cleaned_bytes = _clean_pdf(file_bytes)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(cleaned_bytes)
        tmp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    return _merge_broken_lines(md_text)


def convert_docx_to_markdown(file_bytes: bytes) -> str:
    """Convert DOCX bytes to Markdown via mammoth (DOCX->HTML) + markdownify (HTML->MD).

    Faithful 1:1 conversion — no content modification or summarization.
    """
    buf = io.BytesIO(file_bytes)
    result = mammoth.convert_to_html(buf)
    html = result.value
    md_text = markdownify(html, heading_style="ATX", strip=["img"])
    return md_text
