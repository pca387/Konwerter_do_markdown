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


def _find_recurring_texts(doc: pymupdf.Document) -> set[str]:
    """Find text fragments that repeat at the same position across pages.

    These are headers/footers (e.g. document title, page numbers).
    Returns the set of text strings to strip from output.
    """
    if len(doc) < 2:
        return set()

    # Collect (text, approximate_y_zone) per page
    page_texts: list[set[tuple[str, int]]] = []
    for page in doc:
        texts_on_page: set[tuple[str, int]] = set()
        blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
        page_h = page.rect.height
        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    y_mid = (span["bbox"][1] + span["bbox"][3]) / 2
                    # Zone: top 10% or bottom 10% of page
                    if y_mid < page_h * 0.10:
                        zone = 0  # header zone
                    elif y_mid > page_h * 0.90:
                        zone = 1  # footer zone
                    else:
                        continue  # not in header/footer zone
                    texts_on_page.add((text, zone))
        page_texts.append(texts_on_page)

    # Text is recurring if it appears on 2+ pages in the same zone
    if not page_texts:
        return set()

    all_pairs = page_texts[0]
    for pt in page_texts[1:]:
        all_pairs = all_pairs & pt

    return {text for text, _zone in all_pairs}


_PAGE_NUMBER_RE = re.compile(
    r"^[-–—]?\s*\d{1,4}\s*[-–—]?$"        # "- 1 -", "1", "—2—"
    r"|^(strona|str\.?|page|p\.?)\s*\d+$"  # "strona 1", "page 3"
    r"|^\d+\s*/\s*\d+$",                   # "1 / 5"
    re.IGNORECASE,
)


def _remove_headers_footers(md: str, recurring: set[str]) -> str:
    """Remove recurring header/footer text and page numbers from markdown."""
    if not recurring:
        return md

    lines = md.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().strip("*").strip("#").strip()
        if stripped in recurring:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _remove_page_numbers(md: str) -> str:
    """Remove standalone page number lines."""
    lines = md.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().strip("*").strip()
        if stripped and _PAGE_NUMBER_RE.match(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _clean_header_formatting(md: str) -> str:
    """Fix redundant bold inside headers: '# **text**' -> '# text'."""
    return re.sub(
        r"^(#{1,6})\s+\*\*(.+?)\*\*\s*$",
        r"\1 \2",
        md,
        flags=re.MULTILINE,
    )


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


def _is_list_continuation(text: str, prev_block: str) -> bool:
    """Check if text looks like a continuation of a previous list item."""
    if not prev_block:
        return False
    # Previous block ends with a list item
    last_line = prev_block.strip().split("\n")[-1]
    if not re.match(r"^(\d+\.|[-*+])\s", last_line.strip()):
        return False
    # This text doesn't start a new structural element
    stripped = text.strip()
    if _is_structural_line(stripped):
        return False
    # Looks like a continuation: starts with lowercase, or is indented in source
    if stripped and not stripped[0].isupper():
        return True
    # If it doesn't start with a number/bullet, treat as continuation
    if not re.match(r"^(\d+\.|[-*+])\s", stripped):
        return True
    return False


def _merge_broken_lines(md: str) -> str:
    """Merge lines that were broken by PDF layout into continuous paragraphs.

    pymupdf4llm outputs each PDF text line separated by \\n\\n (double newline).
    Real paragraph breaks appear as \\n\\n\\n (triple). This function:
    1. Splits on real paragraph breaks (3+ newlines)
    2. Within each paragraph block, joins continuation lines with spaces
    3. Preserves headings, lists, tables, code blocks, and other structure
    4. Merges list item continuations across block boundaries
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
            stripped = line.strip()
            is_struct = _is_structural_line(line)
            is_list = bool(re.match(r"^(\d+\.|[-*+])\s", stripped))
            is_heading = stripped.startswith("#")
            is_table = stripped.startswith("|")

            if is_heading or is_table:
                # Headings and tables always stand alone
                if current:
                    merged_lines.append(current)
                    current = ""
                merged_lines.append(stripped)
            elif is_list:
                # New list item — finalize previous, start accumulating
                if current:
                    merged_lines.append(current)
                    current = ""
                current = stripped
            elif current and re.match(r"^(\d+\.|[-*+])\s", current):
                # Continuation of a list item
                current += " " + stripped
            else:
                # Regular text — merge with current
                if current:
                    current += " " + stripped
                else:
                    current = stripped
        if current:
            merged_lines.append(current)

        merged_block = "\n\n".join(merged_lines)

        # Check if this block is a continuation of a previous list item
        if result_blocks and _is_list_continuation(merged_block, result_blocks[-1]):
            result_blocks[-1] = result_blocks[-1] + " " + merged_block.strip()
        else:
            result_blocks.append(merged_block)

    return "\n\n".join(result_blocks).strip() + "\n"


def convert_pdf_to_markdown(file_bytes: bytes) -> str:
    """Convert PDF bytes to Markdown using pymupdf4llm.

    Pipeline:
    1. Detect recurring headers/footers across pages
    2. Remove handwritten margin paraphs
    3. Convert to markdown via pymupdf4llm
    4. Remove headers/footers and page numbers from output
    5. Fix header formatting (redundant bold)
    6. Merge broken lines into paragraphs
    """
    # Detect recurring texts before redaction
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    recurring = _find_recurring_texts(doc)
    doc.close()

    cleaned_bytes = _clean_pdf(file_bytes)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(cleaned_bytes)
        tmp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    md_text = _remove_headers_footers(md_text, recurring)
    md_text = _remove_page_numbers(md_text)
    md_text = _clean_header_formatting(md_text)
    return _merge_broken_lines(md_text)


def _fix_empty_table_headers(md: str) -> str:
    """Fix tables with empty header rows generated by markdownify.

    Promotes the first data row to be the header when the header is empty.
    """
    lines = md.split("\n")
    result = []
    i = 0
    while i < len(lines):
        # Detect pattern: empty header row, separator, data row
        if (
            i + 2 < len(lines)
            and re.match(r"^\|\s*(\|\s*)+$", lines[i].strip())  # empty header
            and re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|?\s*$", lines[i + 1].strip())  # separator
            and lines[i + 2].strip().startswith("|")  # data row
        ):
            # Use first data row as header
            result.append(lines[i + 2])  # promote data row to header
            result.append(lines[i + 1])  # keep separator
            i += 3  # skip the empty header, separator, and promoted row
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def convert_docx_to_markdown(file_bytes: bytes) -> str:
    """Convert DOCX bytes to Markdown via mammoth (DOCX->HTML) + markdownify (HTML->MD).

    Faithful 1:1 conversion — no content modification or summarization.
    """
    buf = io.BytesIO(file_bytes)
    result = mammoth.convert_to_html(buf)
    html = result.value
    md_text = markdownify(html, heading_style="ATX", strip=["img"])
    md_text = _fix_empty_table_headers(md_text)
    return md_text
