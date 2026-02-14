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


_OCR_FONTS = {"glyphlessfont", "cid", "invisible"}


def _needs_ocr(file_bytes: bytes, threshold: float = 0.10) -> bool:
    """Check if a PDF is a pure image scan without a text layer.

    Returns True if fewer than `threshold` fraction of pages contain
    meaningful text (>50 characters). Fast — only extracts text, no OCR.
    """
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    if total == 0:
        doc.close()
        return False
    with_text = sum(1 for p in doc if len(p.get_text().strip()) > 50)
    doc.close()
    return with_text / total < threshold


def _ocr_pdf(file_bytes: bytes, language: str = "pol+eng") -> bytes:
    """Run OCR on a scanned PDF using ocrmypdf CLI, return PDF with text layer.

    Uses --skip-text (safe for mixed documents) and --deskew
    (straightens tilted scans for better OCR quality).

    Called via subprocess because the ocrmypdf Python API conflicts with
    PyMuPDF when running in the same process (produces empty text layers).
    """
    import subprocess

    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_out:
            out_path = tmp_out.name

        result = subprocess.run(
            [
                "ocrmypdf",
                "--skip-text",
                "--deskew",
                "--optimize", "0",
                "-l", language,
                in_path,
                out_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "PriorOcrFoundError" in stderr:
                return file_bytes
            if "tesseract" in stderr.lower() and "not found" in stderr.lower():
                raise RuntimeError(
                    "OCR wymaga zainstalowanego Tesseract. "
                    "macOS: brew install tesseract tesseract-lang | "
                    "Linux: apt install tesseract-ocr tesseract-ocr-pol"
                )
            raise RuntimeError(f"ocrmypdf error: {stderr}")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        if in_path and os.path.exists(in_path):
            os.unlink(in_path)
        if out_path and os.path.exists(out_path):
            os.unlink(out_path)


def _get_font_stats(doc: pymupdf.Document) -> tuple[str, bool]:
    """Return (dominant_font_name, is_ocr_document).

    A document is considered OCR if all text uses OCR-specific fonts
    like GlyphLessFont (Tesseract) or similar invisible text fonts.
    """
    font_counter: Counter[str] = Counter()
    for page in doc:
        blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        font_counter[span["font"]] += len(text)
    if not font_counter:
        return "", False

    dominant = font_counter.most_common(1)[0][0]
    all_ocr = all(f.lower() in _OCR_FONTS for f in font_counter)
    return dominant, all_ocr


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


def _clean_pdf(file_bytes: bytes, is_ocr: bool, dominant_font: str) -> bytes:
    """Remove handwritten margin paraphs from a PDF, return cleaned PDF bytes.

    Skipped for OCR documents (single font — heuristic won't work).
    Keeps dates, amounts, and all main-body text intact.
    """
    if is_ocr:
        return file_bytes

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")

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


def _strip_ocr_backticks(md: str) -> str:
    """Remove inline backtick wrapping that pymupdf4llm adds for monospace/OCR fonts.

    Converts `word` `another` → word another
    Also removes fenced code blocks wrapping normal text.
    """
    # Remove fenced code blocks that wrap regular text (not actual code)
    md = re.sub(r"^```\s*$\n?", "", md, flags=re.MULTILINE)
    # Remove inline backticks around words
    md = re.sub(r"`([^`\n]+?)`", r"\1", md)
    return md


_VOWELS = set("aeiouyąęóAEIOUYĄĘÓ")


def _is_garbage_word(word: str) -> bool:
    """Check if a word looks like garbled OCR output."""
    letters = [c for c in word if c.isalpha()]
    if len(letters) < 3:
        return False
    vowels = sum(1 for c in letters if c in _VOWELS)
    # Real Polish/English words have at least ~20% vowels
    if vowels / len(letters) < 0.15:
        return True
    # Very long words without spaces are likely concatenated garbage
    if len(letters) > 20:
        return True
    # Concatenated words: lowercase→uppercase transition mid-word (e.g. wpisówBrak)
    transitions = sum(1 for i in range(1, len(word) - 1)
                      if word[i].islower() and word[i + 1].isupper())
    if transitions >= 2:
        return True
    return False


def _is_garbage_line(line: str) -> bool:
    """Check if a markdown line is OCR garbage that should be removed.

    Strips markdown formatting before analysis.
    """
    # Strip markdown bold/italic markers for analysis
    clean = re.sub(r"\*{1,2}|_{1,2}", "", line).strip()
    if not clean:
        return False

    # Keep structural markdown
    if _is_structural_line(line.strip()):
        return False

    # Line with very low letter ratio (lots of special chars)
    letters = sum(1 for c in clean if c.isalpha())
    if len(clean) > 5 and letters < len(clean) * 0.4:
        return True

    # Very short line (1-2 meaningful chars)
    if len(clean) <= 2:
        return True

    # Lines with many pipe characters (OCR'd form borders)
    if clean.count("|") >= 3:
        return True

    # Check words
    words = clean.split()

    # Lines with scattered single-letter words (OCR artifact from forms)
    single_letter_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
    if len(words) >= 3 and single_letter_words > len(words) * 0.3:
        return True

    # Lines with long concatenated words (no spaces in OCR)
    long_words = sum(1 for w in words if len(w) > 20)
    if long_words > 0 and len(words) <= 3:
        return True

    # Lines where most words are garbage
    if words:
        garbage_count = sum(1 for w in words if _is_garbage_word(w))
        if len(words) >= 2 and garbage_count > len(words) * 0.4:
            return True
        if len(words) == 1 and _is_garbage_word(words[0]):
            return True

    return False


def _remove_garbage_ocr_lines(md: str) -> str:
    """Remove lines that are clearly OCR garbage from scanned documents.

    Two passes:
    1. Tag each non-empty line as garbage or not
    2. Detect the first point where garbage dominates (rolling window)
       and truncate everything from there onward
    """
    lines = md.split("\n")
    # First pass: tag lines
    tagged: list[tuple[str, bool]] = []  # (line, is_garbage)
    for line in lines:
        if not line.strip():
            tagged.append((line, False))
        else:
            tagged.append((line, _is_garbage_line(line)))

    # Second pass: find truncation point using rolling window
    WINDOW = 8
    non_empty_count = 0
    garbage_in_window: list[bool] = []
    truncate_at = len(tagged)

    for i, (line, is_garb) in enumerate(tagged):
        if not line.strip():
            continue
        garbage_in_window.append(is_garb)
        non_empty_count += 1
        if len(garbage_in_window) > WINDOW:
            garbage_in_window.pop(0)
        # Check window: if >60% is garbage, truncate here
        if (len(garbage_in_window) == WINDOW
                and sum(garbage_in_window) > WINDOW * 0.6):
            # Backtrack to the first garbage line in this window
            backtrack = WINDOW
            for j in range(i, -1, -1):
                if not tagged[j][0].strip():
                    continue
                backtrack -= 1
                if backtrack <= 0:
                    truncate_at = j
                    break
            break

    # Build result: keep lines up to truncation point, skip garbage
    cleaned = []
    for i, (line, is_garb) in enumerate(tagged):
        if i >= truncate_at:
            break
        if is_garb:
            continue
        cleaned.append(line)

    # Trim trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned)


def convert_pdf_to_markdown(file_bytes: bytes, on_status=None) -> str:
    """Convert PDF bytes to Markdown using pymupdf4llm.

    Pipeline:
    0. OCR if no text layer detected (scanned PDF)
    1. Analyze fonts — detect OCR documents
    2. Detect recurring headers/footers across pages
    3. Remove handwritten margin paraphs (non-OCR only)
    4. Convert to markdown via pymupdf4llm
    5. Strip OCR backtick artifacts (OCR only)
    6. Remove garbage OCR lines (OCR only)
    7. Remove headers/footers and page numbers
    8. Fix header formatting (redundant bold)
    9. Merge broken lines into paragraphs
    """
    # Step 0: OCR if no text layer
    if _needs_ocr(file_bytes):
        if on_status:
            on_status("Wykryto skan bez warstwy tekstowej. Uruchamianie OCR...")
        file_bytes = _ocr_pdf(file_bytes)
        if on_status:
            on_status("OCR zakonczone. Przetwarzanie tekstu...")

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    dominant_font, is_ocr = _get_font_stats(doc)
    recurring = _find_recurring_texts(doc)

    # For OCR: skip pages with very low text quality (garbled KRS appendixes, etc.)
    good_pages = None
    if is_ocr:
        good_pages = []
        clean_word_re = re.compile(
            r'^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ.,;:\-()"\'„"]+$'
        )
        for i in range(len(doc)):
            text = doc[i].get_text()
            words = text.split()
            if not words:
                continue
            clean = sum(1 for w in words
                        if clean_word_re.match(w) and 2 <= len(w) <= 30)
            if clean / len(words) >= 0.55:
                good_pages.append(i)
        good_pages = good_pages or None

    doc.close()

    cleaned_bytes = _clean_pdf(file_bytes, is_ocr, dominant_font)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(cleaned_bytes)
        tmp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(
            tmp_path,
            pages=good_pages,
            ignore_code=is_ocr,
        )
    finally:
        os.unlink(tmp_path)

    if is_ocr:
        md_text = _strip_ocr_backticks(md_text)
        md_text = _remove_garbage_ocr_lines(md_text)

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
