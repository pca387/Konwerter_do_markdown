import io
import re
import tempfile
import os

import pymupdf4llm
import mammoth
from markdownify import markdownify


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

    Faithful 1:1 conversion — no content modification or summarization.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
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
