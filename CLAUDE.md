# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Konwerter PDF/DOCX do Markdown — local macOS app for faithful 1:1 document-to-Markdown conversion. All processing is local, no external APIs. Polish-language UI.

## Commands

```bash
# Setup (requires venv on macOS with Homebrew Python)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app/main.py --server.headless=true

# Docker (dev container)
docker compose build
docker compose up
# App at http://localhost:8501
```

## Architecture

Two independent conversion pipelines in `app/converter.py`, Streamlit UI in `app/main.py`.

### PDF Pipeline (`convert_pdf_to_markdown`)

Multi-stage processing:
1. **Recurring text detection** — scans top/bottom 10% of all pages, removes text appearing on 2+ pages (headers/footers)
2. **Paraph removal** (`_clean_pdf`) — identifies handwritten margin annotations using font analysis + position heuristics, redacts them via PyMuPDF. Preserves dates and currency amounts (regex: `_MEANINGFUL_RE`)
3. **Markdown conversion** — `pymupdf4llm.to_markdown()` on cleaned PDF
4. **Post-processing** — removes page numbers (`_PAGE_NUMBER_RE`), fixes `# **bold**` → `# text`, merges broken lines into paragraphs

The line merger (`_merge_broken_lines`) is critical: pymupdf4llm separates every PDF line with `\n\n`. Real paragraph breaks are `\n\n\n`. The function splits on triple newlines, then merges continuation lines within each block. Special handling for list item continuations across block boundaries.

### OCR Pre-processing (scanned PDFs)

When a PDF has no text layer (pure image scan), automatic OCR kicks in before the main pipeline:

1. **Detection** (`_needs_ocr`) — checks if <10% of pages have >50 characters of text
2. **OCR** (`_ocr_pdf`) — runs `ocrmypdf` with `skip_text=True` (safe for mixed docs), `deskew=True` (straightens tilted scans), language `pol+eng`
3. After OCR, the PDF has a GlyphLessFont text layer — the existing OCR pipeline handles it automatically

**System requirements**:
- macOS: `brew install tesseract tesseract-lang`
- Docker: Tesseract packages installed in `.devcontainer/Dockerfile`
- Python: `ocrmypdf` in `requirements.txt`

### DOCX Pipeline (`convert_docx_to_markdown`)

Simpler: `mammoth` (DOCX→HTML) → `markdownify` (HTML→MD) → fix empty table headers.

### Key constants (converter.py)

- `_MARGIN_LEFT/RIGHT/TOP/BOTTOM` — margin zone in points for paraph detection
- `_HANDWRITING_KEYWORDS` — font name fragments identifying handwriting fonts
- `_MEANINGFUL_RE` — regex for dates/amounts to preserve even if handwritten
- `_PAGE_NUMBER_RE` — regex for page number patterns to strip

## Design Principles

- **No content modification** — faithful reproduction, no summarization
- **Only remove metadata** — headers, footers, page numbers, margin paraphs
- Dominant font detection (`_get_dominant_font`) distinguishes main typed content from annotation fonts
- Structural Markdown elements (headings, tables, code blocks, lists) are never merged across boundaries
