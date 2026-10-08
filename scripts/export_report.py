"""docs/REPORT.md -> docs/REPORT.pdf (A4) through headless Chrome.

Usage: uv run python scripts/export_report.py
Markdown -> HTML with the `markdown` package (tables, fenced code), print CSS, then Chrome's --print-to-pdf.
Image paths in the markdown are relative to docs/, so the HTML is written into docs/ (and deleted afterwards).
Prints the page count, because the brief caps the report at 6 pages."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC, PDF = DOCS / "REPORT.md", DOCS / "REPORT.pdf"
HTML = DOCS / "_report_print.html"   # next to REPORT.md so relative image paths resolve

CSS = """
@page { size: A4; margin: 15mm; }
html { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.32; color: #111; }
body { margin: 0; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 11pt 0 3pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 7pt 0 2pt; break-after: avoid; }
p, ul, ol { margin: 0 0 5pt; }
ul, ol { padding-left: 15pt; }
li { margin: 0 0 1.5pt; }
a { color: #1a4f8b; text-decoration: none; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; }
pre { font-size: 8.5pt; background: #f4f4f4; padding: 4pt; }
img { max-width: 100%; max-height: 88mm; display: block; margin: 3pt auto; }
table { border-collapse: collapse; font-size: 9pt; line-height: 1.2; margin: 3pt 0 6pt; width: 100%; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid #bbb; padding: 1.5pt 3.5pt; vertical-align: top; text-align: left; }
th { background: #eee; }
em.cap { display: block; font-size: 8.8pt; color: #444; margin: 0 0 6pt; }
"""

CHROMES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
           "/Applications/Chromium.app/Contents/MacOS/Chromium",
           "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]


def find_chrome() -> str:
    for c in CHROMES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("no Chrome/Chromium found; install one or print docs/_report_print.html by hand")


def page_count(pdf: Path) -> int:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except ImportError:
        return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def main() -> None:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Technical report</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    try:
        subprocess.run([find_chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    finally:
        HTML.unlink(missing_ok=True)
    print(f"{PDF.relative_to(ROOT)}: {page_count(PDF)} pages, {PDF.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
