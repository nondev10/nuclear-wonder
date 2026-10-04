"""
extract_pdfs.py — pull the full text of every reference PDF into a text file
=============================================================================

Each PDF in the current directory becomes a companion `.txt` file under
`docs/pdf_text/`.  Also copies the Python source from `sim/` into
`docs/source/` so the process files are self-contained in one place.

Run from the `simulation/` root:
    python docs/extract_pdfs.py
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # simulation/
PDF_TEXT_DIR = os.path.join(HERE, "docs", "pdf_text")
SOURCE_DIR   = os.path.join(HERE, "docs", "source")
os.makedirs(PDF_TEXT_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)


def slug(name):
    """File name → safe slug for use as a folder/file name."""
    import re
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r'[^A-Za-z0-9_\-]+', ' ', base)
    return re.sub(r'\s+', '_', base).strip().lower() or "document"


def extract_with_pymupdf(pdf_path):
    """Try PyMuPDF first — best layout preservation."""
    import fitz
    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc):
        parts.append(f"\n===== PAGE {i+1} =====\n")
        parts.append(page.get_text("text"))
    doc.close()
    return "".join(parts)


def extract_with_pdftotext(pdf_path):
    """Fallback: external pdftotext binary (present in this env)."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as tmp:
        outpath = tmp.name
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", pdf_path, outpath],
                       capture_output=True)
    with open(outpath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    os.unlink(outpath)
    if r.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {r.stderr.decode(errors='replace')}")
    return text


def extract_with_pypdf(pdf_path):
    """Fallback: pypdf (pure Python)."""
    import pypdf
    r = pypdf.PdfReader(pdf_path)
    parts = [f"\n===== PAGE {i+1} =====\n" for i in range(len(r.pages))]
    text = []
    for i, page in enumerate(r.pages):
        text.append(f"\n===== PAGE {i+1} =====\n")
        text.append((page.extract_text() or "").replace("\xad", ""))
    return "".join(text)


def extract_text(pdf_path):
    for fn in (extract_with_pymupdf, extract_with_pdftotext, extract_with_pypdf):
        try:
            result = fn(pdf_path)
            text = result if isinstance(result, str) else result[0]
            if text and len(text.strip()) > 100:
                return text, (result[1] if isinstance(result, tuple) else fn.__name__)
        except Exception as e:
            print(f"  [warn] {fn.__name__}: {e}")
    # Scanned PDF — no text layer
    return ("[Scanned PDF — no embedded text layer. "
            "OCR not available in this environment. "
            "File preserved as source PDF for manual inspection.]", "scanned")


def extract_html_document(html_path, out_dir):
    """Extract the HTML reference doc into a plain-text process file."""
    import re, html as htmlmod
    raw = open(html_path, encoding="utf-8", errors="ignore").read()
    text = re.sub(r'<script.*?</script>', '', raw, flags=re.S)
    text = re.sub(r'<style.*?</style>', '', text, flags=re.S)
    text = re.sub(r'<[^>]+>', '\n', text)
    text = htmlmod.unescape(text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    out = os.path.join(out_dir, "documentation_and_diagrams_of_the_atomic_bomb.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Source: {os.path.basename(html_path)}\n# Method: html-strip\n")
        f.write(text)
    print(f"Extracted HTML doc: {os.path.basename(html_path)} -> {out}  ({len(text)} chars)")


def main():
    pdfs = sorted(f for f in os.listdir(HERE)
                  if f.lower().endswith(".pdf"))
    print(f"Found {len(pdfs)} PDFs in {HERE}\n")
    for pdf in pdfs:
        print(f"Extracting: {pdf}")
        text, method = extract_text(os.path.join(HERE, pdf))
        out = os.path.join(PDF_TEXT_DIR, f"{slug(pdf)}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# Source: {pdf}\n# Method: {method}\n")
            f.write(text)
        print(f"  -> {out}  ({len(text)} chars)\n")

    # Also extract the HTML reference document
    html_doc = "Documentation and Diagrams of the Atomic Bomb.html"
    html_path = os.path.join(HERE, html_doc)
    if os.path.exists(html_path):
        print(f"Extracting: {html_doc}")
        extract_html_document(html_path, PDF_TEXT_DIR)
    else:
        print(f"[warn] {html_doc} not found in {HERE}")

    # Copy Python source for a self-contained process record
    sim_dir = os.path.join(HERE, "sim")
    for root, _, files in os.walk(sim_dir):
        for fn in files:
            if fn.endswith(".py") or fn == "pytest":
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, HERE)
                dst = os.path.join(SOURCE_DIR, *rel.split(os.sep))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"Copied source: {rel} -> docs/source/{rel}")
    print("\nDone.")


if __name__ == "__main__":
    main()
