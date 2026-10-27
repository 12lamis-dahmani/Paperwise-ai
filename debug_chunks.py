from pathlib import Path
import sys

from paperwise.ingestion import chunk_pages, load_pdf


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


pdf_bytes = Path("data/samples/paper.pdf").read_bytes()
chunks = chunk_pages(load_pdf(pdf_bytes), size=1200, overlap=200)

for chunk in chunks:
    if chunk.page in (2, 3):
        print(f"\n--- Seite {chunk.page}, Chunk {chunk.index} ---")
        print(chunk.text)
