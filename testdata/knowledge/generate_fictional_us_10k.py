"""Generate a fictional US equity 10-K style PDF for knowledge-base RAG tests.

Output is intentionally unique (fake ticker / products / metrics) so retrieval
can be distinguished from model prior knowledge or public web sources.
"""

from __future__ import annotations

from pathlib import Path


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Minimal PDF 1.4 with Helvetica (Latin-1) text."""
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []

    for lines in pages:
        cmds = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
        first = True
        for line in lines:
            text = _escape(line)
            if first:
                cmds.append(f"({text}) Tj")
                first = False
            else:
                cmds.append("T*")
                cmds.append(f"({text}) Tj")
        cmds.append("ET")
        stream = "\n".join(cmds).encode("latin-1", errors="replace")
        content = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        content_ids.append(add(content))

    for cid in content_ids:
        page_ids.append(
            add(
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] "
                f"/Contents {cid} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())

    for i, pid in enumerate(page_ids):
        objects[pid - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_ids[i]} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()

    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(out)


PAGE1 = [
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    "Washington, D.C. 20549",
    "",
    "FORM 10-K  (FICTIONAL TEST FILING - NOT A REAL SEC DOCUMENT)",
    "",
    "HelixOrion Nullweave Systems, Inc.",
    "Commission File Number: 001-99017",
    "Ticker Symbol: NXWV  |  Exchange: NASDAQ Global Select (simulated)",
    "IRS Employer ID (fictional): 98-4472106",
    "HQ: 1187 Emberglass Parkway, Redmond Void District, WA 98052",
    "",
    "================================================================",
    "DOCUMENT PURPOSE (INTERNAL RAG TEST MARKER)",
    "================================================================",
    "This PDF is a synthetic fixture created solely for Research Agent",
    "knowledge-base retrieval tests. It must NOT match any public",
    "company filing. Unique verification phrase:",
    "  emberglass-quantum-yield-ratio of 0.918",
    "If the agent cites this phrase or NXWV / PrismWeave-7, retrieval",
    "from this uploaded document is confirmed.",
    "",
    "Item 1. Business",
    "HelixOrion Nullweave Systems, Inc. (NXWV) designs adaptive optical",
    "interlink fabrics for hyperscale AI clusters. Flagship product:",
    "PrismWeave-7 Adaptive Interlink Array (codename Project Lumenfold-2041).",
    "",
    "Key customers named only in this fixture (not public logos):",
    "  - GlacierByte Compute Cooperative",
    "  - Meridian Lattice Labs",
    "  - Quokka Edge Inference Guild",
    "",
    "CEO: Dr. Maren Quillcroft",
    "CFO: Elias Thornwick",
    "CTO: Priya Vantrel",
]

PAGE2 = [
    "Item 7. Management Discussion (FY2025, fiscal year ended Dec 31, 2025)",
    "",
    "Selected financials (fictional, USD thousands unless noted):",
    "  Total revenue                         $847,320",
    "  Gross profit                          $411,900",
    "  Operating income                      $128,450",
    "  Net income                             $96,220",
    "  Diluted EPS (USD)                         1.47",
    "  Cash and equivalents                 $302,110",
    "  R&D expense                          $188,760",
    "",
    "YoY revenue growth: +31.4% (driven by PrismWeave-7 volume ramps).",
    "Gross margin: 48.6%. Operating margin: 15.2%.",
    "",
    "Segment revenue mix:",
    "  Data-center optical fabric ............. 72%",
    "  Edge inference modules ................. 19%",
    "  Services & firmware subscriptions ......  9%",
    "",
    "Guidance for FY2026 (fictional, as of Feb 14, 2026):",
    "  Revenue midpoint: $1.05 billion",
    "  Non-GAAP operating margin: 16.5%-17.5%",
    "  Capex for Lumenfold-2041 fab tool-in: $210 million",
    "",
    "Risk factor unique to this fixture:",
    "  Supply of 'void-grade erbium wafers' from supplier",
    "  Codename ASH-77 may constrain PrismWeave-7 shipments",
    "  in 2H2026 if dual-source qualification slips past Q3.",
]

PAGE3 = [
    "Item 8. Notes - Product & IP (fictional)",
    "",
    "Patent family (synthetic identifiers, not USPTO-issued):",
    "  US-FICT-9918273-B2  Adaptive phase-locked optical mesh",
    "  US-FICT-9918301-B2  Thermal-null routing for AI backplanes",
    "",
    "Operating metric used internally (do not invent elsewhere):",
    "  Emberglass Quantum Yield Ratio (EQYR) = 0.918 for FY2025",
    "  Target EQYR for FY2026: 0.935",
    "",
    "Board authorization (fictional):",
    "  On Nov 3, 2025 the board approved a $150 million share",
    "  repurchase under Program SILVERTHREAD-9.",
    "",
    "Suggested evaluation questions (Chinese / English):",
    "  1) HelixOrion Nullweave ticker / stock code?",
    "  2) PrismWeave-7 internal project codename?",
    "  3) FY2025 revenue and diluted EPS?",
    "  4) What is emberglass-quantum-yield-ratio?",
    "  5) How does ASH-77 affect 2H2026 shipments?",
    "  6) What is Program SILVERTHREAD-9?",
    "",
    "END OF FICTIONAL TEST FILING — HelixOrion Nullweave Systems, Inc.",
    "Generated for research_agent knowledge ingest validation only.",
]


def main() -> None:
    out = Path(__file__).resolve().parent / ("HelixOrion_Nullweave_NXWV_FY2025_10K_FICTIONAL.pdf")
    pdf = build_pdf([PAGE1, PAGE2, PAGE3])
    out.write_bytes(pdf)
    print(f"wrote {out} ({len(pdf)} bytes)")

    from pypdf import PdfReader

    reader = PdfReader(str(out))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    needles = [
        "NXWV",
        "PrismWeave-7",
        "emberglass-quantum-yield-ratio of 0.918",
        "Lumenfold-2041",
        "847,320",
        "SILVERTHREAD-9",
    ]
    for n in needles:
        ok = n in text
        print(f"  extract {n!r}: {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()
