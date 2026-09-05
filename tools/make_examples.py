"""Generate the synthetic example corpus.

These documents are **synthetic**. No customer or production data appears anywhere in
this repository. They exist so the pipeline is runnable and testable end to end by
anyone who clones it, and they are shaped to exercise the cases that actually matter:

* header fields on page 1, totals on the last page;
* a line-item table that runs across several page groups, with a repeated column
  header on every continuation page and an explicit "continued" marker;
* a value (``total_amount``) that is only correct once the last page is read, so a
  naive first-answer-wins merge gets it wrong;
* an out-of-schema page in the middle, so relevant-page retrieval has something to
  deprioritise.

Run with:  python tools/make_examples.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ROOT / "examples" / "documents"
CORPUS = ROOT / "examples" / "corpus"

VENDORS = [
    ("Northwind Industrial Supply", "USD"),
    ("Meridian Components Ltd", "GBP"),
    ("Baltic Logistics GmbH", "EUR"),
    ("Cascade Manufacturing Co", "USD"),
    ("Anders & Wray Partners", "USD"),
]

BUYERS = [
    "Halcyon Systems Inc",
    "Redpoint Analytics LLC",
    "Foundry Robotics Corp",
    "Lattice Energy Partners",
]

ITEMS = [
    "Precision bearing assembly, 42mm",
    "Hydraulic seal kit, type B",
    "Stainless fastener set (250 ct)",
    "Control module firmware licence",
    "Thermal insulation panel, 1.2m",
    "Calibration service, on-site",
    "Replacement drive belt, HD",
    "Pressure sensor, 0-16 bar",
    "Cable harness, 3m shielded",
    "Mounting bracket, powder-coated",
    "Coolant reservoir, 4L",
    "Diagnostic interface adapter",
]

SYMBOLS = {"USD": "$", "GBP": "£", "EUR": "€"}


def _block(
    block_id: str,
    page: int,
    block_type: str,
    text: str,
    y0: float,
    y1: float,
    *,
    x0: float = 0.08,
    x1: float = 0.92,
    row_index: int | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "block_type": block_type,
        "text": text,
        "bbox": [x0, round(y0, 4), x1, round(y1, 4)],
        "confidence": 0.99,
        "row_index": row_index,
    }


def _money(amount: float, currency: str) -> str:
    return f"{SYMBOLS[currency]}{amount:,.2f}"


def build_invoice(index: int, rng: random.Random) -> dict[str, Any]:
    """One synthetic invoice with a table spanning several pages."""
    vendor, currency = rng.choice(VENDORS)
    buyer = rng.choice(BUYERS)
    invoice_number = f"INV-{2026}{index:04d}"
    invoice_date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    purchase_order = f"PO-{rng.randint(10000, 99999)}"
    terms = rng.choice(["Net 30", "Net 45", "Net 60", "Due on receipt"])

    rows_per_page = 6
    line_count = rng.randint(14, 26)
    pages: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    row_pages: list[int] = []

    # ── page 1: header + first rows ──────────────────────────────────
    page_number = 1
    blocks = [
        _block(f"p{page_number}b0", page_number, "title", vendor, 0.04, 0.09),
        _block(f"p{page_number}b1", page_number, "page_header", f"Page {page_number}", 0.01, 0.035),
        _block(f"p{page_number}b2", page_number, "key_value", f"Invoice No: {invoice_number}", 0.12, 0.15),
        _block(f"p{page_number}b3", page_number, "key_value", f"Invoice Date: {invoice_date}", 0.155, 0.185),
        _block(f"p{page_number}b4", page_number, "key_value", f"Purchase Order: {purchase_order}", 0.19, 0.22),
        _block(f"p{page_number}b5", page_number, "key_value", f"Bill To: {buyer}", 0.225, 0.255),
        _block(f"p{page_number}b6", page_number, "key_value", f"Payment Terms: {terms}", 0.26, 0.29),
        _block(
            f"p{page_number}b7",
            page_number,
            "table_header",
            "line_number  description  quantity  unit_price  amount",
            0.34,
            0.37,
        ),
    ]

    subtotal = 0.0
    line_index = 0
    block_index = 8
    y = 0.39

    while line_index < min(rows_per_page, line_count):
        line_index += 1
        quantity = rng.randint(1, 40)
        unit_price = round(rng.uniform(12.5, 480.0), 2)
        amount = round(quantity * unit_price, 2)
        subtotal += amount
        description = rng.choice(ITEMS)

        blocks.append(
            _block(
                f"p{page_number}b{block_index}",
                page_number,
                "table_row",
                f"{line_index}  {description}  {quantity}  "
                f"{_money(unit_price, currency)}  {_money(amount, currency)}",
                y,
                y + 0.03,
                row_index=line_index,
            )
        )
        gold_rows.append(
            {
                "line_number": line_index,
                "description": description,
                "quantity": float(quantity),
                "unit_price": _money(unit_price, currency),
                "amount": _money(amount, currency),
            }
        )
        row_pages.append(page_number)
        block_index += 1
        y += 0.035

    blocks.append(
        _block(f"p{page_number}b{block_index}", page_number, "page_footer",
               "Continued on next page", 0.95, 0.98)
    )
    pages.append({"page_number": page_number, "width": 1240, "height": 1754, "blocks": blocks})

    # ── continuation pages ───────────────────────────────────────────
    while line_index < line_count:
        page_number += 1
        blocks = [
            _block(f"p{page_number}b0", page_number, "page_header",
                   f"{invoice_number} — Page {page_number}", 0.01, 0.035),
            _block(
                f"p{page_number}b1",
                page_number,
                "table_header",
                "line_number  description  quantity  unit_price  amount",
                0.06,
                0.09,
            ),
        ]
        block_index = 2
        y = 0.11
        page_start = line_index

        while line_index < line_count and (line_index - page_start) < rows_per_page:
            line_index += 1
            quantity = rng.randint(1, 40)
            unit_price = round(rng.uniform(12.5, 480.0), 2)
            amount = round(quantity * unit_price, 2)
            subtotal += amount
            description = rng.choice(ITEMS)

            blocks.append(
                _block(
                    f"p{page_number}b{block_index}",
                    page_number,
                    "table_row",
                    f"{line_index}  {description}  {quantity}  "
                    f"{_money(unit_price, currency)}  {_money(amount, currency)}",
                    y,
                    y + 0.03,
                    row_index=line_index,
                )
            )
            gold_rows.append(
                {
                    "line_number": line_index,
                    "description": description,
                    "quantity": float(quantity),
                    "unit_price": _money(unit_price, currency),
                    "amount": _money(amount, currency),
                }
            )
            row_pages.append(page_number)
            block_index += 1
            y += 0.035

        if line_index < line_count:
            blocks.append(
                _block(f"p{page_number}b{block_index}", page_number, "page_footer",
                       "Continued on next page", 0.95, 0.98)
            )
        pages.append({"page_number": page_number, "width": 1240, "height": 1754, "blocks": blocks})

    # ── an out-of-schema page, to give retrieval something to skip ───
    page_number += 1
    pages.append(
        {
            "page_number": page_number,
            "width": 1240,
            "height": 1754,
            "blocks": [
                _block(f"p{page_number}b0", page_number, "heading",
                       "TERMS AND CONDITIONS OF SALE", 0.05, 0.09),
                _block(
                    f"p{page_number}b1",
                    page_number,
                    "paragraph",
                    "All goods remain the property of the seller until paid for in full. "
                    "Risk passes on delivery. Claims for shortage or damage must be notified "
                    "in writing within seven days of receipt.",
                    0.12,
                    0.22,
                ),
                _block(
                    f"p{page_number}b2",
                    page_number,
                    "paragraph",
                    "The seller shall not be liable for any indirect or consequential loss "
                    "howsoever arising. Nothing in these terms limits liability for death or "
                    "personal injury caused by negligence.",
                    0.24,
                    0.34,
                ),
            ],
        }
    )

    # ── totals page ──────────────────────────────────────────────────
    page_number += 1
    tax_rate = 0.0825
    subtotal = round(subtotal, 2)
    tax_total = round(subtotal * tax_rate, 2)
    total_amount = round(subtotal + tax_total, 2)

    pages.append(
        {
            "page_number": page_number,
            "width": 1240,
            "height": 1754,
            "blocks": [
                _block(f"p{page_number}b0", page_number, "page_header",
                       f"{invoice_number} — Page {page_number}", 0.01, 0.035),
                _block(f"p{page_number}b1", page_number, "key_value",
                       f"Subtotal: {_money(subtotal, currency)}", 0.30, 0.33),
                _block(f"p{page_number}b2", page_number, "key_value",
                       f"Sales Tax (8.25%): {_money(tax_total, currency)}", 0.34, 0.37),
                _block(f"p{page_number}b3", page_number, "key_value",
                       f"Total Amount Due: {_money(total_amount, currency)}", 0.38, 0.42),
                _block(f"p{page_number}b4", page_number, "key_value",
                       f"Currency: {currency}", 0.43, 0.46),
                _block(f"p{page_number}b5", page_number, "paragraph",
                       f"Remit to {vendor}. Payment terms: {terms}.", 0.50, 0.54),
            ],
        }
    )

    document = {
        "document_id": f"invoice_{index:04d}",
        "source_path": None,
        "metadata": {"synthetic": True, "generator": "tools/make_examples.py"},
        "pages": pages,
    }

    gold = {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "purchase_order": purchase_order,
        "vendor_name": vendor,
        "bill_to": buyer,
        "currency": currency,
        "subtotal": _money(subtotal, currency),
        "tax_total": _money(tax_total, currency),
        "total_amount": _money(total_amount, currency),
        "payment_terms": terms,
        "line_items": gold_rows,
    }

    last = pages[-1]["page_number"]
    gold_evidence = {
        "invoice_number": [{"page_number": 1, "block_id": "p1b2"}],
        "invoice_date": [{"page_number": 1, "block_id": "p1b3"}],
        "purchase_order": [{"page_number": 1, "block_id": "p1b4"}],
        "vendor_name": [{"page_number": 1, "block_id": "p1b0"}, {"page_number": last, "block_id": f"p{last}b5"}],
        "bill_to": [{"page_number": 1, "block_id": "p1b5"}],
        "currency": [{"page_number": last, "block_id": f"p{last}b4"}],
        "subtotal": [{"page_number": last, "block_id": f"p{last}b1"}],
        "tax_total": [{"page_number": last, "block_id": f"p{last}b2"}],
        "total_amount": [{"page_number": last, "block_id": f"p{last}b3"}],
        "payment_terms": [{"page_number": 1, "block_id": "p1b6"}, {"page_number": last, "block_id": f"p{last}b5"}],
        "line_items": [{"page_number": page} for page in sorted(set(row_pages))],
    }

    return {
        "document": document,
        "gold": gold,
        "gold_evidence": gold_evidence,
        "row_pages": {"line_items": row_pages},
    }


# ── long service agreements ───────────────────────────────────────────────
CLAUSE_FILLER = [
    "Each party shall perform its obligations under this Agreement with reasonable "
    "skill and care and in accordance with all applicable laws and regulations.",
    "Neither party shall be liable for any failure or delay in performance to the "
    "extent that such failure or delay is caused by an event beyond its reasonable "
    "control, provided that the affected party gives prompt written notice.",
    "The parties shall keep confidential all information disclosed by the other party "
    "which is marked as confidential or which ought reasonably to be regarded as "
    "confidential, and shall not disclose it to any third party.",
    "Any notice given under this Agreement shall be in writing and shall be delivered "
    "by hand, sent by pre-paid first class post, or sent by electronic mail to the "
    "address of the receiving party set out in the preamble.",
    "No variation of this Agreement shall be effective unless it is in writing and "
    "signed by an authorised representative of each of the parties.",
    "This Agreement constitutes the entire agreement between the parties and "
    "supersedes all previous agreements, understandings and arrangements between them.",
    "If any provision of this Agreement is held to be invalid or unenforceable, that "
    "provision shall be severed and the remaining provisions shall continue in force.",
    "Nothing in this Agreement is intended to, or shall be deemed to, establish any "
    "partnership or joint venture between the parties.",
]

SECTION_TITLES = [
    "DEFINITIONS AND INTERPRETATION",
    "SCOPE OF SERVICES",
    "SERVICE LEVELS",
    "CHARGES AND PAYMENT",
    "INTELLECTUAL PROPERTY",
    "DATA PROTECTION",
    "CONFIDENTIALITY",
    "WARRANTIES",
    "INDEMNITIES",
    "FORCE MAJEURE",
    "ASSIGNMENT",
    "DISPUTE RESOLUTION",
    "NOTICES",
    "GENERAL PROVISIONS",
]

PARTY_A = ["Halcyon Systems Inc", "Redpoint Analytics LLC", "Foundry Robotics Corp"]
PARTY_B = ["Northwind Industrial Supply", "Meridian Components Ltd", "Cascade Manufacturing Co"]
JURISDICTIONS = ["State of Delaware", "State of New York", "England and Wales", "State of Georgia"]


def build_agreement(index: int, rng: random.Random) -> dict[str, Any]:
    """A long agreement whose key terms are scattered far apart.

    This is the shape that makes relevant-page retrieval and early exit worth having:
    twenty-odd pages, no required table, and six values buried at unpredictable
    depths. Reading it front to back is mostly wasted work.
    """
    party_a = rng.choice(PARTY_A)
    party_b = rng.choice(PARTY_B)
    effective_date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    term_months = rng.choice([12, 24, 36, 48])
    notice_days = rng.choice([30, 60, 90])
    jurisdiction = rng.choice(JURISDICTIONS)
    liability_cap = round(rng.uniform(250_000, 5_000_000), 2)
    auto_renews = rng.choice([True, False])
    title = "MASTER SERVICES AGREEMENT"

    total_pages = rng.randint(18, 26)
    pages: list[dict[str, Any]] = []
    evidence: dict[str, list[dict[str, Any]]] = {}

    # page 1: cover + parties + effective date
    pages.append(
        {
            "page_number": 1,
            "width": 1240,
            "height": 1754,
            "blocks": [
                _block("p1b0", 1, "title", title, 0.10, 0.16),
                _block("p1b1", 1, "paragraph",
                       f"This Agreement is made by and between {party_a} and {party_b}.",
                       0.24, 0.30),
                _block("p1b2", 1, "key_value", f"Effective Date: {effective_date}", 0.34, 0.37),
                _block("p1b3", 1, "paragraph",
                       "The parties agree to the terms set out on the following pages.",
                       0.42, 0.47),
            ],
        }
    )
    evidence["agreement_title"] = [{"page_number": 1, "block_id": "p1b0"}]
    evidence["parties"] = [{"page_number": 1, "block_id": "p1b1"}]
    evidence["effective_date"] = [{"page_number": 1, "block_id": "p1b2"}]

    # the buried terms, at deliberately awkward depths
    term_page = rng.randint(4, max(5, total_pages // 3))
    liability_page = rng.randint(term_page + 2, max(term_page + 3, total_pages - 6))
    law_page = total_pages - 2

    for page_number in range(2, total_pages - 1):
        blocks = [
            _block(f"p{page_number}b0", page_number, "page_header",
                   f"{title} — Page {page_number}", 0.01, 0.035),
            _block(f"p{page_number}b1", page_number, "heading",
                   SECTION_TITLES[(page_number - 2) % len(SECTION_TITLES)], 0.07, 0.11),
        ]
        block_index = 2
        y = 0.15

        for _ in range(rng.randint(3, 4)):
            blocks.append(
                _block(f"p{page_number}b{block_index}", page_number, "paragraph",
                       rng.choice(CLAUSE_FILLER), y, y + 0.09)
            )
            block_index += 1
            y += 0.11

        if page_number == term_page:
            renew = (
                "This Agreement shall renew automatically for successive periods of "
                "equal length unless terminated."
                if auto_renews
                else "This Agreement shall not renew automatically and shall expire at "
                     "the end of the initial term."
            )
            blocks.append(
                _block(f"p{page_number}b{block_index}", page_number, "paragraph",
                       f"Initial Term: {term_months} months from the Effective Date. {renew}",
                       y, y + 0.08)
            )
            evidence["initial_term_months"] = [
                {"page_number": page_number, "block_id": f"p{page_number}b{block_index}"}
            ]
            evidence["auto_renews"] = [
                {"page_number": page_number, "block_id": f"p{page_number}b{block_index}"}
            ]
            block_index += 1
            y += 0.10
            blocks.append(
                _block(f"p{page_number}b{block_index}", page_number, "paragraph",
                       f"Either party may terminate this Agreement on {notice_days} days "
                       f"written notice to the other party.", y, y + 0.07)
            )
            evidence["termination_notice_days"] = [
                {"page_number": page_number, "block_id": f"p{page_number}b{block_index}"}
            ]
            block_index += 1
            y += 0.09

        if page_number == liability_page:
            blocks.append(
                _block(f"p{page_number}b{block_index}", page_number, "paragraph",
                       f"The aggregate liability of either party under this Agreement "
                       f"shall not exceed ${liability_cap:,.2f}.", y, y + 0.07)
            )
            evidence["liability_cap"] = [
                {"page_number": page_number, "block_id": f"p{page_number}b{block_index}"}
            ]
            block_index += 1
            y += 0.09

        if page_number == law_page:
            blocks.append(
                _block(f"p{page_number}b{block_index}", page_number, "paragraph",
                       f"Governing Law: this Agreement is governed by the laws of the "
                       f"{jurisdiction}.", y, y + 0.07)
            )
            evidence["governing_law"] = [
                {"page_number": page_number, "block_id": f"p{page_number}b{block_index}"}
            ]
            block_index += 1

        pages.append(
            {"page_number": page_number, "width": 1240, "height": 1754, "blocks": blocks}
        )

    # signature page
    last = total_pages
    signed_a = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    pages.append(
        {
            "page_number": last,
            "width": 1240,
            "height": 1754,
            "blocks": [
                _block(f"p{last}b0", last, "page_header", f"{title} — Page {last}", 0.01, 0.035),
                _block(f"p{last}b1", last, "heading", "EXECUTION", 0.08, 0.12),
                _block(f"p{last}b2", last, "signature",
                       f"Signed for and on behalf of {party_a}  "
                       f"Dana Whitfield  Chief Operating Officer  {signed_a}", 0.20, 0.28),
                _block(f"p{last}b3", last, "signature",
                       f"Signed for and on behalf of {party_b}  "
                       f"Emil Sorensen  Managing Director  {signed_a}", 0.32, 0.40),
            ],
        }
    )
    evidence["signatories"] = [{"page_number": last}]

    document = {
        "document_id": f"agreement_{index:04d}",
        "source_path": None,
        "metadata": {"synthetic": True, "generator": "tools/make_examples.py"},
        "pages": pages,
    }
    gold = {
        "agreement_title": title,
        "parties": [party_a, party_b],
        "effective_date": effective_date,
        "initial_term_months": term_months,
        "auto_renews": auto_renews,
        "termination_notice_days": notice_days,
        "governing_law": f"the {jurisdiction}",
        "liability_cap": f"${liability_cap:,.2f}",
        "signatories": [
            {"party": party_a, "name": "Dana Whitfield", "title": "Chief Operating Officer",
             "date_signed": signed_a},
            {"party": party_b, "name": "Emil Sorensen", "title": "Managing Director",
             "date_signed": signed_a},
        ],
    }
    return {"document": document, "gold": gold, "gold_evidence": evidence}


def main() -> None:
    rng = random.Random(20260904)
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)

    agreements = CORPUS.parent / "corpus_agreements"
    agreements.mkdir(parents=True, exist_ok=True)

    for index in range(1, 13):
        payload = build_invoice(index, rng)
        document_id = payload["document"]["document_id"]
        (DOCUMENTS / f"{document_id}.json").write_text(
            json.dumps(payload["document"], indent=2) + "\n", encoding="utf-8"
        )
        (CORPUS / f"{document_id}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    for index in range(1, 9):
        payload = build_agreement(index, rng)
        document_id = payload["document"]["document_id"]
        (DOCUMENTS / f"{document_id}.json").write_text(
            json.dumps(payload["document"], indent=2) + "\n", encoding="utf-8"
        )
        (agreements / f"{document_id}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    pages = sum(
        len(json.loads(path.read_text())["pages"]) for path in DOCUMENTS.glob("*.json")
    )
    print(
        f"wrote 12 invoices + 8 agreements ({pages} pages total)\n"
        f"  documents: {DOCUMENTS}\n"
        f"  invoice corpus:   {CORPUS}\n"
        f"  agreement corpus: {agreements}"
    )


if __name__ == "__main__":
    main()
