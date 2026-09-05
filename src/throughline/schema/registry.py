"""Built-in extraction schemas.

These cover the long-document shapes the pipeline was built for: documents whose
header fields sit on page 1, whose line-item tables run for tens of pages, and whose
totals only reconcile once every continuation page has been read.

Register your own with :func:`register`, or load one from JSON with
:meth:`ExtractionSchema.from_json_file`.
"""

from __future__ import annotations

from throughline.schema.spec import (
    Cardinality,
    ExtractionSchema,
    FieldSpec,
    FieldType,
    TableSpec,
)

# ── invoice ───────────────────────────────────────────────────────────────
INVOICE = ExtractionSchema(
    name="invoice",
    version="1.2.0",
    description=(
        "Commercial invoice. Header fields are concentrated on the first page; the "
        "line-item table routinely continues for many pages and the totals block "
        "appears only on the last."
    ),
    fields=(
        FieldSpec(
            name="invoice_number",
            type=FieldType.STRING,
            description="The invoice's own identifier, not the purchase-order number.",
            required=True,
            page_hint="first page, header block",
            keywords=("invoice no", "invoice number", "invoice #", "inv no"),
        ),
        FieldSpec(
            name="invoice_date",
            type=FieldType.DATE,
            description="Issue date in YYYY-MM-DD or MM/DD/YYYY form.",
            required=True,
            page_hint="first page, header block",
            keywords=("invoice date", "date issued", "issue date"),
        ),
        FieldSpec(
            name="purchase_order",
            type=FieldType.STRING,
            description="Buyer's purchase-order reference, if quoted.",
            cardinality=Cardinality.OPTIONAL,
            page_hint="first page, header block",
            keywords=("purchase order", "po number", "p.o."),
        ),
        FieldSpec(
            name="vendor_name",
            type=FieldType.STRING,
            description="Legal name of the party issuing the invoice.",
            required=True,
            page_hint="first page, letterhead",
            keywords=("remit to", "vendor", "supplier", "seller"),
        ),
        FieldSpec(
            name="bill_to",
            type=FieldType.STRING,
            description="Legal name of the party being billed.",
            page_hint="first page, address block",
            keywords=("bill to", "sold to", "customer", "buyer"),
        ),
        FieldSpec(
            name="currency",
            type=FieldType.ENUM,
            description="ISO currency code of the invoiced amounts.",
            enum_values=("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"),
            page_hint="totals block",
            keywords=("currency", "usd", "eur", "gbp"),
        ),
        FieldSpec(
            name="subtotal",
            type=FieldType.CURRENCY,
            description="Sum of line items before tax.",
            page_hint="last page, totals block",
            keywords=("subtotal", "net amount", "sub total"),
            continues_across_pages=True,
        ),
        FieldSpec(
            name="tax_total",
            type=FieldType.CURRENCY,
            description="Total tax charged.",
            page_hint="last page, totals block",
            keywords=("tax", "vat", "gst", "sales tax"),
        ),
        FieldSpec(
            name="total_amount",
            type=FieldType.CURRENCY,
            description="Final payable amount including tax.",
            required=True,
            page_hint="last page, totals block",
            keywords=("total", "amount due", "balance due", "grand total"),
            continues_across_pages=True,
        ),
        FieldSpec(
            name="payment_terms",
            type=FieldType.STRING,
            description="Stated payment terms, e.g. Net 30.",
            cardinality=Cardinality.OPTIONAL,
            page_hint="first or last page",
            keywords=("terms", "net 30", "payment terms", "due date"),
        ),
    ),
    tables=(
        TableSpec(
            name="line_items",
            description=(
                "One row per billed item. The table often repeats its header at the "
                "top of each continuation page; a repeated header is not a new row."
            ),
            required=True,
            row_key_columns=("line_number", "description"),
            columns=(
                FieldSpec(
                    name="line_number",
                    type=FieldType.INTEGER,
                    description="Sequential line number as printed.",
                ),
                FieldSpec(
                    name="description",
                    type=FieldType.STRING,
                    description="Item description, which may wrap over several printed lines.",
                ),
                FieldSpec(name="quantity", type=FieldType.NUMBER, description="Units billed."),
                FieldSpec(
                    name="unit_price", type=FieldType.CURRENCY, description="Price per unit."
                ),
                FieldSpec(
                    name="amount", type=FieldType.CURRENCY, description="Extended line amount."
                ),
            ),
        ),
    ),
)

# ── account statement ─────────────────────────────────────────────────────
STATEMENT = ExtractionSchema(
    name="account_statement",
    version="1.1.0",
    description=(
        "Periodic account statement. The transaction table is the long part; opening "
        "and closing balances bracket it and must reconcile against it."
    ),
    fields=(
        FieldSpec(
            name="account_number",
            type=FieldType.STRING,
            description="Account identifier, often partially masked.",
            required=True,
            page_hint="every page header",
            keywords=("account number", "account no", "acct"),
        ),
        FieldSpec(
            name="account_holder",
            type=FieldType.STRING,
            description="Name on the account.",
            required=True,
            page_hint="first page",
            keywords=("account holder", "name", "customer"),
        ),
        FieldSpec(
            name="statement_period_start",
            type=FieldType.DATE,
            description="First day covered by the statement.",
            page_hint="first page",
            keywords=("statement period", "from", "period beginning"),
        ),
        FieldSpec(
            name="statement_period_end",
            type=FieldType.DATE,
            description="Last day covered by the statement.",
            page_hint="first page",
            keywords=("statement period", "to", "period ending"),
        ),
        FieldSpec(
            name="opening_balance",
            type=FieldType.CURRENCY,
            description="Balance carried in at the start of the period.",
            page_hint="first page",
            keywords=("opening balance", "beginning balance", "balance forward"),
        ),
        FieldSpec(
            name="closing_balance",
            type=FieldType.CURRENCY,
            description="Balance at the end of the period.",
            required=True,
            page_hint="last page",
            keywords=("closing balance", "ending balance", "new balance"),
            continues_across_pages=True,
        ),
    ),
    tables=(
        TableSpec(
            name="transactions",
            description="One row per posted transaction, in date order.",
            required=True,
            row_key_columns=("date", "description", "amount"),
            columns=(
                FieldSpec(name="date", type=FieldType.DATE, description="Posting date."),
                FieldSpec(
                    name="description", type=FieldType.STRING, description="Transaction narrative."
                ),
                FieldSpec(
                    name="amount",
                    type=FieldType.CURRENCY,
                    description="Signed amount; debits negative.",
                ),
                FieldSpec(
                    name="running_balance",
                    type=FieldType.CURRENCY,
                    description="Balance after the transaction, when printed.",
                ),
            ),
        ),
    ),
)

# ── service agreement ─────────────────────────────────────────────────────
AGREEMENT = ExtractionSchema(
    name="service_agreement",
    version="1.0.0",
    description=(
        "Master service agreement. Key terms are scattered: parties on page 1, "
        "term and termination deep in the body, signatures at the very end."
    ),
    fields=(
        FieldSpec(
            name="agreement_title",
            type=FieldType.STRING,
            description="Title as printed on the cover page.",
            required=True,
            page_hint="cover page",
            keywords=("agreement", "contract", "master service"),
        ),
        FieldSpec(
            name="parties",
            type=FieldType.STRING,
            description="Each contracting party's legal name.",
            cardinality=Cardinality.MANY,
            required=True,
            page_hint="first page, recitals",
            keywords=("between", "by and between", "party", "parties"),
        ),
        FieldSpec(
            name="effective_date",
            type=FieldType.DATE,
            description="Date the agreement takes effect.",
            required=True,
            page_hint="first page or signature block",
            keywords=("effective date", "commencement date", "dated as of"),
        ),
        FieldSpec(
            name="initial_term_months",
            type=FieldType.INTEGER,
            description="Length of the initial term in months.",
            page_hint="term clause",
            keywords=("term", "initial term", "months", "years"),
        ),
        FieldSpec(
            name="auto_renews",
            type=FieldType.BOOLEAN,
            description="Whether the agreement renews automatically.",
            page_hint="term clause",
            keywords=("renew", "automatically", "successive"),
        ),
        FieldSpec(
            name="termination_notice_days",
            type=FieldType.INTEGER,
            description="Days of notice required to terminate.",
            page_hint="termination clause",
            keywords=("terminate", "notice", "days written notice"),
        ),
        FieldSpec(
            name="governing_law",
            type=FieldType.STRING,
            description="Governing jurisdiction.",
            page_hint="miscellaneous clauses, near the end",
            keywords=("governing law", "governed by", "jurisdiction", "laws of"),
        ),
        FieldSpec(
            name="liability_cap",
            type=FieldType.CURRENCY,
            description="Stated cap on aggregate liability.",
            cardinality=Cardinality.OPTIONAL,
            page_hint="limitation of liability clause",
            keywords=("liability", "aggregate", "cap", "shall not exceed"),
        ),
    ),
    tables=(
        TableSpec(
            name="signatories",
            description="One row per signature block.",
            row_key_columns=("party", "name"),
            columns=(
                FieldSpec(name="party", type=FieldType.STRING, description="Entity signed for."),
                FieldSpec(name="name", type=FieldType.STRING, description="Signatory name."),
                FieldSpec(name="title", type=FieldType.STRING, description="Signatory title."),
                FieldSpec(name="date_signed", type=FieldType.DATE, description="Date signed."),
            ),
        ),
    ),
)


_REGISTRY: dict[str, ExtractionSchema] = {
    INVOICE.name: INVOICE,
    STATEMENT.name: STATEMENT,
    AGREEMENT.name: AGREEMENT,
}


def register(schema: ExtractionSchema, *, overwrite: bool = False) -> None:
    """Add a schema to the process-wide registry."""
    if schema.name in _REGISTRY and not overwrite:
        raise ValueError(f"Schema {schema.name!r} is already registered; pass overwrite=True.")
    _REGISTRY[schema.name] = schema


def get(name: str) -> ExtractionSchema:
    """Look up a registered schema by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown schema {name!r}. Registered: {sorted(_REGISTRY)}"
        ) from None


def available() -> list[str]:
    """Names of every registered schema."""
    return sorted(_REGISTRY)
