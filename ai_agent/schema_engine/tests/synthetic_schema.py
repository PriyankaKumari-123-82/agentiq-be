"""Synthetic enterprise schema for testing (50 tables across 5 domains).

Used by unit and integration tests to exercise the full pipeline without
needing a real database connection.
"""

from ai_agent.schema_engine.models import (
    ColumnMetadata,
    DatabaseMetadata,
    ForeignKeyMetadata,
    TableMetadata,
)


def build_synthetic_database(
    table_count: int = 50,
) -> DatabaseMetadata:
    """Build a realistic synthetic schema.

    Contains tables for finance, sales, hr, inventory, and support domains
    with proper FK relationships.
    """
    tables = {}

    # ── Finance ──────────────────────────────────────────────────────────
    tables.update(_finance_tables())
    # ── Sales ────────────────────────────────────────────────────────────
    tables.update(_sales_tables())
    # ── HR ───────────────────────────────────────────────────────────────
    tables.update(_hr_tables())
    # ── Inventory ────────────────────────────────────────────────────────
    tables.update(_inventory_tables())
    # ── Support ──────────────────────────────────────────────────────────
    tables.update(_support_tables())
    # ── Shared / general ─────────────────────────────────────────────────
    tables.update(_shared_tables())

    # Pad to target table_count with generic tables
    idx = 0
    while len(tables) < table_count:
        t = _generic_table(idx)
        tables[t.fqn] = t
        idx += 1

    return DatabaseMetadata(
        server_name="test-db",
        database_name="enterprise_test",
        tables=tables,
    )


def _tbl(
    name: str,
    schema: str = "public",
    columns=None,
    pks=None,
    fks=None,
    domain: str = "general",
    summary: str = "",
    row_count: int = 1000,
) -> TableMetadata:
    cols = columns or []
    pk_list = pks or ([cols[0].name] if cols else [])
    for c in cols:
        if c.name in pk_list:
            c.is_primary_key = True
    fk_cols = {fk.source_column for fk in (fks or [])}
    for c in cols:
        if c.name in fk_cols:
            c.is_foreign_key = True
    return TableMetadata(
        schema_name=schema,
        table_name=name,
        columns=cols,
        primary_keys=pk_list,
        foreign_keys=fks or [],
        row_count=row_count,
        domain=domain,
        summary=summary,
    )


def _col(name: str, dtype: str = "varchar", **kw) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=dtype, **kw)


def _fk(src: str, tgt_table: str, tgt_col: str) -> ForeignKeyMetadata:
    return ForeignKeyMetadata(source_column=src, target_table=tgt_table, target_column=tgt_col)


# ── Domain builders ──────────────────────────────────────────────────────

def _finance_tables():
    invoices = _tbl("invoices", columns=[
        _col("id", "int"), _col("customer_id", "int"), _col("amount", "decimal"),
        _col("status", "varchar"), _col("due_date", "date"), _col("created_at", "timestamp"),
    ], fks=[_fk("customer_id", "customers", "id")], domain="finance",
        summary="Customer invoices for billing")

    payments = _tbl("payments", columns=[
        _col("id", "int"), _col("invoice_id", "int"), _col("amount", "decimal"),
        _col("payment_date", "date"), _col("method", "varchar"),
    ], fks=[_fk("invoice_id", "invoices", "id")], domain="finance",
        summary="Payment records against invoices")

    ledger = _tbl("general_ledger", columns=[
        _col("id", "int"), _col("account_code", "varchar"), _col("debit", "decimal"),
        _col("credit", "decimal"), _col("entry_date", "date"), _col("description", "text"),
    ], domain="finance", summary="General ledger entries")

    accounts = _tbl("accounts", columns=[
        _col("id", "int"), _col("account_code", "varchar"), _col("name", "varchar"),
        _col("type", "varchar"), _col("balance", "decimal"),
    ], domain="finance", summary="Chart of accounts")

    return {t.fqn: t for t in [invoices, payments, ledger, accounts]}


def _sales_tables():
    customers = _tbl("customers", columns=[
        _col("id", "int"), _col("name", "varchar"), _col("email", "varchar"),
        _col("region", "varchar"), _col("created_at", "timestamp"),
    ], domain="sales", summary="Customer master data", row_count=50000)

    orders = _tbl("orders", columns=[
        _col("id", "int"), _col("customer_id", "int"), _col("status", "varchar"),
        _col("total_amount", "decimal"), _col("created_at", "timestamp"),
        _col("shipped_at", "timestamp"),
    ], fks=[_fk("customer_id", "customers", "id")], domain="sales",
        summary="Customer purchase orders", row_count=200000)

    order_items = _tbl("order_items", columns=[
        _col("id", "int"), _col("order_id", "int"), _col("product_id", "int"),
        _col("quantity", "int"), _col("unit_price", "decimal"),
    ], fks=[_fk("order_id", "orders", "id"), _fk("product_id", "products", "id")],
        domain="sales", summary="Line items within orders")

    quotes = _tbl("quotes", columns=[
        _col("id", "int"), _col("customer_id", "int"), _col("total", "decimal"),
        _col("status", "varchar"), _col("valid_until", "date"),
    ], fks=[_fk("customer_id", "customers", "id")], domain="sales",
        summary="Sales quotations")

    return {t.fqn: t for t in [customers, orders, order_items, quotes]}


def _hr_tables():
    employees = _tbl("employees", columns=[
        _col("id", "int"), _col("name", "varchar"), _col("department_id", "int"),
        _col("position", "varchar"), _col("salary", "decimal"), _col("hire_date", "date"),
    ], fks=[_fk("department_id", "departments", "id")], domain="hr",
        summary="Employee records")

    departments = _tbl("departments", columns=[
        _col("id", "int"), _col("name", "varchar"), _col("manager_id", "int"),
        _col("budget", "decimal"),
    ], fks=[_fk("manager_id", "employees", "id")], domain="hr",
        summary="Organisational departments")

    attendance = _tbl("attendance", columns=[
        _col("id", "int"), _col("employee_id", "int"), _col("date", "date"),
        _col("check_in", "timestamp"), _col("check_out", "timestamp"),
    ], fks=[_fk("employee_id", "employees", "id")], domain="hr",
        summary="Daily employee attendance")

    payroll = _tbl("payroll", columns=[
        _col("id", "int"), _col("employee_id", "int"), _col("month", "varchar"),
        _col("gross", "decimal"), _col("deductions", "decimal"), _col("net", "decimal"),
    ], fks=[_fk("employee_id", "employees", "id")], domain="hr",
        summary="Monthly payroll records")

    return {t.fqn: t for t in [employees, departments, attendance, payroll]}


def _inventory_tables():
    products = _tbl("products", columns=[
        _col("id", "int"), _col("sku", "varchar"), _col("name", "varchar"),
        _col("category", "varchar"), _col("unit_price", "decimal"),
        _col("supplier_id", "int"),
    ], fks=[_fk("supplier_id", "suppliers", "id")], domain="inventory",
        summary="Product catalog")

    suppliers = _tbl("suppliers", columns=[
        _col("id", "int"), _col("name", "varchar"), _col("contact_email", "varchar"),
        _col("country", "varchar"),
    ], domain="inventory", summary="Product suppliers")

    stock = _tbl("stock_levels", columns=[
        _col("id", "int"), _col("product_id", "int"), _col("warehouse_id", "int"),
        _col("quantity", "int"), _col("last_updated", "timestamp"),
    ], fks=[_fk("product_id", "products", "id"), _fk("warehouse_id", "warehouses", "id")],
        domain="inventory", summary="Current stock levels per warehouse")

    warehouses = _tbl("warehouses", columns=[
        _col("id", "int"), _col("name", "varchar"), _col("location", "varchar"),
        _col("capacity", "int"),
    ], domain="inventory", summary="Warehouse locations")

    return {t.fqn: t for t in [products, suppliers, stock, warehouses]}


def _support_tables():
    tickets = _tbl("tickets", columns=[
        _col("id", "int"), _col("customer_id", "int"), _col("subject", "varchar"),
        _col("status", "varchar"), _col("priority", "varchar"),
        _col("assigned_to", "int"), _col("created_at", "timestamp"),
    ], fks=[_fk("customer_id", "customers", "id"), _fk("assigned_to", "employees", "id")],
        domain="support", summary="Customer support tickets")

    ticket_comments = _tbl("ticket_comments", columns=[
        _col("id", "int"), _col("ticket_id", "int"), _col("author_id", "int"),
        _col("body", "text"), _col("created_at", "timestamp"),
    ], fks=[_fk("ticket_id", "tickets", "id")], domain="support",
        summary="Comments on support tickets")

    sla_policies = _tbl("sla_policies", columns=[
        _col("id", "int"), _col("name", "varchar"), _col("response_hours", "int"),
        _col("resolution_hours", "int"),
    ], domain="support", summary="SLA policy definitions")

    return {t.fqn: t for t in [tickets, ticket_comments, sla_policies]}


def _shared_tables():
    audit_log = _tbl("audit_log", columns=[
        _col("id", "int"), _col("table_name", "varchar"), _col("record_id", "int"),
        _col("action", "varchar"), _col("user_id", "int"), _col("timestamp", "timestamp"),
    ], domain="general", summary="System-wide audit log")

    settings = _tbl("system_settings", columns=[
        _col("key", "varchar"), _col("value", "text"), _col("updated_at", "timestamp"),
    ], pks=["key"], domain="general", summary="Application settings")

    return {t.fqn: t for t in [audit_log, settings]}


def _generic_table(idx: int) -> TableMetadata:
    return _tbl(f"data_table_{idx:03d}", columns=[
        _col("id", "int"),
        _col(f"field_a_{idx}", "varchar"),
        _col(f"field_b_{idx}", "int"),
        _col("created_at", "timestamp"),
    ], domain="general", summary=f"Generic data table {idx}")
