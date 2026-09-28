import json
import datetime
from typing import Any, Optional, List, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.models import MCPServerConfig

router = APIRouter(prefix="/bajaj", tags=["Bajaj Dashboard"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_company_name(comp_code: Any) -> str:
    if not comp_code:
        return "N/A"
    return str(comp_code).replace(".0", "")

async def get_dynamic_plant_map(db_session: Session) -> Dict[str, str]:
    """Returns a dict mapping plant_code -> company_name dynamically from the DB."""
    query = """
    WITH all_plants AS (
        SELECT plant FROM pending_pr WHERE plant IS NOT NULL
        UNION
        SELECT plant FROM pr_to_po_conversion WHERE plant IS NOT NULL
        UNION
        SELECT plant FROM ordering_trend WHERE plant IS NOT NULL
        UNION
        SELECT plant FROM delivery_analysis WHERE plant IS NOT NULL
    ),
    comp_map AS (
        SELECT DISTINCT plant, company_code FROM ordering_trend WHERE plant IS NOT NULL AND company_code IS NOT NULL
    )
    SELECT CAST(a.plant AS INTEGER) as p, c.company_code 
    FROM all_plants a
    LEFT JOIN comp_map c ON a.plant = c.plant
    ORDER BY p
    """
    raw_data = await execute_bajaj_query(query, db_session)
    plant_map = {}
    for row in raw_data:
        p = row.get("p")
        if not p:
            continue
            
        comp_code = row.get("company_code")
        plant_map[str(p)] = resolve_company_name(comp_code)
        
    return plant_map

async def build_plant_where(
    table_alias: str,
    company: Optional[List[str]],
    plant: Optional[List[str]],
    db_session: Session
) -> str:
    """
    Build a SQL WHERE fragment for plant filtering.
    """
    col = f"{table_alias}.plant" if table_alias else "plant"
    
    if plant and len(plant) > 0:
        plants_str = ", ".join(f"{int(p)}" for p in plant)
        return f"CAST({col} AS INTEGER) IN ({plants_str})"
    
    if company and len(company) > 0:
        plant_map = await get_dynamic_plant_map(db_session)
        matching_plants = [p for p, c in plant_map.items() if c in company]
        if matching_plants:
            plants_str = ", ".join(matching_plants)
            return f"CAST({col} AS INTEGER) IN ({plants_str})"
        else:
            # If no plants match the company, return a condition that is always false
            return "1=0"
    return ""

@router.get("/plants")
async def get_plants(db_session: Session = Depends(db.get_session)):
    """Return all confirmed plant codes grouped by company for the filter UI."""
    plant_map = await get_dynamic_plant_map(db_session)
    res = []
    for p, comp in sorted(plant_map.items()):
        res.append({"id": p, "company": comp, "label": f"{p} ({comp})"})
    return res

async def execute_bajaj_query(query: str, db: Session) -> list[dict[str, Any]]:
    mcp_config = db.exec(select(MCPServerConfig).where(MCPServerConfig.name == "BAJAJ")).first()
    if not mcp_config:
        raise HTTPException(status_code=500, detail="BAJAJ MCP configuration not found")

    mcp_mgr = get_mcp_manager()
    conn_result = await mcp_mgr.connect("BAJAJ", {"name": "BAJAJ", "url": mcp_config.url})
    if conn_result.get("status") == "error":
        raise HTTPException(status_code=500, detail=f"Failed to connect to BAJAJ MCP: {conn_result.get('error')}")

    result = await mcp_mgr.call_tool("BAJAJ", "bajaj_execute_query", {"query": query})
    if result.get("isError"):
        raise HTTPException(status_code=500, detail=f"Query failed: {result}")

    content = result.get("content", [])
    if not content:
        return []

    try:
        return json.loads(content[0].get("text", "[]"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse MCP response")


# ── 1. Pending PR Status ──────────────────────────────────────────────────────

@router.get("/pending-pr-status")
async def get_pending_pr_status(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code e.g. 4004"),
    start_date: Optional[str] = Query(None, description="Requisition date from YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Requisition date to YYYY-MM-DD"),
):
    clauses = []
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"requisition_date >= '{start_date}'")
    if end_date:   clauses.append(f"requisition_date <= '{end_date}'")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    query = f"""
    SELECT
        requisitioner,
        MAX(requisition_date) as max_date,
        COUNT(purchase_requisition) as total_items,
        COUNT(CASE WHEN release_indicator = '2' THEN 1 END) as running_pr,
        COUNT(CASE WHEN release_indicator != '2' OR release_indicator IS NULL THEN 1 END) as u_approval
    FROM pending_pr
    {where}
    GROUP BY requisitioner;
    """
    raw_data = await execute_bajaj_query(query, db_session)

    # Derive year label from max date in result set
    global_max_date = None
    for row in raw_data:
        dt = row.get("max_date")
        if dt and (not global_max_date or dt > global_max_date):
            global_max_date = dt

    year_label = "Till Date"
    if global_max_date:
        try:
            d = datetime.datetime.fromisoformat(str(global_max_date).replace('Z', '+00:00')) \
                if isinstance(global_max_date, str) else global_max_date
            year_label = f"Up to {d.strftime('%B %Y')}"
        except Exception:
            year_label = f"Up to {str(global_max_date)[:7]}"

    members = sorted(list(set(row.get("requisitioner") for row in raw_data if row.get("requisitioner"))))
    if any(not row.get("requisitioner") for row in raw_data):
        members.append(None)

    result = []
    for member in members:
        member_str = str(member) if member else "Unknown"
        row = next((r for r in raw_data if r.get("requisitioner") == member), None)
        
        if row:
            result.append({
                "Year": year_label,
                "Team Member": member_str,
                "U/Enq": "",
                "Under TER": "",
                "U/Negotiation": "",
                "U/Approval": row.get("u_approval", 0),
                "Local offer awaited": "",
                "Consolidated requirement awaited": "",
                "Hold by User / O&M": "",
                "Material code to be changed": "",
                "Drawing Required": "",
                "To be closed": "",
                "Running PR": row.get("running_pr", 0),
                "Site Store": "",
                "Total Items": row.get("total_items", 0),
            })
            year_label = "" # Show year label only on the first row

    result.append({
        "Year": "",
        "Team Member": "Total",
        "U/Enq": "", "Under TER": "", "U/Negotiation": "", "U/Approval": "",
        "Local offer awaited": "", "Consolidated requirement awaited": "",
        "Hold by User / O&M": "", "Material code to be changed": "",
        "Drawing Required": "", "To be closed": "", "Running PR": "",
        "Site Store": "",
        "Total Items": sum(r.get("total_items", 0) for r in raw_data) or "",
    })
    return result


# ── 2. PR-PO Summary ──────────────────────────────────────────────────────────

@router.get("/pr-po-summary")
async def get_pr_po_summary(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code"),
    start_date: Optional[str] = Query(None, description="Requisition date from YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Requisition date to YYYY-MM-DD"),
):
    # NOTE: purchasing_group values are numeric SAP codes — dept name mapping PENDING Bajaj.
    clauses = []
    pf = await build_plant_where("p2p", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"p2p.requisition_date >= '{start_date}'")
    if end_date:   clauses.append(f"p2p.requisition_date <= '{end_date}'")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    query = f"""
    SELECT
        p2p.purchasing_group,
        COUNT(DISTINCT p2p.purchase_requisition) as total_prs,
        COUNT(DISTINCT p2p.purchase_order) as total_pos,
        COUNT(DISTINCT CASE WHEN da.grn_date IS NULL THEN p2p.purchase_order END) as po_placed,
        COUNT(DISTINCT CASE WHEN da.lr_rr_date IS NOT NULL AND da.grn_date IS NULL THEN p2p.purchase_order END) as in_transit,
        COUNT(DISTINCT CASE WHEN da.grn_date IS NOT NULL THEN p2p.purchase_order END) as received
    FROM pr_to_po_conversion p2p
    LEFT JOIN delivery_analysis da ON p2p.purchase_order = da.purchase_order_no
    {where}
    GROUP BY p2p.purchasing_group;
    """
    raw_data = await execute_bajaj_query(query, db_session)

    groups = sorted(list(set(row.get("purchasing_group") for row in raw_data if row.get("purchasing_group"))))
    if any(not row.get("purchasing_group") for row in raw_data):
        groups.append(None)

    result = []
    for i, dept in enumerate(groups, 1):
        dept_str = str(dept) if dept else "Unknown"
        row = next((r for r in raw_data if r.get("purchasing_group") == dept), None)
        
        if row:
            result.append({
                "S. No.": i,
                "Deptt": dept_str,
                "Total PRs": row.get("total_prs", 0),
                "Under Enquiry": "", "U/TER": "", "Under Negotiation": "", "Under Approval": "",
                "PO placed but material not recd": row.get("po_placed", 0),
                "Material is in transit": row.get("in_transit", 0),
                "Material received": row.get("received", 0),
                "Total": "",
            })

    result.append({
        "S. No.": "",
        "Deptt": "Total (All Groups)",
        "Total PRs": sum(r.get("total_prs", 0) for r in raw_data) or "",
        "Under Enquiry": "", "U/TER": "", "Under Negotiation": "", "Under Approval": "",
        "PO placed but material not recd": sum(r.get("po_placed", 0) for r in raw_data),
        "Material is in transit": sum(r.get("in_transit", 0) for r in raw_data),
        "Material received": sum(r.get("received", 0) for r in raw_data),
        "Total": "",
    })
    return result


# ── 3. Delivery Analysis ──────────────────────────────────────────────────────

@router.get("/delivery-analysis")
async def get_delivery_analysis(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code"),
    start_date: Optional[str] = Query(None, description="Delivery date from YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Delivery date to YYYY-MM-DD"),
    status: Optional[str] = Query(None, description="pending | on_time | delayed"),
):
    clauses = []
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"delivery_date >= '{start_date}'")
    if end_date:   clauses.append(f"delivery_date <= '{end_date}'")
    if status == "pending":  clauses.append("grn_date IS NULL")
    if status == "on_time":  clauses.append("grn_date IS NOT NULL AND grn_date <= delivery_date")
    if status == "delayed":  clauses.append("grn_date IS NOT NULL AND grn_date > delivery_date")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    query = f"""
    SELECT
        plant,
        COUNT(*) as total_items,
        SUM(CASE WHEN grn_date IS NULL THEN 1 ELSE 0 END) as delivery_pending,
        SUM(CASE WHEN grn_date <= delivery_date THEN 1 ELSE 0 END) as on_time,
        SUM(CASE WHEN grn_date > delivery_date THEN 1 ELSE 0 END) as delayed,
        SUM(CASE WHEN DATE_PART('day', grn_date - delivery_date) BETWEEN 1 AND 15 THEN 1 ELSE 0 END) as days_1_15,
        SUM(CASE WHEN DATE_PART('day', grn_date - delivery_date) BETWEEN 16 AND 30 THEN 1 ELSE 0 END) as days_16_30,
        SUM(CASE WHEN DATE_PART('day', grn_date - delivery_date) > 30 THEN 1 ELSE 0 END) as days_gt_30
    FROM delivery_analysis
    {where}
    GROUP BY plant;
    """
    raw_data = await execute_bajaj_query(query, db_session)

    # Determine which plant codes are in scope for the current filter
    plant_map = await get_dynamic_plant_map(db_session)
    if plant and len(plant) > 0:
        scope_plants = plant
    elif company and len(company) > 0:
        scope_plants = [k for k, v in plant_map.items() if v in company]
    else:
        scope_plants = list(plant_map.keys())

    # Build result rows — one per plant code, filled only if data exists
    unit_rows = []
    for p in scope_plants:
        row_data = next((r for r in raw_data if str(int(float(r["plant"]))) == str(p)), None)
        has_data = row_data is not None
        comp_name = plant_map.get(str(p), "Unknown")
        label = f"{p} ({comp_name})"
        unit_rows.append({
            "Plant": label,
            "Total Items (Nos)": row_data["total_items"] if has_data else "",
            "Delivery Pending (Nos)": row_data["delivery_pending"] if has_data else "",
            "Total items received (On Time) (Nos)": row_data["on_time"] if has_data else "",
            "Total items delayed (Nos)": row_data["delayed"] if has_data else "",
            "1 - 15 Days": row_data["days_1_15"] if has_data else "",
            "16 - 30 Days": row_data["days_16_30"] if has_data else "",
            "> 30 Days": row_data["days_gt_30"] if has_data else "",
            "_has_data": has_data,
        })

    unit_rows.sort(key=lambda x: (not x["_has_data"], x["Plant"]))
    result = []
    for row in unit_rows:
        row.pop("_has_data", None)
        result.append(row)

    def _s(key): return sum(r[key] for r in unit_rows if isinstance(r.get(key), (int, float)))

    result.append({
        "Plant": "TOTAL",
        "Total Items (Nos)": _s("Total Items (Nos)"),
        "Delivery Pending (Nos)": _s("Delivery Pending (Nos)"),
        "Total items received (On Time) (Nos)": _s("Total items received (On Time) (Nos)"),
        "Total items delayed (Nos)": _s("Total items delayed (Nos)"),
        "1 - 15 Days": _s("1 - 15 Days"),
        "16 - 30 Days": _s("16 - 30 Days"),
        "> 30 Days": _s("> 30 Days"),
    })

    total_items_val = _s("Total Items (Nos)")
    pct = lambda n: f"{(n / total_items_val * 100):.2f}%" if total_items_val else ""
    result.append({
        "Plant": "%",
        "Total Items (Nos)": "100.00%" if total_items_val else "",
        "Delivery Pending (Nos)": pct(_s("Delivery Pending (Nos)")),
        "Total items received (On Time) (Nos)": pct(_s("Total items received (On Time) (Nos)")),
        "Total items delayed (Nos)": pct(_s("Total items delayed (Nos)")),
        "1 - 15 Days": "", "16 - 30 Days": "", "> 30 Days": "",
    })
    return result


# ── 4. Ordering Trend ─────────────────────────────────────────────────────────

@router.get("/ordering-trend")
async def get_ordering_trend(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code"),
    start_month: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end_month: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    # Doc type values confirmed from MCP: 'O&M PO Mat - UNIT', 'O & M PO Ser - UNIT',
    # 'LPGCL_PO_O&M_MAT_U', 'Services-Dom - UNIT', 'Store PO - UNIT', 'MRP PO - UNIT' etc.
    # Fuel commitment: 'Inventory- Furnance Oil & Other Fuel'
    clauses = ["document_date IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_month: clauses.append(f"DATE_TRUNC('month', document_date) >= '{start_month}-01'")
    if end_month:   clauses.append(f"DATE_TRUNC('month', document_date) <= '{end_month}-01'")
    where = "WHERE " + " AND ".join(clauses)

    query = f"""
    SELECT
        plant,
        TO_CHAR(DATE_TRUNC('month', document_date), 'Mon-YY') as month_str,
        SUM(CASE WHEN REPLACE(doc_type_descript, ' ', '') ILIKE '%O&M%' THEN net_order_value ELSE 0 END) as om,
        SUM(CASE WHEN commitment_item_name ILIKE '%Fuel%' OR commitment_item_name ILIKE '%Furnance%' OR commitment_item_name ILIKE '%Furnace%' THEN net_order_value ELSE 0 END) as hsd_ldo,
        SUM(CASE WHEN REPLACE(doc_type_descript, ' ', '') ILIKE '%Ser%' AND REPLACE(doc_type_descript, ' ', '') NOT ILIKE '%O&M%' THEN net_order_value ELSE 0 END) as service,
        SUM(CASE WHEN (REPLACE(doc_type_descript, ' ', '') ILIKE '%Mat%' OR doc_type_descript ILIKE '%Store%' OR doc_type_descript ILIKE '%MRP%') AND REPLACE(doc_type_descript, ' ', '') NOT ILIKE '%O&M%' THEN net_order_value ELSE 0 END) as supply,
        SUM(net_order_value) as grand_total
    FROM ordering_trend
    {where}
    GROUP BY plant, DATE_TRUNC('month', document_date)
    ORDER BY DATE_TRUNC('month', document_date);
    """
    raw_data = await execute_bajaj_query(query, db_session)

    def map_month(db_month: str) -> str:
        if not db_month or '-' not in db_month: return db_month
        parts = db_month.split('-')
        return f"{parts[1]}-{parts[0]}" if len(parts) == 2 else db_month

    # Collect all distinct months from the result set
    all_months_db = sorted({r.get("month_str", "") for r in raw_data if r.get("month_str")})
    months = [map_month(m) for m in all_months_db] if all_months_db else []

    # Determine which companies to show
    companies_to_show = []
    plant_map = await get_dynamic_plant_map(db_session)
    if plant and len(plant) > 0:
        for p in plant:
            companies_to_show.append((plant_map.get(str(p), "ALL"), str(p)))
    elif company and len(company) > 0:
        for c in company:
            companies_to_show.append((c, None))
    else:
        all_comps = list(set(plant_map.values()))
        companies_to_show = [(c, None) for c in all_comps]

    result = []
    for comp_label, filter_plant in companies_to_show:
        result.append({
            "MONTHS": f"{filter_plant} ({comp_label})" if filter_plant else f"{comp_label}",
            "O&M": "", "HSD": "", "LDO": "", "Supply": "", "Service": "",
            "Grand Total": "Amount in Cr.",
        })
        grand_om = grand_hsd = grand_supply = grand_service = grand_total = 0.0
        for m in months:
            m_data = {"om": 0.0, "hsd_ldo": 0.0, "supply": 0.0, "service": 0.0, "grand_total": 0.0}
            found = False
            for row in raw_data:
                db_m = map_month(row.get("month_str", ""))
                p_val = row.get("plant")
                if p_val is None: continue
                row_plant = str(int(float(p_val)))
                if filter_plant:
                    matches = (db_m == m and row_plant == str(filter_plant))
                else:
                    matches = (db_m == m and plant_map.get(row_plant) == comp_label)
                if matches:
                    found = True
                    m_data["om"]          += row.get("om", 0) or 0
                    m_data["hsd_ldo"]     += row.get("hsd_ldo", 0) or 0
                    m_data["supply"]      += row.get("supply", 0) or 0
                    m_data["service"]     += row.get("service", 0) or 0
                    m_data["grand_total"] += row.get("grand_total", 0) or 0

            if found:
                grand_om      += m_data["om"]
                grand_hsd     += m_data["hsd_ldo"]
                grand_supply  += m_data["supply"]
                grand_service += m_data["service"]
                grand_total   += m_data["grand_total"]

            result.append({
                "MONTHS": m,
                "O&M":      f"{m_data['om']:,.2f}"          if found else "",
                "HSD":      f"{m_data['hsd_ldo']:,.2f}"     if found else "",
                "LDO":      "",
                "Supply":   f"{m_data['supply']:,.2f}"      if found else "",
                "Service":  f"{m_data['service']:,.2f}"     if found else "",
                "Grand Total": f"{m_data['grand_total']:,.2f}" if found else "",
            })

        result.append({
            "MONTHS": "GRAND TOTAL (AMT IN CR.)",
            "O&M":    f"{grand_om:,.2f}"      if grand_om      else "-",
            "HSD":    f"{grand_hsd:,.2f}"     if grand_hsd     else "-",
            "LDO":    "-",
            "Supply": f"{grand_supply:,.2f}"  if grand_supply  else "-",
            "Service":f"{grand_service:,.2f}" if grand_service else "-",
            "Grand Total": f"{grand_total:,.2f}" if grand_total else "-",
        })

    return result


# ── 5. Conversion Analysis ────────────────────────────────────────────────────

@router.get("/conversion-analysis")
async def get_conversion_analysis(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code"),
    start_date: Optional[str] = Query(None, description="Requisition date from YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Requisition date to YYYY-MM-DD"),
    conv_time: Optional[str] = Query(None, description="21 Days|30 Days|45 Days|60 Days|More Then 60 Days"),
):
    clauses = ["purchase_order_date IS NOT NULL", "requisition_date IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"requisition_date >= '{start_date}'")
    if end_date:   clauses.append(f"requisition_date <= '{end_date}'")
    where = "WHERE " + " AND ".join(clauses)

    query = f"""
    SELECT
        plant,
        CASE
            WHEN DATE_PART('day', purchase_order_date - requisition_date) <= 21 THEN '21 Days'
            WHEN DATE_PART('day', purchase_order_date - requisition_date) <= 30 THEN '30 Days'
            WHEN DATE_PART('day', purchase_order_date - requisition_date) <= 45 THEN '45 Days'
            WHEN DATE_PART('day', purchase_order_date - requisition_date) <= 60 THEN '60 Days'
            ELSE 'More Then 60 Days'
        END as conv_time,
        COUNT(DISTINCT purchase_requisition) as pr_count,
        COUNT(DISTINCT purchase_order) as po_count,
        SUM(total_value) as total_val
    FROM pr_to_po_conversion
    {where}
    GROUP BY 1, 2;
    """
    raw_data = await execute_bajaj_query(query, db_session)

    plant_map = await get_dynamic_plant_map(db_session)
    if plant and len(plant) > 0:
        companies = list(set([plant_map.get(str(p), "Unknown") for p in plant]))
    elif company and len(company) > 0:
        all_comps = list(set(plant_map.values()))
        companies = [c for c in company if c in all_comps]
    else:
        companies = list(set(plant_map.values()))

    intervals = [conv_time] if conv_time else ["21 Days", "30 Days", "45 Days", "60 Days", "More Then 60 Days"]

    result = []
    for comp in companies:
        comp_prs = comp_pos = comp_amount = 0.0
        for i, interval in enumerate(intervals):
            pr_val = po_val = amt_val = 0
            for row in raw_data:
                row_plant = str(int(float(row["plant"]))) if row.get("plant") else None
                if row_plant and plant_map.get(row_plant) == comp and row.get("conv_time") == interval:
                    pr_val  += row.get("pr_count", 0)
                    po_val  += row.get("po_count", 0)
                    amt_val += row.get("total_val", 0) or 0

            comp_prs    += pr_val
            comp_pos    += po_val
            comp_amount += amt_val

            result.append({
                "Company Name": comp if i == 2 else "",
                "Conv. Time": interval,
                "Nos of PR's": pr_val if pr_val else "",
                "Nos of PO's": po_val if po_val else "",
                "PO Amount": f"{amt_val:,.2f}" if amt_val else "",
            })

        result.append({
            "Company Name": "",
            "Conv. Time": "Total",
            "Nos of PR's": comp_prs,
            "Nos of PO's": comp_pos,
            "PO Amount": f"{comp_amount:,.2f}",
        })

    return result

# ── 6. Quality Analysis ───────────────────────────────────────────────────────

@router.get("/quality-analysis")
async def get_quality_analysis(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code"),
    start_date: Optional[str] = Query(None, description="Delivery date from YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Delivery date to YYYY-MM-DD"),
):
    clauses = ["(accepted_quantity > 0 OR rejected_quantity > 0)"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"delivery_date >= '{start_date}'")
    if end_date:   clauses.append(f"delivery_date <= '{end_date}'")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    query = f"""
    SELECT
        plant,
        SUM(accepted_quantity) as accepted,
        SUM(rejected_quantity) as rejected,
        SUM(accepted_quantity + COALESCE(rejected_quantity, 0)) as total
    FROM delivery_analysis
    {where}
    GROUP BY plant;
    """
    raw_data = await execute_bajaj_query(query, db_session)

    plant_map = await get_dynamic_plant_map(db_session)
    
    result = []
    for row in raw_data:
        p_val = row.get("plant")
        if p_val is None: continue
        row_plant = str(int(float(p_val)))
        comp_name = plant_map.get(row_plant, "Unknown")
        label = f"{row_plant} ({comp_name})"
        
        result.append({
            "Plant": label,
            "Accepted": row.get("accepted", 0) or 0,
            "Rejected": row.get("rejected", 0) or 0,
            "Total": row.get("total", 0) or 0
        })

    return result

# ── 7. Category Spend ────────────────────────────────────────────────────────

@router.get("/spend-by-category")
async def get_spend_by_category(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None, description="BEPL or LPGCL"),
    plant: Optional[List[str]] = Query(None, description="Specific plant code"),
    start_month: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end_month: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    clauses = ["commitment_item_name IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_month: clauses.append(f"DATE_TRUNC('month', document_date) >= '{start_month}-01'")
    if end_month:   clauses.append(f"DATE_TRUNC('month', document_date) <= '{end_month}-01'")
    where = "WHERE " + " AND ".join(clauses)

    query = f"""
    SELECT
        commitment_item_name,
        SUM(net_order_value) as total_value
    FROM ordering_trend
    {where}
    GROUP BY commitment_item_name
    ORDER BY total_value DESC
    LIMIT 10;
    """
    raw_data = await execute_bajaj_query(query, db_session)
    
    result = []
    for row in raw_data:
        val = row.get("total_value", 0) or 0
        if val > 0:
            result.append({
                "Category": row.get("commitment_item_name"),
                "Value": val
            })
            
    return result

# ── 8. Drill-Down Endpoints ──────────────────────────────────────────────────

@router.get("/drill-down/ordering")
async def drill_down_ordering(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None),
    plant: Optional[List[str]] = Query(None),
    month: Optional[str] = Query(None, description="Month in Mon-YY format e.g. Jan-24"),
    category: Optional[str] = Query(None, description="Commitment item name"),
    group_by: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
):
    clauses = ["purchasing_document IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    
    if vendor:
        clauses.append(f"vendor_supplying_plant = '{vendor}'")
    
    if month:
        try:
            import datetime
            d = datetime.datetime.strptime(month, "%b-%y")
            start_date = d.strftime("%Y-%m-01")
            clauses.append(f"DATE_TRUNC('month', document_date) = '{start_date}'")
        except ValueError:
            pass

    if category:
        if category == "O&M":
            clauses.append("REPLACE(doc_type_descript, ' ', '') ILIKE '%O&M%'")
        elif category == "HSD / LDO":
            clauses.append("(commitment_item_name ILIKE '%Fuel%' OR commitment_item_name ILIKE '%Furnance%' OR commitment_item_name ILIKE '%Furnace%')")
        elif category == "Service":
            clauses.append("REPLACE(doc_type_descript, ' ', '') ILIKE '%Ser%' AND REPLACE(doc_type_descript, ' ', '') NOT ILIKE '%O&M%'")
        elif category == "Supply":
            clauses.append("(REPLACE(doc_type_descript, ' ', '') ILIKE '%Mat%' OR doc_type_descript ILIKE '%Store%' OR doc_type_descript ILIKE '%MRP%') AND REPLACE(doc_type_descript, ' ', '') NOT ILIKE '%O&M%'")
        else:
            clauses.append(f"commitment_item_name = '{category}'")
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    
    if group_by:
        query = f"""
        SELECT
            {group_by} as group_name,
            COUNT(DISTINCT purchasing_document) as total_pos,
            SUM(net_order_value) as total_val
        FROM ordering_trend
        {where}
        GROUP BY {group_by}
        ORDER BY total_val DESC
        """
    else:
        query = f"""
        SELECT
            company_code,
            plant,
            vendor_supplying_plant as vendor,
            purchasing_document as purchase_order,
            CAST(document_date AS TEXT) as doc_date,
            commitment_item_name,
            net_order_value
        FROM ordering_trend
        {where}
        ORDER BY net_order_value DESC
        ;
        """
    return await execute_bajaj_query(query, db_session)

@router.get("/drill-down/delivery")
async def drill_down_delivery(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None),
    plant: Optional[List[str]] = Query(None),
    status: Optional[str] = Query(None, description="On Time, Delayed, or Pending"),
    group_by: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    material_code: Optional[str] = Query(None),
):
    clauses = ["purchase_order_no IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    
    if vendor:
        clauses.append(f"vendor_name = '{vendor}'")
    if material_code:
        clauses.append(f"material_code = '{material_code}'")
        
    if status == "Pending":
        clauses.append("grn_date IS NULL")
    elif status == "Delayed":
        clauses.append("grn_date > delivery_date")
    elif status == "On Time":
        clauses.append("grn_date <= delivery_date")
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    
    if group_by:
        query = f"""
        SELECT
            {group_by} as group_name,
            COUNT(DISTINCT purchase_order_no) as total_deliveries,
            SUM(order_quantity) as total_qty,
            SUM(rejected_quantity) as total_rejected
        FROM delivery_analysis
        {where}
        GROUP BY {group_by}
        ORDER BY total_deliveries DESC
        """
    else:
        query = f"""
        SELECT
            plant,
            vendor_name as vendor,
            material_code,
            purchase_order_no,
            CAST(delivery_date AS TEXT) as delivery_date,
            CAST(grn_date AS TEXT) as grn_date,
            accepted_quantity,
            rejected_quantity
        FROM delivery_analysis
        {where}
        ORDER BY delivery_date DESC
        ;
        """
    return await execute_bajaj_query(query, db_session)

@router.get("/drill-down/pending-pr")
async def drill_down_pending_pr(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None),
    plant: Optional[List[str]] = Query(None),
    requisitioner: Optional[str] = Query(None),
    purchasing_group: Optional[str] = Query(None),
    material_group: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="Running PR, U/Approval, Total"),
    group_by: Optional[str] = Query(None),
):
    clauses = ["purchase_requisition IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if requisitioner and requisitioner != "Unknown":
        clauses.append(f"requisitioner = '{requisitioner}'")
    if purchasing_group:
        clauses.append(f"purchasing_group = '{purchasing_group}'")
    if material_group:
        clauses.append(f"material_group = '{material_group}'")
        
    if status == "Running PR":
        clauses.append("release_indicator = '2'")
    elif status == "U/Approval":
        clauses.append("(release_indicator != '2' OR release_indicator IS NULL)")
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    
    if group_by:
        query = f"""
        SELECT
            {group_by} as group_name,
            COUNT(DISTINCT purchase_requisition) as pr_count,
            SUM(total_value) as total_val
        FROM pending_pr
        {where}
        GROUP BY {group_by}
        ORDER BY pr_count DESC
        """
    else:
        query = f"""
        SELECT
            plant,
            purchasing_group,
            purchase_requisition,
            requisition_date,
            requisitioner,
            total_value,
            release_indicator
        FROM pending_pr
        {where}
        ORDER BY requisition_date DESC
        ;
        """
    return await execute_bajaj_query(query, db_session)

@router.get("/drill-down/quality")
async def drill_down_quality(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None),
    plant: Optional[List[str]] = Query(None),
    plant_code: Optional[str] = Query(None, description="Clicked plant code"),
    status: Optional[str] = Query(None, description="Accepted or Rejected"),
):
    clauses = ["purchase_order_no IS NOT NULL"]
    
    # Force plant code to be the clicked one
    if plant_code:
        code = plant_code.split(" ")[0] # extract '4001' from '4001 (BEPL)'
        clauses.append(f"CAST(plant AS INTEGER) = {int(code)}")
    else:
        pf = await build_plant_where("", company, plant, db_session)
        if pf: clauses.append(pf)
        
    if status == "Accepted":
        clauses.append("accepted_quantity > 0")
    elif status == "Rejected":
        clauses.append("rejected_quantity > 0")
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"""
    SELECT
        plant,
        purchase_order_no,
        delivery_date,
        accepted_quantity,
        rejected_quantity
    FROM delivery_analysis
    {where}
    ORDER BY delivery_date DESC
    ;
    """
    return await execute_bajaj_query(query, db_session)


@router.get("/drill-down/pr-po")
async def get_drill_down_pr_po(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None),
    plant: Optional[List[str]] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    deptt: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    clauses = []
    pf = await build_plant_where("p2p", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"p2p.requisition_date >= '{start_date}'")
    if end_date:   clauses.append(f"p2p.requisition_date <= '{end_date}'")
    
    if deptt and deptt != "Unknown": clauses.append(f"p2p.purchasing_group = '{deptt}'")
    elif deptt == "Unknown": clauses.append("p2p.purchasing_group IS NULL")

    if status == 'PO Placed':
        clauses.append("da.grn_date IS NULL")
    elif status == 'In Transit':
        clauses.append("da.lr_rr_date IS NOT NULL AND da.grn_date IS NULL")
    elif status == 'Received':
        clauses.append("da.grn_date IS NOT NULL")
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"""
    SELECT
        p2p.plant,
        p2p.purchasing_group as deptt,
        p2p.purchase_requisition,
        p2p.purchase_order,
        CAST(da.lr_rr_date AS TEXT) as transit_date,
        CAST(da.grn_date AS TEXT) as grn_date
    FROM pr_to_po_conversion p2p
    LEFT JOIN delivery_analysis da ON p2p.purchase_order = da.purchase_order_no
    {where}
    ;
    """
    raw_data = await execute_bajaj_query(query, db_session)
    return raw_data


@router.get("/drill-down/conversion")
async def get_drill_down_conversion(
    db_session: Session = Depends(db.get_session),
    company: Optional[List[str]] = Query(None),
    plant: Optional[List[str]] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    conv_time: Optional[str] = Query(None, description="21 Days|30 Days|45 Days|60 Days|More Then 60 Days"),
    clicked_company: Optional[str] = Query(None),
    group_by: Optional[str] = Query(None, description="Column to aggregate by e.g. requisitioner"),
    requisitioner: Optional[str] = Query(None),
    material_group: Optional[str] = Query(None),
):
    # Support backward compatibility if frontend passed bucket into clicked_company
    buckets = ["21 Days", "30 Days", "45 Days", "60 Days", "More Then 60 Days"]
    if clicked_company in buckets and not conv_time:
        conv_time = clicked_company
        clicked_company = None

    clauses = ["purchase_order_date IS NOT NULL", "requisition_date IS NOT NULL"]
    pf = await build_plant_where("", company, plant, db_session)
    if pf: clauses.append(pf)
    if start_date: clauses.append(f"requisition_date >= '{start_date}'")
    if end_date:   clauses.append(f"requisition_date <= '{end_date}'")
    
    if conv_time:
        if conv_time == '21 Days':
            clauses.append("DATE_PART('day', purchase_order_date - requisition_date) <= 21")
        elif conv_time == '30 Days':
            clauses.append("DATE_PART('day', purchase_order_date - requisition_date) > 21 AND DATE_PART('day', purchase_order_date - requisition_date) <= 30")
        elif conv_time == '45 Days':
            clauses.append("DATE_PART('day', purchase_order_date - requisition_date) > 30 AND DATE_PART('day', purchase_order_date - requisition_date) <= 45")
        elif conv_time == '60 Days':
            clauses.append("DATE_PART('day', purchase_order_date - requisition_date) > 45 AND DATE_PART('day', purchase_order_date - requisition_date) <= 60")
        elif conv_time == 'More Then 60 Days':
            clauses.append("DATE_PART('day', purchase_order_date - requisition_date) > 60")

    if requisitioner:
        clauses.append(f"requisitioner = '{requisitioner}'")
    if material_group:
        clauses.append(f"material_group = '{material_group}'")
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    
    if group_by:
        query = f"""
        SELECT
            {group_by} as group_name,
            COUNT(DISTINCT purchase_order) as total_pos,
            ROUND(AVG(DATE_PART('day', purchase_order_date - requisition_date))::numeric, 1) as avg_delay,
            MAX(DATE_PART('day', purchase_order_date - requisition_date)) as max_delay,
            ROUND(SUM(COALESCE(total_value, 0))::numeric, 2) as total_val
        FROM pr_to_po_conversion
        {where}
        GROUP BY {group_by}
        ORDER BY total_pos DESC
        """
    else:
        query = f"""
        SELECT
            plant,
            requisitioner,
            material_group,
            purchase_requisition,
            purchase_order,
            CAST(requisition_date AS TEXT) as req_date,
            CAST(purchase_order_date AS TEXT) as po_date,
            DATE_PART('day', purchase_order_date - requisition_date) as conv_time_days,
            total_value as value
        FROM pr_to_po_conversion
        {where}
        ORDER BY conv_time_days DESC
        """
        
    raw_data = await execute_bajaj_query(query, db_session)
    
    # Filter by clicked_company (Only applies if filtering by company is needed and not grouped)
    if clicked_company and clicked_company != "Unknown" and clicked_company not in buckets:
        plant_map = await get_dynamic_plant_map(db_session)
        def safe_get_comp(p_val):
            if not p_val: return "Unknown"
            try: return plant_map.get(str(int(float(p_val))), "Unknown")
            except: return "Unknown"
        raw_data = [r for r in raw_data if r.get("plant") is None or safe_get_comp(r.get("plant")) == clicked_company]
        
    return raw_data
