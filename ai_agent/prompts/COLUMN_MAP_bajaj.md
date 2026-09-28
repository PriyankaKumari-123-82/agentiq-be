# Column Mapping Reference

Full mapping of every column across all 4 tables in the `bajaj_poc` database.
Original Excel header → normalized DB column name, data type, and business meaning.

---

## Table: `pending_pr`
> Open Purchase Requisitions not yet converted to a PO as of 07-Aug-2026
> Source: `Pending PR as on 07.08.2026.XLSX` | Rows: 3,463

| # | DB Column Name | Original Excel Header | Data Type | Business Meaning | Relationship |
|---|---|---|---|---|---|
| 1 | `plant` | Plant | TEXT | Plant/location code where the PR was raised (e.g. 3004, 3007) | Links to all 4 tables on `plant` |
| 2 | `document_type` | Document Type | TEXT | SAP document type (e.g. ZP13 = material PO, ZP22 = service PO) | — |
| 3 | `first_release_date` | First Release Date | TIMESTAMP | Date the PR was first approved/released | — |
| 4 | `requisition_date` | Requisition Date | TIMESTAMP | Date the PR was originally raised | — |
| 5 | `purchase_requisition` | Purchase Requisition | TEXT | Unique PR number (e.g. 1300009665) | **Join key** → `pr_to_po_conversion.purchase_requisition`, `ordering_trend.purchase_requisition`, `delivery_analysis.purchase_requisition_no` |
| 6 | `acct_assignment_cat_` | Acct Assignment Cat. | TEXT | Account assignment category (K = cost center, blank = stock) | — |
| 7 | `changed_on` | Changed On | TIMESTAMP | Last date the PR record was modified | — |
| 8 | `item_of_requisition` | Item of requisition | TEXT | Line item number within the PR (10, 20, 30…) | — |
| 9 | `material` | Material | TEXT | SAP material code (e.g. M5375052001) | Links to `ordering_trend.material`, `delivery_analysis.material_code`, `pr_to_po_conversion.material` |
| 10 | `short_text` | Short Text | TEXT | Description of the item/service being requested | — |
| 11 | `quantity_requested` | Quantity requested | DOUBLE PRECISION | Quantity of material requested | — |
| 12 | `unit_of_measure` | Unit of Measure | TEXT | Unit (NO = number, L = litre, AU = activity unit, KG = kg) | — |
| 13 | `total_value` | Total Value | DOUBLE PRECISION | Estimated total value of the PR line item (INR) | — |
| 14 | `purchase_order` | Purchase order | TEXT | PO number if PR has been partially converted (blank = still pending) | Links to `pr_to_po_conversion.purchase_order`, `ordering_trend.purchasing_document`, `delivery_analysis.purchase_order_no` |
| 15 | `purchase_order_date` | Purchase Order Date | TIMESTAMP | Date the PO was created (if converted) | — |
| 16 | `quantity_ordered` | Quantity ordered | DOUBLE PRECISION | Quantity actually ordered in the PO | — |
| 17 | `purchasing_group` | Purchasing Group | TEXT | SAP purchasing group code responsible for procurement | — |
| 18 | `requisitioner` | Requisitioner | TEXT | Name/ID of person who raised the PR | — |
| 19 | `material_group` | Material Group | TEXT | SAP material group/category code (e.g. 5375, 8272) | — |
| 20 | `release_indicator` | Release indicator | TEXT | Approval status (2 = released, 9 = blocked/pending) | — |
| 21 | `purchase_order_item` | Purchase order item | TEXT | Line item number in the PO | — |
| 22 | `procuring_plant` | Procuring Plant | TEXT | Plant that will actually procure the item | — |
| 23 | `purch_option_until` | Purch. option until | TIMESTAMP | Deadline for procurement option | — |
| 24 | `consumption` | Consumption | TEXT | Consumption indicator (V = variable) | — |
| 25 | `order_quantity` | Order Quantity | DOUBLE PRECISION | Final order quantity | — |
| 26 | `commitment_item_name` | Commitment Item Name | TEXT | Budget commitment item description (e.g. Store, Spares & Other material consumption) | — |
| 27 | `material_group_1` | Material Group *(2nd occurrence)* | TEXT | Duplicate material group column from SAP export | — |
| 28 | `commitment_item` | Commitment item | TEXT | SAP commitment item code (e.g. 94000001) | — |
| 29 | `deletion_indicator` | Deletion Indicator | TEXT | Whether the PR line is marked for deletion (false/true) | — |

---

## Table: `pr_to_po_conversion`
> Tracks how long each PR took to become a PO
> Source: `PR TO PO CONVERSION TIME.XLSX` | Rows: 16,245

| # | DB Column Name | Original Excel Header | Data Type | Business Meaning | Relationship |
|---|---|---|---|---|---|
| 1 | `document_type` | Document Type | TEXT | SAP document type (ZP13, ZP22 etc.) | — |
| 2 | `first_release_date` | First Release Date | TIMESTAMP | Date PR was first approved | — |
| 3 | `requisition_date` | Requisition Date | TIMESTAMP | Date PR was raised | — |
| 4 | `purchase_requisition` | Purchase Requisition | TEXT | PR number | **Join key** → `pending_pr.purchase_requisition`, `ordering_trend.purchase_requisition`, `delivery_analysis.purchase_requisition_no` |
| 5 | `acct_assignment_cat_` | Acct Assignment Cat. | TEXT | Account assignment category | — |
| 6 | `changed_on` | Changed On | TIMESTAMP | Last modified date | — |
| 7 | `item_of_requisition` | Item of requisition | TEXT | PR line item number | — |
| 8 | `material` | Material | TEXT | SAP material code | Links to `pending_pr.material`, `ordering_trend.material`, `delivery_analysis.material_code` |
| 9 | `short_text` | Short Text | TEXT | Item/service description | — |
| 10 | `quantity_requested` | Quantity requested | DOUBLE PRECISION | Quantity requested in PR | — |
| 11 | `unit_of_measure` | Unit of Measure | TEXT | Unit of measure | — |
| 12 | `total_value` | Total Value | DOUBLE PRECISION | Total estimated PR value (INR) | — |
| 13 | `purchase_order` | Purchase order | TEXT | PO number created from this PR | **Join key** → `ordering_trend.purchasing_document`, `delivery_analysis.purchase_order_no` |
| 14 | `purchase_order_date` | Purchase Order Date | TIMESTAMP | Date PO was created | — |
| 15 | `quantity_ordered` | Quantity ordered | DOUBLE PRECISION | Quantity in PO | — |
| 16 | `purchasing_group` | Purchasing Group | TEXT | Purchasing group code | — |
| 17 | `requisitioner` | Requisitioner | TEXT | Person who raised the PR | — |
| 18 | `material_group` | Material Group | TEXT | Material group code | — |
| 19 | `release_indicator` | Release indicator | TEXT | PR approval status | — |
| 20 | `purchase_order_item` | Purchase order item | TEXT | PO line item number | — |
| 21 | `procuring_plant` | Procuring Plant | TEXT | Plant handling procurement | — |
| 22 | `purch_option_until` | Purch. option until | TIMESTAMP | Procurement option deadline | — |
| 23 | `consumption` | Consumption | TEXT | Consumption type indicator | — |
| 24 | `order_quantity` | Order Quantity | DOUBLE PRECISION | Final ordered quantity | — |
| 25 | `commitment_item_name` | Commitment Item Name | TEXT | Budget line description | — |
| 26 | `material_group_1` | Material Group *(2nd occurrence)* | TEXT | Duplicate from SAP export | — |
| 27 | `commitment_item` | Commitment item | TEXT | SAP commitment item code | — |
| 28 | `deletion_indicator` | Deletion Indicator | TEXT | Deletion flag | — |
| 29 | `plant` | Plant | TEXT | Plant code | Links to all 4 tables on `plant` |

---

## Table: `ordering_trend`
> Monthly PO ordering trends — what was ordered, when, at what value
> Source: `ORDERING TREND MONTH WISE.XLSX` | Rows: 17,628

| # | DB Column Name | Original Excel Header | Data Type | Business Meaning | Relationship |
|---|---|---|---|---|---|
| 1 | `plant` | Plant | TEXT | Plant code | Links to all 4 tables on `plant` |
| 2 | `doc_type_descript_` | Doc. Type Descript. | TEXT | Human-readable PO type (e.g. O&M PO Ser - UNIT) | — |
| 3 | `purchasing_document` | Purchasing Document | TEXT | PO number | **Join key** → `pr_to_po_conversion.purchase_order`, `delivery_analysis.purchase_order_no` |
| 4 | `document_date` | Document Date | TIMESTAMP | Date the PO was created | — |
| 5 | `purchase_requisition` | Purchase Requisition | TEXT | Source PR number | **Join key** → `pending_pr.purchase_requisition`, `pr_to_po_conversion.purchase_requisition`, `delivery_analysis.purchase_requisition_no` |
| 6 | `changed_on` | Changed On | TIMESTAMP | Last modified date | — |
| 7 | `vendor_supplying_plant` | Vendor/supplying plant | TEXT | Vendor code + name (e.g. "303908 Nirmla Devi") | — |
| 8 | `item` | Item | TEXT | PO line item number | — |
| 9 | `material` | Material | TEXT | SAP material code (blank for services) | Links to `pending_pr.material`, `pr_to_po_conversion.material`, `delivery_analysis.material_code` |
| 10 | `short_text` | Short Text | TEXT | Item/service description | — |
| 11 | `order_unit` | Order Unit | TEXT | Unit of measure | — |
| 12 | `order_quantity` | Order Quantity | DOUBLE PRECISION | Quantity ordered | — |
| 13 | `still_to_be_delivered_qty_` | Still to be delivered (qty) | DOUBLE PRECISION | Outstanding delivery quantity | — |
| 14 | `net_price` | Net price | DOUBLE PRECISION | Unit net price (INR) | — |
| 15 | `net_order_value` | Net Order Value | DOUBLE PRECISION | Total PO value before tax (INR) | — |
| 16 | `net_value_tax` | Net Value + Tax | DOUBLE PRECISION | Total PO value including tax (INR) | — |
| 17 | `to_be_delivered_tax` | To be delivered + TAX | DOUBLE PRECISION | Pending delivery value including tax (INR) | — |
| 18 | `company_code` | Company Code | TEXT | SAP company code (e.g. 3000) | — |
| 19 | `commitment_item_name` | Commitment Item Name | TEXT | Budget line description | — |
| 20 | `creator_name` | Creator Name | TEXT | Name of person who created the PO | — |
| 21 | `deletion_indicator` | Deletion Indicator | TEXT | Deletion flag | — |
| 22 | `storage_location` | Storage Location | TEXT | Warehouse/storage location code | — |
| 23 | `delivery_date` | Delivery Date | TIMESTAMP | Expected delivery date on PO | — |

---

## Table: `delivery_analysis`
> Full delivery lifecycle — from PO to GRN to MIRO (invoice)
> Source: `Delivery Analysis report.XLSX` | Rows: 44,747

| # | DB Column Name | Original Excel Header | Data Type | Business Meaning | Relationship |
|---|---|---|---|---|---|
| 1 | `plant` | Plant | TEXT | Plant code | Links to all 4 tables on `plant` |
| 2 | `po_creator` | PO Creator | TEXT | Name of person who created the PO | — |
| 3 | `purchase_requisition_no` | Purchase Requisition No | TEXT | Source PR number | **Join key** → `pending_pr.purchase_requisition`, `pr_to_po_conversion.purchase_requisition`, `ordering_trend.purchase_requisition` |
| 4 | `purchase_requisition_item_no` | Purchase Requisition Item No | TEXT | PR line item number | — |
| 5 | `purchase_req_change_date` | Purchase Req Change Date | TIMESTAMP | Date PR was last changed | — |
| 6 | `purchase_order_no` | Purchase Order No | TEXT | PO number | **Join key** → `pr_to_po_conversion.purchase_order`, `ordering_trend.purchasing_document` |
| 7 | `document_date` | Document Date | TIMESTAMP | PO creation date | — |
| 8 | `delivery_date` | Delivery Date | TIMESTAMP | Promised delivery date on PO | — |
| 9 | `lr_rr_date` | LR/RR Date | TIMESTAMP | Lorry Receipt / Railway Receipt date (dispatch date from vendor) | — |
| 10 | `vendor_code` | Vendor Code | TEXT | SAP vendor code (e.g. 0000102949) | — |
| 11 | `vendor_name` | Vendor Name | TEXT | Vendor company name | — |
| 12 | `material_code` | Material Code | TEXT | SAP material code | **Join key** → `pending_pr.material`, `pr_to_po_conversion.material`, `ordering_trend.material` |
| 13 | `item_text` | Item Text | TEXT | Description of item | — |
| 14 | `order_quantity` | Order Quantity | DOUBLE PRECISION | Quantity ordered in PO | — |
| 15 | `igp_date` | IGP Date | TIMESTAMP | Inward Gate Pass date (when goods entered the gate) | — |
| 16 | `manual_gate_pass_no` | Manual Gate Pass No | TEXT | Manual gate pass reference number | — |
| 17 | `manual_gate_pass_date` | Manual Gate Pass Date | TIMESTAMP | Date of manual gate pass | — |
| 18 | `inbound_date` | Inbound Date | TIMESTAMP | Date goods were logged as inbound | — |
| 19 | `grn_date` | GRN Date | TIMESTAMP | Goods Receipt Note date (when goods were officially received) | — |
| 20 | `grn_entry_date` | GRN Entry Date | TIMESTAMP | Date GRN was entered in SAP | — |
| 21 | `miro_date` | MIRO Date | TIMESTAMP | Material Invoice Receipt date (when invoice was processed) | — |
| 22 | `miro_entry_date` | MIRO Entry Date | TIMESTAMP | Date MIRO was entered in SAP | — |
| 23 | `quality_date` | Quality Date | TIMESTAMP | Date quality inspection was completed | — |
| 24 | `u_o_m` | UoM | TEXT | Unit of measure | — |
| 25 | `unit_price` | Unit Price | DOUBLE PRECISION | Price per unit (INR) | — |
| 26 | `net_price` | Net Price | DOUBLE PRECISION | Total net price (INR) | — |
| 27 | `lr_rr_detail` | LR/RR Detail | TEXT | Lorry/Railway receipt reference number | — |
| 28 | `igp_no_` | IGP No. | TEXT | Inward Gate Pass number | — |
| 29 | `inbound_quantity` | Inbound Quantity | DOUBLE PRECISION | Quantity recorded at inbound | — |
| 30 | `grn_no_` | GRN No. | TEXT | GRN reference number | — |
| 31 | `bill_of_ladding` | Bill of Ladding | TEXT | Bill of lading reference (shipping document) | — |
| 32 | `delivery_note` | Delivery Note | TEXT | Vendor delivery note number | — |
| 33 | `grn_quantity` | GRN Quantity | DOUBLE PRECISION | Quantity received and accepted in GRN | — |
| 34 | `grn_value` | GRN Value | DOUBLE PRECISION | Value of goods received (INR) | — |
| 35 | `accepted_quantity` | Accepted Quantity | DOUBLE PRECISION | Quantity accepted after quality check | — |
| 36 | `rejected_quantity` | Rejected Quantity | DOUBLE PRECISION | Quantity rejected after quality check | — |
| 37 | `miro_no_` | MIRO No. | TEXT | Invoice reference number in SAP | — |
| 38 | `miro_quantity` | MIRO Quantity | DOUBLE PRECISION | Quantity invoiced | — |
| 39 | `miro_value` | MIRO Value | DOUBLE PRECISION | Invoice value processed (INR) | — |
| 40 | `ordering_plant` | Ordering Plant | TEXT | Plant that placed the order (may differ from receiving plant) | — |

---

## Cross-Table Join Keys Summary

| Join | Left Table . Column | Right Table . Column | Meaning |
|---|---|---|---|
| PR linkage | `pending_pr.purchase_requisition` | `pr_to_po_conversion.purchase_requisition` | Match open PRs to conversion records |
| PR linkage | `pending_pr.purchase_requisition` | `ordering_trend.purchase_requisition` | Match open PRs to POs placed |
| PR linkage | `pending_pr.purchase_requisition` | `delivery_analysis.purchase_requisition_no` | Match open PRs to delivery records |
| PO linkage | `pr_to_po_conversion.purchase_order` | `ordering_trend.purchasing_document` | Match conversion records to ordering data |
| PO linkage | `pr_to_po_conversion.purchase_order` | `delivery_analysis.purchase_order_no` | Match conversion records to delivery data |
| PO linkage | `ordering_trend.purchasing_document` | `delivery_analysis.purchase_order_no` | Match ordering trend to delivery data |
| Material | `pending_pr.material` | `delivery_analysis.material_code` | Track material across procurement lifecycle |
| Plant | `any_table.plant` | `any_table.plant` | Filter by plant/location |

---

## Key Business Metrics You Can Derive

| Metric | Tables Needed | Formula |
|---|---|---|
| PR to PO conversion time | `pr_to_po_conversion` | `purchase_order_date - requisition_date` |
| Delivery delay | `delivery_analysis` | `grn_date - delivery_date` |
| Invoice processing lag | `delivery_analysis` | `miro_date - grn_date` |
| Gate to GRN time | `delivery_analysis` | `grn_date - igp_date` |
| Pending PR value | `pending_pr` | `SUM(total_value) WHERE purchase_order IS NULL` |
| Rejection rate | `delivery_analysis` | `SUM(rejected_quantity) / SUM(order_quantity)` |
| Spend by vendor | `delivery_analysis` | `SUM(miro_value) GROUP BY vendor_name` |
| Monthly order value | `ordering_trend` | `SUM(net_order_value) GROUP BY DATE_TRUNC('month', document_date)` |
