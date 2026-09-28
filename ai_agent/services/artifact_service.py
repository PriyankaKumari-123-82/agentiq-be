"""Reusable service for generating and storing artifacts.

This module centralizes artifact creation so it can be invoked from both:

- The ``/artifacts/generate/{run_id}`` HTTP route (user-facing /
  programmatic generation).
- The agent runtime's built-in ``export_file`` tool (so the agent can
  produce a downloadable file as part of its plan).

Supported artifact types:
    csv, xlsx, json, txt, md, html, xml, pdf, docx,
    chart, chart_png, chart_svg, chart_html

Column filtering (external / MCP data only)
-------------------------------------------
When ``requires_query_building=False`` (i.e. the data came from an external
API via MCP, *not* from a DB query) and a ``user_query`` string is provided,
``create_artifact`` makes a lightweight LLM call via ``filter_columns_via_llm``
to select only the columns relevant to the user's question before building the
Excel / CSV file.  DB query results (``requires_query_building=True``) are
never filtered — every column the SQL selected is intentional.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_agent.core.llm_gateway import LLMGateway

import pandas as pd
from sqlmodel import Session

from ai_agent.core.aws_s3 import s3_client
from ai_agent.core.chart_service import (
    ChartConfig,
    ChartService,
    ChartType,
    OutputFormat,
)
from ai_agent.core.enums import ArtifactType
from ai_agent.core.generate_documents import document_generator
from ai_agent.core.response_schemas import ArtifactSummary
from ai_agent.core.response_schemas import BlockType
from ai_agent.models import AgentRun, Artifact
from ai_agent.helpers.excel_formatter import filter_columns_via_llm, filter_rows_to_columns, apply_cell_flattening, expand_list_columns
from ai_agent.helpers.date_formatter import transform_dates_in_obj

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIME_TYPES: Dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "json": "application/json",
    "txt": "text/plain",
    "md": "text/markdown",
    "html": "text/html",
    "xml": "application/xml",
    "chart_json": "application/vnd.agentiq.chart+json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "svg": "image/svg+xml",
    "webp": "image/webp",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "mp4": "video/mp4",
}

SUPPORTED_ARTIFACT_TYPES: List[str] = [
    ArtifactType.CSV.value,
    ArtifactType.XLSX.value,
    ArtifactType.JSON.value,
    ArtifactType.TXT.value,
    ArtifactType.MD.value,
    ArtifactType.HTML.value,
    ArtifactType.XML.value,
    ArtifactType.PDF.value,
    ArtifactType.DOCX.value,
    ArtifactType.CHART.value,
    ArtifactType.CHART_PNG.value,
    ArtifactType.CHART_SVG.value,
    ArtifactType.CHART_HTML.value,
]


def get_mime_type(artifact_type: str) -> str:
    """Return the MIME type for an artifact type, falling back to octet-stream."""
    return _MIME_TYPES.get(artifact_type.lower(), "application/octet-stream")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _json_safe_dict(value: Any) -> Dict[str, Any]:
    safe_value = _json_safe(value)
    if isinstance(safe_value, dict):
        return json.loads(json.dumps(safe_value, default=str))
    return {"value": safe_value}


def bind_artifacts_to_message(
    db_session: Session,
    *,
    artifacts: List[ArtifactSummary],
    message_id: int,
) -> None:
    """Attach this run's ``export_file`` artifacts to the assistant's reply
    message. They're created mid-run (before the assistant message exists)
    against the user's input message as a placeholder, so once the assistant
    message is committed the chat UI needs them rebound here to render on
    the correct bubble."""
    if not artifacts:
        return
    for artifact_summary in artifacts:
        artifact = db_session.get(Artifact, artifact_summary.id)
        if artifact:
            artifact.message_id = message_id
    db_session.commit()


def artifact_payload(artifact: Artifact) -> Dict[str, Any]:
    """Serialize an Artifact row for chat/history responses."""
    return {
        "id": artifact.id,
        "artifact_type": artifact.artifact_type,
        "mime_type": artifact.mime_type,
        "filename": artifact.filename,
        "size_bytes": artifact.size_bytes,
        "description": artifact.description,
        "extra_metadata": artifact.extra_metadata,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }


def create_chart_metadata_artifact(
    db_session: Session,
    *,
    run_id: str,
    user_id: int,
    agent_id: Optional[str],
    session_id: Optional[int],
    conversation_id: Optional[str],
    message_id: int,
    chart: Any,
) -> Artifact:
    """Store inline chart JSON in artifacts.metadata for chat reload."""
    chart_payload = _json_safe_dict(chart)
    artifact_id = str(uuid.uuid4())
    metadata = {
        "kind": "chart",
        "schema_version": 1,
        "chart": chart_payload,
    }
    artifact = Artifact(
        id=artifact_id,
        run_id=run_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        conversation_id=conversation_id,
        message_id=message_id,
        artifact_type=ArtifactType.CHART_JSON.value,
        mime_type=get_mime_type(ArtifactType.CHART_JSON.value),
        filename=None,
        storage_key=None,
        size_bytes=len(json.dumps(metadata, ensure_ascii=False, default=str).encode("utf-8")),
        extra_metadata=metadata,
        description=chart_payload.get("title") or "Chart",
        expires_at=None,
    )
    db_session.add(artifact)
    db_session.commit()
    db_session.refresh(artifact)
    return artifact


def persist_chart_artifacts_from_blocks(
    db_session: Session,
    *,
    blocks: List[Any],
    run_id: str,
    user_id: int,
    agent_id: Optional[str],
    session_id: Optional[int],
    conversation_id: Optional[str],
    message_id: int,
) -> List[Artifact]:
    """Persist all chart content blocks as metadata-only artifacts."""
    artifacts: List[Artifact] = []
    for block in blocks or []:
        if getattr(block, "type", None) != BlockType.CHART:
            continue
        artifacts.append(
            create_chart_metadata_artifact(
                db_session,
                run_id=run_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                conversation_id=conversation_id,
                message_id=message_id,
                chart=block.content,
            )
        )
    return artifacts


# ---------------------------------------------------------------------------
# Generators (pure: data -> (bytes, default_filename))
# ---------------------------------------------------------------------------

def _coerce_rows(data: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of a list-of-dicts from agent-provided data."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            return [r for r in data["rows"] if isinstance(r, dict)]
        if isinstance(data.get("value"), list):
            return [r for r in data["value"] if isinstance(r, dict)]
        if isinstance(data.get("data"), list):
            return [r for r in data["data"] if isinstance(r, dict)]
        return [data] if data else []
    return []


def _coerce_text(data: Any) -> str:
    """Extract text content from agent-provided data for txt/md/html."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("content", "text", "markdown", "body"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# Common acronyms kept uppercase when humanizing raw DB/API column names
# (e.g. "job_id" -> "Job ID", not "Job Id"). Mirrors the same convention the
# chat synthesis prompt uses for markdown tables (see agent_runtime.py's
# column_display_hint) so exported files and in-chat previews read the same way.
_HEADER_ACRONYMS = {"id", "url", "api", "sql", "uuid", "ip"}


def _humanize_header(key: str) -> str:
    """Convert a raw column name (snake_case, camelCase, PascalCase, or
    kebab-case) into a human-readable header, e.g. "job_no" -> "Job No",
    "openDate" -> "Open Date". Falls back to the original key if it doesn't
    look like an identifier."""
    if not isinstance(key, str) or not key:
        return key
    # Split camelCase/PascalCase boundaries, then snake/kebab-case separators.
    spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', key)
    spaced = spaced.replace('_', ' ').replace('-', ' ')
    words = [w for w in spaced.split(' ') if w]
    if not words:
        return key
    return ' '.join(
        w.upper() if w.lower() in _HEADER_ACRONYMS else w.capitalize()
        for w in words
    )


def _humanize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {_humanize_header(k): v for k, v in row.items()}
        for row in rows
    ]


def _generate_csv(data: Any) -> bytes:
    rows = _humanize_rows(_coerce_rows(data))
    if not rows:
        raise ValueError("CSV generation requires at least one row.")
    df = pd.DataFrame(rows)
    df = df.where(pd.notna(df), "Not Available")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _generate_xlsx(data: Any, sheet_title: str = "Sheet1") -> bytes:
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    rows = _humanize_rows(_coerce_rows(data))
    if not rows:
        raise ValueError("XLSX generation requires at least one row.")
    df = pd.DataFrame(rows)
    df = df.where(pd.notna(df), "Not Available")
    buf = io.BytesIO()

    sheet_name = sheet_title[:31] if sheet_title else "Sheet1"

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]

        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True)
        center_all  = Alignment(horizontal="center", vertical="center")
        thin_border = Border(bottom=Side(style="thin"))

        ws.freeze_panes = "A2"

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_all
            cell.border = thin_border

        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=0)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 50)

    return buf.getvalue()


def _generate_json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _generate_pdf(data: Any, default_filename: str) -> bytes:
    """Generate a PDF from markdown content via Pandoc."""
    markdown = _coerce_text(data)
    if not markdown.strip():
        raise ValueError("PDF generation requires non-empty 'content' (markdown).")

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, default_filename)
        result = document_generator.generate_from_markdown(
            markdown_text=markdown,
            output_path=output_path,
        )
        if not result or not os.path.exists(result):
            raise RuntimeError("Pandoc failed to produce a PDF file.")
        with open(result, "rb") as fh:
            return fh.read()


def _generate_docx(data: Any, default_filename: str) -> bytes:
    """Generate a DOCX from markdown content via Pandoc."""
    markdown = _coerce_text(data)
    if not markdown.strip():
        raise ValueError("DOCX generation requires non-empty 'content' (markdown).")

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, default_filename)
        result = document_generator.generate_from_markdown(
            markdown_text=markdown,
            output_path=output_path,
        )
        if not result or not os.path.exists(result):
            raise RuntimeError("Pandoc failed to produce a DOCX file.")
        with open(result, "rb") as fh:
            return fh.read()


def _generate_chart(artifact_type: str, data: Any) -> bytes:
    """Generate a chart (PNG/SVG/HTML) via ChartService."""
    payload = data if isinstance(data, dict) else {}
    chart_type_str = payload.get("chart_type", "bar")
    try:
        chart_type = ChartType(chart_type_str)
    except ValueError:
        chart_type = ChartType.BAR

    config = ChartConfig(
        chart_type=chart_type,
        title=payload.get("title", "Chart"),
        x_column=payload.get("x_column"),
        y_column=payload.get("y_column"),
        width=payload.get("width", 800),
        height=payload.get("height", 400),
        theme=payload.get("theme", "light"),
        data=payload.get("data", []),
    )

    chart_service = ChartService()
    if artifact_type == ArtifactType.CHART_SVG.value:
        content = chart_service.generate_chart(config, OutputFormat.SVG)
    elif artifact_type == ArtifactType.CHART_HTML.value:
        content = chart_service.generate_chart(config, OutputFormat.HTML)
    else:  # chart or chart_png
        content = chart_service.generate_chart(config, OutputFormat.PNG)
    if isinstance(content, str):
        return content.encode("utf-8")
    return content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_artifact(
    db_session: Session,
    *,
    run_id: str,
    user_id: int,
    agent_id: Optional[str] = None,
    session_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    message_id: Optional[int] = None,
    artifact_type: str,
    data: Any,
    filename: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    presigned_expiration: int = 3600,
    storage_lifetime_days: int = 30,
    # ── LLM-powered column filtering (external/MCP sources only) ────────────
    requires_query_building: bool = False,
    user_query: str = "",
    llm: Optional["LLMGateway"] = None,
    litellm_model_string: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> ArtifactSummary:
    """Generate an artifact, store it in S3, persist its DB record, and return
    a summary including a fresh presigned download URL.

    Parameters
    ----------
    requires_query_building:
        Pass ``True`` when the data originates from a database query (via an
        MCP server with ``requires_query_building=True``).  In that case the
        LLM column-filtering step is skipped entirely so no DB columns are
        dropped.  Defaults to ``False`` (i.e. external/API/MCP data).
    user_query:
        The user's original question.  Only used when
        ``requires_query_building=False`` to feed context to the column-
        selection LLM call.  Safe to omit — filtering is skipped when empty.
    llm:
        An ``LLMGateway`` instance.  Required for column filtering; when
        ``None`` (or when ``requires_query_building=True``) filtering is
        silently skipped.
    litellm_model_string:
        LiteLLM model string forwarded to the column-selection LLM call.
    llm_api_key:
        API key for the column-selection LLM call.

    Raises ``ValueError`` for unsupported types or invalid input.
    """
    artifact_type_norm = artifact_type.lower().strip()
    if artifact_type_norm not in SUPPORTED_ARTIFACT_TYPES:
        raise ValueError(
            f"Unsupported artifact type: {artifact_type}. "
            f"Supported: {', '.join(SUPPORTED_ARTIFACT_TYPES)}"
        )

    # Resolve agent_id from the run if not provided (HTTP callers usually
    # don't pass it; runtime callers do).
    if not agent_id or not session_id or not conversation_id:
        run = db_session.get(AgentRun, run_id)
        if not run:
            raise ValueError(f"AgentRun {run_id} not found")
        agent_id = agent_id or run.agent_id
        session_id = session_id or run.session_id
        conversation_id = conversation_id or run.conversation_id

    artifact_id = str(uuid.uuid4())
    default_extension_map = {
        ArtifactType.CHART.value: "png",
        ArtifactType.CHART_PNG.value: "png",
        ArtifactType.CHART_SVG.value: "svg",
        ArtifactType.CHART_HTML.value: ArtifactType.HTML.value,
    }
    default_ext = default_extension_map.get(artifact_type_norm, artifact_type_norm)
    safe_filename = filename or f"export_{run_id}.{default_ext}"

    # ── 1. List expansion — before flattening (all sources) ────────────────────
    # Must run FIRST so list-of-dicts (Attendees, Categories, etc.) get split
    # into numbered columns BEFORE the flattener joins them into one big string.
    # expand_list_columns internally flattens each list item via flatten_cell_value.
    if artifact_type_norm in (ArtifactType.CSV.value, ArtifactType.XLSX.value):
        try:
            list_rows = _coerce_rows(data)
            if list_rows:
                data = expand_list_columns(list_rows)
                log.info("create_artifact: list column expansion applied to %d rows", len(list_rows))
        except Exception as exc:
            log.warning("create_artifact: list column expansion failed (%s); using raw data", exc)

    # ── 2. Flatten remaining nested dicts to plain scalars (all sources) ────────
    # Runs after list expansion so remaining non-list dicts (User, Pull Request,
    # Location, emailAddress, etc.) become clean human-readable strings.
    if artifact_type_norm in (ArtifactType.CSV.value, ArtifactType.XLSX.value):
        try:
            flat_rows = _coerce_rows(data)
            if flat_rows:
                data = apply_cell_flattening(flat_rows)
                log.info("create_artifact: cell flattening applied to %d rows", len(flat_rows))
        except Exception as exc:
            log.warning("create_artifact: cell flattening failed (%s); using raw data", exc)

    # ── 3. LLM column filter — only for MCP/external data ─────────────────────
    # Runs on fully opened flat data so LLM sees:
    #   - "Attendee 1", "Attendee 2", ... (not raw Attendees list)
    #   - "pk-pb" for User (not a dict)
    # DB query results (requires_query_building=True) are NEVER filtered.
    if (
        artifact_type_norm in (ArtifactType.CSV.value, ArtifactType.XLSX.value)
        and not requires_query_building
        and user_query
        and llm is not None
    ):
        flat_rows = _coerce_rows(data)
        if flat_rows:
            try:
                columns_to_keep = await filter_columns_via_llm(
                    rows=flat_rows,
                    user_query=user_query,
                    llm=llm,
                    litellm_model_string=litellm_model_string,
                    api_key=llm_api_key,
                )
                data = filter_rows_to_columns(flat_rows, columns_to_keep)
                log.info(
                    "create_artifact: LLM column filtering applied — "
                    "%d columns kept from %d available",
                    len(columns_to_keep),
                    len(flat_rows[0]) if flat_rows else 0,
                )
            except Exception as exc:
                log.warning("create_artifact: column filtering failed (%s); proceeding with all columns", exc)

    # ── 4. IST date conversion on final columns ────────────────────────────────
    # Runs last so we only convert dates in the columns we're actually keeping.
    if artifact_type_norm in (ArtifactType.CSV.value, ArtifactType.XLSX.value):
        try:
            converted_rows = _coerce_rows(data)
            if converted_rows:
                data = transform_dates_in_obj(converted_rows)
                log.info("create_artifact: UTC → IST date conversion applied to %d rows", len(converted_rows))
        except Exception as exc:
            log.warning("create_artifact: date conversion failed (%s); using raw data", exc)

    # Generate bytes
    try:
        if artifact_type_norm == ArtifactType.CSV.value:
            content = _generate_csv(data)
        elif artifact_type_norm == ArtifactType.XLSX.value:
            sheet_title = safe_filename.rsplit(".", 1)[0].replace("_", " ").title()
            content = _generate_xlsx(data, sheet_title=sheet_title)
        elif artifact_type_norm == ArtifactType.JSON.value:
            content = _generate_json(data)
        elif artifact_type_norm in (
            ArtifactType.TXT.value,
            ArtifactType.MD.value,
            ArtifactType.HTML.value,
            ArtifactType.XML.value,
        ):
            content = _coerce_text(data).encode("utf-8")
        elif artifact_type_norm == ArtifactType.PDF.value:
            content = _generate_pdf(data, safe_filename)
        elif artifact_type_norm == ArtifactType.DOCX.value:
            content = _generate_docx(data, safe_filename)
        elif artifact_type_norm in (
            ArtifactType.CHART.value,
            ArtifactType.CHART_PNG.value,
            ArtifactType.CHART_SVG.value,
            ArtifactType.CHART_HTML.value,
        ):
            content = _generate_chart(artifact_type_norm, data)
        else:
            # Should not reach here because of the early check above.
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
    except ValueError:
        raise
    except Exception as e:
        log.exception("Artifact generation failed for type=%s", artifact_type_norm)
        raise RuntimeError(f"Failed to generate {artifact_type_norm}: {e}") from e

    mime_type = get_mime_type(default_ext)
    storage_key = f"artifacts/{user_id}/{run_id}/{artifact_id}/{safe_filename}"

    try:
        await s3_client.upload_fileobj(
            file_obj=io.BytesIO(content),
            object_name=storage_key,
            content_type=mime_type,
        )
    except Exception as e:
        log.exception("S3 upload failed for artifact %s", artifact_id)
        raise RuntimeError(f"Failed to store artifact: {e}") from e

    artifact = Artifact(
        id=artifact_id,
        run_id=run_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        conversation_id=conversation_id,
        message_id=message_id,
        artifact_type=artifact_type_norm,
        mime_type=mime_type,
        filename=safe_filename,
        storage_key=storage_key,
        size_bytes=len(content),
        extra_metadata=metadata or {},
        description=description,
        expires_at=datetime.now(timezone.utc) + timedelta(days=storage_lifetime_days),
    )
    db_session.add(artifact)
    db_session.commit()
    db_session.refresh(artifact)

    try:
        download_url = s3_client.generate_presigned_url(
            object_name=storage_key,
            expiration=presigned_expiration,
        )
    except Exception as e:
        log.warning("Failed to generate presigned URL for %s: %s", artifact_id, e)
        download_url = None

    return ArtifactSummary(
        id=artifact_id,
        artifact_type=artifact_type_norm,
        mime_type=mime_type,
        filename=safe_filename,
        size_bytes=len(content),
        description=description,
        download_url=download_url,
    )
