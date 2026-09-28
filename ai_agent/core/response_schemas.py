"""Response schemas for block-based agent outputs.

This module defines the structured response format for Agent IQ,
supporting multiple content types: markdown, tables, artifacts,
citations, code blocks, and more.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Types of content blocks in agent responses."""
    MARKDOWN = "markdown"
    TABLE = "table"
    ARTIFACT = "artifact"
    CITATION = "citation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CODE = "code"
    REASONING = "reasoning"
    CHART = "chart"


class TableColumn(BaseModel):
    """Definition of a table column."""
    name: str = Field(description="Column header name")
    type: str = Field(default="string", description="Data type: string/number/date/boolean")
    format: Optional[str] = Field(default=None, description="Format string for display")


class TableBlock(BaseModel):
    """Structured table data block."""
    columns: List[TableColumn] = Field(description="Column definitions")
    rows: List[List[Any]] = Field(description="Row data as arrays matching column order")
    title: Optional[str] = Field(default=None, description="Table title")
    source: Optional[str] = Field(default=None, description="Source of the data")
    download_formats: List[str] = Field(default=["csv", "json"], description="Available download formats")


class ChartDataset(BaseModel):
    """A single dataset (series) inside a chart."""
    label: str = Field(description="Series label shown in legend")
    data: List[Any] = Field(description="Numeric values aligned with ChartBlock.labels")
    color: Optional[str] = Field(default=None, description="Optional hex/CSS color hint for frontend")


class ChartBlock(BaseModel):
    """Structured chart data block.

    The frontend receives this as a ContentBlock with type=CHART and renders
    it using its preferred charting library (Recharts, etc.).
    Labels + datasets follow the same convention as Chart.js so the frontend
    can pass the payload through with minimal transformation.
    """
    chart_type: str = Field(
        description="Chart variety: bar | line | pie | doughnut | scatter | area"
    )
    title: str = Field(description="Chart title shown above the visualization")
    labels: List[str] = Field(
        description="X-axis tick labels (or slice labels for pie/doughnut)"
    )
    datasets: List[ChartDataset] = Field(
        description="One entry per data series"
    )
    x_label: Optional[str] = Field(default=None, description="X-axis label")
    y_label: Optional[str] = Field(default=None, description="Y-axis label")

# ─────────────────────────────────────────────────────────────────────────────


class ArtifactSummary(BaseModel):
    """Summary of an artifact (file output)."""
    id: str = Field(description="Artifact ID")
    artifact_type: str = Field(description="Type: pdf/csv/xlsx/json/image/audio/etc")
    mime_type: str = Field(description="MIME type")
    filename: Optional[str] = Field(default=None, description="Original filename")
    size_bytes: int = Field(description="File size in bytes")
    description: Optional[str] = Field(default=None)
    download_url: Optional[str] = Field(default=None, description="Presigned download URL")
    companion_excel: Optional["ArtifactSummary"] = Field(default=None, description="Auto-generated Excel export of the chart data")


ArtifactSummary.model_rebuild()


class ArtifactBlock(BaseModel):
    """Artifact reference block."""
    artifact: ArtifactSummary = Field(description="Artifact summary")
    display_mode: str = Field(default="link", description="Display: link/preview/embed")


class CitationBlock(BaseModel):
    """Citation/reference block for RAG sources."""
    citation_id: str = Field(description="Unique citation ID")
    source_type: str = Field(description="Type: document/web/tool/database")
    source_id: Optional[str] = Field(default=None, description="Source document/tool ID")
    source_name: str = Field(description="Human-readable source name")
    content: str = Field(description="Quoted content")
    page: Optional[int] = Field(default=None)
    section: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ToolCallBlock(BaseModel):
    """Tool call execution block."""
    tool_name: str = Field(description="Name of the tool called")
    tool_id: str = Field(description="Unique tool call ID")
    arguments: Dict[str, Any] = Field(description="Tool arguments")
    original_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments exactly as the selector produced them, before "
                    "any runtime mutation (e.g. the SQL guardrail rewriting the "
                    "query). Used for tool-loop dedup so a repeated selection "
                    "still matches the stored call.",
    )
    status: str = Field(default="pending", description="pending/running/completed/failed")
    latency_ms: Optional[int] = Field(default=None)


class ToolResultBlock(BaseModel):
    """Tool execution result block."""
    tool_call_id: str = Field(description="Reference to tool call")
    status: str = Field(description="success/failure")
    result: Any = Field(description="Tool output data")
    error: Optional[str] = Field(default=None)
    output_type: str = Field(default="text", description="text/json/table/binary")


class CodeBlock(BaseModel):
    """Code snippet block."""
    language: str = Field(description="Programming language")
    code: str = Field(description="Code content")
    filename: Optional[str] = Field(default=None)
    executable: bool = Field(default=False, description="Whether code can be executed")


class ReasoningBlock(BaseModel):
    """Agent reasoning/thinking block."""
    reasoning: str = Field(description="Reasoning content")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    alternatives_considered: Optional[List[str]] = Field(default=None)


class ContentBlock(BaseModel):
    """Generic content block that can hold any block type."""
    type: BlockType = Field(description="Type of content block")
    content: Union[
        str, TableBlock, ChartBlock, ArtifactBlock, CitationBlock,
        ToolCallBlock, ToolResultBlock, CodeBlock, ReasoningBlock,
        Dict[str, Any],
    ] = Field(description="Block content")


class UsageInfo(BaseModel):
    """Token and cost usage information."""
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cached_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    audio_input_seconds: float = Field(default=0.0)
    audio_output_seconds: float = Field(default=0.0)


class ResponseMetadata(BaseModel):
    """Metadata about the agent response."""
    run_id: str = Field(description="Agent run ID")
    message_id: int = Field(description="Message ID in database")
    session_id: str = Field(description="Session ID")
    conversation_id: str = Field(description="Conversation ID")
    agent_id: str = Field(description="Agent ID")
    provider: str = Field(description="LLM provider used")
    model: str = Field(description="LLM model used")
    latency_ms: int = Field(description="Total response latency")
    usage: UsageInfo = Field(description="Token/cost usage")
    citations: List[CitationBlock] = Field(default=[], description="RAG citations")


class AgentResponse(BaseModel):
    """Complete agent response with block-based content."""
    message_id: int = Field(description="Database message ID")
    run_id: str = Field(description="Agent run ID")
    session_id: str = Field(description="User session ID")
    conversation_id: str = Field(description="Conversation ID")
    
    # Response content as blocks
    blocks: List[ContentBlock] = Field(description="Content blocks")
    
    # Artifacts generated
    artifacts: List[ArtifactSummary] = Field(default=[], description="Generated artifacts")
    
    # Explainability
    citations: List[CitationBlock] = Field(default=[], description="RAG citations")
    reasoning: Optional[ReasoningBlock] = Field(default=None, description="Agent reasoning")
    
    # Tools used
    tool_calls: List[ToolCallBlock] = Field(default=[], description="Tool calls made")
    tool_results: List[ToolResultBlock] = Field(default=[], description="Tool results")
    
    # Usage and metadata
    usage: UsageInfo = Field(description="Usage statistics")
    metadata: ResponseMetadata = Field(description="Response metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Streaming support
    is_complete: bool = Field(default=True, description="Whether response is complete")
    final_block_index: Optional[int] = Field(default=None, description="Last block index for streaming")
    
    # Follow-up FAQ suggestions
    follow_up_faqs: List[str] = Field(default_factory=list, description="Follow-up FAQ suggestions")


class StreamingEvent(BaseModel):
    """Event for SSE streaming responses."""
    event_type: str = Field(description="Event type: run.started/message.delta/tool_call.started/etc")
    run_id: str = Field(description="Run ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(description="Event-specific data")


def _format_cell(value: Any) -> str:
    """Render a single table cell as a markdown-safe single-line string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        try:
            import json as _json
            text = _json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    # Escape pipes and collapse newlines so the markdown table stays well-formed.
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def table_block_to_markdown(table: "TableBlock") -> str:
    """Render a TableBlock as a GitHub-flavored markdown table."""
    if not table.columns:
        return ""
    headers = [c.name for c in table.columns]
    header_row = "| " + " | ".join(_format_cell(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_format_cell(cell) for cell in row) + " |"
        for row in table.rows
    ]
    parts: List[str] = []
    if table.title:
        parts.append(f"**{table.title}**")
        parts.append("")
    parts.append(header_row)
    parts.append(separator)
    parts.extend(body_rows)
    return "\n".join(parts)


def content_block_to_markdown(block: "ContentBlock") -> str:
    """Render any ContentBlock as a markdown string suitable for the chat UI.

    The agent runtime emits structured blocks (TableBlock, CodeBlock, etc.).
    The chat UI renders markdown, so non-markdown blocks must be converted
    here rather than passed through ``str(block.content)`` (which produces a
    Pydantic repr and breaks rendering).
    """
    content = block.content
    btype = block.type

    if isinstance(content, str):
        return content

    if btype == BlockType.TABLE and isinstance(content, TableBlock):
        return table_block_to_markdown(content)

    if btype == BlockType.CODE and isinstance(content, CodeBlock):
        lang = content.language or ""
        return f"```{lang}\n{content.code}\n```"

    if btype == BlockType.REASONING and isinstance(content, ReasoningBlock):
        return f"_Reasoning:_ {content.reasoning}"

    if btype == BlockType.CITATION and isinstance(content, CitationBlock):
        label = content.source_name or content.source_id or "source"
        if content.url:
            return f"[{label}]({content.url}): {content.content}"
        return f"**{label}**: {content.content}"

    # ── CHART: representation 
    if btype == BlockType.CHART:
        return ""

    if btype == BlockType.ARTIFACT:
        # Artifact content may be an ArtifactBlock model, an ArtifactSummary
        # model, or a flat dict (the shape the agent runtime emits today).
        artifact_dict: Dict[str, Any] = {}
        if hasattr(content, "model_dump"):
            artifact_dict = content.model_dump()
            # Unwrap ArtifactBlock -> ArtifactSummary if needed.
            if "artifact" in artifact_dict and isinstance(artifact_dict["artifact"], dict):
                artifact_dict = artifact_dict["artifact"]
        elif isinstance(content, dict):
            artifact_dict = content
        filename = artifact_dict.get("filename") or artifact_dict.get("name") or "file"
        url = artifact_dict.get("download_url")
        artifact_type = (artifact_dict.get("artifact_type") or "").upper()
        size = artifact_dict.get("size_bytes")
        size_str = ""
        if isinstance(size, (int, float)) and size > 0:
            if size < 1024:
                size_str = f" ({int(size)} B)"
            elif size < 1024 * 1024:
                size_str = f" ({size / 1024:.1f} KB)"
            else:
                size_str = f" ({size / (1024 * 1024):.1f} MB)"
        label_parts = [filename]
        if artifact_type:
            label_parts.append(artifact_type)
        label = " — ".join(label_parts) + size_str
        if url:
            return f"[\U0001F4E5 Download {label}]({url})"
        return f"**\U0001F4E5 Attachment:** {label}"

    # Fallback: pretty-print structured content as JSON inside a code fence
    # so it at least renders cleanly instead of leaking Python repr.
    try:
        import json as _json
        if hasattr(content, "model_dump"):
            payload = content.model_dump()
        elif isinstance(content, dict):
            payload = content
        else:
            payload = {"value": str(content)}
        return f"```json\n{_json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```"
    except Exception:
        return str(content)


# Legacy compatibility wrapper for plain text responses
class LegacyResponseModel(BaseModel):
    """Backwards-compatible simple response model."""
    message_id: int
    response: str
    session_id: str
    conversation_id: str
