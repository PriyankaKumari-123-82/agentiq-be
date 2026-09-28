"""Shared enum definitions."""
from enum import Enum


class ActionType(str, Enum):
    """Allowed actions in the activity log."""
    CREATE = "created"
    UPDATE = "updated"
    DELETE = "deleted"
    UPLOAD = "uploaded"
    START = "started"
    COMPLETE = "completed"


class EntityType(str, Enum):
    """Allowed entity types in the activity log."""
    DOCUMENT = "document"
    AGENT = "agent"
    API_KEY = "api_key"
    USER = "user"
    SETTINGS = "settings"
    CONVERSATION = "conversation"
    LLM_SETTINGS = "llm_settings"
    VECTOR_DB_SETTINGS = "vector_db_settings"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    PHONE = "phone"
    ADDRESS = "address"


class ArtifactType(str, Enum):
    """Types of artifacts that can be generated or referenced."""
    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    XML = "xml"
    DOCX = "docx"
    CHART = "chart"
    CHART_JSON = "chart_json"
    CHART_PNG = "chart_png"
    CHART_SVG = "chart_svg"
    CHART_HTML = "chart_html"
    PNG = "png"
    SVG = "svg"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    MARKDOWN = "markdown"
    ARTIFACT = "artifact"
    CODE = "code"
    CITATION = "citation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"


class LLMOperation(str, Enum):
    """Standardized names for tracked LLM operations."""
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    SYNTHESIS = "synthesis"
    EMBEDDING = "embedding"


class SourceType(str, Enum):
    """Supported evidence-source categories for attribution and citations."""

    DOCUMENT = "document"
    WEB = "web"
    MCP = "mcp"
