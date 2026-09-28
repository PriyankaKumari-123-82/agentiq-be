import logging
import uuid as uuid_mod
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, List, Dict, Any, ClassVar
from sqlmodel import Field, SQLModel, Relationship, Column, String, ForeignKey, Integer, JSON
from sqlalchemy import UniqueConstraint, CheckConstraint
import sqlalchemy as sa

from ai_agent.core.security import get_encryption_manager
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Text


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"



class ConversationRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class ConversationType(str, Enum):
    TEXT = "text"
    VOICE = "voice"


class DeliveryMethod(str, Enum):
    DOWNLOAD = "download"
    EMAIL = "email"


class Organization(SQLModel, table=True):
    """An organization that groups users together."""
    __tablename__ = "organizations"

    id: int | None = Field(default=None, primary_key=True, index=True)
    name: str = Field(sa_column=sa.Column(sa.String(), nullable=False), description="Organization display name")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    pluggy_tenant_id: str | None = Field(default=None, description="Pluggy tenant UUID linked to this organization")
    is_active: bool = Field(default=True, index=True, sa_column_kwargs={"server_default": "true"})

    memberships: List["OrganizationMembership"] = Relationship(back_populates="organization")
    document_org_links: List["OrganizationDocument"] = Relationship(back_populates="organization", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    agents: List["AIAgent"] = Relationship(back_populates="organization")
    settings: Optional["OrganizationSettings"] = Relationship(back_populates="organization")


class OrganizationSettings(SQLModel, table=True):
    """Organization-level configuration. Currently holds the embedding model
    shared by every agent in the organization for both indexing and retrieval."""
    __tablename__ = "organization_settings"

    id: int | None = Field(default=None, primary_key=True)
    organization_id: int = Field(foreign_key="organizations.id", unique=True, index=True)
    embedding_model: str = Field(description="Embedding model used for RAG indexing/retrieval across the organization")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    organization: "Organization" = Relationship(back_populates="settings")


class VectorCollection(SQLModel, table=True):
    __tablename__ = "vector_collection"

    id: int | None = Field(default=None, primary_key=True)
    organization_id: int = Field(
        foreign_key="organizations.id", unique=True, index=True, nullable=False,
    )
    collection_name: str = Field(
        default_factory=lambda: f"org_{uuid_mod.uuid4().hex}",
        sa_column=sa.Column(sa.String(255), nullable=False, unique=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


class Document(SQLModel, table=True):
    """Stores metadata about uploaded documents.

    Organization and agent associations are managed exclusively via the
    OrganizationDocument and AgentDocument junction tables.
    A single Document row represents one physical file regardless of how many
    organizations it belongs to.
    """
    __tablename__ = "documents"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    file_name: str = Field(description="Original name of the uploaded file")
    description: str | None = Field(default=None, description="Optional description of the document")
    file_path: str = Field(description="Path where the file is stored in MinIO")
    file_size: int = Field(description="Size of the file in bytes")
    mime_type: str = Field(description="MIME type of the file")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    deleted_at: datetime | None = Field(
        default=None,
        index=True,
        description="Timestamp when the document was soft-deleted, null if not deleted"
    )
    is_active: bool = Field(default=True, index=True, sa_column_kwargs={"server_default": "true"})

    # Relationships
    user: "User" = Relationship(back_populates="documents")
    org_links: List["OrganizationDocument"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    agent_links: List["AgentDocument"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    session_links: List["SessionDocument"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class OrganizationDocument(SQLModel, table=True):
    """Links a document to an organization (many-to-many).

    Each row means:
    - The document is embedded in the organization's Milvus collection (``org_{org_id}``)
    - Users/agents of that org can potentially access this document
    """
    __tablename__ = "organization_documents"
    __table_args__ = (
        UniqueConstraint("document_id", "organization_id", name="uq_org_doc_link"),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True, nullable=False)
    organization_id: int = Field(foreign_key="organizations.id", index=True, nullable=False)
    user_id: int | None = Field(default=None, foreign_key="users.id", index=True,
                                description="User who created this org link")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = Field(default=True, index=True, sa_column_kwargs={"server_default": "true"})

    document: "Document" = Relationship(back_populates="org_links")
    organization: "Organization" = Relationship(back_populates="document_org_links")


class AgentDocument(SQLModel, table=True):
    """Links a document to an agent (many-to-many).

    Each row means the agent is allowed to retrieve chunks from this document
    during RAG. Org-level isolation is enforced at query time by searching
    only the org's Milvus collection (``org_{org_id}``).
    """
    __tablename__ = "agent_documents"
    __table_args__ = (
        UniqueConstraint("document_id", "agent_id", name="uq_agent_doc_link"),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True, nullable=False)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)
    user_id: int | None = Field(default=None, foreign_key="users.id", index=True,
                                description="User who created this document-agent link")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    document: "Document" = Relationship(back_populates="agent_links")
    agent: "AIAgent" = Relationship(back_populates="document_links")


class SessionDocument(SQLModel, table=True):
    """Links a document to a user session for chat-scoped retrieval."""
    __tablename__ = "session_documents"
    __table_args__ = (
        UniqueConstraint("document_id", "session_id", name="uq_session_doc_link"),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True, nullable=False)
    session_id: int = Field(foreign_key="user_sessions.id", index=True, nullable=False)
    user_id: int | None = Field(default=None, foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    document: "Document" = Relationship(back_populates="session_links")
    session: "UserSession" = Relationship(back_populates="document_links")


class InvitationStatus(str, Enum):
    """Status of an organization invitation."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OrganizationMembership(SQLModel, table=True):
    """Links a user to an organization with a single role, plus ownership/active-state flags.

    Replaces the old single `User.organization_id` FK (one org per user) and the old
    `UserRole` many-to-many link (role assignment now lives here, one role per membership).
    """
    __tablename__ = "organization_memberships"
    __table_args__ = (UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),)

    id: int | None = Field(default=None, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    organization_id: int = Field(foreign_key="organizations.id", index=True, nullable=False)
    role_id: int = Field(foreign_key="roles.id", index=True, nullable=False)
    is_owner: bool = Field(default=False, description="True for the user who created the organization")
    is_active: bool = Field(default=True, index=True, description="False when the member has been removed (soft-delete)")
    removed_at: datetime | None = Field(default=None)
    joined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    user: "User" = Relationship(back_populates="memberships")
    organization: "Organization" = Relationship(back_populates="memberships")
    role: "Role" = Relationship(back_populates="memberships")


class OrganizationInvitation(SQLModel, table=True):
    """A pending/actioned invitation for an email address to join an organization with a role."""
    __tablename__ = "organization_invitations"

    id: int | None = Field(default=None, primary_key=True, index=True)
    organization_id: int = Field(foreign_key="organizations.id", index=True, nullable=False)
    email: str = Field(index=True, nullable=False, description="Invitee email, lowercased")
    role_id: int = Field(foreign_key="roles.id", nullable=False)
    invited_by_user_id: int = Field(foreign_key="users.id", nullable=False)
    token_hash: str = Field(unique=True, index=True, nullable=False, description="SHA-256 hash of the raw invite token")
    status: str = Field(default=InvitationStatus.PENDING, index=True)
    expires_at: datetime = Field(nullable=False)
    accepted_at: datetime | None = Field(default=None)
    accepted_user_id: int | None = Field(default=None, foreign_key="users.id")
    cancelled_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    organization: "Organization" = Relationship()
    role: "Role" = Relationship()


class User(SQLModel, table=True):
    """User account information."""
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True, index=True)
    email: str = Field(unique=True, index=True)
    first_name: str = Field(..., description="User's first name")
    last_name: str | None = None
    phone: str | None = None
    address: str | None = None
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    is_subscribed: bool = Field(
        default=False,
        index=True,
        sa_column_kwargs={"server_default": "false"},
        description="True once the user has converted off the free trial and brings their own LLM API key",
    )
    active_organization_id: int | None = Field(default=None, foreign_key="organizations.id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    settings: "UserSettings" = Relationship(back_populates="user")
    agents: List["AIAgent"] = Relationship(back_populates="owner")
    documents: List["Document"] = Relationship(back_populates="user")
    sessions: List["UserSession"] = Relationship(back_populates="user")
    mcp_server_configs: List["MCPServerConfig"] = Relationship(back_populates="user")
    memberships: List["OrganizationMembership"] = Relationship(back_populates="user")


class Role(SQLModel, table=True):
    """An organizational role (e.g. Admin) assigned to users."""
    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_role_org_name"),)
    id: int | None = Field(default=None, primary_key=True, index=True)
    organization_id: int = Field(foreign_key="organizations.id", index=True)
    name: str
    description: str | None = None
    is_system: bool = Field(default=False)
    permissions: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    is_active: bool = Field(default=True, index=True, sa_column_kwargs={"server_default": "true"})
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    organization: "Organization" = Relationship()
    memberships: List["OrganizationMembership"] = Relationship(back_populates="role")


class UserSession(SQLModel, table=True):
    """User session information."""
    __tablename__ = "user_sessions"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    agent_id: str = Field(
        foreign_key="ai_agents.id",
        index=True,
        nullable=False,
        description="ID of the AI agent associated with this session"
    )
    session_id: str = Field(index=True, unique=True, nullable=False)
    title: Optional[str] = None
    status: str = Field(default=SessionStatus.ACTIVE)
    is_active: bool = Field(default=True, index=True, description="False when the session is soft-deleted")
    is_archived: bool = Field(default=False, index=True, description="True when the conversation is archived")
    # WhatsApp sender phone — set only for webhook-initiated sessions so we can
    # match returning senders to their existing chat without a platform User lookup.
    whatsapp_phone: str | None = Field(
        default=None,
        index=True,
        description="Normalized sender phone (digits only) for WhatsApp-initiated sessions"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    
    # Relationships
    user: "User" = Relationship(back_populates="sessions")
    messages: List["Message"] = Relationship(back_populates="session")
    agent: "AIAgent" = Relationship()
    document_links: List["SessionDocument"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Message(SQLModel, table=True):
    """Represents a single message in a conversation."""
    __tablename__ = "messages"
    
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="user_sessions.id", index=True, nullable=False)
    agent_id: str | None = Field(foreign_key="ai_agents.id", index=True, nullable=True)
    conversation_id: str = Field(index=True, description="Unique identifier for the conversation this message belongs to")
    role: str = Field(index=True, description="Role of the message sender (user/assistant)")
    prompt: str | None = Field(nullable=True, description="The original prompt that generated this message (if any)")
    content: str = Field(description="The message content")
    counter: int = Field(description="Message position in the conversation")
    conversation_type: str = Field(
        default=ConversationType.TEXT,
        description="Type of conversation: 'text' or 'voice'"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session: "UserSession" = Relationship(back_populates="messages")
    agent: Optional["AIAgent"] = Relationship()
    feedbacks: List["MessageFeedback"] = Relationship(back_populates="message")


class MessageFeedback(SQLModel, table=True):
    """Feedback for chat messages (thumbs up/down)."""
    __tablename__ = "message_feedbacks"
    
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    message_id: int = Field(foreign_key="messages.id", index=True, nullable=False)
    is_positive: bool = Field(description="True for thumbs up, False for thumbs down")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user: "User" = Relationship()
    message: "Message" = Relationship(back_populates="feedbacks")


class UserSettings(SQLModel, table=True):
    """User preferences and settings."""
    __tablename__ = "user_settings"
    
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)
    theme: str = Field(default="light")
    language: str = Field(default="en")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    
    # Relationships
    user: "User" = Relationship(back_populates="settings")


class AIAgent(SQLModel, table=True):
    """AI Agent configuration and metadata."""
    __tablename__ = "ai_agents"
    
    id: str = Field(primary_key=True, index=True)
    name: str = Field(index=True)
    description: str | None = None
    owner_id: int = Field(foreign_key="users.id", index=True)
    organization_id: int | None = Field(default=None, foreign_key="organizations.id", index=True)
    is_active: bool = Field(default=True)
    deleted: bool = Field(default=False, index=True, sa_column_kwargs={"server_default": "false"})
    is_cold_build: bool = Field(default=False, sa_column_kwargs={"server_default": "false"})

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    
    # Relationships
    owner: "User" = Relationship(back_populates="agents")
    settings: "AIAgentSettings" = Relationship(back_populates="agent")
    organization: Optional["Organization"] = Relationship(back_populates="agents")
    document_links: List["AgentDocument"] = Relationship(back_populates="agent", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    whatsapp_numbers: List["AgentWhatsappNumber"] = Relationship(back_populates="agent", sa_relationship_kwargs={"cascade": "all, delete-orphan"})


class AIAgentSettings(SQLModel, table=True):
    """Configuration settings for an AI agent."""
    __tablename__ = "ai_agent_settings"
    __table_args__ = (
        CheckConstraint('llm_temperature >= 0.0 AND llm_temperature <= 2.0', name='chk_agent_llm_temperature_range'),
    )
    
    id: int | None = Field(default=None, primary_key=True)
    agent_id: str = Field(foreign_key="ai_agents.id", unique=True, index=True)
    
    # Relationships
    agent: "AIAgent" = Relationship(back_populates="settings")
    mcp_links: List["AgentMCPConfigLink"] = Relationship(back_populates="agent_settings")

    
    # LLM Configuration
    llm_provider: str = Field(description="LLM provider (e.g., 'openai', 'gemini')")
    llm_model: str = Field(description="Model to use from provider")
    llm_temperature: float = Field(description="Controls randomness (0-1)")
    llm_max_tokens: int = Field(description="Max tokens to generate")
    llm_system_prompt: str | None = Field(description="System instructions for the model")
    llm_json_prompt: bool = Field(default=False, description="Whether to use JSON format for model prompts")
    preserve_memory: bool = Field(default=True, description="Whether to preserve conversation history for this agent")
    allow_websearch: bool = Field(default=True, description="Whether the agent is allowed to search the web for answers")
    websearch_provider: str | None = Field(default=None, description="Selected web search provider")
    websearch_api_key_id: int | None = Field(default=None, foreign_key="api_keys.id", index=True, description="API key row for the web search provider")
    websearch_model: str | None = Field(default=None, description="Selected model for web search (e.g., gpt-5.4-mini)")
    llm_voice_prompt: str | None = Field(default=None, description="Custom system prompt for voice conversations (falls back to llm_system_prompt if null)")
    api_key: int | None = Field(
        default=None,
        foreign_key="api_keys.id",
        index=True,
        description="Selected organization API key row for this agent's LLM provider",
    )


    has_data_store: bool = Field(
        default=False,
        description="Whether this agent has a vector data store for document storage"
    )
    is_mcp_enabled: bool = Field(
        default=False,
        description="Whether this agent has MCP server integration enabled"
    )
    whatsapp_enabled: bool = Field(
        default=False,
        description="Whether WhatsApp integration is enabled for this agent"
    )
    # The business WhatsApp number that receives incoming messages for this agent.
    # When set, the webhook routes by this number first (business-number-first routing).
    business_number: str | None = Field(
        default=None,
        nullable=True,
        index=True,
        unique=True,
        description="Business WhatsApp number that receives messages for this agent (E.164 digits)"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


class AgentWhatsappNumber(SQLModel, table=True):
    """Stores WhatsApp (user/sender) numbers linked to an agent.

    Each row represents one sender phone number allowed to message this agent.
    A single agent can have many numbers. The same phone number CAN be
    registered to multiple agents (via different business numbers), but
    cannot be registered to the same agent twice.
    """
    __tablename__ = "agent_whatsapp_numbers"
    __table_args__ = (
        # Relaxed from global uniqueness: same user phone can be linked to
        # different agents (each agent has its own business number).
        UniqueConstraint("phone_number", "agent_id", name="uq_wa_number_phone_agent"),
    )

    id: int | None = Field(default=None, primary_key=True)

    # The agent this number belongs to
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)

    # The logged-in user who added this number (used as user_id in message pipeline)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False,
                         description="Platform user who registered this number")

    phone_number: str = Field(
        nullable=False,
        description="Sender phone in E.164 digits, e.g. '919876543210'"
    )
    display_label: str | None = Field(
        default=None,
        description="Human-readable label, e.g. 'Support Line'"
    )

    is_active: bool = Field(
        default=True,
        index=True,
        description="Set to False to temporarily disable this number without deleting it"
    )

    welcome_message_sent: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"server_default": "false"},
        description="True once the Gallabox welcome_template has been successfully dispatched for this number"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    agent: "AIAgent" = Relationship(back_populates="whatsapp_numbers")
    user: "User" = Relationship()

    
class ModelProvider(SQLModel, table=True):
    """DB-backed model catalog for LLM and embedding dropdowns."""
    __tablename__ = "model_provider"
    __table_args__ = (
        sa.CheckConstraint("type IN ('llm', 'embeddings')", name="ck_model_provider_type"),
        UniqueConstraint("provider", "model_name", "type", name="uq_model_provider_catalog"),
        sa.Index("ix_model_provider_provider_type", "provider", "type"),
    )

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(nullable=False)
    model_name: str = Field(nullable=False)
    type: str = Field(nullable=False)
    dimension: int | None = Field(default=None, nullable=True)
    litellm_prefix: str = Field(nullable=False)
    is_default: bool = Field(default=False, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    is_llm: bool = Field(default=True, nullable=False)
    is_websearch: bool = Field(default=False, nullable=False)
    # False for models that reject the temperature parameter entirely (e.g. o1-preview, o3-mini).
    # The UI reads this flag to disable the temperature slider rather than relying on silent retries.
    supports_temperature: bool = Field(default=True, nullable=False)
    max_llm_temperature: float = Field(default=2.0, nullable=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)


class ApiKey(SQLModel, table=True):
    """Stores encrypted API keys for different providers."""
    __tablename__ = "api_keys"
    
    id: int | None = Field(default=None, primary_key=True)
    organization_id: int = Field(foreign_key="organizations.id", index=True)
    provider: str = Field(index=True, description="Service provider (e.g., 'openai', 'gemini')")
    name: str = Field(default="", description="Name or label for this API key")
    is_default: bool = Field(default=False, index=True, nullable=False)
    is_active: bool = Field(default=True, index=True, nullable=False)
    deleted_at: datetime | None = Field(default=None, nullable=True)
    encrypted_key: str = Field(
        sa_column=sa.Column("encrypted_key", sa.Text(), nullable=False),
        exclude=True
    )
    last_used: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    
    # Encryption manager for API keys
    _encryption_manager: ClassVar = get_encryption_manager()
    
    def encrypt(self, key: str) -> None:
        """Encrypt and store the API key."""
        if not key:
            raise ValueError("API key cannot be empty")
        self.encrypted_key = self._encryption_manager.encrypt(key)
        self.updated_at = datetime.now(timezone.utc)
    
    def decrypt(self) -> str:
        """Decrypt and return the API key."""
        if not self.encrypted_key:
            raise ValueError("No encrypted key found")
        self.last_used = datetime.now(timezone.utc)
        return self._encryption_manager.decrypt(self.encrypted_key)

    @property
    def masked_key(self) -> str:
        """Return a masked representation of the API key."""
        try:
            raw = self._encryption_manager.decrypt(self.encrypted_key)
            if not raw:
                return "••••••••"
            if len(raw) <= 8:
                return "••••" + raw[-2:] if len(raw) > 2 else "••••"
            return raw[:4] + "..." + raw[-4:]
        except Exception:
            return "••••••••"


class MCPServerConfig(SQLModel, table=True):
    """Stores MCP server configurations owned by a user."""
    __tablename__ = "mcp_server_configs"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    name: str = Field(index=True, description="Display name for this MCP server")
    url: str = Field(nullable=False, description="URL for SSE/HTTP transport")
    description: str = Field(
        nullable=False,
        sa_column_kwargs={"server_default": ""},
        description="Human-readable description of what this MCP server exposes.",
    )
    requires_query_building: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"server_default": "false"},
        description=(
            "If True, this MCP server exposes a generic database and the agent should "
            "fetch schema + generate SQL. If False, the agent should treat the server "
            "as a set of purpose-built tools."
        ),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    user: "User" = Relationship(back_populates="mcp_server_configs")
    agent_links: List["AgentMCPConfigLink"] = Relationship(back_populates="mcp_server_config")


class AgentMCPConfigLink(SQLModel, table=True):
    """Link table between AIAgentSettings and MCPServerConfig (many-to-many)."""
    __tablename__ = "agent_mcp_config_links"

    id: int | None = Field(default=None, primary_key=True)
    agent_settings_id: int = Field(foreign_key="ai_agent_settings.id", index=True, nullable=False)
    mcp_server_config_id: int = Field(foreign_key="mcp_server_configs.id", index=True, nullable=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    agent_settings: "AIAgentSettings" = Relationship(back_populates="mcp_links")
    mcp_server_config: "MCPServerConfig" = Relationship(back_populates="agent_links")


# ============================================================================
# Agent Runtime & Observability Models
# ============================================================================

class RunStatus(str, Enum):
    """Status of an agent run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_APPROVAL = "requires_approval"


class RunMode(str, Enum):
    """Execution mode for agent runs."""
    CHAT = "chat"
    TOOL_USE = "tool_use"
    PLANNER = "planner"
    AUTONOMOUS = "autonomous"
    WORKFLOW = "workflow"


class AgentRun(SQLModel, table=True):
    """Represents a complete agent execution run."""
    __tablename__ = "agent_runs"

    id: str = Field(primary_key=True, index=True, description="Unique run ID (UUID)")
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)
    session_id: int = Field(foreign_key="user_sessions.id", index=True, nullable=False)
    conversation_id: str = Field(index=True, nullable=False)
    input_message_id: int | None = Field(foreign_key="messages.id", nullable=True)
    output_message_id: int | None = Field(foreign_key="messages.id", nullable=True)

    # Execution configuration
    mode: str = Field(default=RunMode.CHAT, description="Execution mode")
    llm_provider: str = Field(description="LLM provider used")
    llm_model: str = Field(description="LLM model used")

    # Execution limits
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    max_tool_calls: int = Field(default=5, description="Maximum tool calls allowed")
    max_tokens: int | None = Field(default=None, description="Maximum tokens allowed")
    timeout_seconds: int = Field(default=120, description="Timeout in seconds")

    # Execution state
    status: str = Field(default=RunStatus.PENDING, index=True)
    current_step: int = Field(default=0, description="Current step number")
    total_steps: int = Field(default=0, description="Total steps executed")

    # Usage tracking
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cached_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)

    # Audio usage
    audio_input_seconds: float = Field(default=0.0)
    audio_output_seconds: float = Field(default=0.0)

    # Performance
    latency_ms: int | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Error information
    error: str | None = Field(default=None)
    error_code: str | None = Field(default=None)

    # Relationships
    user: "User" = Relationship()
    agent: "AIAgent" = Relationship()
    session: "UserSession" = Relationship()
    steps: List["AgentRunStep"] = Relationship(back_populates="run")
    usage_events: List["LLMUsageEvent"] = Relationship(back_populates="run")
    artifacts: List["Artifact"] = Relationship(back_populates="run")


class StepType(str, Enum):
    """Types of steps in an agent run."""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    PLANNER = "planner"
    ROUTER = "router"
    GUARDRAIL = "guardrail"
    HUMAN_APPROVAL = "human_approval"
    OBSERVATION = "observation"
    FORMATTING = "formatting"


class AgentRunStep(SQLModel, table=True):
    """Represents a single step in an agent run."""
    __tablename__ = "agent_run_steps"

    id: str = Field(primary_key=True, index=True)
    run_id: str = Field(foreign_key="agent_runs.id", index=True, nullable=False)
    step_index: int = Field(index=True, description="Step number within the run")

    step_type: str = Field(description="Type of step")
    name: str = Field(description="Human-readable name for the step")

    # Input/output
    input_data: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column(JSON))
    output_data: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column(JSON))

    # Status
    status: str = Field(default="running", description="pending/running/completed/failed")
    error: str | None = Field(default=None)

    # Usage
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    latency_ms: int | None = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)

    # Relationships
    run: "AgentRun" = Relationship(back_populates="steps")


class LLMUsageEvent(SQLModel, table=True):
    """Tracks individual LLM API calls for detailed usage analysis."""
    __tablename__ = "llm_usage_events"

    id: str = Field(primary_key=True, index=True)
    run_id: str | None = Field(foreign_key="agent_runs.id", index=True, nullable=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    organization_id: int | None = Field(foreign_key="organizations.id", index=True, nullable=True)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)
    session_id: int | None = Field(foreign_key="user_sessions.id", index=True, nullable=True)
    conversation_id: str | None = Field(index=True, nullable=True)
    message_id: int | None = Field(foreign_key="messages.id", index=True, nullable=True)

    # LLM call details
    provider: str = Field(index=True, description="LLM provider")
    model: str = Field(index=True, description="Model used")
    operation: str = Field(index=True, description="Operation type: chat/embedding/stt/tts/realtime")
    temperature: float | None = Field(default=None, description="Sampling temperature used for the call")

    # Token usage
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cached_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)
    tool_tokens: int = Field(default=0)
    embedding_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    # Audio usage (for voice operations)
    audio_input_seconds: float = Field(default=0.0)
    audio_output_seconds: float = Field(default=0.0)

    # Cost
    cost_usd: float = Field(default=0.0)

    # Metadata
    latency_ms: int | None = Field(default=None)
    status: str = Field(default="success", index=True, description="success/failure")
    error_message: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    run: "AgentRun" = Relationship(back_populates="usage_events")


class Artifact(SQLModel, table=True):
    """Stores generated artifacts (files, outputs) from agent runs."""
    __tablename__ = "artifacts"

    id: str = Field(primary_key=True, index=True)
    run_id: str = Field(foreign_key="agent_runs.id", index=True, nullable=False)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)
    session_id: int | None = Field(foreign_key="user_sessions.id", index=True, nullable=True)
    conversation_id: str | None = Field(index=True, nullable=True)
    message_id: int | None = Field(foreign_key="messages.id", index=True, nullable=True)

    # Artifact metadata
    artifact_type: str = Field(index=True, description="Type of artifact")
    mime_type: str = Field(description="MIME type")
    filename: str | None = Field(default=None, description="Original filename")
    storage_key: str | None = Field(default=None, description="Storage location (S3/MinIO path)")
    size_bytes: int = Field(default=0, description="Size in bytes")

    # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy reserved attr)
    extra_metadata: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column("metadata", JSON))
    description: str | None = Field(default=None)

    # Expiration
    expires_at: datetime | None = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    run: "AgentRun" = Relationship(back_populates="artifacts")


class AuditAction(str, Enum):
    """Types of audit actions."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"


class AuditLog(SQLModel, table=True):
    """Comprehensive audit logging for security and compliance."""
    __tablename__ = "audit_logs"

    id: str = Field(primary_key=True, index=True)
    actor_user_id: int | None = Field(foreign_key="users.id", index=True, nullable=True)
    action: str = Field(index=True, description="Action performed")

    # Resource being acted upon
    resource_type: str = Field(index=True, description="Type of resource: agent/message/document/etc")
    resource_id: str | None = Field(index=True, nullable=True)

    # Change tracking
    before_state: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column(JSON))
    after_state: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column(JSON))

    # Request context
    ip_address: str | None = Field(default=None)
    user_agent: str | None = Field(default=None)
    request_path: str | None = Field(default=None)
    request_method: str | None = Field(default=None)

    # Additional context (renamed from 'metadata' to avoid SQLAlchemy reserved attr)
    extra_metadata: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column("metadata", JSON))
    reason: str | None = Field(default=None, description="Reason for the action")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentIngestionStatus(str, Enum):
    """Status of document ingestion jobs."""
    PENDING = "pending"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentIngestionJob(SQLModel, table=True):
    """Tracks background document ingestion jobs."""
    __tablename__ = "document_ingestion_jobs"

    id: str = Field(primary_key=True, index=True)
    document_id: int = Field(foreign_key="documents.id", index=True, nullable=False)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)

    # Job status
    status: str = Field(default=DocumentIngestionStatus.PENDING, index=True)
    progress_percent: int = Field(default=0, ge=0, le=100)

    # Processing details
    total_chunks: int = Field(default=0)
    processed_chunks: int = Field(default=0)
    failed_chunks: int = Field(default=0)

    # Configuration used
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    embedding_model: str = Field(default="text-embedding-3-small")

    # Error tracking
    error_message: str | None = Field(default=None)
    error_details: Dict[str, Any] | None = Field(default=None, sa_column=sa.Column(JSON))

    # Processing metadata
    celery_task_id: str | None = Field(default=None, index=True)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


class UsageQuota(SQLModel, table=True):
    """Tracks usage quotas for users/agents."""
    __tablename__ = "usage_quotas"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    agent_id: str | None = Field(foreign_key="ai_agents.id", index=True, nullable=True)

    # Quota type
    quota_type: str = Field(index=True, description="tokens/cost/requests/audio_seconds")
    period: str = Field(index=True, description="daily/weekly/monthly")

    # Limits
    limit_value: float = Field(description="Quota limit")
    used_value: float = Field(default=0.0, description="Used amount")

    # Period tracking
    period_start: datetime = Field(index=True)
    period_end: datetime = Field(index=True)

    # Enforcement
    is_enforced: bool = Field(default=True)
    action_on_exceed: str = Field(default="block", description="block/warn/allow")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    user: "User" = Relationship()


class FAQSource(str, Enum):
    """Source of a generated FAQ."""
    AGENT_CONFIG = "agent_config"
    CHAT_HISTORY = "chat_history"


class AgentFAQ(SQLModel, table=True):
    """Cached dynamically-generated FAQ for an AI agent.

    FAQs are generated per-agent (not per-user) and stored for reuse.
    They are regenerated when the agent's configuration changes or enough
    new conversation data accumulates.
    """
    __tablename__ = "agent_faqs"

    id: int | None = Field(default=None, primary_key=True)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)
    question: str = Field(description="The FAQ question text")
    source: str = Field(
        default=FAQSource.AGENT_CONFIG,
        index=True,
        description="How this FAQ was generated: 'agent_config' or 'chat_history'",
    )
    version: int = Field(
        default=1,
        description="Monotonically increasing version; bumped when FAQs are regenerated",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    agent: "AIAgent" = Relationship()

# Monitor domain — models live in custom_model/monitor.py
from ai_agent.custom_model.monitor import MonitorType, Monitor, MonitorResult, MonitorResultType, MonitorResultStatus  # noqa: E402, F401

# Query Assistant history — model lives in custom_model/query_history.py
from ai_agent.custom_model.query_history import QueryHistoryEntry  # noqa: E402, F401


class OrganizationTokenPolicy(SQLModel, table=True):
    """Token usage policy constraints at the organization level.

    Daily, weekly and monthly caps are independent and can be enabled
    simultaneously (e.g. a hard daily ceiling alongside a monthly budget).
    """
    __tablename__ = "organization_token_policies"

    id: int = Field(default=None, primary_key=True)
    organization_id: int = Field(foreign_key="organizations.id", index=True, unique=True, nullable=False)

    daily_limit_enabled: bool = Field(default=False)
    daily_max_tokens: int = Field(default=0)

    weekly_limit_enabled: bool = Field(default=False)
    weekly_max_tokens: int = Field(default=0)

    monthly_limit_enabled: bool = Field(default=False)
    monthly_max_tokens: int = Field(default=0)

    alerts_enabled: bool = Field(default=False)
    alert_threshold_percentage: int = Field(default=80)
    notification_email: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default=None)


class AgentTokenPolicy(SQLModel, table=True):
    """Token usage policy constraints at the agent level.

    Daily, weekly and monthly caps are independent and can be enabled
    simultaneously (e.g. a hard daily ceiling alongside a monthly budget).
    """
    __tablename__ = "agent_token_policies"

    id: int = Field(default=None, primary_key=True)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, unique=True, nullable=False)

    daily_limit_enabled: bool = Field(default=False)
    daily_max_tokens: int = Field(default=0)

    weekly_limit_enabled: bool = Field(default=False)
    weekly_max_tokens: int = Field(default=0)

    monthly_limit_enabled: bool = Field(default=False)
    monthly_max_tokens: int = Field(default=0)

    alerts_enabled: bool = Field(default=False)
    alert_threshold_percentage: int = Field(default=80)
    notification_email: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default=None)


class UserTokenPolicy(SQLModel, table=True):
    """Token usage policy constraints at the user level.

    Daily, weekly and monthly caps are independent and can be enabled
    simultaneously (e.g. a hard daily ceiling alongside a monthly budget).
    """
    __tablename__ = "user_token_policies"

    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, unique=True, nullable=False)

    daily_limit_enabled: bool = Field(default=False)
    daily_max_tokens: int = Field(default=0)

    weekly_limit_enabled: bool = Field(default=False)
    weekly_max_tokens: int = Field(default=0)

    monthly_limit_enabled: bool = Field(default=False)
    monthly_max_tokens: int = Field(default=0)

    alerts_enabled: bool = Field(default=False)
    alert_threshold_percentage: int = Field(default=80)
    notification_email: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default=None)

