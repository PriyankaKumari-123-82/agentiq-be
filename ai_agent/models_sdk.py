"""Models for Embeddable SDK and Public API Management.

This module contains models for:
- Public API Keys (for SDK embedding)
- SDK Configuration
- Webhook endpoints
- Rate limiting tracking
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlmodel import Field, SQLModel, Relationship, Column, JSON
import sqlalchemy as sa


class PublicAPIKey(SQLModel, table=True):
    """Public API keys for embedding agents in external websites/apps.
    
    These are scoped, restricted keys meant for client-side use (websites,
    mobile apps) - NOT for server-to-server API calls.
    """
    __tablename__ = "public_api_keys"
    
    id: str = Field(primary_key=True, index=True)  # pk_live_abc123 / pk_test_xyz789
    
    # Ownership
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    agent_id: str = Field(foreign_key="ai_agents.id", index=True, nullable=False)
    
    # Identification
    name: str = Field(description="Human-readable name: 'Website Widget', 'Mobile App'")
    description: Optional[str] = Field(default=None)
    
    # Key type
    key_type: str = Field(default="live", description="live/test")
    is_active: bool = Field(default=True, index=True)
    
    # Domain restrictions (CORS)
    allowed_domains: List[str] = Field(default=[], sa_column=Column(JSON))
    # Examples: ["example.com", "*.example.com", "localhost:3000"]
    
    # IP restrictions (optional additional security)
    allowed_ips: List[str] = Field(default=[], sa_column=Column(JSON))
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)
    rate_limit_tokens_per_day: Optional[int] = Field(default=None)
    
    # Feature flags (what this key can do)
    enable_voice: bool = Field(default=True)
    enable_file_upload: bool = Field(default=False)
    enable_streaming: bool = Field(default=True)
    enable_artifacts: bool = Field(default=True)
    allowed_modes: List[str] = Field(default=["chat", "tool_use"], sa_column=Column(JSON))
    
    # Embed widget customization (stored per-key so each deployment can have its own branding)
    embed_config: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=sa.Column("embed_config", JSON),
        description="Embed widget appearance settings (colors, dimensions, branding, etc.)"
    )
    logo_url: Optional[str] = Field(
        default=None,
        description="S3 URI (s3://...) for the custom logo; resolved to presigned URL on read"
    )

    # Usage tracking
    total_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    last_used_at: Optional[datetime] = Field(default=None)
    
    # Expiration
    expires_at: Optional[datetime] = Field(default=None, index=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    
    # Relationships
    sessions: List["SDKSession"] = Relationship(back_populates="api_key")
    webhook_endpoints: List["WebhookEndpoint"] = Relationship(back_populates="api_key")


class SDKSession(SQLModel, table=True):
    """Anonymous user sessions created via the SDK.
    
    These are lighter-weight than full UserSessions and don't require
    user accounts. They're tied to a PublicAPIKey.
    """
    __tablename__ = "sdk_sessions"
    
    id: str = Field(primary_key=True, index=True)  # sess_abc123
    
    # Link to API key
    api_key_id: str = Field(foreign_key="public_api_keys.id", index=True, nullable=False)
    
    # Anonymous user identifier (generated client-side)
    anonymous_user_id: Optional[str] = Field(default=None, index=True)
    
    # Optional: link to real user if they later authenticate
    user_id: Optional[int] = Field(foreign_key="users.id", nullable=True)
    
    # Context
    user_agent: Optional[str] = Field(default=None)
    ip_address: Optional[str] = Field(default=None)
    referrer: Optional[str] = Field(default=None)
    
    # Session state (renamed from 'metadata' to avoid SQLAlchemy reserved attr)
    extra_metadata: Dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))
    
    # Usage
    message_count: int = Field(default=0)
    token_count: int = Field(default=0)
    
    # Status
    is_active: bool = Field(default=True, index=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + __import__('datetime').timedelta(hours=24)
    )
    
    # Relationships
    api_key: "PublicAPIKey" = Relationship(back_populates="sessions")
    messages: List["SDKMessage"] = Relationship(back_populates="session")


class SDKMessage(SQLModel, table=True):
    """Messages in SDK sessions.
    
    These mirror regular Messages but are linked to SDKSession.
    """
    __tablename__ = "sdk_messages"
    
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sdk_sessions.id", index=True, nullable=False)
    
    role: str = Field(index=True)  # user/assistant/system
    content: str = Field()
    
    # Usage
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    
    # For rich responses
    has_artifacts: bool = Field(default=False)
    artifact_ids: List[str] = Field(default=[], sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session: "SDKSession" = Relationship(back_populates="messages")


class WebhookEndpoint(SQLModel, table=True):
    """Webhooks for async events from SDK usage.
    
    Allows external systems to receive real-time events:
    - message.created
    - session.started
    - run.completed
    - quota.approaching_limit
    """
    __tablename__ = "webhook_endpoints"
    
    id: str = Field(primary_key=True, index=True)
    
    # Ownership
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    api_key_id: Optional[str] = Field(foreign_key="public_api_keys.id", nullable=True)
    
    # Configuration
    url: str = Field(description="Webhook URL")
    secret: str = Field(description="Webhook signing secret")
    
    # Event types to subscribe to
    events: List[str] = Field(sa_column=Column(JSON))
    # ["message.created", "run.completed", "session.ended"]
    
    # Status
    is_active: bool = Field(default=True, index=True)
    
    # Delivery tracking
    last_delivery_at: Optional[datetime] = Field(default=None)
    last_delivery_status: Optional[str] = Field(default=None)  # success/failure
    failure_count: int = Field(default=0)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    
    # Relationships
    api_key: Optional["PublicAPIKey"] = Relationship(back_populates="webhook_endpoints")
    deliveries: List["WebhookDelivery"] = Relationship(back_populates="webhook")


class WebhookDelivery(SQLModel, table=True):
    """Log of webhook delivery attempts."""
    __tablename__ = "webhook_deliveries"
    
    id: str = Field(primary_key=True, index=True)
    webhook_id: str = Field(foreign_key="webhook_endpoints.id", index=True, nullable=False)
    
    event_type: str = Field()
    event_id: str = Field(index=True)  # ID of the event being delivered
    
    # Request details
    payload: Dict = Field(sa_column=Column(JSON))
    headers: Dict = Field(sa_column=Column(JSON))
    
    # Response
    response_status: Optional[int] = Field(default=None)
    response_body: Optional[str] = Field(default=None)
    
    # Status
    status: str = Field(index=True)  # pending/delivered/failed
    error_message: Optional[str] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = Field(default=None)
    
    # Retry tracking
    attempt_count: int = Field(default=0)
    next_retry_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    webhook: "WebhookEndpoint" = Relationship(back_populates="deliveries")


class RateLimitLog(SQLModel, table=True):
    """Track rate limit hits for monitoring and alerting."""
    __tablename__ = "rate_limit_logs"
    
    id: int | None = Field(default=None, primary_key=True)
    
    api_key_id: Optional[str] = Field(foreign_key="public_api_keys.id", index=True)
    user_id: Optional[int] = Field(foreign_key="users.id", index=True)
    
    limit_type: str = Field()  # requests_per_minute / tokens_per_day
    limit_value: int = Field()
    current_usage: int = Field()
    
    action_taken: str = Field()  # blocked / warned / allowed
    
    # Request context
    ip_address: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SDKConfigPreset(SQLModel, table=True):
    """Pre-configured SDK appearance/settings presets.
    
    Allows users to save and reuse widget configurations.
    """
    __tablename__ = "sdk_config_presets"
    
    id: str = Field(primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    
    name: str = Field()
    description: Optional[str] = Field(default=None)
    
    # Appearance
    theme: str = Field(default="light")  # light/dark/auto
    primary_color: Optional[str] = Field(default=None)  # Hex color
    logo_url: Optional[str] = Field(default=None)
    
    # Widget settings
    position: str = Field(default="bottom-right")  # bottom-right/bottom-left/inline
    initial_message: Optional[str] = Field(default=None)
    placeholder_text: str = Field(default="Type your message...")
    
    # Features
    show_file_upload: bool = Field(default=False)
    show_voice_button: bool = Field(default=True)
    show_powered_by: bool = Field(default=True)
    
    # Suggested prompts
    suggested_prompts: List[str] = Field(default=[], sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
