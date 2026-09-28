"""Restore SDK model tables

Revision ID: 2026_06_04_0000
Revises: 2026_05_26_0000
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2026_06_04_0000"
down_revision: Union[str, Sequence[str], None] = "2026_05_26_0000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "public_api_keys",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("key_type", sa.String(), nullable=False, server_default="live"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("allowed_domains", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("allowed_ips", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("rate_limit_requests_per_minute", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("rate_limit_tokens_per_day", sa.Integer(), nullable=True),
        sa.Column("enable_voice", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("enable_file_upload", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("enable_streaming", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("enable_artifacts", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("allowed_modes", sa.JSON(), nullable=False, server_default='["chat", "tool_use"]'),
        sa.Column("total_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["ai_agents.id"], name="fk_public_api_keys_agent"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_public_api_keys_user"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_public_api_keys_agent_id", "public_api_keys", ["agent_id"])
    op.create_index("ix_public_api_keys_is_active", "public_api_keys", ["is_active"])
    op.create_index("ix_public_api_keys_user_id", "public_api_keys", ["user_id"])

    op.create_table(
        "sdk_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=False),
        sa.Column("anonymous_user_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("referrer", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["public_api_keys.id"], name="fk_sdk_sessions_api_key", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_sdk_sessions_user"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sdk_sessions_anonymous_user_id", "sdk_sessions", ["anonymous_user_id"])
    op.create_index("ix_sdk_sessions_api_key_id", "sdk_sessions", ["api_key_id"])
    op.create_index("ix_sdk_sessions_is_active", "sdk_sessions", ["is_active"])

    op.create_table(
        "sdk_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_artifacts", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("artifact_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sdk_sessions.id"], name="fk_sdk_messages_session", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sdk_messages_role", "sdk_messages", ["role"])
    op.create_index("ix_sdk_messages_session_id", "sdk_messages", ["session_id"])

    op.create_table(
        "webhook_endpoints",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("secret", sa.String(), nullable=False),
        sa.Column("events", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("last_delivery_at", sa.DateTime(), nullable=True),
        sa.Column("last_delivery_status", sa.String(), nullable=True),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["public_api_keys.id"], name="fk_webhook_endpoints_api_key", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_webhook_endpoints_user"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_webhook_endpoints_api_key_id", "webhook_endpoints", ["api_key_id"])
    op.create_index("ix_webhook_endpoints_is_active", "webhook_endpoints", ["is_active"])
    op.create_index("ix_webhook_endpoints_user_id", "webhook_endpoints", ["user_id"])

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("webhook_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("headers", sa.JSON(), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("delivered_at", sa.DateTime(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["webhook_id"], ["webhook_endpoints.id"], name="fk_webhook_deliveries_webhook", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_webhook_deliveries_event_id", "webhook_deliveries", ["event_id"])
    op.create_index("ix_webhook_deliveries_status", "webhook_deliveries", ["status"])
    op.create_index("ix_webhook_deliveries_webhook_id", "webhook_deliveries", ["webhook_id"])

    op.create_table(
        "rate_limit_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("limit_type", sa.String(), nullable=False),
        sa.Column("limit_value", sa.Integer(), nullable=False),
        sa.Column("current_usage", sa.Integer(), nullable=False),
        sa.Column("action_taken", sa.String(), nullable=False),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["public_api_keys.id"], name="fk_rate_limit_logs_api_key", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_rate_limit_logs_user"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rate_limit_logs_api_key_id", "rate_limit_logs", ["api_key_id"])
    op.create_index("ix_rate_limit_logs_created_at", "rate_limit_logs", ["created_at"])
    op.create_index("ix_rate_limit_logs_user_id", "rate_limit_logs", ["user_id"])

    op.create_table(
        "sdk_config_presets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("theme", sa.String(), nullable=False, server_default="light"),
        sa.Column("primary_color", sa.String(), nullable=True),
        sa.Column("logo_url", sa.String(), nullable=True),
        sa.Column("position", sa.String(), nullable=False, server_default="bottom-right"),
        sa.Column("initial_message", sa.Text(), nullable=True),
        sa.Column("placeholder_text", sa.String(), nullable=False, server_default="Type your message..."),
        sa.Column("show_file_upload", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("show_voice_button", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("show_powered_by", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("suggested_prompts", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_sdk_config_presets_user"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sdk_config_presets_user_id", "sdk_config_presets", ["user_id"])


def downgrade() -> None:
    op.drop_table("sdk_config_presets")
    op.drop_table("rate_limit_logs")
    op.drop_table("webhook_deliveries")
    op.drop_table("webhook_endpoints")
    op.drop_table("sdk_messages")
    op.drop_table("sdk_sessions")
    op.drop_table("public_api_keys")
