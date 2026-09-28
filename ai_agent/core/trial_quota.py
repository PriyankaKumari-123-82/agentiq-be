"""Free-trial token quota.

A subscribed user brings their own LLM API key (existing behavior, see
services/api_keys.py) and is never subject to the trial balance. A user still
on the trial is served on AgentIQ's own env-configured key, capped at
FREE_TRIAL_TOKEN_LIMIT tokens.

The balance is computed live from LLMUsageEvent — the ledger AgentRuntime
already writes to on every LLM call — rather than kept as a separately
maintained counter, so there is nothing that can drift out of sync with
actual recorded usage.
"""
from sqlalchemy import func
from sqlmodel import Session, select

from ai_agent.core.config import settings
from ai_agent.models import AIAgentSettings, LLMUsageEvent, User


class TokenQuotaExhausted(Exception):
    """Raised when a non-subscribed user has used up their free-trial balance."""


def get_tokens_used(db: Session, user_id: int) -> int:
    """Lifetime total tokens recorded for this user across all LLM calls."""
    total = db.exec(
        select(func.sum(LLMUsageEvent.total_tokens)).where(
            LLMUsageEvent.user_id == user_id
        )
    ).first()
    return total or 0


def has_balance(db: Session, user_id: int) -> bool:
    """True if the user still has free-trial tokens remaining.

    Always true when the feature flag is off, so trial gating stays fully
    inert until FREE_TRIAL_QUOTA_ENABLED is turned on (prod only).
    """
    if not settings.FREE_TRIAL_QUOTA_ENABLED:
        return True
    return get_tokens_used(db, user_id) < settings.FREE_TRIAL_TOKEN_LIMIT


def resolve_api_key(db: Session, user: User, agent_settings: AIAgentSettings | None) -> str | None:
    """Decide which LLM API key a call on this user's behalf should use.

    Subscribed users keep the existing bring-your-own-key behavior, untouched
    and never quota-checked. Everyone else is on the trial: served on the
    platform's env key as long as they still have balance.

    With FREE_TRIAL_QUOTA_ENABLED off, this always takes the DB-key branch
    regardless of is_subscribed — so behavior is identical to today for every
    user, existing or new, until the flag is deliberately turned on in prod
    (by which point existing users must already be backfilled to
    is_subscribed=True, or they'd be newly gated the moment it flips).

    Raises:
        TokenQuotaExhausted: non-subscribed user with no balance left
        (only possible when the flag is on).
    """
    if not settings.FREE_TRIAL_QUOTA_ENABLED or user.is_subscribed:
        from ai_agent.services.api_keys import get_api_key
        api_key_id = agent_settings.api_key if agent_settings else None
        return get_api_key(db, api_key_id)

    if not has_balance(db, user.id):
        raise TokenQuotaExhausted(
            f"User {user.id} has exhausted their free-trial token balance."
        )

    return settings.OPENAI_API_KEY
