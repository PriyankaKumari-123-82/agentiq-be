"""Usage tracking and quota enforcement service.

This module provides:
- Token usage recording
- Cost tracking
- Quota enforcement
- Usage analytics
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from decimal import Decimal

from sqlalchemy import func
from sqlmodel import Session, select

from ai_agent.core.enums import LLMOperation
from ai_agent.models import (
    LLMUsageEvent, UsageQuota, AgentRun,
    User, AIAgent
)
from ai_agent.core.config import settings

log = logging.getLogger(__name__)


class UsageTracker:
    """Tracks and enforces usage quotas."""

    # Cost per 1K tokens (approximate, should be updated from pricing APIs)
    COST_PER_1K_TOKENS = {
        "openai/gpt-4o": {"input": 0.0025, "output": 0.01},
        "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "openai/gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "anthropic/claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "google/gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
        "google/gemini-1.5-flash": {"input": 0.00035, "output": 0.00105},
    }

    def __init__(self, db_session: Session):
        """Initialize the usage tracker.
        
        Args:
            db_session: Database session
        """
        self.db = db_session

    def record_llm_usage(
        self,
        user_id: int,
        agent_id: str,
        provider: str,
        model: str,
        operation: Union[LLMOperation, str],
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        tool_tokens: int = 0,
        embedding_tokens: int = 0,
        audio_input_seconds: float = 0.0,
        audio_output_seconds: float = 0.0,
        run_id: Optional[str] = None,
        session_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[int] = None,
        latency_ms: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        organization_id: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMUsageEvent:
        """Record an LLM usage event.
        
        Returns:
            The created LLMUsageEvent record
        """
        operation_name = operation.value if isinstance(operation, LLMOperation) else operation or "unknown"

        # Calculate cost
        cost_usd = self._calculate_cost(
            provider, model, input_tokens, output_tokens,
            audio_input_seconds, audio_output_seconds
        )

        event = LLMUsageEvent(
            id=str(__import__('uuid').uuid4()),
            run_id=run_id,
            user_id=user_id,
            organization_id=organization_id,
            agent_id=agent_id,
            session_id=session_id,
            conversation_id=conversation_id,
            message_id=message_id,
            provider=provider,
            model=model,
            operation=operation_name,
            temperature=temperature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            tool_tokens=tool_tokens,
            embedding_tokens=embedding_tokens,
            audio_input_seconds=audio_input_seconds,
            audio_output_seconds=audio_output_seconds,
            total_tokens=input_tokens + output_tokens + tool_tokens + embedding_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

        self.db.add(event)
        self.db.commit()

        log.info(
            f"Recorded usage: {user_id}/{agent_id} - {provider}/{model} - "
            f"{input_tokens}+{output_tokens} tokens - ${cost_usd:.4f}"
        )

        return event

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        audio_input_seconds: float = 0.0,
        audio_output_seconds: float = 0.0,
    ) -> float:
        """Calculate cost in USD for an LLM call."""
        # Build key from provider/model
        key = f"{provider}/{model}"
        
        # Get pricing
        pricing = self.COST_PER_1K_TOKENS.get(key)
        if not pricing:
            # Try generic lookup
            for k, v in self.COST_PER_1K_TOKENS.items():
                if model in k or k in model:
                    pricing = v
                    break
        
        if not pricing:
            # Default pricing
            pricing = {"input": 0.001, "output": 0.002}
        
        # Calculate text cost
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        text_cost = input_cost + output_cost
        
        # Calculate audio cost (approximate)
        # OpenAI Realtime: $0.006/min input, $0.024/min output
        audio_input_cost = (audio_input_seconds / 60) * 0.006
        audio_output_cost = (audio_output_seconds / 60) * 0.024
        audio_cost = audio_input_cost + audio_output_cost
        
        return round(text_cost + audio_cost, 6)

    def check_quota(
        self,
        user_id: int,
        agent_id: Optional[str] = None,
        quota_type: str = "tokens",
        requested_amount: float = 0,
    ) -> Dict[str, Any]:
        """Check if a request is within quota limits.
        
        Returns:
            Dict with allowed (bool), remaining, and message
        """
        # Get active quotas for user
        now = datetime.utcnow()
        
        quotas = self.db.exec(
            select(UsageQuota)
            .where(UsageQuota.user_id == user_id)
            .where(UsageQuota.quota_type == quota_type)
            .where(UsageQuota.period_start <= now)
            .where(UsageQuota.period_end >= now)
            .where(UsageQuota.is_enforced == True)
        ).all()

        if not quotas:
            # No quotas set, allow
            return {"allowed": True, "remaining": None, "message": None}

        for quota in quotas:
            # Check specific agent quota if applicable
            if quota.agent_id and quota.agent_id != agent_id:
                continue

            remaining = quota.limit_value - quota.used_value
            
            if remaining < requested_amount:
                return {
                    "allowed": False,
                    "remaining": remaining,
                    "limit": quota.limit_value,
                    "message": f"Quota exceeded: {remaining:.0f} {quota_type} remaining of {quota.limit_value:.0f} {quota_type}",
                    "action": quota.action_on_exceed,
                }

        return {"allowed": True, "remaining": remaining, "message": None}

    def increment_quota(
        self,
        user_id: int,
        agent_id: Optional[str],
        quota_type: str,
        amount: float,
    ):
        """Increment quota usage."""
        now = datetime.utcnow()
        
        quotas = self.db.exec(
            select(UsageQuota)
            .where(UsageQuota.user_id == user_id)
            .where(UsageQuota.quota_type == quota_type)
            .where(UsageQuota.period_start <= now)
            .where(UsageQuota.period_end >= now)
        ).all()

        for quota in quotas:
            if quota.agent_id and quota.agent_id != agent_id:
                continue
            
            quota.used_value += amount
            self.db.add(quota)
        
        self.db.commit()

    def get_usage_summary(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get usage summary for a user."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

        # Build base query
        query = select(LLMUsageEvent).where(
            LLMUsageEvent.user_id == user_id
        ).where(
            LLMUsageEvent.created_at >= start_date
        ).where(
            LLMUsageEvent.created_at <= end_date
        )

        if agent_id:
            query = query.where(LLMUsageEvent.agent_id == agent_id)

        events = self.db.exec(query).all()

        # Calculate totals
        total_input_tokens = sum(e.input_tokens for e in events)
        total_output_tokens = sum(e.output_tokens for e in events)
        total_tool_tokens = sum(e.tool_tokens for e in events)
        total_embedding_tokens = sum(e.embedding_tokens for e in events)
        total_cost = sum(e.cost_usd for e in events)
        
        # By provider
        by_provider = {}
        for e in events:
            key = f"{e.provider}/{e.model}"
            if key not in by_provider:
                by_provider[key] = {"tokens": 0, "cost": 0.0, "calls": 0}
            by_provider[key]["tokens"] += e.total_tokens
            by_provider[key]["cost"] += e.cost_usd
            by_provider[key]["calls"] += 1

        # By operation
        by_operation = {}
        for e in events:
            if e.operation not in by_operation:
                by_operation[e.operation] = {"tokens": 0, "cost": 0.0}
            by_operation[e.operation]["tokens"] += e.total_tokens
            by_operation[e.operation]["cost"] += e.cost_usd

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "totals": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "tool_tokens": total_tool_tokens,
                "embedding_tokens": total_embedding_tokens,
                "total_tokens": total_input_tokens + total_output_tokens + total_tool_tokens + total_embedding_tokens,
                "cost_usd": round(total_cost, 4),
                "calls": len(events),
            },
            "by_provider": by_provider,
            "by_operation": by_operation,
        }

    def get_run_usage(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get usage for a specific run."""
        run = self.db.get(AgentRun, run_id)
        if not run:
            return None

        return {
            "run_id": run_id,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "total_tokens": run.total_tokens,
            "cost_usd": run.cost_usd,
            "audio_input_seconds": run.audio_input_seconds,
            "audio_output_seconds": run.audio_output_seconds,
            "latency_ms": run.latency_ms,
        }

    def create_quota(
        self,
        user_id: int,
        quota_type: str,
        limit: float,
        period: str = "monthly",
        agent_id: Optional[str] = None,
        action_on_exceed: str = "block",
    ) -> UsageQuota:
        """Create a new quota for a user."""
        now = datetime.utcnow()
        
        if period == "daily":
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
        elif period == "weekly":
            period_start = now - timedelta(days=now.weekday())
            period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(weeks=1)
        else:  # monthly
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Approximate month end
            if now.month == 12:
                period_end = now.replace(year=now.year + 1, month=1, day=1)
            else:
                period_end = now.replace(month=now.month + 1, day=1)

        quota = UsageQuota(
            user_id=user_id,
            agent_id=agent_id,
            quota_type=quota_type,
            period=period,
            limit_value=limit,
            used_value=0.0,
            period_start=period_start,
            period_end=period_end,
            is_enforced=True,
            action_on_exceed=action_on_exceed,
        )

        self.db.add(quota)
        self.db.commit()
        self.db.refresh(quota)

        return quota


# Factory
def get_usage_tracker(db_session: Session) -> UsageTracker:
    """Get a UsageTracker instance."""
    return UsageTracker(db_session)
