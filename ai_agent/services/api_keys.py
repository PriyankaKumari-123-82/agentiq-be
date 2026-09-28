from sqlmodel import Session

from ai_agent.models import ApiKey


def normalize_provider(provider: str | None) -> str:
    """Normalize provider names used by organization API-key routes."""
    return provider.strip().lower() if provider else ""


def get_api_key(db_session: Session, api_key_id: int | None) -> str | None:
    """Decrypt and return an active organization API key by its ID."""
    if api_key_id is None:
        return None
    key = db_session.get(ApiKey, api_key_id)
    if key and key.is_active and key.deleted_at is None:
        return key.decrypt()
    return None
