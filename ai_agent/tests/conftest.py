"""
Shared pytest fixtures for the FAQ module test suite.

Uses an in-memory SQLite database so tests run without any external dependencies.
All heavy I/O (LLM calls, MCP connections) is patched at the service boundary.
"""
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

# In-memory SQLite engine — created once per test session

SQLITE_URL = "sqlite://"  # in-memory


@pytest.fixture(scope="session")
def engine():
    """Create a shared in-memory SQLite engine with all tables."""
    _engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Import all models so SQLModel.metadata is populated
    import ai_agent.models  # noqa: F401
    SQLModel.metadata.create_all(_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture()
def db(engine) -> Generator[Session, None, None]:
    """Yield a fresh session for each test; roll back changes afterwards."""
    with Session(engine) as session:
        yield session
        session.rollback()

# Domain object factories

def make_user(db: Session, *, email: str | None = None) -> "User":  # type: ignore[name-defined]
    from ai_agent.models import User

    user = User(
        email=email or f"user_{uuid.uuid4().hex[:8]}@test.com",
        first_name="Test",
        last_name="User",
    )
    db.add(user)
    db.flush()
    return user


def make_agent(db: Session, owner_id: int, *, cold_build: bool = False) -> "AIAgent":  # type: ignore[name-defined]
    from ai_agent.models import AIAgent

    agent = AIAgent(
        id=str(uuid.uuid4()),
        name="Test Agent",
        description="A test agent",
        owner_id=owner_id,
        is_cold_build=cold_build,
    )
    db.add(agent)
    db.flush()
    return agent


def make_agent_settings(db: Session, agent_id: str, **kwargs) -> "AIAgentSettings":  # type: ignore[name-defined]
    from ai_agent.models import AIAgentSettings

    defaults = dict(
        agent_id=agent_id,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.7,
        llm_max_tokens=1024,
        llm_system_prompt="You are a helpful assistant.",
    )
    defaults.update(kwargs)
    settings = AIAgentSettings(**defaults)
    db.add(settings)
    db.flush()
    return settings


def make_faq(
    db: Session,
    agent_id: str,
    question: str = "Test question?",
    source: "FAQSource" = "agent_config",  # type: ignore[name-defined]
    version: int = 1,
) -> "AgentFAQ":  # type: ignore[name-defined]
    """Create and persist an AgentFAQ."""
    from ai_agent.models import AgentFAQ

    faq = AgentFAQ(
        agent_id=agent_id,
        question=question,
        source=source,
        version=version,
    )
    db.add(faq)
    db.flush()
    return faq


def make_session_with_messages(
    db: Session,
    user_id: int,
    agent_id: str,
    messages: list[str],
) -> None:
    """Create a user session and a set of user messages for history-based FAQ tests."""
    from ai_agent.models import Message, UserSession

    session = UserSession(
        user_id=user_id,
        agent_id=agent_id,
        session_id=str(uuid.uuid4()),
    )
    db.add(session)
    db.flush()

    for i, content in enumerate(messages):
        msg = Message(
            session_id=session.id,
            agent_id=agent_id,
            conversation_id=str(uuid.uuid4()),
            role="user",
            content=content,
            counter=i,
        )
        db.add(msg)
    db.flush()

# Re-usable pytest fixtures

@pytest.fixture()
def user(db):
    return make_user(db)


@pytest.fixture()
def agent(db, user):
    return make_agent(db, user.id)


@pytest.fixture()
def agent_settings(db, agent):
    return make_agent_settings(db, agent.id)


@pytest.fixture()
def cold_agent(db, user):
    return make_agent(db, user.id, cold_build=True)


@pytest.fixture()
def cold_agent_settings(db, cold_agent):
    return make_agent_settings(db, cold_agent.id)

# LLM mock — avoids real API calls in every test

@pytest.fixture()
def mock_llm(monkeypatch):
    """
    Patch get_llm_gateway() so every chat() call returns a canned list of 6
    FAQ questions as valid JSON. Tests that need different output can
    override ``mock_llm.chat.return_value.content`` directly.
    """
    import json

    default_questions = [
        "How do I check inventory?",
        "What is the current order status?",
        "Can you generate a sales report?",
        "How many active customers are there?",
        "What products are low in stock?",
        "Show me this week's revenue",
    ]
    response = MagicMock()
    response.content = json.dumps(default_questions)

    gateway = MagicMock()
    gateway.chat = AsyncMock(return_value=response)

    monkeypatch.setattr(
        "ai_agent.workflows.faq.faq.get_llm_gateway",
        lambda: gateway,
    )
    return gateway

# FastAPI TestClient with DB override

@pytest.fixture()
def client(engine):
    """
    Returns a TestClient backed by the in-memory SQLite engine.
    Security dependency is bypassed so tests don't need real JWTs.
    """
    from fastapi.security import HTTPAuthorizationCredentials
    from ai_agent.core.db import get_session
    from ai_agent.core.token_validation import security
    from ai_agent.main import app

    def override_get_session():
        with Session(engine) as session:
            yield session

    def override_security():
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token")

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[security] = override_security

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
