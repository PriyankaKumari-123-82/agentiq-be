"""
Integration tests for the Monitor Type catalog endpoint.

Endpoints covered:
  GET  /api/v1/monitor/type
"""
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from datetime import datetime, timezone

from ai_agent.models import MonitorType, User

PREFIX = "/api/v1"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    _engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import ai_agent.models  # noqa: F401
    SQLModel.metadata.create_all(_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture(scope="module")
def seeded_engine(engine):
    with Session(engine) as s:
        seed_user = User(email="seed@test.com", first_name="Seed")
        s.add(seed_user)
        s.flush()
        for mt in _make_monitor_types(seed_user.id):
            s.add(mt)
        s.commit()
    return engine


@pytest.fixture()
def client(seeded_engine):
    from ai_agent.core.db import get_session
    from ai_agent.core.deps import get_current_user
    from ai_agent.main import app

    def _override_session():
        with Session(seeded_engine) as session:
            yield session

    def _override_user():
        return User(id=9999, email="test@example.com", first_name="Test")

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_current_user] = _override_user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Seed helpers ──────────────────────────────────────────────────────────────

def _make_monitor_types(created_by: int) -> list[MonitorType]:
    now = datetime.now(timezone.utc)
    return [
        MonitorType(name="COUNT",           title="Count",           description="Returns a count.", result_type="NUMBER",  configuration_schema='{"type":"object"}', is_active=True,  created_by=created_by, created_at=now, updated_at=now),
        MonitorType(name="LIST",            title="List",            description="Returns rows.",   result_type="JSON",    configuration_schema='{"type":"object"}', is_active=True,  created_by=created_by, created_at=now, updated_at=now),
        MonitorType(name="TREND",           title="Trend",           description="Time-series.",    result_type="JSON",    configuration_schema="{}",                is_active=True,  created_by=created_by, created_at=now, updated_at=now),
        MonitorType(name="ALERT_THRESHOLD", title="Alert/Threshold", description="Bool result.",    result_type="BOOLEAN", configuration_schema="{}",                is_active=True,  created_by=created_by, created_at=now, updated_at=now),
        MonitorType(name="AI_QUERY",        title="AI Query",        description="Free text.",      result_type="TEXT",    configuration_schema="{}",                is_active=True,  created_by=created_by, created_at=now, updated_at=now),
        MonitorType(name="PIE_BREAKDOWN",   title="Pie Breakdown",   description="Pie data.",       result_type="JSON",    configuration_schema="{}",                is_active=True,  created_by=created_by, created_at=now, updated_at=now),
    ]


# ── GET /monitor/type ─────────────────────────────────────────────────────────

class TestListMonitorTypes:
    def test_returns_200(self, client):
        assert client.get(f"{PREFIX}/monitor/type").status_code == 200

    def test_returns_a_list(self, client):
        """Response is a plain list, not wrapped in a key."""
        resp = client.get(f"{PREFIX}/monitor/type")
        assert isinstance(resp.json(), list)

    def test_returns_all_types(self, client):
        resp = client.get(f"{PREFIX}/monitor/type")
        names = [item["name"] for item in resp.json()]
        for expected in ["COUNT", "LIST", "TREND", "ALERT_THRESHOLD", "AI_QUERY", "PIE_BREAKDOWN"]:
            assert expected in names

    def test_each_item_has_required_fields(self, client):
        resp = client.get(f"{PREFIX}/monitor/type")
        required = {"id", "name", "title", "description",
                    "configuration_schema", "result_type", "is_active",
                    "created_by", "created_at", "updated_at"}
        for item in resp.json():
            assert required.issubset(item.keys())

    def test_result_type_is_string(self, client):
        """result_type column exists and holds a string value."""
        resp = client.get(f"{PREFIX}/monitor/type")
        for item in resp.json():
            assert isinstance(item["result_type"], str)

    def test_id_is_integer(self, client):
        for item in client.get(f"{PREFIX}/monitor/type").json():
            assert isinstance(item["id"], int)

    def test_is_active_field_present(self, client):
        for item in client.get(f"{PREFIX}/monitor/type").json():
            assert "is_active" in item
            assert isinstance(item["is_active"], bool)

    def test_requires_auth(self, seeded_engine):
        from ai_agent.core.db import get_session
        from ai_agent.main import app

        def _override_session():
            with Session(seeded_engine) as session:
                yield session

        app.dependency_overrides[get_session] = _override_session
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get(f"{PREFIX}/monitor/type")
            assert resp.status_code in (401, 403, 422)
        finally:
            app.dependency_overrides.clear()
