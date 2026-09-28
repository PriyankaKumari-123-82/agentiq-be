"""
Integration tests for POST /api/v1/monitor/seed-from-db.

Auto-creates starter monitors (row-count + recent-rows) from the table
names exposed by the organization's connected DB MCP server, instead of
relying on static mock data in the Intelligence Hub.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from ai_agent.models import Monitor, MonitorResult, MonitorType, User

PREFIX = "/api/v1"


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


@pytest.fixture()
def db_session(engine):
    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture()
def org_user(db_session) -> User:
    user = User(
        email=f"user_{uuid.uuid4().hex[:8]}@test.com",
        first_name="Test",
        active_organization_id=1,
    )
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture()
def no_org_user(db_session) -> User:
    user = User(email=f"noorg_{uuid.uuid4().hex[:8]}@test.com", first_name="NoOrg")
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture()
def seed_monitor_types(db_session, org_user) -> None:
    """The real seed migration creates COUNT and LIST types — replicate here."""
    db_session.add(MonitorType(
        name="COUNT", title="Count", description="Returns a count.",
        result_type="NUMBER", configuration_schema="{}",
        is_active=True, created_by=org_user.id,
    ))
    db_session.add(MonitorType(
        name="LIST", title="List", description="Returns rows.",
        result_type="JSON", configuration_schema="{}",
        is_active=True, created_by=org_user.id,
    ))
    db_session.commit()


def _client(engine, user):
    from ai_agent.core.db import get_session
    from ai_agent.core.deps import get_current_user
    from ai_agent.main import app

    def _override_session():
        with Session(engine) as session:
            yield session

    def _override_user():
        return user

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_current_user] = _override_user
    return TestClient(app)


class TestSeedMonitorsFromDb:
    def test_400_when_no_active_organization(self, engine, no_org_user):
        with _client(engine, no_org_user) as c:
            resp = c.post(f"{PREFIX}/monitor/seed-from-db")
        assert resp.status_code == 400

    def test_empty_when_no_db_tables_found(self, engine, org_user, seed_monitor_types):
        with patch(
            "ai_agent.services.monitor_service.list_org_db_tables",
            new=AsyncMock(return_value=[]),
        ):
            with _client(engine, org_user) as c:
                resp = c.post(f"{PREFIX}/monitor/seed-from-db")
        assert resp.status_code == 201
        assert resp.json()["items"] == []

    def test_creates_monitors_from_table_names(self, engine, org_user, seed_monitor_types):
        with (
            patch(
                "ai_agent.services.monitor_service.list_org_db_tables",
                new=AsyncMock(return_value=["orders", "vendors"]),
            ),
            patch(
                "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
                new=AsyncMock(),
            ),
        ):
            with _client(engine, org_user) as c:
                resp = c.post(f"{PREFIX}/monitor/seed-from-db")

        assert resp.status_code == 201
        items = resp.json()["items"]
        names = {m["name"] for m in items}
        assert "orders — Row Count" in names
        assert "orders — Recent Rows" in names
        assert "vendors — Row Count" in names
        assert all(m["is_active"] for m in items)

    def test_is_idempotent_across_calls(self, engine, org_user, seed_monitor_types):
        with (
            patch(
                "ai_agent.services.monitor_service.list_org_db_tables",
                new=AsyncMock(return_value=["orders"]),
            ),
            patch(
                "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
                new=AsyncMock(),
            ),
        ):
            with _client(engine, org_user) as c:
                first = c.post(f"{PREFIX}/monitor/seed-from-db")
                second = c.post(f"{PREFIX}/monitor/seed-from-db")

        assert len(first.json()["items"]) > 0
        assert second.json()["items"] == []

    def test_new_monitors_appear_in_list_endpoint_once_they_succeed(
        self, engine, org_user, seed_monitor_types,
    ):
        """Seeded monitors reach the hub only after their background run works.

        execute_monitor is mocked out here, so nothing succeeds on its own —
        the SUCCESS result is written by hand to stand in for that run.
        """
        with (
            patch(
                "ai_agent.services.monitor_service.list_org_db_tables",
                new=AsyncMock(return_value=["shipments"]),
            ),
            patch(
                "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
                new=AsyncMock(),
            ),
        ):
            with _client(engine, org_user) as c:
                created = c.post(f"{PREFIX}/monitor/seed-from-db").json()["items"]
                before = c.get(f"{PREFIX}/monitor").json()["items"]

                row_count = next(
                    m for m in created if m["name"] == "shipments — Row Count"
                )
                with Session(engine) as s:
                    s.add(MonitorResult(
                        monitor_id=row_count["id"],
                        executed_at=datetime.now(timezone.utc),
                        status="SUCCESS",
                        execution_time_ms=1,
                        result_type="NUMBER",
                        result='{"value": 12}',
                        is_active=True,
                    ))
                    s.commit()

                after = c.get(f"{PREFIX}/monitor").json()["items"]

        assert "shipments — Row Count" not in {m["name"] for m in before}
        assert "shipments — Row Count" in {m["name"] for m in after}
