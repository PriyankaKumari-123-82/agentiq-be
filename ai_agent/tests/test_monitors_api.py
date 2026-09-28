"""
Integration tests for Monitor CRUD endpoints.

Endpoints covered:
  POST   /api/v1/monitor
  GET    /api/v1/monitor          (paginated: ?page=1&limit=20&search=)
  GET    /api/v1/monitor/{id}
  PATCH  /api/v1/monitor/{id}
  DELETE /api/v1/monitor/{id}
"""
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from ai_agent.models import Monitor, MonitorResult, MonitorType, User

PREFIX = "/api/v1"


# ── Engine + fixtures ─────────────────────────────────────────────────────────

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
def test_user(db_session) -> User:
    user = User(email=f"user_{uuid.uuid4().hex[:8]}@test.com", first_name="Test")
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture()
def other_user(db_session) -> User:
    user = User(email=f"other_{uuid.uuid4().hex[:8]}@test.com", first_name="Other")
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture()
def monitor_type(db_session, test_user) -> MonitorType:
    now = datetime.now(timezone.utc)
    mt = MonitorType(
        name=f"COUNT_{uuid.uuid4().hex[:4].upper()}",
        title="Count",
        description="Returns a count.",
        result_type="NUMBER",
        configuration_schema="{}",
        is_active=True,
        created_by=test_user.id,
        created_at=now,
        updated_at=now,
    )
    db_session.add(mt)
    db_session.flush()
    return mt


@pytest.fixture()
def client(engine, test_user):
    from ai_agent.core.db import get_session
    from ai_agent.core.deps import get_current_user
    from ai_agent.main import app

    def _override_session():
        with Session(engine) as session:
            yield session

    def _override_user():
        return test_user

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_current_user] = _override_user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def _mark_succeeded(engine, monitor_id: int) -> None:
    """Give a monitor a SUCCESS result so GET /monitor considers it finished."""
    with Session(engine) as s:
        s.add(MonitorResult(
            monitor_id=monitor_id,
            executed_at=datetime.now(timezone.utc),
            status="SUCCESS",
            execution_time_ms=1,
            result_type="NUMBER",
            result='{"value": 1}',
            is_active=True,
        ))
        s.commit()


def _valid_body(monitor_type_id: int, **overrides) -> dict:
    """Build a valid MonitorCreate payload — monitors are org-scoped, not agent-scoped."""
    payload = {
        "name": "Daily Sales Count",
        "monitor_type": monitor_type_id,
        "natural_query": "How many sales happened today?",
        "refresh_interval": 3600,
    }
    payload.update(overrides)
    return payload


# ── POST /monitor ─────────────────────────────────────────────────────────────

class TestCreateMonitor:
    def test_returns_201(self, client, monitor_type):
        assert client.post(f"{PREFIX}/monitor",
                           json=_valid_body(monitor_type.id)).status_code == 201

    def test_id_is_integer(self, client, monitor_type):
        resp = client.post(f"{PREFIX}/monitor", json=_valid_body(monitor_type.id))
        assert isinstance(resp.json()["id"], int)

    def test_is_active_true_by_default(self, client, monitor_type):
        resp = client.post(f"{PREFIX}/monitor", json=_valid_body(monitor_type.id))
        assert resp.json()["is_active"] is True

    def test_scheduler_frequency_set_from_refresh_interval(self, client, monitor_type):
        """refresh_interval in the request body maps onto scheduler_frequency."""
        resp = client.post(f"{PREFIX}/monitor",
                           json=_valid_body(monitor_type.id, refresh_interval=7200))
        assert resp.json()["scheduler_frequency"] == 7200

    def test_strips_whitespace_from_name(self, client, monitor_type):
        resp = client.post(f"{PREFIX}/monitor",
                           json=_valid_body(monitor_type.id, name="  My Monitor  "))
        assert resp.json()["name"] == "My Monitor"

    def test_404_for_unknown_monitor_type(self, client):
        resp = client.post(f"{PREFIX}/monitor", json=_valid_body(99999))
        assert resp.status_code == 404
        assert "monitor type" in resp.json()["detail"].lower()

    def test_422_when_name_missing(self, client, monitor_type):
        body = _valid_body(monitor_type.id)
        body.pop("name")
        assert client.post(f"{PREFIX}/monitor", json=body).status_code == 422

    def test_422_when_refresh_interval_below_60(self, client, monitor_type):
        assert client.post(f"{PREFIX}/monitor",
                           json=_valid_body(monitor_type.id, refresh_interval=30)).status_code == 422


# ── GET /monitor ──────────────────────────────────────────────────────────────

class TestListMonitors:
    def test_returns_200(self, client):
        assert client.get(f"{PREFIX}/monitor").status_code == 200

    def test_response_shape(self, client):
        """Response matches PaginatedMonitorsOut: items, page, limit, total, has_more."""
        resp = client.get(f"{PREFIX}/monitor").json()
        assert all(k in resp for k in ("items", "page", "limit", "total", "has_more"))

    def test_items_is_a_list(self, client):
        assert isinstance(client.get(f"{PREFIX}/monitor").json()["items"], list)

    def test_excludes_inactive(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id, name="ToDelete")).json()["id"]
        client.delete(f"{PREFIX}/monitor/{monitor_id}")
        ids = [m["id"] for m in client.get(f"{PREFIX}/monitor").json()["items"]]
        assert monitor_id not in ids

    def test_returns_only_current_users_monitors(self, client, engine, other_user, monitor_type):
        with Session(engine) as s:
            now = datetime.now(timezone.utc)
            other = Monitor(name="Other", monitor_type_id=monitor_type.id,
                            natural_query="q", created_by=other_user.id, created_at=now, updated_at=now)
            s.add(other)
            s.commit()
            other_id = other.id

        ids = [m["id"] for m in client.get(f"{PREFIX}/monitor").json()["items"]]
        assert other_id not in ids

    def test_pagination_defaults(self, client):
        resp = client.get(f"{PREFIX}/monitor").json()
        assert resp["page"] == 1
        assert resp["limit"] == 10

    def test_pagination_page_param(self, client, monitor_type):
        resp = client.get(f"{PREFIX}/monitor?page=1&limit=5")
        assert resp.status_code == 200
        assert resp.json()["limit"] == 5
        assert len(resp.json()["items"]) <= 5

    def test_total_and_has_more_fields(self, client):
        resp = client.get(f"{PREFIX}/monitor").json()
        assert isinstance(resp["total"], int)
        assert isinstance(resp["has_more"], bool)

    def test_search_param_filters_by_name(self, client, engine, monitor_type):
        unique = f"SearchableMonitor_{uuid.uuid4().hex[:6]}"
        monitor_id = client.post(
            f"{PREFIX}/monitor", json=_valid_body(monitor_type.id, name=unique),
        ).json()["id"]
        _mark_succeeded(engine, monitor_id)
        resp = client.get(f"{PREFIX}/monitor?search={unique}")
        names = [m["name"] for m in resp.json()["items"]]
        assert unique in names

    def test_excludes_monitor_that_never_succeeded(self, client, monitor_type):
        """A monitor with no SUCCESS result isn't finished — the hub skips it."""
        monitor_id = client.post(
            f"{PREFIX}/monitor", json=_valid_body(monitor_type.id, name="NeverRan"),
        ).json()["id"]
        ids = [m["id"] for m in client.get(f"{PREFIX}/monitor").json()["items"]]
        assert monitor_id not in ids

    def test_excludes_monitor_awaiting_followup(self, client, engine, monitor_type):
        """A monitor still asking the user a question must not reach the hub."""
        monitor_id = client.post(
            f"{PREFIX}/monitor", json=_valid_body(monitor_type.id, name="NeedsInfo"),
        ).json()["id"]
        _mark_succeeded(engine, monitor_id)
        with Session(engine) as s:
            m = s.get(Monitor, monitor_id)
            m.followup_question = "Which date range did you mean?"
            s.add(m)
            s.commit()

        ids = [m["id"] for m in client.get(f"{PREFIX}/monitor").json()["items"]]
        assert monitor_id not in ids

    def test_search_returns_empty_items_for_no_match(self, client):
        resp = client.get(f"{PREFIX}/monitor?search=ZZZNONEXISTENT999")
        assert resp.json()["items"] == []
        assert resp.json()["total"] == 0


# ── GET /monitor/{id} ─────────────────────────────────────────────────────────

class TestGetMonitor:
    def test_returns_200(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        assert client.get(f"{PREFIX}/monitor/{monitor_id}").status_code == 200

    def test_returns_correct_data(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id, name="UniqueXYZ")).json()["id"]
        resp = client.get(f"{PREFIX}/monitor/{monitor_id}")
        assert resp.json()["name"] == "UniqueXYZ"

    def test_404_for_unknown_id(self, client):
        assert client.get(f"{PREFIX}/monitor/999999").status_code == 404

    def test_404_after_delete(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        client.delete(f"{PREFIX}/monitor/{monitor_id}")
        assert client.get(f"{PREFIX}/monitor/{monitor_id}").status_code == 404

    def test_404_for_other_users_monitor(self, client, engine, other_user, monitor_type):
        with Session(engine) as s:
            now = datetime.now(timezone.utc)
            other = Monitor(name="Private", monitor_type_id=monitor_type.id,
                            natural_query="private", created_by=other_user.id,
                            created_at=now, updated_at=now)
            s.add(other)
            s.commit()
            other_id = other.id
        assert client.get(f"{PREFIX}/monitor/{other_id}").status_code == 404


# ── PATCH /monitor/{id} ───────────────────────────────────────────────────────

class TestUpdateMonitor:
    def test_returns_200(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        assert client.patch(f"{PREFIX}/monitor/{monitor_id}", json={"name": "Updated"}).status_code == 200

    def test_updates_name(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        client.patch(f"{PREFIX}/monitor/{monitor_id}", json={"name": "New Name"})
        assert client.get(f"{PREFIX}/monitor/{monitor_id}").json()["name"] == "New Name"

    def test_updates_refresh_interval(self, client, monitor_type):
        """PATCH refresh_interval maps onto scheduler_frequency without crashing."""
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        resp = client.patch(f"{PREFIX}/monitor/{monitor_id}", json={"refresh_interval": 1800})
        assert resp.status_code == 200
        assert resp.json()["scheduler_frequency"] == 1800

    def test_changing_query_clears_scheduler_frequency(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id, refresh_interval=3600)).json()["id"]
        client.patch(f"{PREFIX}/monitor/{monitor_id}", json={"natural_query": "How many users?"})
        assert client.get(f"{PREFIX}/monitor/{monitor_id}").json()["scheduler_frequency"] is None

    def test_empty_patch_is_no_op(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id, name="Stable")).json()["id"]
        resp = client.patch(f"{PREFIX}/monitor/{monitor_id}", json={})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Stable"

    def test_404_for_bad_monitor_type(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        assert client.patch(f"{PREFIX}/monitor/{monitor_id}",
                            json={"monitor_type": 99999}).status_code == 404

    def test_422_for_bad_refresh_interval(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        assert client.patch(f"{PREFIX}/monitor/{monitor_id}",
                            json={"refresh_interval": 5}).status_code == 422


# ── DELETE /monitor/{id} ──────────────────────────────────────────────────────

class TestDeleteMonitor:
    def test_returns_204(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        assert client.delete(f"{PREFIX}/monitor/{monitor_id}").status_code == 204

    def test_sets_is_active_false_in_db(self, client, engine, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        client.delete(f"{PREFIX}/monitor/{monitor_id}")
        with Session(engine) as s:
            m = s.get(Monitor, monitor_id)
            assert m.is_active is False

    def test_deleted_not_in_list(self, client, monitor_type):
        monitor_id = client.post(f"{PREFIX}/monitor",
                                  json=_valid_body(monitor_type.id)).json()["id"]
        client.delete(f"{PREFIX}/monitor/{monitor_id}")
        ids = [m["id"] for m in client.get(f"{PREFIX}/monitor").json()["items"]]
        assert monitor_id not in ids

    def test_404_for_unknown_id(self, client):
        assert client.delete(f"{PREFIX}/monitor/999999").status_code == 404

    def test_404_for_other_users_monitor(self, client, engine, other_user, monitor_type):
        with Session(engine) as s:
            now = datetime.now(timezone.utc)
            other = Monitor(name="Not mine", monitor_type_id=monitor_type.id,
                            natural_query="private", created_by=other_user.id,
                            created_at=now, updated_at=now)
            s.add(other)
            s.commit()
            other_id = other.id
        assert client.delete(f"{PREFIX}/monitor/{other_id}").status_code == 404
