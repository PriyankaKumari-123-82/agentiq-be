"""
Integration tests for monitor execution endpoints.

Endpoints covered:
  POST  /api/v1/monitor/{id}/run       Trigger execution (202 + background task)
  GET   /api/v1/monitor/{id}/results   Paginated execution history

Unit tests covered:
  _format_result()                     Result formatting by result_type
  load_monitor_with_context()          Service layer loading
  persist_monitor_result()             Service layer write
  get_latest_result()                  Service layer read
  get_result_history()                 Service layer read + pagination
  execute_monitor()                    Full workflow (LLM calls mocked)
"""
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from ai_agent.custom_model.monitor import Monitor, MonitorResult, MonitorType
from ai_agent.models import User

PREFIX = "/api/v1"


# ── Shared engine + fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    _engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import ai_agent.models  # noqa: F401 — registers all tables
    import ai_agent.custom_model.monitor  # noqa: F401
    SQLModel.metadata.create_all(_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture()
def db_session(engine):
    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture()
def user(db_session) -> User:
    u = User(email=f"u_{uuid.uuid4().hex[:8]}@test.com", first_name="Test")
    db_session.add(u)
    db_session.flush()
    return u


@pytest.fixture()
def monitor_type(db_session, user) -> MonitorType:
    mt = MonitorType(
        name=f"COUNT_{uuid.uuid4().hex[:4].upper()}",
        title="Count",
        description="Returns a row count.",
        result_type="NUMBER",
        configuration_schema="{}",
        is_active=True,
        created_by=user.id,
    )
    db_session.add(mt)
    db_session.flush()
    return mt


@pytest.fixture()
def monitor(db_session, user, monitor_type) -> Monitor:
    m = Monitor(
        name="Daily Sales Count",
        monitor_type_id=monitor_type.id,
        natural_query="How many sales happened today?",
        scheduler_frequency=3600,
        is_active=True,
        created_by=user.id,
    )
    db_session.add(m)
    db_session.flush()
    return m


@pytest.fixture()
def monitor_result(db_session, monitor, monitor_type) -> MonitorResult:
    r = MonitorResult(
        monitor_id=monitor.id,
        result_type="NUMBER",
        result=json.dumps({"value": 42}),
        status="SUCCESS",
        is_active=True,
    )
    db_session.add(r)
    db_session.flush()
    return r


@pytest.fixture()
def client(engine, user):
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

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── POST /monitor/{id}/run ────────────────────────────────────────────────────

class TestRunMonitor:
    def test_returns_202(self, client, monitor):
        with patch("ai_agent.workflows.monitor.monitor_agent.execute_monitor", new=AsyncMock()):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run")
        assert resp.status_code == 202

    def test_response_contains_monitor_id(self, client, monitor):
        with patch("ai_agent.workflows.monitor.monitor_agent.execute_monitor", new=AsyncMock()):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run")
        assert resp.json()["monitor_id"] == monitor.id

    def test_response_contains_message(self, client, monitor):
        with patch("ai_agent.workflows.monitor.monitor_agent.execute_monitor", new=AsyncMock()):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run")
        assert "message" in resp.json()
        assert len(resp.json()["message"]) > 0

    def test_404_for_unknown_monitor(self, client):
        resp = client.post(f"{PREFIX}/monitor/999999/run")
        assert resp.status_code == 404

    def test_404_for_inactive_monitor(self, client, engine, user, monitor_type):
        """Soft-deleted monitor returns 404 on /run."""
        with Session(engine) as s:
            m = Monitor(
                name="Deleted",
                monitor_type_id=monitor_type.id,
                natural_query="q",
                created_by=user.id,
                is_active=False,
            )
            s.add(m)
            s.commit()
            deleted_id = m.id
        resp = client.post(f"{PREFIX}/monitor/{deleted_id}/run")
        assert resp.status_code == 404

    def test_sync_run_returns_200_and_success_status(self, client, monitor):
        """?sync=true awaits the run so the caller sees the real outcome."""
        with patch(
            "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
            new=AsyncMock(return_value={"status": "success", "result": None}),
        ):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run?sync=true")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_sync_run_surfaces_error_without_500(self, client, monitor):
        """A failed run is reported as a normal 200 payload with status=error,
        so the UI can preview it instead of hitting an exception page.
        """
        with patch(
            "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
            new=AsyncMock(return_value={
                "status": "error",
                "error": "Table 'not_crm.sales' doesn't exist",
                "result": None,
            }),
        ):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run?sync=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "doesn't exist" in body["error"]

    def test_sync_run_surfaces_followup(self, client, monitor):
        with patch(
            "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
            new=AsyncMock(return_value={
                "status": "followup",
                "followup_question": "Which quarter?",
                "result": None,
            }),
        ):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run?sync=true")
        assert resp.json()["status"] == "followup"
        assert resp.json()["followup_question"] == "Which quarter?"

    def test_sync_run_converts_unexpected_exception_into_error_payload(self, client, monitor):
        with patch(
            "ai_agent.workflows.monitor.monitor_agent.execute_monitor",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            resp = client.post(f"{PREFIX}/monitor/{monitor.id}/run?sync=true")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
        assert "boom" in resp.json()["error"]

    def test_404_for_other_users_monitor(self, client, engine, monitor_type):
        """Cannot trigger another user's monitor."""
        other = User(email=f"other_{uuid.uuid4().hex[:8]}@test.com", first_name="Other")
        with Session(engine) as s:
            s.add(other)
            s.flush()
            m = Monitor(
                name="Theirs",
                monitor_type_id=monitor_type.id,
                natural_query="q",
                created_by=other.id,
                is_active=True,
            )
            s.add(m)
            s.commit()
            other_id = m.id
        assert client.post(f"{PREFIX}/monitor/{other_id}/run").status_code == 404


# ── GET /monitor/{id}/results ─────────────────────────────────────────────────

class TestListMonitorResults:
    def test_returns_200(self, client, monitor, monitor_result):
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results")
        assert resp.status_code == 200

    def test_response_shape(self, client, monitor, monitor_result):
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results").json()
        assert all(k in resp for k in ("items", "page", "limit", "total", "has_more"))

    def test_items_is_a_list(self, client, monitor, monitor_result):
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results").json()
        assert isinstance(resp["items"], list)

    def test_result_appears_in_items(self, client, monitor, monitor_result):
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results").json()
        ids = [r["id"] for r in resp["items"]]
        assert monitor_result.id in ids

    def test_default_pagination(self, client, monitor, monitor_result):
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results").json()
        assert resp["page"] == 1
        assert resp["limit"] == 20

    def test_limit_param(self, client, engine, monitor, monitor_type, user):
        """Create 5 results; request limit=2 and check items <= 2."""
        with Session(engine) as s:
            for _ in range(5):
                s.add(MonitorResult(
                    monitor_id=monitor.id,
                    result_type="NUMBER",
                    result=json.dumps({"value": 1}),
                    status="SUCCESS",
                    is_active=True,
                ))
            s.commit()
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results?limit=2").json()
        assert len(resp["items"]) <= 2
        assert resp["limit"] == 2

    def test_has_more_when_results_exceed_limit(self, client, engine, monitor):
        """has_more=True when total > limit."""
        with Session(engine) as s:
            for _ in range(6):
                s.add(MonitorResult(
                    monitor_id=monitor.id,
                    result_type="NUMBER",
                    result=json.dumps({"value": 0}),
                    status="SUCCESS",
                    is_active=True,
                ))
            s.commit()
        resp = client.get(f"{PREFIX}/monitor/{monitor.id}/results?limit=2").json()
        assert resp["has_more"] is True

    def test_returns_newest_first(self, client, engine, monitor):
        with Session(engine) as s:
            for i in range(3):
                s.add(MonitorResult(
                    monitor_id=monitor.id,
                    result_type="NUMBER",
                    result=json.dumps({"value": i}),
                    status="SUCCESS",
                    is_active=True,
                ))
            s.commit()
        items = client.get(f"{PREFIX}/monitor/{monitor.id}/results").json()["items"]
        ids = [r["id"] for r in items]
        assert ids == sorted(ids, reverse=True)

    def test_404_for_unknown_monitor(self, client):
        assert client.get(f"{PREFIX}/monitor/999999/results").status_code == 404

    def test_404_for_other_users_monitor(self, client, engine, monitor_type):
        other = User(email=f"o_{uuid.uuid4().hex[:6]}@test.com", first_name="X")
        with Session(engine) as s:
            s.add(other)
            s.flush()
            m = Monitor(
                name="Private",
                monitor_type_id=monitor_type.id,
                natural_query="q",
                created_by=other.id,
                is_active=True,
            )
            s.add(m)
            s.commit()
            other_monitor_id = m.id
        assert client.get(f"{PREFIX}/monitor/{other_monitor_id}/results").status_code == 404


# ── Unit: _format_result ──────────────────────────────────────────────────────

class TestFormatResult:
    """Tests for the _format_result() helper in workflows/monitor/monitor_agent.py.

    Current signature: _format_result(monitor_type: MonitorType, raw_text: str) -> dict
    """

    def _make_monitor_type(self, result_type: str, configuration_schema: str = "{}") -> MonitorType:
        return MonitorType(
            id=1,
            name="T",
            title="T",
            description="",
            configuration_schema=configuration_schema,
            result_type=result_type,
            is_active=True,
            created_by=1,
        )

    def _make_chart_monitor_type(self, chart_type: str) -> MonitorType:
        return self._make_monitor_type(
            "JSON",
            json.dumps({"format": "chart", "chart_type": chart_type, "fields": ["label", "value"]}),
        )

    def test_number_from_plain_string(self):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        result = _format_result(self._make_monitor_type("NUMBER"), "42")
        assert result == {"value": 42.0}

    def test_number_falls_back_to_value_key(self):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        result = _format_result(self._make_monitor_type("NUMBER"), json.dumps({"value": 99}))
        assert result["value"] == 99.0

    def test_number_from_single_cell_row(self):
        """SELECT COUNT(*) via a DB tool comes back as [{"count": 5}]."""
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        result = _format_result(self._make_monitor_type("NUMBER"), json.dumps([{"count": 5}]))
        assert result == {"value": 5.0}

    def test_number_unparseable_raises(self):
        """Prose must never be stored as if it were a NUMBER result."""
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        raw = "The latest sales figure could not be determined from the table."
        with pytest.raises(MonitorExecutionError, match="expects a single number"):
            _format_result(self._make_monitor_type("NUMBER"), raw)

    def test_refusal_raises_even_for_text_monitors(self):
        """'I'm unable to access...' is a failure, not a TEXT answer."""
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        raw = "I'm unable to access specific data for the last 10 days."
        with pytest.raises(MonitorExecutionError, match="could not answer"):
            _format_result(self._make_monitor_type("TEXT"), raw)

    def test_long_text_answer_survives_a_mid_prose_refusal_phrase(self):
        """A real summary may mention 'no data available' without being a refusal.

        The marker scan only covers the opening, and a TEXT answer long enough
        to carry real content is trusted — otherwise a valid AI_QUERY summary
        gets thrown away because of one unlucky phrase.
        """
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        raw = (
            "Vendor performance improved 12% this quarter. Global Tech Supply "
            "led on delivery at 93%, followed by ABC Engineering at 90%. "
            "Two vendors were excluded because no data available was recorded "
            "for them in the source system during this period. Overall spend "
            "fell 4% while on-time delivery rose, which suggests the "
            "renegotiated contracts are holding. Prime Components remains the "
            "main risk, with three late shipments logged in the last month."
        )
        assert _format_result(self._make_monitor_type("TEXT"), raw)["value"] == raw

    def test_short_pure_refusal_still_raises_for_text(self):
        """A refusal that IS the whole answer is still a failure."""
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="could not answer"):
            _format_result(self._make_monitor_type("TEXT"), "I do not have access to that.")

    def test_refusal_scan_only_covers_the_opening(self):
        """A marker past the scan window is prose, not a refusal."""
        from ai_agent.workflows.monitor.monitor_agent import _looks_like_refusal
        assert _looks_like_refusal("I'm unable to reach the database.")
        assert not _looks_like_refusal("Sales rose. " + "x" * 200 + " i cannot say more")

    def test_boolean_rejects_non_boolean_prose(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="yes/no"):
            _format_result(self._make_monitor_type("BOOLEAN"), "maybe, it depends")

    def test_json_rejects_unparseable_output(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="structured/tabular"):
            _format_result(self._make_monitor_type("JSON"), "not json at all")

    def test_boolean_true(self):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        result = _format_result(self._make_monitor_type("BOOLEAN"), "true")
        assert result["value"] is True
        assert "message" in result

    def test_boolean_false_for_unknown_value(self):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        result = _format_result(self._make_monitor_type("BOOLEAN"), "no")
        assert result["value"] is False

    def test_text_wraps_response(self):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        result = _format_result(self._make_monitor_type("TEXT"), "Sales are up 10%")
        assert result == {"value": "Sales are up 10%"}

    def test_json_parses_valid_json(self):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        payload = {"key": "val"}
        result = _format_result(self._make_monitor_type("JSON"), json.dumps(payload))
        assert result == {"value": payload}

    def test_empty_raw_raises(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="no output"):
            _format_result(self._make_monitor_type("TEXT"), "")


class TestChartFormatting:
    """PIE_CHART / BAR_CHART / LINE_CHART monitor types wrap JSON rows into
    {"chartType": ..., "data": [...]} for the Intelligence Hub's chart renderer.
    """

    def _make_chart_monitor_type(self, chart_type: str) -> MonitorType:
        return MonitorType(
            id=1, name="T", title="T", description="",
            configuration_schema=json.dumps(
                {"format": "chart", "chart_type": chart_type, "fields": ["label", "value"]}
            ),
            result_type="JSON", is_active=True, created_by=1,
        )

    @pytest.mark.parametrize("chart_type", ["pie", "bar", "line"])
    def test_wraps_bare_row_list(self, chart_type):
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        rows = [{"category": "Ghost", "count": 3}, {"category": "Poison", "count": 1}]
        result = _format_result(self._make_chart_monitor_type(chart_type), json.dumps(rows))
        assert result == {"value": {"chartType": chart_type, "data": rows}}

    def test_unwraps_already_shaped_chart_json(self):
        """If the LLM already returned {"chartType": ..., "data": [...]}, the data
        is kept but chart_type is always taken from the monitor's own config —
        never trusted from the model.
        """
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        raw = json.dumps({"chartType": "totally_wrong", "data": [{"a": "x", "b": 1}]})
        result = _format_result(self._make_chart_monitor_type("pie"), raw)
        assert result == {"value": {"chartType": "pie", "data": [{"a": "x", "b": 1}]}}

    def test_non_chart_json_monitor_is_unaffected(self):
        """A plain JSON (LIST) monitor type must not be chart-wrapped."""
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        rows = [{"a": 1}, {"a": 2}]
        result = _format_result(self._make_monitor_type_helper(), json.dumps(rows))
        assert result == {"value": rows}

    def test_rejects_non_list_data(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="did not return a list"):
            _format_result(self._make_chart_monitor_type("bar"), json.dumps({"count": 5}))

    def test_rejects_empty_row_list(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="no rows to chart"):
            _format_result(self._make_chart_monitor_type("bar"), "[]")

    def test_rejects_single_column_rows(self):
        """A row needs a label AND a value — one column isn't chartable."""
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _format_result,
        )
        with pytest.raises(MonitorExecutionError, match="label and a numeric value"):
            _format_result(self._make_chart_monitor_type("pie"), json.dumps([{"category": "Ghost"}]))

    def test_get_chart_type_returns_none_for_non_chart_monitor(self):
        from ai_agent.workflows.monitor.monitor_agent import _get_chart_type
        assert _get_chart_type(self._make_monitor_type_helper()) is None

    def _make_monitor_type_helper(self):
        return MonitorType(
            id=1, name="T", title="T", description="",
            configuration_schema="{}", result_type="JSON",
            is_active=True, created_by=1,
        )

    def test_chart_instructions_empty_for_non_chart_monitor(self):
        from ai_agent.workflows.monitor.monitor_agent import _chart_instructions
        assert _chart_instructions(self._make_monitor_type_helper()) == ""

    def test_chart_instructions_mentions_chart_type(self):
        from ai_agent.workflows.monitor.monitor_agent import _chart_instructions
        text = _chart_instructions(self._make_chart_monitor_type("line"))
        assert "line chart" in text


# ── Unit: schema handling / never asking the user for table names ─────────────

class TestSchemaQuestionGuard:
    """A business user must never be asked which table or column to use —
    the agent is handed the schema and has to resolve that itself.
    """

    @pytest.mark.parametrize("question", [
        "Could you specify which table contains the repair statuses and their turnaround times?",
        "What table holds the sales data?",
        "Which column stores the order total?",
        "Please confirm the table name for vendors.",
        "Which database should I query?",
    ])
    def test_detects_schema_questions(self, question):
        from ai_agent.workflows.monitor.monitor_agent import _is_schema_question
        assert _is_schema_question(question) is True

    @pytest.mark.parametrize("question", [
        "Which quarter's sales data would you like?",
        "Do you want open or closed purchase orders?",
        "What date range should I use?",
        "",
    ])
    def test_allows_genuine_business_questions(self, question):
        from ai_agent.workflows.monitor.monitor_agent import _is_schema_question
        assert _is_schema_question(question) is False

    @pytest.mark.asyncio
    async def test_schema_question_triggers_retry_and_can_recover(
        self, db_session, monitor, monitor_type
    ):
        """First reply asks which table; the agent pushes back and the retry
        produces a real query, which is what the user sees.
        """
        from ai_agent.workflows.monitor.monitor_agent import MonitorAgent

        agent = MonitorAgent(
            monitor=monitor, monitor_type=monitor_type,
            mcp_tools=[{
                "tool_name": "run_sql", "server_name": "srv",
                "requires_query_building": True, "input_schema": {}, "description": "",
            }],
            db_session=db_session, api_key="k",
            schema_context="Table: svc_job (id int, status varchar)",
        )

        replies = [
            json.dumps({"followup": "Which table contains the repair statuses?"}),
            json.dumps({"query": "SELECT COUNT(*) FROM svc_job", "tool_name": "run_sql"}),
        ]
        with patch(
            "ai_agent.workflows.monitor.monitor_agent._call_llm",
            new=AsyncMock(side_effect=replies),
        ) as mock_llm:
            result = await agent.generate_query()

        assert mock_llm.await_count == 2
        assert result["status"] == "query"
        assert result["query"] == "SELECT COUNT(*) FROM svc_job"

    @pytest.mark.asyncio
    async def test_persistent_schema_question_becomes_error_not_a_user_prompt(
        self, db_session, monitor, monitor_type
    ):
        """If it keeps asking, fail with a useful message rather than making
        the user name their own tables.
        """
        from ai_agent.workflows.monitor.monitor_agent import MonitorAgent

        agent = MonitorAgent(
            monitor=monitor, monitor_type=monitor_type,
            mcp_tools=[{
                "tool_name": "run_sql", "server_name": "srv",
                "requires_query_building": True, "input_schema": {}, "description": "",
            }],
            db_session=db_session, api_key="k",
            schema_context="Table: svc_job (id int)",
        )

        stubborn = json.dumps({"followup": "Which table should I use?"})
        with patch(
            "ai_agent.workflows.monitor.monitor_agent._call_llm",
            new=AsyncMock(return_value=stubborn),
        ):
            result = await agent.generate_query()

        assert result["status"] == "error"
        assert "Could not identify which tables" in result["error"]

    @pytest.mark.asyncio
    async def test_business_followup_is_still_passed_through(
        self, db_session, monitor, monitor_type
    ):
        from ai_agent.workflows.monitor.monitor_agent import MonitorAgent

        agent = MonitorAgent(
            monitor=monitor, monitor_type=monitor_type,
            mcp_tools=[{
                "tool_name": "run_sql", "server_name": "srv",
                "requires_query_building": True, "input_schema": {}, "description": "",
            }],
            db_session=db_session, api_key="k", schema_context="Table: orders (id int)",
        )

        with patch(
            "ai_agent.workflows.monitor.monitor_agent._call_llm",
            new=AsyncMock(return_value=json.dumps({"followup": "Which quarter do you mean?"})),
        ) as mock_llm:
            result = await agent.generate_query()

        assert mock_llm.await_count == 1  # no correction retry
        assert result["status"] == "followup"
        assert result["followup"] == "Which quarter do you mean?"


class TestValueCoercion:
    """A DB tool can express "the answer is 20" in many shapes depending on the
    MCP server. All of them must be accepted; only genuine ambiguity rejected.
    """

    @pytest.mark.parametrize("raw,expected", [
        # The exact payload that was wrongly rejected in production:
        ('{"COUNT(*)": 20}', 20.0),
        # Plain scalars
        ("20", 20.0),
        ("  20  ", 20.0),
        ("20.5", 20.5),
        ("0", 0.0),
        ("-3", -3.0),
        # Envelopes
        ('{"value": 99}', 99.0),
        ('{"rows": [[20]], "columns": ["c"]}', 20.0),
        ('{"rows": [{"total": 7}]}', 7.0),
        ('{"data": [{"n": 4}]}', 4.0),
        ('{"result": 11}', 11.0),
        # Row shapes
        ('[{"count": 5}]', 5.0),
        ("[[20]]", 20.0),
        ("[20]", 20.0),
        # Formatted numbers
        ("1,234", 1234.0),
        ("$1,500", 1500.0),
        ("45%", 45.0),
        ('{"total": "1,234"}', 1234.0),
        # Fenced JSON from an LLM
        ('```json\n{"COUNT(*)": 8}\n```', 8.0),
    ])
    def test_accepts_number_shape(self, raw, expected):
        from ai_agent.workflows.monitor.monitor_agent import _coerce_number
        assert _coerce_number(raw) == expected

    @pytest.mark.parametrize("raw", [
        '[{"a": 1}, {"a": 2}]',      # two rows — which one?
        '{"a": 1, "b": 2}',          # two cells — which one?
        "not a number",
        "",
        "[]",
        "{}",
        "true",                      # a bool is not a number
    ])
    def test_rejects_ambiguous_or_non_numeric(self, raw):
        from ai_agent.workflows.monitor.monitor_agent import _coerce_number
        assert _coerce_number(raw) is None

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("false", False),
        ("YES", True), ("No", False),
        ("1", True), ("0", False),
        ('{"EXISTS(1)": 1}', True),
        ('{"EXISTS(1)": 0}', False),
        ("[[0]]", False),
        ('[{"has_overdue": true}]', True),
        ('{"value": false}', False),
    ])
    def test_accepts_boolean_shape(self, raw, expected):
        from ai_agent.workflows.monitor.monitor_agent import _coerce_boolean
        assert _coerce_boolean(raw) is expected

    @pytest.mark.parametrize("raw", ["maybe", '{"a": 1, "b": 0}', "", "7"])
    def test_rejects_non_boolean(self, raw):
        from ai_agent.workflows.monitor.monitor_agent import _coerce_boolean
        assert _coerce_boolean(raw) is None

    def test_count_star_flows_through_format_result(self):
        """End-to-end for the reported bug: a COUNT(*) payload becomes a value,
        not a 'expects a single number' error.
        """
        from ai_agent.workflows.monitor.monitor_agent import _format_result
        mt = MonitorType(
            id=1, name="COUNT", title="Count", description="",
            configuration_schema="{}", result_type="NUMBER",
            is_active=True, created_by=1,
        )
        assert _format_result(mt, '{"COUNT(*)": 20}') == {"value": 20.0}


class TestErrorMessages:
    """Raw engine errors must never reach the UI verbatim."""

    @pytest.mark.parametrize("raw,expect_substring,must_not_contain", [
        ("Unknown column 'close_date' in 'field list'",
         "close_date", "field list"),
        # "doesn't exist" survives as plain English — what matters is that the
        # message is rewritten for a business user, not echoed verbatim.
        ("Table 'not_crm.sales' doesn't exist",
         "not_crm.sales", None),
        ("You have an error in your SQL syntax near 'SELCT'",
         "couldn't understand", "SELCT"),
        ("This version of MariaDB doesn't yet support 'LIMIT & IN/ALL/ANY/SOME subquery'",
         "doesn't support", "IN/ALL/ANY/SOME"),
        ("Column 'id' in field list is ambiguous",
         "ambiguous", None),
        # Postgres phrases it the other way round.
        ('column reference "id" is ambiguous',
         "more than one table", None),
        ("Access denied for user 'bot'@'localhost'",
         "permission", "localhost"),
        ("Lock wait timeout exceeded",
         "too long", None),
    ])
    def test_translates_engine_errors(self, raw, expect_substring, must_not_contain):
        from ai_agent.workflows.monitor.monitor_agent import _humanize_error
        out = _humanize_error(raw)
        assert expect_substring.lower() in out.lower()
        if must_not_contain:
            assert must_not_contain not in out

    def test_result_shape_error_drops_raw_payload(self):
        """The user shouldn't see a JSON blob like {"COUNT(*)":20} in the UI."""
        from ai_agent.workflows.monitor.monitor_agent import _humanize_error
        out = _humanize_error(
            'This monitor expects a single number, but the query returned: {"COUNT(*)":20}'
        )
        assert "{" not in out
        assert "single number" in out

    def test_our_own_messages_pass_through(self):
        from ai_agent.workflows.monitor.monitor_agent import _humanize_error
        msg = "Could not identify which tables in your database answer this question."
        assert _humanize_error(msg) == msg

    @pytest.mark.parametrize("msg", [
        "The request is ambiguous, please clarify which period you mean.",
        "Your question was ambiguous.",
    ])
    def test_bare_ambiguous_is_not_rewritten_as_a_database_fault(self, msg):
        """'ambiguous' alone is not a join collision — don't blame the database."""
        from ai_agent.workflows.monitor.monitor_agent import _humanize_error
        assert "more than one table" not in _humanize_error(msg)

    def test_long_unknown_error_is_truncated(self):
        from ai_agent.workflows.monitor.monitor_agent import _humanize_error
        out = _humanize_error("x" * 900)
        assert len(out) <= 201

    def test_empty_error_gets_a_message(self):
        from ai_agent.workflows.monitor.monitor_agent import _humanize_error
        assert _humanize_error("") == "This monitor could not be run."

    @pytest.mark.asyncio
    async def test_ui_never_receives_raw_engine_error(
        self, db_session, monitor, monitor_type, user
    ):
        """End-to-end: the persisted error_message and API payload are both
        the humanized text, not MySQL's wording.
        """
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        monitor.organization_id = 1
        db_session.add(monitor)
        db_session.commit()

        mock_tools = [{
            "tool_name": "run_sql", "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}}, "description": "",
        }]
        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="k"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="job_card\n  (id:int PK)")),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=json.dumps(
                      {"query": "SELECT close_date FROM job_card", "tool_name": "run_sql"}))),
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
        ):
            mgr.return_value.call_tool = AsyncMock(return_value={
                "isError": True,
                "content": [{"text": "Unknown column 'close_date' in 'field list'"}],
            })
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "error"
        assert "field list" not in outcome["error"]
        assert "close_date" in outcome["error"]
        assert "field list" not in outcome["result"].error_message


class TestSqlSelfRepair:
    """When the database rejects the SQL, the error is fed back and the query
    is rewritten once. This is what makes dialect quirks self-healing without
    detecting the dialect up front.
    """

    MARIADB_ERROR = (
        "This version of MariaDB doesn't yet support "
        "'LIMIT & IN/ALL/ANY/SOME subquery'"
    )
    BAD_SQL = (
        "SELECT status FROM job_card WHERE id IN "
        "(SELECT id FROM job_card ORDER BY opened_at DESC LIMIT 10)"
    )
    GOOD_SQL = (
        "SELECT status FROM job_card WHERE id IN "
        "(SELECT id FROM (SELECT id FROM job_card ORDER BY opened_at DESC LIMIT 10) AS x)"
    )

    def _agent(self, db_session, monitor, monitor_type):
        from ai_agent.workflows.monitor.monitor_agent import MonitorAgent
        return MonitorAgent(
            monitor=monitor, monitor_type=monitor_type,
            mcp_tools=[{
                "tool_name": "run_sql", "server_name": "srv",
                "requires_query_building": True,
                "input_schema": {"properties": {"query": {}}}, "description": "",
            }],
            db_session=db_session, api_key="k",
            schema_context="job_card\n  (id:int PK, status:varchar, opened_at:datetime)",
        )

    @pytest.mark.asyncio
    async def test_mariadb_limit_subquery_error_is_repaired(
        self, db_session, monitor, monitor_type
    ):
        """The exact failure the user hit: rewritten and re-run automatically."""
        agent = self._agent(db_session, monitor, monitor_type)

        call_tool = AsyncMock(side_effect=[
            {"isError": True, "content": [{"text": self.MARIADB_ERROR}]},
            {"isError": False, "content": [{"text": "7"}]},
        ])
        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=json.dumps(
                      {"query": self.GOOD_SQL, "tool_name": "run_sql"}))) as mock_llm,
        ):
            mgr.return_value.call_tool = call_tool
            result, used = await agent.run_and_format(self.BAD_SQL, "run_sql")

        assert result == {"value": 7.0}
        assert used == self.GOOD_SQL          # caller persists the FIXED sql
        assert call_tool.await_count == 2
        # The DB's own error text must reach the repair prompt.
        assert self.MARIADB_ERROR in mock_llm.await_args_list[0].args[0][0]["content"]

    @pytest.mark.asyncio
    async def test_stops_when_model_re_proposes_a_query_already_known_to_fail(
        self, db_session, monitor, monitor_type
    ):
        """Repeating a query that already failed would just burn attempts."""
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        agent = self._agent(db_session, monitor, monitor_type)
        call_tool = AsyncMock(return_value={
            "isError": True, "content": [{"text": self.MARIADB_ERROR}],
        })
        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=json.dumps(
                      {"query": self.GOOD_SQL, "tool_name": "run_sql"}))),
        ):
            mgr.return_value.call_tool = call_tool
            with pytest.raises(MonitorExecutionError, match="MariaDB"):
                await agent.run_and_format(self.BAD_SQL, "run_sql")

        # original + one repair; the model then re-proposed GOOD_SQL which had
        # already failed, so the loop stopped rather than re-running it.
        assert call_tool.await_count == 2

    @pytest.mark.asyncio
    async def test_converges_over_several_distinct_errors(
        self, db_session, monitor, monitor_type
    ):
        """The reported symptom was 'a different error every time'. Successive
        distinct failures must converge within the attempt budget, and each
        repair must see the full history of what already failed.
        """
        agent = self._agent(db_session, monitor, monitor_type)

        call_tool = AsyncMock(side_effect=[
            {"isError": True, "content": [{"text": "Unknown column 'close_date' in 'field list'"}]},
            {"isError": True, "content": [{"text": self.MARIADB_ERROR}]},
            {"isError": False, "content": [{"text": '{"COUNT(*)": 20}'}]},
        ])
        repairs = [
            json.dumps({"query": "SELECT a FROM job_card", "tool_name": "run_sql"}),
            json.dumps({"query": "SELECT COUNT(*) FROM job_card", "tool_name": "run_sql"}),
        ]
        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(side_effect=repairs)) as mock_llm,
        ):
            mgr.return_value.call_tool = call_tool
            result, used = await agent.run_and_format(self.BAD_SQL, "run_sql")

        assert result == {"value": 20.0}
        assert used == "SELECT COUNT(*) FROM job_card"
        assert call_tool.await_count == 3

        # The second repair prompt must list the first failed attempt so the
        # model doesn't cycle back to it.
        second_prompt = mock_llm.await_args_list[1].args[0][0]["content"]
        assert "Already tried" in second_prompt
        assert self.BAD_SQL in second_prompt

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, db_session, monitor, monitor_type):
        """Bounded: distinct failures still stop at MAX_SQL_REPAIR_ATTEMPTS."""
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorAgent, MonitorExecutionError,
        )

        agent = self._agent(db_session, monitor, monitor_type)
        call_tool = AsyncMock(return_value={
            "isError": True, "content": [{"text": self.MARIADB_ERROR}],
        })
        # Every repair proposes a brand-new (still broken) query.
        counter = {"n": 0}

        async def _new_query(*_a, **_kw):
            counter["n"] += 1
            return json.dumps({
                "query": f"SELECT c{counter['n']} FROM job_card", "tool_name": "run_sql",
            })

        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm", new=_new_query),
        ):
            mgr.return_value.call_tool = call_tool
            with pytest.raises(MonitorExecutionError):
                await agent.run_and_format(self.BAD_SQL, "run_sql")

        # original + MAX repairs, then stop
        assert call_tool.await_count == MonitorAgent.MAX_SQL_REPAIR_ATTEMPTS + 1

    @pytest.mark.asyncio
    async def test_non_repairable_error_is_not_retried(
        self, db_session, monitor, monitor_type
    ):
        """'Returned no data' can't be fixed by rewriting SQL — don't burn a call."""
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        agent = self._agent(db_session, monitor, monitor_type)
        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock()) as mock_llm,
        ):
            mgr.return_value.call_tool = AsyncMock(
                return_value={"isError": False, "content": []}
            )
            with pytest.raises(MonitorExecutionError, match="returned no data"):
                await agent.run_and_format("SELECT 1 FROM job_card", "run_sql")

        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_result_shape_mismatch_is_repairable(
        self, db_session, monitor, monitor_type
    ):
        """A NUMBER monitor given rows rewrites the SQL to aggregate instead."""
        agent = self._agent(db_session, monitor, monitor_type)

        call_tool = AsyncMock(side_effect=[
            {"isError": False, "content": [{"text": json.dumps([{"a": 1}, {"a": 2}])}]},
            {"isError": False, "content": [{"text": "2"}]},
        ])
        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=json.dumps(
                      {"query": "SELECT COUNT(*) FROM job_card", "tool_name": "run_sql"}))),
        ):
            mgr.return_value.call_tool = call_tool
            result, used = await agent.run_and_format("SELECT a FROM job_card", "run_sql")

        assert result == {"value": 2.0}
        assert used == "SELECT COUNT(*) FROM job_card"

    @pytest.mark.asyncio
    async def test_identical_repair_is_rejected(self, db_session, monitor, monitor_type):
        """If the model returns the same SQL, don't re-run it pointlessly."""
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        agent = self._agent(db_session, monitor, monitor_type)
        call_tool = AsyncMock(return_value={
            "isError": True, "content": [{"text": self.MARIADB_ERROR}],
        })
        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=json.dumps(
                      {"query": self.BAD_SQL, "tool_name": "run_sql"}))),
        ):
            mgr.return_value.call_tool = call_tool
            with pytest.raises(MonitorExecutionError, match="MariaDB"):
                await agent.run_and_format(self.BAD_SQL, "run_sql")

        assert call_tool.await_count == 1  # never re-ran the identical query

    @pytest.mark.asyncio
    async def test_repaired_sql_is_persisted_for_future_runs(
        self, db_session, monitor, monitor_type, user
    ):
        """End-to-end: the monitor caches the FIXED sql, so the next scheduled
        run doesn't repeat the failure.
        """
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        monitor.organization_id = 1
        monitor.llm_query = json.dumps({"query": self.BAD_SQL, "tool_name": "run_sql"})
        db_session.add(monitor)
        db_session.commit()

        mock_tools = [{
            "tool_name": "run_sql", "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}}, "description": "",
        }]

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="k"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="job_card\n  (id:int PK, status:varchar)")),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=json.dumps(
                      {"query": self.GOOD_SQL, "tool_name": "run_sql"}))),
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mgr,
        ):
            mgr.return_value.call_tool = AsyncMock(side_effect=[
                {"isError": True, "content": [{"text": self.MARIADB_ERROR}]},
                {"isError": False, "content": [{"text": "7"}]},
            ])
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "success"
        db_session.refresh(monitor)
        assert json.loads(monitor.llm_query)["query"] == self.GOOD_SQL


class TestSchemaEngineIntegration:
    """Monitors use the same schema_engine retrieval pipeline as /chat, so a
    business user is never asked to name a table in a large database.
    """

    ENGINE_CTX = (
        "job_card -- Stores repair job records\n"
        "  (id:int PK, status:varchar, opened_at:datetime)\n"
        "  FK: cust_id->customer.id\n\n"
        "svc_job_status -- Lookup of repair statuses\n"
        "  (code:int PK, label:varchar)"
    )

    def _tools(self):
        return [{
            "server_name": "srv", "tool_name": "run_sql",
            "requires_query_building": True, "input_schema": {}, "description": "",
        }]

    @pytest.mark.asyncio
    async def test_uses_schema_engine_and_passes_no_agent_id(self):
        """Monitors are org-scoped, so agent_id must be None — that only skips
        is_cold_build bookkeeping, retrieval is keyed by server_name.
        """
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        orch = MagicMock()
        orch.build_context = AsyncMock(return_value=self.ENGINE_CTX)
        orch.table_count = 412

        with (
            patch("ai_agent.schema_engine.orchestrator.get_orchestrator",
                  new=AsyncMock(return_value=orch)) as mock_get,
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"),
        ):
            ctx = await _load_schema_context(
                self._tools(),
                natural_query="repair statuses and turnaround times",
                api_key="k",
            )

        assert ctx == self.ENGINE_CTX
        assert mock_get.await_args.kwargs["agent_id"] is None
        assert mock_get.await_args.kwargs["server_name"] == "srv"
        orch.build_context.assert_awaited_once_with("repair statuses and turnaround times")

    @pytest.mark.asyncio
    async def test_engine_context_still_yields_table_names_for_sql_validation(self):
        """The engine emits a different DDL shape than the legacy loader —
        table-existence validation must keep working against it.
        """
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _extract_table_names, _validate_sql,
        )

        known = _extract_table_names(self.ENGINE_CTX)
        assert {"job_card", "svc_job_status"} <= known

        _validate_sql("SELECT COUNT(*) FROM job_card", known)
        with pytest.raises(MonitorExecutionError, match="do not exist"):
            _validate_sql("SELECT * FROM not_crm.sales", known)

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_loader_when_engine_raises(self):
        """Engine not provisioned / Milvus down must not break monitors."""
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        payload = json.dumps([{
            "table_name": "orders",
            "columns": json.dumps([{"column_name": "id", "data_type": "int"}]),
        }])

        with (
            patch("ai_agent.schema_engine.orchestrator.get_orchestrator",
                  new=AsyncMock(side_effect=RuntimeError("milvus unavailable"))),
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mock_mgr,
        ):
            mock_mgr.return_value.list_tools = AsyncMock(return_value=[{"name": "get_schema"}])
            mock_mgr.return_value.call_tool = AsyncMock(
                return_value={"content": [{"text": payload}]}
            )
            ctx = await _load_schema_context(
                self._tools(), natural_query="how many orders", api_key="k",
            )

        assert "Table: orders" in ctx

    @pytest.mark.asyncio
    async def test_falls_back_when_engine_returns_empty_context(self):
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        payload = json.dumps([{
            "table_name": "orders",
            "columns": json.dumps([{"column_name": "id", "data_type": "int"}]),
        }])
        orch = MagicMock()
        orch.build_context = AsyncMock(return_value="")
        orch.table_count = 0

        with (
            patch("ai_agent.schema_engine.orchestrator.get_orchestrator",
                  new=AsyncMock(return_value=orch)),
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mock_mgr,
        ):
            mock_mgr.return_value.list_tools = AsyncMock(return_value=[{"name": "get_schema"}])
            mock_mgr.return_value.call_tool = AsyncMock(
                return_value={"content": [{"text": payload}]}
            )
            ctx = await _load_schema_context(
                self._tools(), natural_query="how many orders", api_key="k",
            )

        assert "Table: orders" in ctx

    @pytest.mark.asyncio
    async def test_no_db_servers_returns_empty_without_touching_engine(self):
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        with patch(
            "ai_agent.schema_engine.orchestrator.get_orchestrator",
            new=AsyncMock(),
        ) as mock_get:
            ctx = await _load_schema_context(
                [{"server_name": "srv", "requires_query_building": False}],
                natural_query="q", api_key="k",
            )

        assert ctx == ""
        mock_get.assert_not_awaited()


class TestSchemaNarrowing:
    """Fallback loader: large schemas are narrowed by relevance, never
    truncated alphabetically. Only used when the schema engine is unavailable.
    """

    def _schema_payload(self, table_names, cols_each=1):
        return json.dumps([
            {
                "table_name": name,
                "columns": json.dumps([
                    {"column_name": f"col_{i}", "data_type": "varchar"}
                    for i in range(cols_each)
                ]),
            }
            for name in table_names
        ])

    def test_parse_schema_payload(self):
        from ai_agent.workflows.monitor.monitor_agent import _parse_schema_payload
        tables = _parse_schema_payload(self._schema_payload(["orders", "vendors"]))
        assert set(tables) == {"orders", "vendors"}
        assert "col_0 varchar" in tables["orders"]

    def test_parse_schema_payload_returns_empty_for_unknown_shape(self):
        from ai_agent.workflows.monitor.monitor_agent import _parse_schema_payload
        assert _parse_schema_payload("just some text") == {}
        assert _parse_schema_payload(json.dumps({"not": "a list"})) == {}

    @pytest.mark.asyncio
    async def test_small_schema_is_returned_in_full(self):
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        payload = self._schema_payload(["orders", "vendors"])
        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_mgr:
            mock_mgr.return_value.list_tools = AsyncMock(return_value=[{"name": "get_schema"}])
            mock_mgr.return_value.call_tool = AsyncMock(
                return_value={"content": [{"text": payload}]}
            )
            ctx = await _load_schema_context(
                [{"server_name": "srv", "requires_query_building": True}]
            )

        assert "Table: orders" in ctx
        assert "Table: vendors" in ctx

    @pytest.mark.asyncio
    async def test_large_schema_is_narrowed_to_relevant_tables(self):
        """The old code truncated alphabetically, hiding late-alphabet tables
        entirely — which is why the agent asked the user to name them.
        """
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        # ~400 tables with fat column lists → well over the size budget.
        names = [f"tbl_{i:03d}" for i in range(400)] + ["svc_job_status"]
        payload = self._schema_payload(names, cols_each=12)

        with (
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mock_mgr,
            patch(
                "ai_agent.workflows.monitor.monitor_agent._call_llm",
                new=AsyncMock(return_value=json.dumps({"tables": ["svc_job_status"]})),
            ),
        ):
            mock_mgr.return_value.list_tools = AsyncMock(return_value=[{"name": "get_schema"}])
            mock_mgr.return_value.call_tool = AsyncMock(
                return_value={"content": [{"text": payload}]}
            )
            ctx = await _load_schema_context(
                [{"server_name": "srv", "requires_query_building": True}],
                natural_query="repair statuses and turnaround times",
                api_key="k",
            )

        # The late-alphabet table survives with its full columns...
        assert "Table: svc_job_status (col_0 varchar" in ctx
        # ...and every table name is still listed so nothing is invisible.
        assert "svc_job_status" in ctx
        assert "tbl_399" in ctx

    @pytest.mark.asyncio
    async def test_large_schema_falls_back_to_names_only_without_api_key(self):
        """No narrowing available → list every table name rather than truncate."""
        from ai_agent.workflows.monitor.monitor_agent import _load_schema_context

        names = [f"tbl_{i:03d}" for i in range(400)]
        payload = self._schema_payload(names, cols_each=12)

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_mgr:
            mock_mgr.return_value.list_tools = AsyncMock(return_value=[{"name": "get_schema"}])
            mock_mgr.return_value.call_tool = AsyncMock(
                return_value={"content": [{"text": payload}]}
            )
            ctx = await _load_schema_context(
                [{"server_name": "srv", "requires_query_building": True}]
            )

        assert "Table: tbl_000" in ctx
        assert "Table: tbl_399" in ctx  # nothing dropped

    @pytest.mark.asyncio
    async def test_select_relevant_tables_ignores_hallucinated_names(self):
        from ai_agent.workflows.monitor.monitor_agent import _select_relevant_tables

        with patch(
            "ai_agent.workflows.monitor.monitor_agent._call_llm",
            new=AsyncMock(return_value=json.dumps({"tables": ["orders", "does_not_exist"]})),
        ):
            picked = await _select_relevant_tables("q", ["orders", "vendors"], "k")

        assert picked == ["orders"]

    @pytest.mark.asyncio
    async def test_select_relevant_tables_returns_empty_on_llm_failure(self):
        from ai_agent.workflows.monitor.monitor_agent import _select_relevant_tables

        with patch(
            "ai_agent.workflows.monitor.monitor_agent._call_llm",
            new=AsyncMock(side_effect=RuntimeError("api down")),
        ):
            assert await _select_relevant_tables("q", ["orders"], "k") == []


# ── Unit: MonitorAgent.execute_query ──────────────────────────────────────────

class TestValidateSql:
    """Generated SQL is validated before it is ever sent to the database, so
    the user sees a real error instead of the DB's raw complaint stored as a
    monitor value.
    """

    def test_accepts_plain_select(self):
        from ai_agent.workflows.monitor.monitor_agent import _validate_sql
        _validate_sql("SELECT COUNT(*) FROM orders", {"orders"})

    def test_accepts_cte(self):
        from ai_agent.workflows.monitor.monitor_agent import _validate_sql
        _validate_sql(
            "WITH recent AS (SELECT * FROM orders) SELECT COUNT(*) FROM recent",
            {"orders"},
        )

    def test_rejects_empty_query(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _validate_sql,
        )
        with pytest.raises(MonitorExecutionError, match="did not generate"):
            _validate_sql("", {"orders"})

    def test_rejects_non_select(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _validate_sql,
        )
        with pytest.raises(MonitorExecutionError, match="read-only SELECT"):
            _validate_sql("DROP TABLE orders", {"orders"})

    def test_rejects_write_statement_hidden_after_select(self):
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _validate_sql,
        )
        with pytest.raises(MonitorExecutionError, match="forbidden DELETE"):
            _validate_sql("SELECT 1; DELETE FROM orders", {"orders"})

    def test_allows_columns_that_contain_keyword_substrings(self):
        """updated_at / created_by must not trip the forbidden-keyword guard."""
        from ai_agent.workflows.monitor.monitor_agent import _validate_sql
        _validate_sql("SELECT updated_at, created_by FROM orders", {"orders"})

    def test_rejects_unknown_table(self):
        """This is the 'Table not_crm.sales doesn't exist' case, caught up front."""
        from ai_agent.workflows.monitor.monitor_agent import (
            MonitorExecutionError, _validate_sql,
        )
        with pytest.raises(MonitorExecutionError, match="do not exist"):
            _validate_sql("SELECT * FROM sales", {"orders", "vendors"})

    def test_ignores_schema_prefix_when_matching_tables(self):
        from ai_agent.workflows.monitor.monitor_agent import _validate_sql
        _validate_sql("SELECT * FROM mydb.orders", {"orders"})

    def test_skips_table_check_when_schema_unknown(self):
        """No schema available → only the SELECT-only guard applies."""
        from ai_agent.workflows.monitor.monitor_agent import _validate_sql
        _validate_sql("SELECT * FROM anything_at_all", set())

    def test_extract_table_names_from_minified_schema(self):
        from ai_agent.workflows.monitor.monitor_agent import _extract_table_names
        schema = "Table: orders (id int, total decimal)\nTable: vendors (id int)"
        assert _extract_table_names(schema) == {"orders", "vendors"}

    def test_extract_table_names_from_raw_json_schema(self):
        from ai_agent.workflows.monitor.monitor_agent import _extract_table_names
        schema = '[{"table_name":"orders","columns":"..."},{"table_name":"sales"}]'
        assert _extract_table_names(schema) == {"orders", "sales"}


class TestExecuteQuery:
    """The prompt allows the LLM to return tool_name: null alongside a SQL
    query. That query must still be executed against the DB tool, not echoed
    back as text — regression test for the bug where generated SQL leaked
    straight into the monitor result.
    """

    def _make_agent(self, db_session, monitor, monitor_type, mcp_tools, schema_context=""):
        from ai_agent.workflows.monitor.monitor_agent import MonitorAgent
        return MonitorAgent(
            monitor=monitor,
            monitor_type=monitor_type,
            mcp_tools=mcp_tools,
            db_session=db_session,
            api_key="fake-key",
            schema_context=schema_context,
        )

    @pytest.mark.asyncio
    async def test_null_tool_name_falls_back_to_available_db_tool(
        self, db_session, monitor, monitor_type
    ):
        from ai_agent.core.mcp_manager import get_mcp_manager

        mock_tools = [{
            "tool_name": "run_sql",
            "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}},
            "description": "runs SQL",
        }]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_get_mgr:
            mock_mgr = mock_get_mgr.return_value
            mock_mgr.call_tool = AsyncMock(return_value={"content": [{"text": "5"}]})

            result = await agent.execute_query("SELECT COUNT(*) FROM sales", tool_name=None)

        assert result == "5"
        mock_mgr.call_tool.assert_awaited_once_with(
            "srv", "run_sql", {"query": "SELECT COUNT(*) FROM sales"}
        )

    @pytest.mark.asyncio
    async def test_null_tool_name_skips_schema_tool_when_query_tool_also_present(
        self, db_session, monitor, monitor_type
    ):
        """requires_query_building is set per-SERVER (gather_org_mcp_tools), so the
        schema/introspection tool carries the same flag as the real query tool.
        The fallback must not pick the schema tool — regression test for the bug
        where a monitor's result became the entire DB schema dump instead of the
        query's actual answer.
        """
        from ai_agent.core.mcp_manager import get_mcp_manager

        mock_tools = [
            {
                "tool_name": "get_schema",
                "server_name": "srv",
                "requires_query_building": True,
                "input_schema": {"properties": {}},
                "description": "returns DB schema",
            },
            {
                "tool_name": "run_sql",
                "server_name": "srv",
                "requires_query_building": True,
                "input_schema": {"properties": {"query": {}}},
                "description": "runs SQL",
            },
        ]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_get_mgr:
            mock_mgr = mock_get_mgr.return_value
            mock_mgr.call_tool = AsyncMock(return_value={"content": [{"text": "5"}]})

            result = await agent.execute_query("SELECT COUNT(*) FROM sales", tool_name=None)

        assert result == "5"
        mock_mgr.call_tool.assert_awaited_once_with(
            "srv", "run_sql", {"query": "SELECT COUNT(*) FROM sales"}
        )

    @pytest.mark.asyncio
    async def test_null_tool_name_with_no_db_tool_returns_query_as_is(
        self, db_session, monitor, monitor_type
    ):
        """No requires_query_building tool available → falls back to old behavior."""
        mock_tools = [{
            "tool_name": "send_email",
            "server_name": "srv",
            "requires_query_building": False,
            "input_schema": {},
            "description": "sends an email",
        }]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)
        result = await agent.execute_query("Just a direct answer", tool_name=None)
        assert result == "Just a direct answer"

    @pytest.mark.asyncio
    async def test_mcp_is_error_raises_instead_of_returning_error_text(
        self, db_session, monitor, monitor_type
    ):
        """Regression: "Table 'not_crm.sales' doesn't exist" was being stored as
        a SUCCESS monitor value because isError was ignored.
        """
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        mock_tools = [{
            "tool_name": "run_sql",
            "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}},
            "description": "runs SQL",
        }]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_get_mgr:
            mock_mgr = mock_get_mgr.return_value
            mock_mgr.call_tool = AsyncMock(return_value={
                "isError": True,
                "content": [{"text": "Error: Table 'not_crm.sales' doesn't exist"}],
            })

            with pytest.raises(MonitorExecutionError, match="doesn't exist"):
                await agent.execute_query("SELECT * FROM sales", tool_name="run_sql")

    @pytest.mark.asyncio
    async def test_tool_exception_raises(self, db_session, monitor, monitor_type):
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        mock_tools = [{
            "tool_name": "run_sql",
            "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}},
            "description": "runs SQL",
        }]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_get_mgr:
            mock_mgr = mock_get_mgr.return_value
            mock_mgr.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))

            with pytest.raises(MonitorExecutionError, match="connection lost"):
                await agent.execute_query("SELECT 1 FROM orders", tool_name="run_sql")

    @pytest.mark.asyncio
    async def test_empty_tool_output_raises(self, db_session, monitor, monitor_type):
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        mock_tools = [{
            "tool_name": "run_sql",
            "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}},
            "description": "runs SQL",
        }]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_get_mgr:
            mock_mgr = mock_get_mgr.return_value
            mock_mgr.call_tool = AsyncMock(return_value={"isError": False, "content": []})

            with pytest.raises(MonitorExecutionError, match="returned no data"):
                await agent.execute_query("SELECT 1 FROM orders", tool_name="run_sql")

    @pytest.mark.asyncio
    async def test_unknown_tool_name_raises(self, db_session, monitor, monitor_type):
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        mock_tools = [{
            "tool_name": "run_sql",
            "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}},
            "description": "runs SQL",
        }]
        agent = self._make_agent(db_session, monitor, monitor_type, mock_tools)

        with pytest.raises(MonitorExecutionError, match="not connected"):
            await agent.execute_query("SELECT 1", tool_name="nonexistent_tool")

    @pytest.mark.asyncio
    async def test_invalid_sql_is_rejected_before_reaching_the_database(
        self, db_session, monitor, monitor_type
    ):
        from ai_agent.workflows.monitor.monitor_agent import MonitorExecutionError

        mock_tools = [{
            "tool_name": "run_sql",
            "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}},
            "description": "runs SQL",
        }]
        agent = self._make_agent(
            db_session, monitor, monitor_type, mock_tools,
            schema_context="Table: orders (id int)",
        )

        with patch(
            "ai_agent.workflows.monitor.monitor_agent.get_mcp_manager"
        ) as mock_get_mgr:
            mock_mgr = mock_get_mgr.return_value
            mock_mgr.call_tool = AsyncMock()

            with pytest.raises(MonitorExecutionError, match="do not exist"):
                await agent.execute_query("SELECT * FROM sales", tool_name="run_sql")

            mock_mgr.call_tool.assert_not_awaited()


# ── Unit: monitor_service ─────────────────────────────────────────────────────

class TestMonitorService:
    def test_load_monitor_with_context_returns_monitor_and_type(self, db_session, monitor, monitor_type):
        from ai_agent.services.monitor_service import load_monitor_with_context
        db_session.commit()  # make rows visible
        m, mt = load_monitor_with_context(monitor.id, db_session)
        assert m is not None
        assert mt is not None

    def test_load_monitor_returns_none_for_missing(self, db_session):
        from ai_agent.services.monitor_service import load_monitor_with_context
        m, mt = load_monitor_with_context(999999, db_session)
        assert m is None
        assert mt is None

    def test_load_monitor_returns_none_for_inactive(self, db_session, user, monitor_type):
        from ai_agent.services.monitor_service import load_monitor_with_context
        inactive = Monitor(
            name="Inactive",
            monitor_type_id=monitor_type.id,
            natural_query="q",
            created_by=user.id,
            is_active=False,
        )
        db_session.add(inactive)
        db_session.flush()
        m, _ = load_monitor_with_context(inactive.id, db_session)
        assert m is None

    def test_persist_monitor_result_creates_row(self, db_session, monitor, monitor_type):
        from ai_agent.services.monitor_service import persist_monitor_result
        row = persist_monitor_result(
            monitor=monitor,
            monitor_type=monitor_type,
            result={"value": 7},
            status="SUCCESS",
            execution_time_ms=100,
            db_session=db_session,
        )
        assert row is not None
        assert row.id is not None
        assert row.status == "SUCCESS"
        assert row.execution_time_ms == 100
        assert json.loads(row.result) == {"value": 7}

    def test_persist_monitor_result_stores_error(self, db_session, monitor, monitor_type):
        from ai_agent.services.monitor_service import persist_monitor_result
        row = persist_monitor_result(
            monitor=monitor,
            monitor_type=monitor_type,
            result={},
            status="FAILED",
            execution_time_ms=50,
            db_session=db_session,
            error_message="Tool timed out",
        )
        assert row.status == "FAILED"
        assert row.error_message == "Tool timed out"

    def test_get_latest_result_returns_most_recent(self, db_session, monitor):
        from ai_agent.services.monitor_service import get_latest_result
        for i in range(3):
            db_session.add(MonitorResult(
                monitor_id=monitor.id,
                result_type="NUMBER",
                result=json.dumps({"value": i}),
                status="SUCCESS",
                is_active=True,
            ))
        db_session.flush()
        latest = get_latest_result(monitor.id, db_session)
        assert latest is not None

    def test_get_latest_result_returns_none_when_no_results(self, db_session, user, monitor_type):
        from ai_agent.services.monitor_service import get_latest_result
        empty = Monitor(
            name="Empty",
            monitor_type_id=monitor_type.id,
            natural_query="q",
            created_by=user.id,
            is_active=True,
        )
        db_session.add(empty)
        db_session.flush()
        assert get_latest_result(empty.id, db_session) is None

    def test_get_result_history_returns_tuple(self, db_session, monitor, monitor_result):
        from ai_agent.services.monitor_service import get_result_history
        items, total = get_result_history(monitor.id, db_session)
        assert isinstance(items, list)
        assert isinstance(total, int)
        assert total >= 1

    def test_get_result_history_pagination(self, db_session, monitor):
        from ai_agent.services.monitor_service import get_result_history
        for _ in range(5):
            db_session.add(MonitorResult(
                monitor_id=monitor.id,
                result_type="TEXT",
                result=json.dumps({"value": "ok"}),
                status="SUCCESS",
                is_active=True,
            ))
        db_session.flush()
        items_p1, total = get_result_history(monitor.id, db_session, page=1, limit=2)
        items_p2, _ = get_result_history(monitor.id, db_session, page=2, limit=2)
        assert len(items_p1) <= 2
        assert len(items_p2) <= 2
        # pages must not overlap
        p1_ids = {r.id for r in items_p1}
        p2_ids = {r.id for r in items_p2}
        assert p1_ids.isdisjoint(p2_ids)


# ── Unit: execute_monitor workflow ────────────────────────────────────────────

class TestExecuteMonitor:
    """Tests for the module-level execute_monitor() workflow function.

    The OpenAI key check and the LLM call are mocked — these tests verify the
    orchestration logic (query generation, execution, result persistence)
    without any real LLM/MCP calls.
    """

    @pytest.mark.asyncio
    async def test_execute_monitor_returns_success_result(self, db_session, monitor, monitor_type, user):
        """No org MCP tools configured → falls back to a direct LLM answer."""
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="fake-key"),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm", new=AsyncMock(return_value="99")),
        ):
            db_session.commit()
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "success"
        assert outcome["result"] is not None
        assert outcome["result"].status == "SUCCESS"
        assert json.loads(outcome["result"].result) == {"value": 99.0}

    @pytest.mark.asyncio
    async def test_execute_monitor_stores_failed_result_on_invalid_llm_response(
        self, db_session, monitor, monitor_type, user
    ):
        """With org MCP tools available, an unparseable LLM response → FAILED result."""
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        monitor.organization_id = 1
        db_session.add(monitor)
        db_session.commit()

        mock_tools = [{"tool_name": "get_count", "description": "counts rows",
                       "input_schema": {}, "server_name": "srv", "requires_query_building": True}]

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="fake-key"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="")),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value="not valid json")),
        ):
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "error"
        assert outcome["result"] is not None
        assert outcome["result"].status == "FAILED"

    @pytest.mark.asyncio
    async def test_execute_monitor_returns_followup_when_llm_asks_for_clarification(
        self, db_session, monitor, monitor_type, user
    ):
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        monitor.organization_id = 1
        db_session.add(monitor)
        db_session.commit()

        mock_tools = [{"tool_name": "get_count", "description": "counts rows",
                       "input_schema": {}, "server_name": "srv", "requires_query_building": True}]
        followup_json = json.dumps({"followup": "Which quarter do you mean?"})

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="fake-key"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="")),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=followup_json)),
        ):
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "followup"
        assert outcome["followup_question"] == "Which quarter do you mean?"
        assert outcome["result"] is None

        db_session.refresh(monitor)
        assert monitor.followup_question == "Which quarter do you mean?"

    @pytest.mark.asyncio
    async def test_execute_monitor_raises_for_missing_monitor(self, db_session):
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor
        with pytest.raises(ValueError, match="not found"):
            await execute_monitor(999999, db_session)

    @pytest.mark.asyncio
    async def test_database_error_is_persisted_as_failed_not_success(
        self, db_session, monitor, monitor_type, user
    ):
        """End-to-end regression for the reported bug: a DB error must produce a
        FAILED row with error_message — never a SUCCESS row whose value is the
        error text.
        """
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        monitor.organization_id = 1
        db_session.add(monitor)
        db_session.commit()

        mock_tools = [{
            "tool_name": "run_sql", "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}}, "description": "runs SQL",
        }]
        query_json = json.dumps({"query": "SELECT COUNT(*) FROM sales", "tool_name": "run_sql"})

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="k"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="")),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=query_json)),
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mock_get_mgr,
        ):
            mock_get_mgr.return_value.call_tool = AsyncMock(return_value={
                "isError": True,
                "content": [{"text": "Table 'not_crm.sales' doesn't exist"}],
            })
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "error"
        assert outcome["result"].status == "FAILED"
        assert "doesn't exist" in outcome["result"].error_message
        # The error text must NOT have been stored as the monitor's value.
        assert not outcome["result"].result or "doesn't exist" not in str(
            json.loads(outcome["result"].result).get("value", "")
        )

    @pytest.mark.asyncio
    async def test_bad_cached_llm_query_is_cleared_so_next_run_regenerates(
        self, db_session, monitor, monitor_type, user
    ):
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        monitor.organization_id = 1
        monitor.llm_query = json.dumps(
            {"query": "SELECT * FROM gone", "tool_name": "run_sql"}
        )
        db_session.add(monitor)
        db_session.commit()

        mock_tools = [{
            "tool_name": "run_sql", "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}}, "description": "runs SQL",
        }]

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="k"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="Table: orders (id int)")),
        ):
            outcome = await execute_monitor(monitor.id, db_session)

        assert outcome["status"] == "error"
        db_session.refresh(monitor)
        assert not monitor.llm_query

    @pytest.mark.asyncio
    async def test_chart_monitor_end_to_end_produces_chartable_result(
        self, db_session, user
    ):
        """Full pipeline for a PIE_CHART-style monitor: DB tool returns rows,
        result is wrapped as {"chartType": "pie", "data": [...]}.
        """
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor

        chart_type = MonitorType(
            name=f"PIE_{uuid.uuid4().hex[:4]}", title="Pie Chart", description="",
            result_type="JSON",
            configuration_schema=json.dumps(
                {"format": "chart", "chart_type": "pie", "fields": ["label", "value"]}
            ),
            is_active=True, created_by=user.id,
        )
        db_session.add(chart_type)
        db_session.flush()

        chart_monitor = Monitor(
            name="Vendor Spend Breakdown",
            monitor_type_id=chart_type.id,
            natural_query="Show spend by vendor",
            organization_id=1,
            is_active=True,
            created_by=user.id,
        )
        db_session.add(chart_monitor)
        db_session.commit()

        mock_tools = [{
            "tool_name": "run_sql", "server_name": "srv",
            "requires_query_building": True,
            "input_schema": {"properties": {"query": {}}}, "description": "runs SQL",
        }]
        query_json = json.dumps({
            "query": "SELECT vendor, SUM(amount) FROM orders GROUP BY vendor",
            "tool_name": "run_sql",
        })
        rows = [{"vendor": "Acme", "total": 500}, {"vendor": "Globex", "total": 300}]

        with (
            patch("ai_agent.workflows.monitor.monitor_agent._get_openai_key", return_value="k"),
            patch("ai_agent.workflows.monitor.monitor_agent.gather_org_mcp_tools",
                  new=AsyncMock(return_value=mock_tools)),
            patch("ai_agent.workflows.monitor.monitor_agent._load_schema_context",
                  new=AsyncMock(return_value="Table: orders (vendor varchar, amount int)")),
            patch("ai_agent.workflows.monitor.monitor_agent._call_llm",
                  new=AsyncMock(return_value=query_json)),
            patch("ai_agent.workflows.monitor.monitor_agent.get_mcp_manager") as mock_get_mgr,
        ):
            mock_get_mgr.return_value.call_tool = AsyncMock(return_value={
                "isError": False,
                "content": [{"text": json.dumps(rows)}],
            })
            outcome = await execute_monitor(chart_monitor.id, db_session)

        assert outcome["status"] == "success"
        assert outcome["result"].status == "SUCCESS"
        value = json.loads(outcome["result"].result)["value"]
        assert value == {"chartType": "pie", "data": rows}
