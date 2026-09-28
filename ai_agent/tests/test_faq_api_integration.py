"""
Integration tests for FAQ API endpoints.

Tests exercise the full FastAPI request → service → DB round-trip using
an in-memory SQLite engine and a mocked LLM gateway.  No real network
calls are made.

Endpoints covered:
  GET  /api/v1/agents/{agent_id}/faqs
  POST /api/v1/agents/faqs/refresh
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from ai_agent.models import AgentFAQ
from ai_agent.tests.conftest import (
    make_agent,
    make_agent_settings,
    make_faq,
    make_user,
)

PREFIX = "/api/v1"

# Helpers

def _seed_agent(engine):
    """Create a user + agent + settings in the given engine; return (user, agent)."""
    with Session(engine) as db:
        user = make_user(db)
        agent = make_agent(db, user.id, cold_build=True)
        make_agent_settings(db, agent.id)
        db.commit()
        db.refresh(user)
        db.refresh(agent)
        return user, agent


def _patch_llm(monkeypatch, questions=None):
    if questions is None:
        questions = [
            "How do I check inventory?",
            "What is the order status?",
            "Generate a sales report?",
            "How many customers today?",
            "Which products are low in stock?",
            "Show me this week's revenue?",
        ]
    response = MagicMock()
    response.content = json.dumps(questions)
    gateway = MagicMock()
    gateway.chat = AsyncMock(return_value=response)
    monkeypatch.setattr("ai_agent.workflows.faq.faq.get_llm_gateway", lambda: gateway)
    return gateway


# GET /agents/{agent_id}/faqs

class TestGetAgentFaqs:
    def test_returns_cached_faqs(self, client, engine):
        user, agent = _seed_agent(engine)
        with Session(engine) as db:
            for i in range(6):
                make_faq(db, agent.id, question=f"Cached FAQ {i}?")
            db.commit()

        resp = client.get(f"{PREFIX}/agents/{agent.id}/faqs")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 6
        assert all("question" in item for item in data)
        assert all("source" in item for item in data)

    def test_returns_empty_list_for_agent_with_no_faqs(
        self, client, engine, monkeypatch
    ):
        _patch_llm(monkeypatch, questions=[])  # LLM returns nothing
        user, agent = _seed_agent(engine)

        resp = client.get(f"{PREFIX}/agents/{agent.id}/faqs")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_404_for_unknown_agent(self, client):
        resp = client.get(f"{PREFIX}/agents/nonexistent-id/faqs")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_404_for_deleted_agent(self, client, engine):
        user, agent = _seed_agent(engine)
        with Session(engine) as db:
            a = db.get(type(agent), agent.id)
            a.deleted = True
            db.add(a)
            db.commit()

        resp = client.get(f"{PREFIX}/agents/{agent.id}/faqs")
        assert resp.status_code == 404

    def test_refresh_param_triggers_regeneration(
        self, client, engine, monkeypatch
    ):
        gateway = _patch_llm(monkeypatch)
        user, agent = _seed_agent(engine)
        # Seed stale FAQs
        with Session(engine) as db:
            for i in range(6):
                make_faq(db, agent.id, question=f"Old FAQ {i}?")
            db.commit()

        resp = client.get(f"{PREFIX}/agents/{agent.id}/faqs?refresh=true")

        assert resp.status_code == 200
        gateway.chat.assert_called()

    def test_source_field_present_and_valid(self, client, engine):
        valid_sources = {"agent_config", "chat_history"}
        user, agent = _seed_agent(engine)
        with Session(engine) as db:
            make_faq(db, agent.id, question="FAQ?", source="agent_config")
            db.commit()

        resp = client.get(f"{PREFIX}/agents/{agent.id}/faqs")

        assert resp.status_code == 200
        for item in resp.json():
            assert item["source"] in valid_sources


# POST /agents/faqs/refresh — single-agent mode

class TestRefreshFaqsSingleAgent:
    def test_single_agent_refresh_success(self, client, engine, monkeypatch):
        _patch_llm(monkeypatch)
        user, agent = _seed_agent(engine)

        resp = client.post(
            f"{PREFIX}/agents/faqs/refresh",
            json={"agent_ids": [agent.id]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["agents_processed"] == 1
        assert body["agents_failed"] == []

    def test_single_agent_refresh_persists_faqs(self, client, engine, monkeypatch):
        _patch_llm(monkeypatch)
        user, agent = _seed_agent(engine)

        client.post(
            f"{PREFIX}/agents/faqs/refresh",
            json={"agent_ids": [agent.id]},
        )

        with Session(engine) as db:
            faqs = db.exec(
                select(AgentFAQ).where(AgentFAQ.agent_id == agent.id)
            ).all()
        assert len(faqs) > 0

    def test_single_agent_refresh_returns_200_with_failed_for_unknown_agent(self, client):
        resp = client.post(
            f"{PREFIX}/agents/faqs/refresh",
            json={"agent_ids": ["does-not-exist"]},
        )
        assert resp.status_code == 200
        assert resp.json()["agents_processed"] == 0
        assert "does-not-exist" in resp.json()["agents_failed"]

# POST /agents/faqs/refresh — bulk mode

class TestRefreshFaqsBulk:
    def test_bulk_refresh_multiple_agents(self, client, engine, monkeypatch):
        _patch_llm(monkeypatch)
        with Session(engine) as db:
            user = make_user(db)
            agent1 = make_agent(db, user.id, cold_build=True)
            agent2 = make_agent(db, user.id, cold_build=False)
            make_agent_settings(db, agent1.id)
            make_agent_settings(db, agent2.id)
            db.commit()
            
            agent_ids = [agent1.id, agent2.id]

        resp = client.post(
            f"{PREFIX}/agents/faqs/refresh",
            json={"agent_ids": agent_ids},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["agents_processed"] == 2
        assert body["agents_failed"] == []

# POST /agents/faqs/refresh — validation

class TestRefreshFaqsValidation:
    def test_422_when_agent_ids_empty(self, client):
        resp = client.post(
            f"{PREFIX}/agents/faqs/refresh",
            json={"agent_ids": []},
        )
        # Assuming Pydantic catches min_length=1
        assert resp.status_code == 422

# Response schema shape

class TestRefreshResponseSchema:
    def test_response_has_all_required_fields(self, client, engine, monkeypatch):
        _patch_llm(monkeypatch)
        user, agent = _seed_agent(engine)

        resp = client.post(
            f"{PREFIX}/agents/faqs/refresh",
            json={"agent_ids": [agent.id]},
        )

        body = resp.json()
        assert "success" in body
        assert "message" in body
        assert "agents_processed" in body
        assert "agents_failed" in body
        assert isinstance(body["agents_failed"], list)

    def test_faq_item_has_question_and_source(self, client, engine):
        # Seed exactly TARGET_FAQ_COUNT FAQs so the cache is satisfied
        # and no LLM regeneration happens
        with Session(engine) as db:
            user = make_user(db)
            agent = make_agent(db, user.id)
            for i in range(6):
                q = "Schema test FAQ?" if i == 0 else f"Other FAQ {i}?"
                make_faq(db, agent.id, question=q, source="agent_config")
            db.commit()
            agent_id = agent.id

        resp = client.get(f"{PREFIX}/agents/{agent_id}/faqs")

        assert resp.status_code == 200
        items = resp.json()
        first = items[0]
        assert "question" in first
        assert "source" in first
        assert first["question"] == "Schema test FAQ?"
        assert first["source"] == "agent_config"
