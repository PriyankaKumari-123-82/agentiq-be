"""Tests for tool-selection loop termination (duplicate-call dedupe).

Reproduces the bug where the selector LLM re-emits an identical tool call
every iteration (same tool_name + same arguments) and the runtime happily
re-executes it until max_tool_calls is exhausted — burning 5x tokens and,
for write-tools, risking duplicate side effects.

The fix: _node_select_tools must NOT queue a (tool_name, arguments) pair that
has already been executed in this run; queueing nothing makes
_has_pending_tool_call route to format_response, terminating the loop.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_agent.core.agent_runtime import AgentRuntime, AgentState
from ai_agent.core.response_schemas import ToolCallBlock, ToolResultBlock


class _FakeLLM:
    """Minimal async LLM gateway stub returning a fixed selector response."""

    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self._content)


JIRA_TOOL = {
    "tool_name": "jira_prismberry_list_epic_by_project_name",
    "description": "List epics by project name",
    "input_schema": {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
    },
}
OTHER_TOOL = {
    "tool_name": "jira_prismberry_create_issue",
    "description": "Create an issue",
    "input_schema": {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
    },
}

EPIC_ARGS = {"payload": {"ProjectName": "Vision IQ"}}
EPIC_SELECTION_JSON = (
    '{"tool_name": "jira_prismberry_list_epic_by_project_name",'
    ' "arguments": {"payload": {"ProjectName": "Vision IQ"}}}'
)


def _make_state(**overrides) -> AgentState:
    defaults = dict(
        run_id="test-run",
        user_id=1,
        agent_id="test-agent",
        session_id=1,
        conversation_id="conv-1",
        messages=[{"role": "user", "content": "List all EPIC of Vision IQ."}],
        mcp_tools=[JIRA_TOOL, OTHER_TOOL],
    )
    defaults.update(overrides)
    return AgentState(**defaults)


def _executed_call(tool_name: str, arguments: dict, tool_id: str = "tc_1",
                   original_arguments: dict = None):
    """A tool call that has already run in this run.

    ``original_arguments`` mirrors what the runtime stamps at selection time
    (defaults to ``arguments``); pass it explicitly to simulate the SQL
    guardrail having rewritten ``arguments`` at execution time.
    """
    return ToolCallBlock(
        tool_name=tool_name,
        tool_id=tool_id,
        arguments=arguments,
        original_arguments=original_arguments if original_arguments is not None else arguments,
        status="completed",
    )


def _pending_calls(state: AgentState):
    return [tc for tc in state.tool_calls if tc.status == "pending"]


@pytest.mark.asyncio
async def test_duplicate_selection_is_not_requeued():
    """Selector repeats an identical, already-executed call -> queue nothing,
    so the loop terminates instead of re-running the tool."""
    runtime = AgentRuntime(
        db_session=MagicMock(), llm_gateway=_FakeLLM(EPIC_SELECTION_JSON)
    )
    state = _make_state(
        tool_calls=[_executed_call(JIRA_TOOL["tool_name"], EPIC_ARGS)],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_1",
                status="success",
                result=[{"type": "text", "text": '{"issues": []}'}],
                output_type="json",
            )
        ],
    )

    out = await runtime._node_select_tools(state)

    assert _pending_calls(out) == [], (
        "identical repeated call must not be re-queued"
    )
    assert runtime._has_pending_tool_call(out) == "format_response"


@pytest.mark.asyncio
async def test_duplicate_after_sql_guardrail_mutation_is_not_requeued():
    """Dedup must key on ORIGINAL args. The SQL guardrail rewrites the stored
    ToolCallBlock.arguments at execution time (e.g. drops a trailing ';'), but
    the selector re-emits the original SQL. Signing on original_arguments keeps
    the repeat matching so the loop still terminates for DB tools."""
    original = {"payload": {"sql": "SELECT * FROM issues;"}}   # what the LLM emits
    mutated = {"payload": {"sql": "SELECT * FROM issues"}}     # guardrail-cleaned
    selection = (
        '{"tool_name": "jira_prismberry_list_epic_by_project_name",'
        ' "arguments": {"payload": {"sql": "SELECT * FROM issues;"}}}'
    )
    runtime = AgentRuntime(db_session=MagicMock(), llm_gateway=_FakeLLM(selection))
    state = _make_state(
        tool_calls=[
            _executed_call(JIRA_TOOL["tool_name"], mutated, original_arguments=original)
        ],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_1",
                status="success",
                result=[{"type": "text", "text": "{}"}],
                output_type="json",
            )
        ],
    )

    out = await runtime._node_select_tools(state)

    assert _pending_calls(out) == [], (
        "repeat of the original SQL must dedup against original_arguments, not "
        "the guardrail-mutated stored arguments"
    )
    assert runtime._has_pending_tool_call(out) == "format_response"


@pytest.mark.asyncio
async def test_failed_call_can_be_retried():
    """A prior call whose RESULT was a failure (MCP isError=true, recorded with
    ToolCallBlock.status == 'completed') must NOT be treated as a duplicate —
    the selector is allowed to retry it."""
    runtime = AgentRuntime(
        db_session=MagicMock(), llm_gateway=_FakeLLM(EPIC_SELECTION_JSON)
    )
    state = _make_state(
        tool_calls=[_executed_call(JIRA_TOOL["tool_name"], EPIC_ARGS)],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_1",
                status="failure",
                result={},
                error="upstream 503",
            )
        ],
    )

    out = await runtime._node_select_tools(state)

    pending = _pending_calls(out)
    assert len(pending) == 1, "a failed call must remain retryable, not deduped"
    assert pending[0].tool_name == JIRA_TOOL["tool_name"]
    assert runtime._has_pending_tool_call(out) == "execute_tool"


@pytest.mark.asyncio
async def test_same_tool_different_args_is_queued():
    """Same tool with DIFFERENT arguments is legitimate and must still run."""
    selection = (
        '{"tool_name": "jira_prismberry_list_epic_by_project_name",'
        ' "arguments": {"payload": {"ProjectName": "AgentIQ"}}}'
    )
    runtime = AgentRuntime(db_session=MagicMock(), llm_gateway=_FakeLLM(selection))
    state = _make_state(
        tool_calls=[_executed_call(JIRA_TOOL["tool_name"], EPIC_ARGS)],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_1",
                status="success",
                result=[{"type": "text", "text": '{"issues": []}'}],
                output_type="json",
            )
        ],
    )

    out = await runtime._node_select_tools(state)

    pending = _pending_calls(out)
    assert len(pending) == 1
    assert pending[0].arguments == {"payload": {"ProjectName": "AgentIQ"}}
    assert runtime._has_pending_tool_call(out) == "execute_tool"


@pytest.mark.asyncio
async def test_new_tool_call_is_queued():
    """Regression guard: a fresh selection must queue a pending call."""
    runtime = AgentRuntime(
        db_session=MagicMock(), llm_gateway=_FakeLLM(EPIC_SELECTION_JSON)
    )
    state = _make_state()

    out = await runtime._node_select_tools(state)

    pending = _pending_calls(out)
    assert len(pending) == 1
    assert pending[0].tool_name == JIRA_TOOL["tool_name"]
    assert pending[0].arguments == EPIC_ARGS
    assert runtime._has_pending_tool_call(out) == "execute_tool"


@pytest.mark.parametrize(
    "done_variant",
    ["DONE", "DONE.", "`DONE`", '"DONE"', "Done", "DONE — all parts satisfied"],
)
@pytest.mark.asyncio
async def test_done_variants_terminate(done_variant):
    """Decorated DONE responses must terminate, not fall into tool matching."""
    runtime = AgentRuntime(db_session=MagicMock(), llm_gateway=_FakeLLM(done_variant))
    state = _make_state()

    out = await runtime._node_select_tools(state)

    assert _pending_calls(out) == []
    assert runtime._has_pending_tool_call(out) == "format_response"


@pytest.mark.asyncio
async def test_empty_response_queues_nothing():
    """Empty selector output must NOT match the first tool via '' substring
    fallback (reasoning models can return content='')."""
    runtime = AgentRuntime(db_session=MagicMock(), llm_gateway=_FakeLLM(""))
    state = _make_state()

    out = await runtime._node_select_tools(state)

    assert _pending_calls(out) == [], (
        "empty response must not select any tool"
    )
    assert runtime._has_pending_tool_call(out) == "format_response"
