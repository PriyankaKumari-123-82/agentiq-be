import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from ai_agent.core.agent_runtime import AgentRuntime, AgentState
from ai_agent.core.response_schemas import ToolCallBlock, ToolResultBlock, BlockType


@pytest.mark.asyncio
async def test_format_response_hides_schema_tools():
    """
    Test that schema/discovery tools do NOT output generic data tables
    and trigger a clean fallback response rather than dumping raw JSON.
    """
    db_session_mock = MagicMock()
    runtime = AgentRuntime(db_session=db_session_mock)

    # 1. Test case: ONLY schema tools ran
    state = AgentState(
        run_id="test-run-1",
        user_id=1,
        agent_id="test-agent",
        session_id=1,
        conversation_id="conv-1",
        tool_results_processed=False,
        response_content="",
        plan_text="",
        tool_calls=[
            ToolCallBlock(tool_name="get_schema", tool_id="tc_1", arguments={}),
        ],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_1",
                status="success",
                result=[{"column_name": "id", "data_type": "int"}],
                output_type="json"
            )
        ]
    )

    new_state = await runtime._node_format_response(state)

    # Assert fallback message is used instead of "See results below"
    assert new_state.response_content == "I have analyzed the database structure to understand its capabilities."
    
    # Assert NO table blocks were added
    table_blocks = [b for b in new_state.content_blocks if b.type == BlockType.TABLE]
    assert len(table_blocks) == 0


@pytest.mark.asyncio
async def test_format_response_uses_plan_text_for_schema_tools():
    """
    Test that if only schema tools ran, but we have a plan_text from the planner,
    that plan_text is used instead of the generic fallback.
    """
    db_session_mock = MagicMock()
    runtime = AgentRuntime(db_session=db_session_mock)

    state = AgentState(
        run_id="test-run-2",
        user_id=1,
        agent_id="test-agent",
        session_id=1,
        conversation_id="conv-1",
        tool_results_processed=False,
        response_content="",
        plan_text="I can assist you with tracking device lifecycles.",
        tool_calls=[
            ToolCallBlock(tool_name="describe_table", tool_id="tc_2", arguments={}),
        ],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_2",
                status="success",
                result=[{"col": "val"}],
                output_type="json"
            )
        ]
    )

    new_state = await runtime._node_format_response(state)

    assert new_state.response_content == "I can assist you with tracking device lifecycles."
    
    # Assert NO table blocks were added
    table_blocks = [b for b in new_state.content_blocks if b.type == BlockType.TABLE]
    assert len(table_blocks) == 0


@pytest.mark.asyncio
async def test_format_response_renders_data_but_hides_schema():
    """
    Test that if BOTH schema tools and data tools run, the schema tools
    are excluded from table rendering, but data tools are rendered.
    """
    db_session_mock = MagicMock()
    runtime = AgentRuntime(db_session=db_session_mock)

    state = AgentState(
        run_id="test-run-3",
        user_id=1,
        agent_id="test-agent",
        session_id=1,
        conversation_id="conv-1",
        tool_results_processed=False,
        response_content="",
        plan_text="",
        tool_calls=[
            ToolCallBlock(tool_name="get_schema", tool_id="tc_schema", arguments={}),
            ToolCallBlock(tool_name="query_db", tool_id="tc_data", arguments={}),
        ],
        tool_results=[
            ToolResultBlock(
                tool_call_id="tc_schema",
                status="success",
                result=[{"column_name": "id", "data_type": "int"}],
                output_type="json"
            ),
            ToolResultBlock(
                tool_call_id="tc_data",
                status="success",
                result=[{"customer_name": "Alice", "revenue": 100}],
                output_type="json"
            )
        ]
    )

    new_state = await runtime._node_format_response(state)

    # Since it's not ONLY schema tools, it should use the generic fallback for data
    assert new_state.response_content == "I retrieved the requested data. See the results below."
    
    # Only ONE table block should be added (for query_db, not get_schema)
    table_blocks = [b for b in new_state.content_blocks if b.type == BlockType.TABLE]
    assert len(table_blocks) == 1
    
    # The table block should contain the data from query_db
    table_block = table_blocks[0].content
    assert table_block.columns[0].name == "customer_name"
    assert table_block.columns[1].name == "revenue"
    assert table_block.rows[0] == ["Alice", 100]
