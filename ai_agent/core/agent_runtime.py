"""Agent Runtime using LangGraph for agentic execution.

This module provides a state-machine-based agent runtime that supports:
- Multi-step planning and execution
- Tool calling with MCP integration
- RAG with LlamaIndex
- Streaming responses
- Usage tracking and observability
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Callable, Union
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sqlmodel import Session, select

from ai_agent.core.llm_gateway import LLMGateway, get_llm_gateway, is_llm_quota_error
from ai_agent.core.usage_tracker import get_usage_tracker
from ai_agent.core.enums import LLMOperation
from ai_agent.core.response_schemas import (
    AgentResponse, ContentBlock, BlockType, TableBlock, TableColumn,
    ChartBlock, ChartDataset,                      
    ArtifactSummary, CitationBlock, ToolCallBlock, ToolResultBlock,
    ReasoningBlock, UsageInfo, ResponseMetadata, StreamingEvent
)
from ai_agent.services.source_attribution import SourceRegistry
from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.models import (
    AIAgent, AIAgentSettings, UserSession, Message, AgentRun, AgentRunStep,
    RunStatus, RunMode, StepType, LLMUsageEvent, Artifact, ModelProvider
)
from ai_agent.core.config import settings
from ai_agent.services.web_search import get_web_search_service
from ai_agent.helpers.date_formatter import transform_dates_in_obj
from ai_agent.core.url_shortener import shorten_urls_in_obj

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentState:
    """State maintained throughout an agent run."""
    # Run context
    run_id: str
    user_id: int
    agent_id: str
    session_id: Optional[int]
    conversation_id: str
    input_message_id: Optional[int] = None
    mode: Optional[str] = None
    
    # Agent configuration
    agent: Optional[AIAgent] = None
    agent_settings: Optional[AIAgentSettings] = None
    api_key: Optional[str] = None
    litellm_model_string: Optional[str] = None
    
    # Messages and context
    messages: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    context_from_retrieval: str = ""
    # Transitional web-search context. Keep this until web results are fully
    # supplied through the source-attribution registry.
    context_from_web_search: str = ""
    # Evidence available to the answer LLM during this run.  This remains
    # internal until a later citation-validation step selects used sources.
    source_registry: SourceRegistry = field(default_factory=SourceRegistry)

    # Chart specification decided by the LLM.
    # Important: this is NOT chart data. It only describes intent/type/title
    # and which tool-result columns should be mapped into a ChartBlock.
    chart_spec: Optional[Dict[str, Any]] = None
    
    # Execution tracking
    current_step: int = 0
    max_steps: int = 10
    tool_calls_made: int = 0
    max_tool_calls: int = 5
    
    # Results
    # ``plan_text`` holds internal planner output (never shown to users).
    # ``response_content`` holds the user-visible answer produced by the
    # observe node after processing tool results (or a direct LLM answer).
    plan_text: str = ""
    response_content: str = ""
    content_blocks: List[ContentBlock] = field(default_factory=list)
    citations: List[CitationBlock] = field(default_factory=list)
    tool_calls: List[ToolCallBlock] = field(default_factory=list)
    tool_results: List[ToolResultBlock] = field(default_factory=list)
    artifacts: List[ArtifactSummary] = field(default_factory=list)
    
    # Usage tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # User identity (populated from Keycloak token at run start)
    user_email: Optional[str] = None

    # Status
    status: str = "pending"
    error: Optional[str] = None
    tool_results_processed: bool = False
    # Accumulated human-readable observations from all tool results so far.
    # Populated incrementally by _node_observe; consumed once by _node_format_response.
    observations: List[str] = field(default_factory=list)

    # MCP tools
    mcp_tools: List[Dict] = field(default_factory=list)
    mcp_servers: Dict[str, bool] = field(default_factory=dict)

    # DB-aware MCP: cached schema context injected into planning/selection
    # prompts so the LLM can generate SQL without re-fetching the schema on
    # every turn.  ``db_server_names`` is the set of MCP server names that
    # have ``requires_query_building=True``; the runtime applies SQL
    # guardrails before executing tools on these servers.
    schema_context: str = ""
    db_server_names: set = field(default_factory=set)

    # Per-step wall-clock timings (populated by the runtime timing wrappers).
    # Key: node name, Value: elapsed seconds (float).
    step_timings: Dict[str, float] = field(default_factory=dict)
    # How many times each node was invoked.
    step_counts: Dict[str, int] = field(default_factory=dict)
    # Ordered list of node names in the sequence they were called.
    step_sequence: List[str] = field(default_factory=list)


class AgentRuntime:
    """LangGraph-based agent runtime for agentic execution."""

    # Max time to wait for a single MCP tool call before giving up. Pluggy
    # has been observed to leave a chunked response half-read indefinitely
    # on unbounded queries; this bounds the damage to one failed tool call
    # instead of hanging the whole request. Comfortably above the slowest
    # legitimate query seen in testing (~7s) with headroom for real variance.
    MCP_CALL_TOOL_TIMEOUT_S = 45.0

    # Cap unbounded list-query results so a single SELECT * can't pull
    # megabytes of data through the MCP round trip. format_response only
    # ever shows the user a preview of the first 10 rows anyway.
    MAX_SQL_ROW_LIMIT = 100

    def __init__(self, db_session: Session, llm_gateway: Optional[LLMGateway] = None):
        """Initialize the Agent Runtime.
        
        Args:
            db_session: SQLModel database session
            llm_gateway: Optional LLM gateway instance
        """
        self.db = db_session
        self.llm = llm_gateway or get_llm_gateway()
        self.mcp_manager = get_mcp_manager()
        self.usage_tracker = get_usage_tracker(db_session)
        self.web_search_service = get_web_search_service(db_session)
        
        # Build the execution graph
        self.graph = self._build_graph()

        # Wall-clock start time set at the beginning of each run() call so
        # the final timing summary can include total elapsed time.
        self._run_start_time: float = 0.0

        log.info("AgentRuntime initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for agent execution."""
        
        # Define the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (each wrapped with a timing decorator)
        workflow.add_node("load_context", self._timed("load_context", self._node_load_context))
        workflow.add_node("web_search", self._timed("web_search", self._node_web_search))
        workflow.add_node("retrieve", self._timed("retrieve", self._node_retrieve))
        workflow.add_node("plan", self._timed("plan", self._node_plan))
        workflow.add_node("select_tools", self._timed("select_tools", self._node_select_tools))
        workflow.add_node("execute_tool", self._timed("execute_tool", self._node_execute_tool))
        workflow.add_node("observe", self._timed("observe", self._node_observe))
        workflow.add_node("format_response", self._timed("format_response", self._node_format_response))
        workflow.add_node("guardrail", self._timed("guardrail", self._node_guardrail))
        
        # Define edges
        workflow.set_entry_point("load_context")
        
        # From load_context: check if web search is enabled
        workflow.add_conditional_edges(
            "load_context",
            lambda state: (
                "web_search" 
                if (state.agent_settings 
                    and state.agent_settings.allow_websearch 
                    and state.agent_settings.websearch_provider)
                else "retrieve"
            ),
            {
                "web_search": "web_search",
                "retrieve": "retrieve",
            }
        )
        
        workflow.add_edge("web_search", "retrieve")
        
        # From retrieve: always plan
        workflow.add_edge("retrieve", "plan")
        
        # From plan: select tools or format directly
        workflow.add_conditional_edges(
            "plan",
            self._should_use_tools,
            {
                "select_tools": "select_tools",
                "format_response": "format_response",
            }
        )
        
        # From select_tools: execute tools if a new pending call was added,
        # otherwise the LLM said DONE so we proceed straight to formatting.
        workflow.add_conditional_edges(
            "select_tools",
            self._has_pending_tool_call,
            {
                "execute_tool": "execute_tool",
                "format_response": "format_response",
            }
        )

        # From execute_tool: observe
        workflow.add_edge("execute_tool", "observe")

        # From observe: keep looping back to select_tools as long as we
        # haven't hit max_tool_calls. select_tools (and its conditional edge)
        # decides each step whether another tool is needed or the plan is
        # complete.
        workflow.add_conditional_edges(
            "observe",
            self._should_continue_tools,
            {
                "select_tools": "select_tools",
                "format_response": "format_response"
            }
        )
        
        # From format_response: guardrail check
        workflow.add_edge("format_response", "guardrail")
        
        # From guardrail: end
        workflow.add_edge("guardrail", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def _check_token_policy(self, state: AgentState) -> None:
        """Enforce token limits before making an LLM call."""
        if state.mode == "monitor":
            return
        try:
            from ai_agent.core.token_governance import check_token_policy
            from ai_agent.models import User
            user = self.db.get(User, state.user_id)
            organization_id = user.active_organization_id if user else None
            check_token_policy(self.db, state.user_id, state.agent_id, organization_id)
        except Exception as e:
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                raise
            log.error(f"[{state.run_id}] Token policy check failed: {e}")

    def _track_llm_usage(
        self,
        state: AgentState,
        response: Any,
        *,
        operation: LLMOperation,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Record token/cost usage for an LLM response returned by self.llm.chat().

        Updates state.total_input_tokens/output_tokens/cost_usd and writes a
        matching LLMUsageEvent row. Call this once right after any
        self.llm.chat(...) call, from any node.
        """
        def _get(field, default=None):
            if isinstance(response, dict):
                return response.get(field, default)
            return getattr(response, field, default)

        input_tokens = _get("input_tokens", 0) or 0
        output_tokens = _get("output_tokens", 0) or 0
        cost_usd = _get("cost_usd", 0.0) or 0.0
        cached_tokens = _get("cached_tokens", 0) or 0
        reasoning_tokens = _get("reasoning_tokens", 0) or 0
        latency_ms = _get("latency_ms", None)

        state.total_input_tokens += input_tokens
        state.total_output_tokens += output_tokens
        state.total_cost_usd += cost_usd

        # Monitor runs do not create an AgentRun row, so inserting an
        # LLMUsageEvent with that run_id would violate the FK constraint and
        # poison the DB session.  Skip the DB write for monitor mode; the
        # in-memory token/cost counters above are still updated.
        if state.mode == "monitor":
            return

        try:
            # Get user's organization_id
            from ai_agent.models import User
            user = self.db.get(User, state.user_id)
            organization_id = user.active_organization_id if user else None

            self.usage_tracker.record_llm_usage(
                user_id=state.user_id,
                agent_id=state.agent_id,
                provider=provider or settings.LLM_PROVIDER,
                model=model or settings.LLM_MODEL,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                run_id=state.run_id,
                session_id=state.session_id,
                conversation_id=state.conversation_id,
                message_id=state.input_message_id,
                latency_ms=latency_ms,
                organization_id=organization_id,
                temperature=temperature,
            )
        except Exception:
            self.db.rollback()
            operation_name = operation.value if isinstance(operation, LLMOperation) else operation
            log.exception(f"[{state.run_id}] Failed to record LLM usage event ({operation_name})")

    async def _node_load_context(self, state: AgentState) -> AgentState:
        """Load conversation context and agent configuration."""
        log.info(f"[{state.run_id}] Loading context")
        
        # Load agent settings
        agent = self.db.get(AIAgent, state.agent_id)
        agent_settings = self.db.query(AIAgentSettings).filter(
            AIAgentSettings.agent_id == state.agent_id
        ).first()
        
        state.agent = agent
        state.agent_settings = agent_settings
        if agent_settings:
            model_provider = self.db.exec(
                select(ModelProvider).where(
                    ModelProvider.provider == agent_settings.llm_provider,
                    ModelProvider.model_name == agent_settings.llm_model,
                    ModelProvider.type == "llm",
                    ModelProvider.is_active == True,
                )
            ).first()
            state.litellm_model_string = (
                f"{model_provider.litellm_prefix.strip()}/{model_provider.model_name}"
            )
        
        # Build system prompt
        if agent_settings and agent_settings.llm_system_prompt:
            state.system_prompt = agent_settings.llm_system_prompt
        else:
            state.system_prompt = settings.LLM_SYSTEM_PROMPT or "You are a helpful AI assistant."
        
        # Load MCP tools if enabled
        if agent_settings and agent_settings.is_mcp_enabled:
            try:
                state.mcp_tools = await self._load_mcp_tools(agent_settings)
            except Exception as e:
                log.warning(f"Failed to load MCP tools: {e}")
                state.mcp_tools = []

        # Register built-in tools that are always available regardless of MCP
        # configuration. These let the agent produce artifacts (downloadable
        # files) as part of its plan. They are executed by the runtime itself
        # (not the MCP manager) and are distinguished by ``server_name ==
        # "__builtin__"``.
        state.mcp_tools.extend(self._builtin_tools())

        # -----------------------------------------------------------------
        # DB-aware MCP: detect servers with requires_query_building=True,
        # fetch their schemas (from cache when possible), and build an
        # optimized schema context for the LLM.  This happens once per run
        # and the schema_context is injected into the planning and tool-
        # selection prompts so the agent can generate SQL without separate
        # schema-fetching tool calls.
        # -----------------------------------------------------------------
        db_servers: set[str] = set()
        for tool in state.mcp_tools:
            if tool.get("requires_query_building"):
                db_servers.add(tool["server_name"])

        if db_servers:
            state.db_server_names = db_servers
            conversation_lines = []

            for msg in reversed(state.messages):
                if msg.get("role") == "user":
                    conversation_lines.append(msg.get("content", ""))
                    if len(conversation_lines) == 20:
                        break

            conversation_lines.reverse()

            user_query = "\n".join(conversation_lines)
            await self._load_db_schema_context(state, db_servers, user_query)

        state.current_step += 1
        return state

    async def _node_web_search(self, state_data) -> AgentState:
        """Execute web search and add context to state."""
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data
            
        log.info(f"[{state.run_id}] Executing web search")
        
        # Check if web search is enabled
        if not (state.agent_settings and state.agent_settings.allow_websearch):
            log.info(f"[{state.run_id}] Web search disabled, skipping")
            state.current_step += 1
            return state
        
        try:
            # Get last user message
            user_message = None
            for msg in reversed(state.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            
            # Execute web search
            search_result = await self.web_search_service.search(
                query=user_message,
                agent_settings=state.agent_settings,
                max_results=5,
            )
            
            # Store results in state - extract content from results
            if search_result.results:
                # Combine all result content into context
                content_parts = []
                for item in search_result.results:
                    content_parts.append(f"Source: {item.title}\n{item.content}")
                
                state.context_from_web_search = "\n\n".join(content_parts)
                log.info(f"[{state.run_id}] Web search found {len(search_result.results)} results, {len(state.context_from_web_search)} chars")
                
        except Exception as e:
            log.error(f"[{state.run_id}] Web search failed: {e}", exc_info=True)
            
            
        state.current_step += 1
        return state

    async def _node_retrieve(self, state_data) -> AgentState:
        """Retrieve relevant context from vector store using LlamaIndex.
        
        """
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data
            
        log.info(f"[{state.run_id}] Vector retrieval")
        
        from ai_agent.core.rag_service import RAGService
        
        rag_service = RAGService()
        
        try:
            # Get last user message
            user_message = None
            for msg in reversed(state.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if user_message and state.agent:
                result = await rag_service.retrieve(
                    query=user_message,
                    agent_id=state.agent_id,
                    agent_settings=state.agent_settings,
                    top_k=settings.VECTOR_MAX_CHUNKS,
                    db_session=self.db,
                    session_id=state.session_id,
                )
                
                state.context_from_retrieval = result.get("context", "")
                state.citations = result.get("citations", [])
                
                # Add to total tokens (approximate)
                state.total_input_tokens += result.get("tokens_used", 0)
                
                log.info(f"[{state.run_id}] Retrieved {len(state.citations)} citations")
                
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            # Don't fail the run, just continue without context
            
        state.current_step += 1
        return state

    async def _node_plan(self, state_data) -> AgentState:
        """Plan the approach to answer the user."""
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data

        log.info(f"[{state.run_id}] Planning")

        # Build messages with system prompt and context
        messages = [{"role": "system", "content": state.system_prompt}]

        # Add web search context if available
        if state.context_from_web_search:
            messages.append({
                "role": "system",
                "content": f"Web Search Results:\n{state.context_from_web_search}"
            })

        # Add retrieval context if available (currently disabled)
        if state.context_from_retrieval:
            messages.append({
                "role": "system",
                "content": f"Retrieved context:\n{state.context_from_retrieval}"
            })

        # Add conversation history
        messages.extend(state.messages)

        # Check if we need tools
        needs_tools = (
            bool(state.agent_settings and state.agent_settings.is_mcp_enabled)
            and bool(state.mcp_tools)
            and state.tool_calls_made < state.max_tool_calls
        )

        if needs_tools:
            # Add planning prompt (includes chart guidance)
            messages.append({
                "role": "system",
                "content": self._build_planning_prompt(state)
            })
        else:
            # LLM-chart: no MCP tools available. Inject chart hint.
            # If RAG context was retrieved, instruct LLM to prioritise it.
            messages.append({
                "role": "system",
                "content": self._build_chart_prompt("llm_chart"),
            })

        try:
            self._check_token_policy(state)
            temperature = state.agent_settings.llm_temperature
            response = await self.llm.chat(
                messages=messages,
                provider=state.agent_settings.llm_provider if state.agent_settings else None,
                model=state.litellm_model_string,
                temperature=temperature,
                user_id=str(state.user_id),
                trace_id=state.run_id,
                api_key=state.api_key,
            )

            self._track_llm_usage(
                state, response,
                operation=LLMOperation.PLANNING,
                provider=state.agent_settings.llm_provider if state.agent_settings else None,
                model=state.agent_settings.llm_model if state.agent_settings else None,
                temperature=temperature,
            )

            # Handle both dict and object response types
            if isinstance(response, dict):
                state.response_content = response.get("content", "")
            else:
                state.response_content = getattr(response, "content", "")

            # Store plan in plan_text (internal only) so the selector LLM
            # can read it, but never expose it as user-visible response_content.
            if needs_tools and state.response_content:
                log.info(f"[{state.run_id}] Plan response content length: {len(state.response_content)}")
                log.info(f"[{state.run_id}] Plan response (first 500 chars): {state.response_content[:500]}")
                state.plan_text = state.response_content
                state.response_content = ""  # Clear so it never leaks as the user-visible answer

        except Exception as e:
            if is_llm_quota_error(e):
                raise
            log.error(f"Planning failed: {e}")
            state.error = str(e)

        state.current_step += 1
        return state

    async def _node_select_tools(self, state_data) -> AgentState:
        """Select which tools to call based on the plan and current state."""
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data

        log.info(f"[{state.run_id}] Selecting tools")

        # Build tool selection prompt based on plan and previous results.
        # Include the input schema so the LLM uses the correct argument shape.
        import json as _json
        tool_descriptions = []
        for tool in state.mcp_tools:
            schema = tool.get("input_schema") or tool.get("inputSchema") or {}
            try:
                schema_str = _json.dumps(schema, ensure_ascii=False)
            except Exception:
                schema_str = str(schema)
            tool_descriptions.append(
                f"- {tool['tool_name']}: {tool.get('description', 'No description')}\n"
                f"  input_schema: {schema_str}"
            )

        # Use the dedicated plan_text field so we never accidentally serve
        # internal planning output to the user via response_content.
        plan = state.plan_text if state.plan_text else (state.response_content if state.response_content else "No plan available")

        # Get the CURRENT (most recent) user message. Earlier user messages
        # are conversation history/context only and must not drive intent for
        # the current turn.
        user_message = ""
        if state.messages:
            for msg in reversed(state.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

        # Include a concise summary of already-executed tool calls so the
        # selector LLM can pick the NEXT step (e.g. follow a data-fetch with
        # an export_file call) and knows when to say DONE.
        previous_results = ""
        export_file_already_done = False
        if state.tool_results:
            id_to_name = {tc.tool_id: tc.tool_name for tc in state.tool_calls}
            lines = []
            for result in state.tool_results[-5:]:
                tname = id_to_name.get(getattr(result, "tool_call_id", None), "unknown")
                status = getattr(result, "status", "?")
                preview = str(result.result)[:200] if result.result else "(empty)"
                lines.append(f"- {tname} [{status}]: {preview}")
                if tname == self.EXPORT_FILE_TOOL and status == "success":
                    export_file_already_done = True
            previous_results = "\n\nAlready-executed tool calls (in order):\n" + "\n".join(lines)
        if export_file_already_done:
            previous_results += "\n\nIMPORTANT: export_file has already been called successfully. Do NOT call export_file again. Respond with DONE."
        else:
            # Whenever the last data-fetch tool returned actual rows, back
            # it with a real downloadable file — regardless of row count.
            # The synthesis prompt always tells the user "the complete data
            # is available in the attached download" (whether it's showing
            # a 10-row preview of a larger set, or all 5 rows of a small
            # one), so that promise needs to hold even for small result sets.
            #
            # This used to be a prompt instruction telling the selector LLM
            # "your NEXT step MUST be export_file" — in practice GPT-4o would
            # sometimes see that instruction and still reply DONE, so the
            # download silently never happened. Forcing the call in code
            # instead of asking the LLM to remember to do it removes that
            # failure mode entirely.
            row_count = self._count_rows_in_last_tool_result(state)
            if row_count is not None and row_count > 0:
                last_tool_name, _ = self._last_data_tool_result(state)
                filename = self._default_export_filename(last_tool_name, "xlsx")
                log.info(
                    f"[{state.run_id}] Forcing export_file: last tool result "
                    f"had {row_count} rows"
                )
                export_args = {"artifact_type": "xlsx", "filename": filename}
                state.tool_calls.append(ToolCallBlock(
                    tool_name=self.EXPORT_FILE_TOOL,
                    tool_id=str(uuid4()),
                    arguments=export_args,
                    original_arguments=export_args,
                    status="pending",
                ))
                state.current_step += 1
                return state

        # Once a real data-fetch tool has already succeeded this run, the
        # remaining iterations only decide between export_file and DONE —
        # neither needs the DB schema. Skip injecting it (~3k tokens) on
        # those calls; only the very first call (deciding what SQL to run)
        # needs it.
        _SCHEMA_TOOL_KEYWORDS = ("get_schema", "get_tables", "describe", "schema")
        data_fetch_already_done = False
        if state.tool_results:
            id_to_name_for_schema = {tc.tool_id: tc.tool_name for tc in state.tool_calls}
            for result in state.tool_results:
                tname = id_to_name_for_schema.get(getattr(result, "tool_call_id", None), "")
                if (
                    getattr(result, "status", None) == "success"
                    and tname != self.EXPORT_FILE_TOOL
                    and not any(kw in tname.lower() for kw in _SCHEMA_TOOL_KEYWORDS)
                ):
                    data_fetch_already_done = True
                    break

        # When DB-aware servers are present, inject schema context so the
        # LLM can write correct SQL inside tool arguments.
        db_schema_section = ""
        if state.schema_context and not data_fetch_already_done:
            db_schema_section = f"""

--- DATABASE SCHEMA (for writing SQL arguments) ---
{state.schema_context}

SQL RULES (MUST FOLLOW)

1. READ ONLY
- Only generate SELECT or WITH ... SELECT queries.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, EXEC, MERGE, or any write operation.

2. ENUM VALUES ARE THE SOURCE OF TRUTH
- If a column shows [possible_values: 'X', 'Y', 'Z'] in the schema, use ONLY those exact values.
- The values are shown quoted; an empty string is shown as '' and must be matched with = '' (NOT IS NULL).
- Never invent, normalize, translate, or guess values.
- Example:
  warranty_status [possible_values: 'IN', 'OUT']
  → valid: warranty_status = 'IN'
  → invalid: warranty_status = 'Under Warranty'

3. UNKNOWN VALUES
- If a column has no [possible_values: ...] information and the user's request requires filtering on that column:
  - DO NOT guess string values.
  - DO NOT invent status names.
  - Prefer returning grouped/aggregated results for that column so the actual values can be inspected.
  - If a filter cannot be safely inferred, omit the filter rather than hallucinating a value.

4. COUNTS
- "How many records" → COUNT(*)
- "How many unique customers/products/etc." → COUNT(DISTINCT column)
- Do not use COUNT(DISTINCT ...) unless the user is explicitly asking for unique entities.

5. RANKINGS
- "Top", "highest", "most", "best", "largest"
  → GROUP BY + ORDER BY aggregate DESC
- Use LIMIT 1 when the user asks for a single winner.

6. PERCENTAGES
- Use:
  (SUM(CASE WHEN condition THEN 1 ELSE 0 END) * 100.0 / COUNT(*))
- Never hardcode percentage calculations.

7. JOINS
- Use only relationships provided in the schema context.
- Prefer explicit JOIN syntax.
- Do not invent joins.
- If no relationship is known, avoid joining.

8. NULL HANDLING
- Use IS NULL / IS NOT NULL.
- Never compare NULL using = or !=.

9. LIMITS
- List queries: LIMIT 100
- Aggregate/count queries: no LIMIT
- Ranking queries: use appropriate LIMIT

10. TABLES AND COLUMNS
- Use only tables and columns present in the schema context.
- Never invent table names.
- Never invent column names.

11. NO SCHEMA DISCOVERY
- Do not call schema, describe, metadata, or discovery tools.
- The schema provided above is the authoritative source.

12. WHEN IN DOUBT
- Prefer a broader query over a guessed query.
- Missing data is better than hallucinated data.
- Accuracy is more important than aggressiveness.

13. CARRY FORWARD FILTERS FROM THE PLAN
- The Plan above may reference a filter (a date, location, status, etc.)
  that was established earlier in the conversation even though the current
  "User message" does not repeat it.
- If the Plan states or implies such a filter, include it in the SQL WHERE
  clause — do not drop it just because the current user message is a bare
  follow-up (e.g. "show me status wise count").
- If the current user message introduces its own filter, that overrides
  anything carried from the Plan.

--- END DATABASE SCHEMA ---"""

        messages = [
            {
                "role": "system",
                "content": f"""You are a tool selector. Based on the plan, user message, and current state, determine the next tool to call and extract the required arguments.

CRITICAL: The "arguments" object you return MUST exactly match the tool's input_schema (use the exact property names and nesting shown in the schema). Do NOT invent argument names. If the schema has a nested object (e.g. "payload"), wrap your values inside that object.

Available tools (with their input_schema):
{chr(10).join(tool_descriptions)}

User message:
{user_message}

Plan:
{plan}
{previous_results}
{db_schema_section}

Guidance:
- If a previous tool already returned the data needed and the user asked for a downloadable file (CSV, Excel, PDF, JSON, Markdown, DOCX), your NEXT step MUST be the built-in `export_file` tool. Pass the rows you already retrieved via `rows` (for csv/xlsx), `content` (markdown for pdf/docx/md/txt/html), or `data` (for json). Choose a sensible `filename`.
- Do NOT call any tool again with the same arguments if it already appears in the "Already-executed tool calls" list above — that step is complete and its result is already available. Re-running it is never correct.
- As SOON as every part of the user's request has been satisfied (all needed data fetched AND any requested file produced), you MUST reply with the single literal word `DONE` and nothing else. Do NOT call another tool to double-check, re-verify, or "make sure". If every tool you would consider has already been executed, the answer is `DONE`.
- BEFORE writing the SQL: re-read the Plan above for any filter it mentions (a date, location, status, customer, etc.) that the short "User message" itself does not repeat. If the Plan mentions such a filter, your SQL's WHERE clause MUST include it — a bare follow-up like "show me status wise count" or "how many are still open" is asking you to apply a NEW aggregation/metric to the SAME filtered scope from the Plan, not to remove the filter.
  Example: Plan says "...jobsheets logged on 2026-06-29..." and User message is "show me status wise count" →
  correct: SELECT repair_status, COUNT(*) FROM jobsheet_data_view WHERE open_date = '2026-06-29' GROUP BY repair_status
  wrong (missing carried-over date filter): SELECT repair_status, COUNT(*) FROM jobsheet_data_view GROUP BY repair_status

Respond ONLY with a JSON object matching this format (no markdown, no commentary):
{{
  "tool_name": "<exact_tool_name>",
  "arguments": <object matching the tool's input_schema>
}}

If no further tool call is needed — including when the tools you would pick have already been executed — reply with the single literal word DONE and nothing else."""
            },
            {
                # A user turn is required: ZAI/GLM (and other strict providers)
                # reject a system-only messages array with "messages parameter
                # is illegal". OpenAI tolerates it, which is why this only broke
                # for non-OpenAI agents.
                "role": "user",
                "content": user_message or "Select the next tool to call, or respond DONE.",
            },
        ]

        try:
            # ``max_tokens`` must accommodate tool-arguments JSON. For most
            # tools this is small; for ``export_file`` the runtime auto-fills
            # the data payload from the previous tool result (see
            # ``_execute_builtin_tool``), so the LLM doesn't have to
            # regurgitate large datasets through JSON.
            #
            # Use the agent's configured provider/model (like plan and
            # synthesis do). Previously this call passed neither, so it
            # defaulted to settings.LLM_PROVIDER/LLM_MODEL (OpenAI/gpt-4o) —
            # meaning a Z.AI agent's tool selection was silently sent to OpenAI.
            sel_provider = state.agent_settings.llm_provider if state.agent_settings else None
            sel_model = state.agent_settings.llm_model if state.agent_settings else None
            temperature = state.agent_settings.llm_temperature if state.agent_settings else 0.0
            
            self._check_token_policy(state)
            response = await self.llm.chat(
                messages=messages,
                model=state.litellm_model_string,
                temperature=temperature,
                max_tokens=2000,
                api_key=state.api_key,
            )

            self._track_llm_usage(
                state, response,
                operation=LLMOperation.TOOL_SELECTION,
                provider=sel_provider,
                model=sel_model,
                temperature=temperature,
            )

            # Safely extract content from response
            if hasattr(response, 'content'):
                response_content = response.content.strip() if response.content else ""
            elif isinstance(response, dict):
                response_content = response.get('content', '').strip()
            else:
                response_content = str(response).strip()

            log.info(f"[{state.run_id}] LLM response: {response_content}")

            # Terminate the loop when the selector signals completion. Accept
            # the common decorated variants models emit ("DONE.", "`DONE`",
            # '"DONE"', "Done", "DONE — all parts satisfied") so a finished run
            # ends here instead of falling through to tool matching and burning
            # the remaining loop budget on repeated calls.
            _done_probe = response_content.strip().strip('`"\'*. ').strip()
            if re.match(r"^done\b", _done_probe, re.IGNORECASE):
                log.info(f"[{state.run_id}] LLM indicated no more tools needed")
                state.current_step += 1
                return state

            # An empty selector response must not reach the substring fallback
            # below, where "" matches the first tool via `"" in tool_name`.
            # Reasoning models can legitimately return empty content.
            if not response_content:
                log.info(f"[{state.run_id}] Empty selector response — no tool selected")
                state.current_step += 1
                return state

            # Try to parse JSON response by extracting the first balanced {...}
            def _extract_json_object(text: str):
                start = text.find('{')
                if start == -1:
                    return None
                depth = 0
                in_str = False
                escape = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if in_str:
                        if escape:
                            escape = False
                        elif ch == '\\':
                            escape = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return text[start:i + 1]
                return None

            json_str = _extract_json_object(response_content)
            if json_str:
                try:
                    import json
                    tool_selection = json.loads(json_str)
                    selected_tool = tool_selection.get('tool_name')
                    arguments = tool_selection.get('arguments', {})

                    log.info(f"[{state.run_id}] LLM selected tool: {selected_tool} with args: {arguments}")

                    # Signatures of tool calls already executed in THIS run.
                    # The selector LLM frequently re-emits an identical
                    # (tool_name + arguments) call every iteration; without this
                    # guard the runtime re-executes it until max_tool_calls is
                    # exhausted (5x tokens, and duplicate side effects for write
                    # tools). Queueing nothing makes _has_pending_tool_call route
                    # to format_response, terminating the loop.
                    def _call_sig(name, args):
                        try:
                            return (name, json.dumps(args or {}, sort_keys=True, ensure_ascii=False))
                        except Exception:
                            return (name, str(args))

                    # Only a SUCCESSFUL prior execution should suppress a repeat.
                    # execute_tool records ToolCallBlock.status == "completed"
                    # even when the tool RESULT came back as a failure (MCP
                    # isError=true), so keying on call status alone would wrongly
                    # block a legitimate retry. Map each call to its result
                    # status and dedup only against successful ones.
                    result_status_by_call = {
                        getattr(r, "tool_call_id", None): getattr(r, "status", None)
                        for r in state.tool_results
                    }

                    # Sign on the ORIGINAL (pre-guardrail) arguments. The SQL
                    # guardrail rewrites ToolCallBlock.arguments in place at
                    # execution time (strips fences/comments/trailing ';'), so
                    # comparing against the mutated stored args would miss a
                    # selector that re-emits the same original SQL — letting DB
                    # tool loops run to max iterations.
                    executed_sigs = set()
                    for tc in state.tool_calls:
                        if result_status_by_call.get(tc.tool_id) != "success":
                            continue
                        orig = getattr(tc, "original_arguments", None)
                        executed_sigs.add(
                            _call_sig(tc.tool_name, orig if orig is not None else tc.arguments)
                        )

                    # Find the tool
                    for tool in state.mcp_tools:
                        if tool["tool_name"] == selected_tool or (selected_tool and selected_tool.lower() in tool["tool_name"].lower()):
                            if _call_sig(tool["tool_name"], arguments) in executed_sigs:
                                log.info(
                                    f"[{state.run_id}] Skipping duplicate tool call already "
                                    f"executed this run: {tool['tool_name']} args={arguments}"
                                )
                                break
                            tool_call = ToolCallBlock(
                                tool_name=tool["tool_name"],
                                tool_id=str(uuid4()),
                                arguments=arguments,
                                original_arguments=arguments,
                                status="pending"
                            )
                            state.tool_calls.append(tool_call)
                            log.info(f"[{state.run_id}] Added tool call: {tool['tool_name']} with args: {arguments}")
                            break
                    else:
                        log.warning(f"[{state.run_id}] Tool not found: {selected_tool}")

                except Exception as e:
                    log.error(f"[{state.run_id}] Failed to parse tool selection JSON: {e}")
                    # Fallback: just use the tool name
                    selected_tool = response_content
                    for tool in state.mcp_tools:
                        if tool["tool_name"] == selected_tool or selected_tool.lower() in tool["tool_name"].lower():
                            tool_call = ToolCallBlock(
                                tool_name=tool["tool_name"],
                                tool_id=str(uuid4()),
                                arguments={},
                                status="pending"
                            )
                            state.tool_calls.append(tool_call)
                            log.info(f"[{state.run_id}] Added tool call (fallback): {tool['tool_name']}")
                            break
            else:
                # Fallback: treat entire response as tool name
                selected_tool = response_content
                for tool in state.mcp_tools:
                    if tool["tool_name"] == selected_tool or selected_tool.lower() in tool["tool_name"].lower():
                        tool_call = ToolCallBlock(
                            tool_name=tool["tool_name"],
                            tool_id=str(uuid4()),
                            arguments={},
                            status="pending"
                        )
                        state.tool_calls.append(tool_call)
                        log.info(f"[{state.run_id}] Added tool call (fallback): {tool['tool_name']}")
                        break
                else:
                    log.warning(f"[{state.run_id}] Tool not found: {selected_tool}")

        except Exception as e:
            if is_llm_quota_error(e):
                raise
            log.error(f"Tool selection failed: {e}")

        state.current_step += 1
        return state

    async def _node_execute_tool(self, state_data) -> AgentState:
        """Execute selected tools via MCP."""
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data
            
        log.info(f"[{state.run_id}] Executing tools")
        
        for tool_call in state.tool_calls:
            if tool_call.status != "pending":
                continue
                
            tool_call.status = "running"
            
            try:
                # Find the tool in MCP tools
                tool_def = None
                for tool in state.mcp_tools:
                    if tool["tool_name"] == tool_call.tool_name:
                        tool_def = tool
                        break
                
                if not tool_def:
                    raise ValueError(f"Tool {tool_call.tool_name} not found")

                # Built-in tools (e.g. ``export_file``) are executed directly
                # by the runtime so we can hand them the live AgentState
                # without round-tripping through MCP.
                if tool_def.get("server_name") == self.BUILTIN_SERVER:
                    result = await self._execute_builtin_tool(
                        state=state,
                        tool_name=tool_call.tool_name,
                        arguments=tool_call.arguments,
                    )
                else:
                    # -------------------------------------------------
                    # SQL guardrail for DB-aware MCP servers.
                    # Before sending the tool call to MCP, validate that
                    # any SQL argument is a safe, read-only query.
                    # -------------------------------------------------
                    server_name = tool_def.get("server_name", "")
                    if server_name in state.db_server_names:
                        tool_call.arguments = self._apply_sql_guardrail(
                            state, tool_call, tool_def,
                        )
                        if tool_call.status == "failed":
                            # Guardrail blocked this call; skip execution
                            # but consume budget to prevent infinite retries.
                            state.tool_calls_made += 1
                            continue

                    # Execute via MCP manager. Wrapped in a timeout so a
                    # hung/stuck MCP response (observed: Pluggy can leave a
                    # chunked HTTP response half-read indefinitely on an
                    # unbounded query) fails fast with a clear error instead
                    # of hanging the request forever.
                    try:
                        result = await asyncio.wait_for(
                            self.mcp_manager.call_tool(
                                server_name=tool_def["server_name"],
                                tool_name=tool_call.tool_name,
                                args=tool_call.arguments,
                            ),
                            timeout=self.MCP_CALL_TOOL_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"MCP call to '{tool_call.tool_name}' on server "
                            f"'{tool_def['server_name']}' timed out after "
                            f"{self.MCP_CALL_TOOL_TIMEOUT_S}s"
                        )

                # Create result block
                tool_result = ToolResultBlock(
                    tool_call_id=tool_call.tool_id,
                    status="success" if not result.get("isError") else "failure",
                    result=result.get("content", []),
                    output_type="json" if isinstance(result.get("content"), list) else "text"
                )
                
                state.tool_results.append(tool_result)
                tool_call.status = "completed"
                state.tool_calls_made += 1
                
            except Exception as e:
                log.error(f"Tool execution failed: {e}")
                tool_call.status = "failed"
                state.tool_results.append(ToolResultBlock(
                    tool_call_id=tool_call.tool_id,
                    status="failure",
                    result={},
                    error=str(e)
                ))
        
        state.current_step += 1
        return state

    async def _node_observe(self, state_data) -> AgentState:
        """Accumulate tool results into state.observations for later synthesis.

        The LLM synthesis (answer formulation) is intentionally deferred to
        _node_format_response, which runs exactly once after the tool loop
        completes. This eliminates the N redundant LLM calls that previously
        happened on every intermediate observe step.
        """
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data

        log.info(f"[{state.run_id}] Observing tool results")

        # Only process results that haven't been observed yet by tracking the
        # count we've already added to state.observations.
        already_observed = len(state.observations)
        new_observations = []

        # Keywords that identify schema/discovery tools whose raw output is only
        # needed during SQL generation (select_tools) — NOT for final synthesis.
        # select_tools reads from state.messages (updated below), so these results
        # are still fully available there; we only exclude them from state.observations
        # which feeds the synthesis LLM in _node_format_response.
        _SCHEMA_TOOL_KEYWORDS = ("get_schema", "get_tables", "describe", "schema")
        _MAX_OBS_CHARS = 40_000

        # Build lookup: tool_call_id -> tool_name for filtering.
        id_to_tool_name: dict = {tc.tool_id: tc.tool_name for tc in state.tool_calls}

        for result in state.tool_results[already_observed:]:
            tool_name = id_to_tool_name.get(getattr(result, "tool_call_id", None), "")

            # Exclude schema/discovery results from synthesis — they're internal
            # helpers for SQL generation and not needed in the final answer prompt.
            if any(kw in tool_name.lower() for kw in _SCHEMA_TOOL_KEYWORDS):
                log.info(
                    f"[{state.run_id}] Excluded '{tool_name}' result from synthesis (discovery tool)"
                )
                continue

            if result.status == "success":
                # Extract actual data from generic MCP format if present
                result_data = result.result
                if isinstance(result_data, list) and len(result_data) > 0:
                    first_item = result_data[0]
                    if isinstance(first_item, dict) and set(first_item.keys()) == {"type", "text", "data", "mimeType"}:
                        # Extract odata JSON from the "text" field
                        if first_item.get("type") == "text" and first_item.get("text"):
                            result_data = first_item["text"]
                            log.info(f"[{state.run_id}] Extracted odata JSON from MCP format")

                # Programmatically convert any UTC timestamps to IST in tool
                # outputs before the synthesis LLM ever sees the data.
                result_data = transform_dates_in_obj(result_data)

                # Shorten all http/https URLs in the tool result concurrently.
                # This is the single global hook — every downstream consumer
                # (LLM synthesis, table preview, Excel export, WhatsApp) reads
                # from result.result, so they all receive short TinyURLs for free.
                result_data = await shorten_urls_in_obj(result_data)

                # Write the transformed data back so the export_file built-in
                # (and any other consumer that reads result.result directly)
                # also gets the clean, shortened version.
                result.result = result_data

                # Strip empty/null values from tabular results so the LLM
                # doesn't report columns as "unavailable" when they are merely
                # blank in the source data.
                if isinstance(result_data, list) and result_data and isinstance(result_data[0], dict):
                    result_data = [
                        {k: v for k, v in row.items() if v is not None and v != ""}
                        for row in result_data
                    ]
                obs = f"Tool result: {json.dumps(result_data, indent=2)}"
                if len(obs) > _MAX_OBS_CHARS:
                    original_len = len(obs)
                    obs = obs[:_MAX_OBS_CHARS] + f"\n...[truncated, {original_len - _MAX_OBS_CHARS} characters omitted]"
                    log.info(
                        f"[{state.run_id}] Truncated observation from '{tool_name}': "
                        f"{original_len} -> {_MAX_OBS_CHARS} chars"
                    )
                new_observations.append(obs)
            else:
                new_observations.append(f"Tool error: {result.error}")

        if new_observations:
            state.observations.extend(new_observations)
            # Keep the message history up-to-date so select_tools sees latest
            # results when deciding the next step.
            state.messages.append({
                "role": "system",
                "content": f"Tool execution results:\n{chr(10).join(new_observations)}"
            })
            log.info(
                f"[{state.run_id}] Accumulated {len(new_observations)} new observation(s); "
                f"total={len(state.observations)}"
            )

        state.current_step += 1
        return state

    async def _node_format_response(self, state_data) -> AgentState:
        """Format the final response with content blocks."""
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data

        log.info(f"[{state.run_id}] Formatting response")
        log.info(f"[{state.run_id}] response_content length: {len(state.response_content) if state.response_content else 0}")
        log.info(f"[{state.run_id}] tool_results_processed: {state.tool_results_processed}")
        log.info(f"[{state.run_id}] tool_results count: {len(state.tool_results)}")

        # ---------------------------------------------------------------
        # LLM synthesis — runs exactly ONCE here, after all tool iterations
        # are complete. state.observations holds every unwrapped tool result
        # accumulated by _node_observe across all loop iterations.
        # ---------------------------------------------------------------
        if state.observations and not state.tool_results_processed:
            # Get the CURRENT (most recent) user message.
            user_message = ""
            for msg in reversed(state.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            # If the user is asking for a downloadable file (CSV / Excel /
            # PDF / JSON / Markdown / DOCX), the runtime will produce a
            # proper artifact via the export_file tool and render a download
            # card in the UI. In that case, this answer-formulation step
            # should NOT inline the file contents in a code block; it should
            # produce a brief natural-language confirmation instead, leaving
            # the actual file delivery to the artifact pipeline.
            file_request_hint = (
                "If the user asked for a table or Excel/CSV export, ALWAYS "
                "display the data as a markdown table in your response. If "
                "the dataset has more than 10 rows, show only the first 10 "
                "rows in the table (a preview) — do NOT inline the full "
                "dataset. Follow the preview table with a short line telling "
                "the user this is a preview and the complete data is "
                "available in the attached download (e.g. 'Here's a preview "
                "of the first 10 rows — download the file below for the full "
                "data.'). If the dataset has 10 rows or fewer, show all rows "
                "and just add a short confirmation that the file is attached "
                "for download. For other file types (PDF, DOCX, Markdown), "
                "do NOT paste the file contents — just give a short "
                "confirmation and let the download card handle delivery."
            )
            column_display_hint = (
                "When presenting tabular data to the user, make column headers "
                "human-readable instead of copying raw database field names. "
                "Convert snake_case, camelCase, PascalCase, and kebab-case "
                "headers into display labels. Examples: job_id -> Job ID, "
                "job_no -> Job No, location_code -> Location Code, "
                "open_date -> Open Date, createdAt -> Created At, "
                "userURL -> User URL. Keep common acronyms such as ID, URL, "
                "API, SQL, UUID, and IP uppercase. Do not show raw database "
                "column names in user-facing tables unless the user explicitly "
                "asks for raw schema, SQL, or exact field names. "
                "Do NOT include 'Show As' (or showAs / show_as) columns in event or calendar tables."
            )
            missing_value_hint = (
                "Never display null, None, NaN, or missing empty values to the "
                "user. In markdown tables, show missing values as "
                "'Not Available'. In prose, omit missing fields unless they "
                "are directly relevant to the answer."
            )

            # ── CHART SPEC INSTRUCTION ───────────────────────────────────────
            # For MCP/tool-backed data, the LLM must decide ONLY the chart
            # intent/type/title/column mapping. The backend will build the
            # actual ChartBlock from raw MCP/tool rows. This avoids LLM-created
            # chart data.
            chart_hint = self._build_chart_prompt("runtime_chart")

            process_prompt = f"""Answer the user's question directly using the data below. Do not mention tools, plans, or how you obtained the data. Speak naturally and confidently.

{file_request_hint}

{column_display_hint}

{missing_value_hint}

{chart_hint}

User's question: {user_message}

Available data:
{chr(10).join(state.observations)}

Provide a clear, direct answer. If calculations are needed, perform them and present the result. If specific data isn't available, briefly note what's missing without asking the user to run tools."""

            try:
                log.info(f"[{state.run_id}] Calling LLM to synthesize tool results (once)")
                log.info(f"[{state.run_id}] api_key is set: {state.api_key is not None}")
                log.info(f"[{state.run_id}] agent_settings is set: {state.agent_settings is not None}")

                temperature = state.agent_settings.llm_temperature
                self._check_token_policy(state)
                response = await self.llm.chat(
                    messages=[{"role": "user", "content": process_prompt}],
                    provider=state.agent_settings.llm_provider if state.agent_settings else None,
                    model=state.litellm_model_string,
                    temperature=temperature,
                    user_id=str(state.user_id),
                    trace_id=state.run_id,
                    api_key=state.api_key,
                )

                self._track_llm_usage(
                    state, response,
                    operation=LLMOperation.SYNTHESIS,
                    provider=state.agent_settings.llm_provider if state.agent_settings else None,
                    model=state.agent_settings.llm_model if state.agent_settings else None,
                    temperature=temperature,
                )

                if hasattr(response, 'content'):
                    state.response_content = response.content
                elif isinstance(response, dict):
                    state.response_content = response.get('content', '')

                chart_spec, cleaned_content = self._extract_chart_spec_from_response(
                    state.response_content or ""
                )
                if chart_spec is not None:
                    state.chart_spec = chart_spec
                    state.response_content = cleaned_content
                    log.info(f"[{state.run_id}] Extract chart_spec from LLM response")

                log.info(f"[{state.run_id}] Updated response_content based on tool results")
                state.tool_results_processed = True

            except Exception as e:
                if is_llm_quota_error(e):
                    raise
                log.error(f"[{state.run_id}] Failed to synthesize tool results: {e}")
                # Safe fallback: present raw observations rather than leaking plan text.
                fallback_lines = []
                for obs in state.observations:
                    if obs.startswith("Tool error:"):
                        fallback_lines.append(obs)
                    else:
                        fallback_lines.append(obs[len("Tool result: "):] if obs.startswith("Tool result: ") else obs)
                state.response_content = "Here are the results:\n\n" + "\n\n".join(fallback_lines[:3])
                state.tool_results_processed = True
                log.warning(f"[{state.run_id}] Using safe fallback answer after LLM synthesis failure")

        # Create markdown block for main response.
        # If response_content is empty but tool_results exist and were NOT
        # processed (e.g. max_tool_calls reached without an observe LLM call),
        # produce a clean neutral message so the plan text never leaks.
        if not state.response_content and not state.tool_results_processed and state.tool_results:
            # If ONLY schema/discovery tools were run, don't output the generic data table fallback
            _SCHEMA_TOOL_KEYWORDS = ("get_schema", "get_tables", "describe", "schema")
            id_to_tool_name = {tc.tool_id: tc.tool_name for tc in state.tool_calls}
            only_schema = all(
                any(kw in id_to_tool_name.get(getattr(r, "tool_call_id", None), "").lower() for kw in _SCHEMA_TOOL_KEYWORDS)
                for r in state.tool_results
            )
            
            if only_schema:
                state.response_content = state.plan_text if state.plan_text else "I have analyzed the database structure to understand its capabilities."
            else:
                state.response_content = "I retrieved the requested data. See the results below."
        elif not state.response_content and not state.tool_results and state.plan_text:
            # No tools ran — the planner already produced a direct conversational
            # answer (stored in plan_text). Use it as the user-visible response.
            state.response_content = state.plan_text

        if state.response_content:
            chart_spec, cleaned_content = self._extract_chart_spec_from_response(
                state.response_content
            )
            if chart_spec is not None:
                state.chart_spec = chart_spec
                state.response_content = cleaned_content
                log.info(f"[{state.run_id}] Extracted chart_spec from response_content")

        chart_added=False
        # extract <chart> tag from response_content 
        # If MCP/tool data is available and chart_spec exists, the deterministic
        # backend chart below will override this LLM-generated chart block.
        if state.response_content:
            chart_block, cleaned_content = self._extract_chart_from_response(
                state.response_content
            )
            if chart_block is not None:
                state.response_content = cleaned_content
                state.content_blocks.append(ContentBlock(
                    type=BlockType.CHART,
                    content=chart_block,
                ))
                log.info(f"[{state.run_id}] Added chart block from LLM response")
                chart_added=True

        # ──(MCP/tool data): build chart directly from raw tool result ──
        # If data came from MCP/tool results and the current user asked for a
        # chart, prefer deterministic backend-generated ChartBlock over the
        # LLM-generated <chart> JSON.
        tool_result_chart_block = self._try_chart_from_tool_results(state)
        if tool_result_chart_block is not None:
            state.content_blocks = [
                block for block in state.content_blocks
                if block.type != BlockType.CHART
            ]
            state.content_blocks.append(ContentBlock(
                type=BlockType.CHART,
                content=tool_result_chart_block,
            ))
            log.info(f"[{state.run_id}] Added chart block from tool result data")
            chart_added=True
        # Create markdown block for main response
        if not chart_added and state.response_content:
            state.content_blocks.append(ContentBlock(
                type=BlockType.MARKDOWN,
                content=state.response_content
            ))
            log.info(f"[{state.run_id}] Added markdown block")
        else:
            log.warning(f"[{state.run_id}] No response_content to add as markdown block")
        
        # Add citations as a block if available
        if state.citations:
            # Citations are already in the response metadata
            pass
        
        # Add structured table blocks from tool results only when LLM has NOT
        # already synthesised them into prose (tool_results_processed=False).
        # Always unwrap the MCP wire envelope before attempting to format so
        # raw {type, text, data, mimeType} dicts are never emitted as blocks.
        if not state.tool_results_processed:
            _SCHEMA_TOOL_KEYWORDS = ("get_schema", "get_tables", "describe", "schema")
            id_to_tool_name = {tc.tool_id: tc.tool_name for tc in state.tool_calls}

            for result in state.tool_results:
                tool_name = id_to_tool_name.get(getattr(result, "tool_call_id", None), "")
                
                # Exclude schema tools from being rendered as generic UI tables
                if any(kw in tool_name.lower() for kw in _SCHEMA_TOOL_KEYWORDS):
                    log.info(f"[{state.run_id}] Excluded '{tool_name}' result from UI table rendering")
                    continue

                # Unwrap MCP envelope: [{"type":"text","text":"<payload>",...}]
                raw = result.result
                unwrapped_result = raw
                if isinstance(raw, list) and raw:
                    first_item = raw[0]
                    if (
                        isinstance(first_item, dict)
                        and first_item.get("type") == "text"
                        and first_item.get("text")
                        and set(first_item.keys()) <= {"type", "text", "data", "mimeType"}
                    ):
                        unwrapped_result = first_item["text"]

                # Try to parse JSON string into a structured object for table formatting.
                if isinstance(unwrapped_result, str):
                    try:
                        unwrapped_result = json.loads(unwrapped_result)
                    except Exception:
                        pass  # Keep as string; won't format as a table

                if self._looks_like_table(unwrapped_result):
                    table_block = self._format_as_table(unwrapped_result)
                    if table_block:
                        state.content_blocks.append(ContentBlock(
                            type=BlockType.TABLE,
                            content=table_block
                        ))

        # Emit one ARTIFACT content block per generated artifact so the UI
        # (BlockRenderer / ArtifactBlock) can render a download card inside
        # the chat bubble. Content is a flat dict matching the frontend's
        # ArtifactContent shape (id, artifact_type, filename, download_url,
        # size_bytes, description, mime_type).
        for summary in state.artifacts:
            artifact_payload = summary.model_dump()
            # Ensure both 'id' and 'artifact_id' are present so frontend code
            # that reads either key works.
            artifact_payload.setdefault("artifact_id", summary.id)
            state.content_blocks.append(ContentBlock(
                type=BlockType.ARTIFACT,
                content=artifact_payload,
            ))
            log.info(f"[{state.run_id}] Added artifact block for {summary.id}")

        state.current_step += 1
        return state

    async def _node_guardrail(self, state_data) -> AgentState:
        """Apply safety guardrails to the response."""
        # Convert dict to AgentState if needed
        if isinstance(state_data, dict):
            state = AgentState(**state_data)
        else:
            state = state_data
            
        log.info(f"[{state.run_id}] Running guardrails")
        
        # TODO: Implement guardrails (PII detection, content moderation)
        # For now, just pass through
        
        state.status = "completed"
        state.current_step += 1
        return state

    # Conditional edge functions
    def _should_retrieve(self, state: AgentState) -> str:
        """Every agent has knowledge-base access via the org's shared collection,
        so retrieval always runs — RAGService.retrieve() returns empty context
        gracefully if the agent has no linked documents."""
        return "retrieve"

    def _should_use_tools(self, state: AgentState) -> str:
        """Determine if we should use tools.

        If MCP is disabled for the agent, do not enter the tool-selection loop.
        """
        if not (state.agent_settings and state.agent_settings.is_mcp_enabled):
            log.info(f"[{state.run_id}] MCP disabled — LLM-chart path")
            return "format_response"

        if state.mcp_tools and state.tool_calls_made < state.max_tool_calls:
            log.info(f"[{state.run_id}] MCP enabled and tools available, will use tools")
            return "select_tools"

        log.info(f"[{state.run_id}] No tools available or needed or max tool call reached — LLM-chart path")
        return "format_response"

    def _should_continue_tools(self, state: AgentState) -> str:
        """After observing a tool result, decide whether to loop back to the
        selector for another tool or finalize the response.

        We always route back to ``select_tools`` while we still have budget
        because the selector LLM is the only place that can decide the plan
        is complete (by responding ``DONE``). The post-``select_tools`` edge
        (``_has_pending_tool_call``) skips ``execute_tool`` when no new call
        was queued, which keeps the loop bounded.
        """
        if state.tool_calls_made >= state.max_tool_calls:
            log.info(f"[{state.run_id}] Max tool calls reached, formatting response")
            return "format_response"

        log.info(f"[{state.run_id}] Looping back to select_tools to consider next step")
        return "select_tools"

    def _has_pending_tool_call(self, state: AgentState) -> str:
        """After ``select_tools``, route to ``execute_tool`` only if the
        selector queued a new pending call. Otherwise the selector either
        said DONE or failed to pick a tool, and we should finalize."""
        for tc in state.tool_calls:
            if tc.status == "pending":
                return "execute_tool"
        log.info(f"[{state.run_id}] No pending tool call after select_tools; finalizing")
        return "format_response"

    def _needs_more_tools(self, result: Any) -> bool:
        """Determine if tool results indicate need for more tool calls."""
        # Placeholder: implement logic based on result content
        return False


    def _extract_chart_from_response(
        self, content: str
    ) -> tuple[Optional[ChartBlock], str]:
        """Extract the first <chart>…</chart> block from an LLM response.

        Returns (ChartBlock, cleaned_content) where cleaned_content has the
        tag removed.  If no valid tag is found returns (None, original content).
        """
        match = re.search(r"<chart>(.*?)</chart>", content, re.DOTALL | re.IGNORECASE)
        if not match:
            return None, content

        json_str = match.group(1).strip()
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as exc:
            log.warning("Could not parse <chart> JSON: %s", exc)
            return None, content

        try:
            datasets = [
                ChartDataset(
                    label=ds.get("label", ""),
                    data=ds.get("data", []),
                    color=ds.get("color"),
                )
                for ds in raw.get("datasets", [])
            ]
            chart = ChartBlock(
                chart_type=raw.get("chart_type", "bar"),
                title=raw.get("title", "Chart"),
                labels=raw.get("labels", []),
                datasets=datasets,
                x_label=raw.get("x_label"),
                y_label=raw.get("y_label"),
            )
        except Exception as exc:
            log.warning("Could not build ChartBlock from <chart> tag: %s", exc)
            return None, content

        # Remove the tag (and any surrounding blank lines) from the text.
        cleaned = content[: match.start()] + content[match.end() :]
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return chart, cleaned

    def _extract_chart_spec_from_response(
        self, content: str
    ) -> tuple[Optional[Dict[str, Any]], str]:
        """Extract the <chart_spec>…</chart_spec> block from an LLM response.

        chart_spec is used only for intent/type/title/column mapping. It must
        not contain chart data. Actual ChartBlock labels/datasets are built
        from MCP/tool result rows by _try_chart_from_tool_results().
        """
        match = re.search(
            r"<chart_spec>(.*?)</chart_spec>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return None, content

        json_str = match.group(1).strip()
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as exc:
            log.warning("Could not parse <chart_spec> JSON: %s", exc)
            return None, content

        if not isinstance(raw, dict):
            log.warning("<chart_spec> JSON must be an object")
            return None, content

        allowed_chart_types = {"bar", "line", "pie", "doughnut", "scatter", "area"}

        wants_chart = bool(raw.get("wants_chart", True))
        chart_type = str(raw.get("chart_type", "bar")).strip().lower()
        if chart_type not in allowed_chart_types:
            chart_type = "bar"

        value_keys = raw.get("value_keys", [])
        if isinstance(value_keys, str):
            value_keys = [value_keys]
        elif not isinstance(value_keys, list):
            value_keys = []

        spec = {
            "wants_chart": wants_chart,
            "chart_type": chart_type,
            "title": str(raw.get("title") or "Results"),
            "label_key": raw.get("label_key"),
            "value_keys": [str(v) for v in value_keys if v is not None],
            "x_label": raw.get("x_label"),
            "y_label": raw.get("y_label"),
        }

        cleaned = content[: match.start()] + content[match.end() :]
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return spec, cleaned

    def _try_chart_from_tool_results(
        self, state: AgentState
    ) -> Optional[ChartBlock]:
        """Build a ChartBlock directly from the latest successful tool result.

        Important:
        - Chart intent/type/title/column mapping comes from state.chart_spec,
        which is decided by the LLM.
        - Chart labels/datasets come from MCP/tool result rows only.
        - No query-specific checks and no keyword-based chart type detection.
        """
        spec = state.chart_spec or {}
        if not spec or not spec.get("wants_chart"):
            return None

        # Find the most recent successful non-builtin tool result.
        id_to_name = {tc.tool_id: tc.tool_name for tc in state.tool_calls}

        chosen = None
        for result in reversed(state.tool_results):
            tool_name = id_to_name.get(
                getattr(result, "tool_call_id", None),
                "",
            )

            # Skip file-export tool results. We need the data-fetching result.
            if tool_name == self.EXPORT_FILE_TOOL:
                continue

            if getattr(result, "status", None) != "success":
                continue

            chosen = result
            break

        if chosen is None:
            return None

        rows = self._extract_rows_from_tool_result(chosen.result)
        if not rows:
            return None

        keys = list(rows[0].keys())
        if len(keys) < 2:
            return None

        def is_number(value: Any) -> bool:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if isinstance(value, str):
                try:
                    float(value.replace(",", "").strip())
                    return True
                except Exception:
                    return False
            return False

        def to_number(value: Any) -> float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                try:
                    return float(value.replace(",", "").strip())
                except Exception:
                    return 0
            return 0

        # LLM-decided label key. Fallback is generic and data-based only.
        label_key = spec.get("label_key")
        if not isinstance(label_key, str) or label_key not in keys:
            label_key = None
            for key in keys:
                non_empty_values = [
                    row.get(key)
                    for row in rows
                    if row.get(key) not in (None, "")
                ]
                if non_empty_values and not all(
                    is_number(value) for value in non_empty_values
                ):
                    label_key = key
                    break

        if label_key is None:
            label_key = keys[0]

        # LLM-decided value keys. Keep only keys that exist and are numeric.
        requested_value_keys = spec.get("value_keys") or []
        if isinstance(requested_value_keys, str):
            requested_value_keys = [requested_value_keys]
        if not isinstance(requested_value_keys, list):
            requested_value_keys = []

        numeric_keys: List[str] = []
        for key in requested_value_keys:
            if not isinstance(key, str) or key not in keys or key == label_key:
                continue

            non_empty_values = [
                row.get(key)
                for row in rows
                if row.get(key) not in (None, "")
            ]
            if non_empty_values and all(
                is_number(value) for value in non_empty_values
            ):
                numeric_keys.append(key)

        # Generic fallback when LLM did not provide usable value_keys.
        if not numeric_keys:
            for key in keys:
                if key == label_key:
                    continue

                non_empty_values = [
                    row.get(key)
                    for row in rows
                    if row.get(key) not in (None, "")
                ]
                if non_empty_values and all(
                    is_number(value) for value in non_empty_values
                ):
                    numeric_keys.append(key)

        if not numeric_keys:
            return None

        chart_type = str(spec.get("chart_type") or "bar").strip().lower()
        if chart_type not in {"bar", "line", "pie", "doughnut", "scatter", "area"}:
            chart_type = "bar"

        # Pie/doughnut charts usually need one dataset.
        if chart_type in ("pie", "doughnut") and len(numeric_keys) > 1:
            numeric_keys = [numeric_keys[0]]

        labels = [str(row.get(label_key, "")) for row in rows]

        datasets = [
            ChartDataset(
                label=key,
                data=[to_number(row.get(key, 0)) for row in rows],
            )
            for key in numeric_keys
        ]

        return ChartBlock(
            chart_type=chart_type,
            title=str(spec.get("title") or "Results"),
            labels=labels,
            datasets=datasets,
            x_label=spec.get("x_label") or label_key,
            y_label=spec.get("y_label") or ", ".join(numeric_keys),
        )

    def _extract_rows_from_tool_result(self, raw: Any) -> List[Dict[str, Any]]:
        """Extract clean row dictionaries from common MCP/tool result shapes."""
        # Unwrap common MCP envelope:
        # [{"type": "text", "text": "...json...", "data": ..., "mimeType": "..."}]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            first = raw[0]
            if first.get("type") == "text":
                # Prefer structured data if the MCP server provides it.
                if isinstance(first.get("data"), (list, dict)):
                    raw = first.get("data")
                elif isinstance(first.get("text"), str):
                    raw = first.get("text")

        # Parse JSON string if needed.
        parsed: Any = raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return []

        rows: List[Dict[str, Any]] = []

        if isinstance(parsed, list):
            rows = [item for item in parsed if isinstance(item, dict)]

        elif isinstance(parsed, dict):
            rows = self._rows_from_dict(parsed) or []

            # Single flat object fallback.
            if not rows and all(
                not isinstance(value, (list, dict))
                for value in parsed.values()
            ):
                rows = [parsed]

        if not rows:
            return []

        metadata_fields = {
            "@odata.context", "@odata.metadata", "@odata.type",
            "@odata.id", "@odata.etag",
        }

        cleaned_rows: List[Dict[str, Any]] = []
        for row in rows:
            cleaned = {
                key: value
                for key, value in row.items()
                if key not in metadata_fields
            }
            if cleaned:
                cleaned_rows.append(cleaned)

        return cleaned_rows


    def _looks_like_table(self, data: Any) -> bool:
        """Check if data looks like it should be formatted as a table."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # Check if it has odata metadata
                if any(key.startswith("@odata") for key in data[0].keys()):
                    return True
                # Standard list of dicts
                return True
        # Check for odata XML format
        if isinstance(data, str) and ("<entry>" in data or "<content>" in data or "<d:" in data):
            return True
        # Check for odata JSON format
        if isinstance(data, str) and ("@odata.context" in data or "@odata.metadata" in data or '"value"' in data):
            return True
        # Check for odata dict format
        if isinstance(data, dict) and ("@odata.context" in data or "@odata.metadata" in data or "value" in data):
            return True
        return False

    def _format_as_table(self, data: Any) -> Optional[TableBlock]:
        """Format data as a table block."""
        try:
            # Handle odata JSON format (as string)
            if isinstance(data, str) and ("@odata.context" in data or "@odata.metadata" in data or '"value"' in data):
                return self._parse_odata_json_to_table(data)

            # Handle odata JSON format (as dict/list - already parsed)
            if isinstance(data, dict) and ("@odata.context" in data or "@odata.metadata" in data or "value" in data):
                return self._format_odata_dict_to_table(data)

            # Handle odata JSON format (as list of dicts with odata metadata)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                first_item = data[0]
                if any(key.startswith("@odata") for key in first_item.keys()):
                    return self._format_odata_list_to_table(data)

            # Handle odata XML format
            if isinstance(data, str) and ("<entry>" in data or "<content>" in data or "<d:" in data):
                return self._parse_odata_to_table(data)

            # Handle list of dicts (standard table format)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                columns = [TableColumn(name=k, type="string") for k in data[0].keys()]
                rows = [[item.get(col.name, "") for col in columns] for item in data]

                return TableBlock(
                    columns=columns,
                    rows=rows,
                    title="Results",
                    download_formats=["csv", "json", "xlsx"]
                )
        except Exception as e:
            log.error(f"Failed to format as table: {e}")
        return None

    def _parse_odata_to_table(self, xml_string: str) -> Optional[TableBlock]:
        """Parse odata XML format into a table block."""
        try:
            import xml.etree.ElementTree as ET
            import re

            # Parse the XML
            root = ET.fromstring(xml_string)

            # Find all entry elements
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                # Try without namespace
                entries = root.findall(".//entry")

            if not entries:
                log.warning("No entries found in odata XML")
                return None

            # Extract column names and data from entries
            columns = []
            rows = []

            for entry in entries:
                # Find content/properties
                content = entry.find(".//{http://www.w3.org/2005/Atom}content")
                if content is None:
                    content = entry.find(".//content")

                if content is not None:
                    properties = content.find(".//{http://docs.oasis-open.org/odata/ns/metadata}properties")
                    if properties is None:
                        properties = content.find(".//properties")

                    if properties is not None:
                        # Get all property elements
                        props = properties.findall(".//{http://docs.oasis-open.org/odata/ns/metadata}*")
                        if not props:
                            props = properties.findall(".//*")

                        row_data = {}
                        for prop in props:
                            # Remove namespace from tag name
                            tag_name = re.sub(r'\{.*\}', '', prop.tag)
                            if tag_name and tag_name != 'properties':
                                value = prop.text or ""
                                row_data[tag_name] = value
                                if tag_name not in [col.name for col in columns]:
                                    columns.append(TableColumn(name=tag_name, type="string"))

                        if row_data:
                            rows.append([row_data.get(col.name, "") for col in columns])

            if not columns or not rows:
                log.warning("Could not extract table data from odata XML")
                return None

            return TableBlock(
                columns=columns,
                rows=rows,
                title="Results",
                download_formats=["csv", "json", "xlsx"]
            )
        except Exception as e:
            log.error(f"Failed to parse odata XML: {e}")
            return None

    def _parse_odata_json_to_table(self, json_string: str) -> Optional[TableBlock]:
        """Parse odata JSON format into a table block."""
        try:
            import json

            # Parse the JSON
            data = json.loads(json_string)

            # Odata JSON typically has a "value" array containing the actual data
            if isinstance(data, dict):
                # Check for odata structure with "value" array
                if "value" in data and isinstance(data["value"], list):
                    items = data["value"]
                else:
                    # If no "value" array, treat the dict itself as a single item
                    items = [data]
            elif isinstance(data, list):
                items = data
            else:
                log.warning("Invalid odata JSON structure")
                return None

            if not items:
                log.warning("No items found in odata JSON")
                return None

            # Extract columns from the first item, excluding odata metadata fields
            metadata_fields = {"@odata.context", "@odata.metadata", "@odata.type", "@odata.id", "@odata.etag"}
            first_item = items[0]

            if isinstance(first_item, dict):
                columns = []
                for key in first_item.keys():
                    if key not in metadata_fields:
                        columns.append(TableColumn(name=key, type="string"))

                if not columns:
                    log.warning("No columns found in odata JSON after filtering metadata")
                    return None

                # Build rows
                rows = []
                for item in items:
                    if isinstance(item, dict):
                        row = []
                        for col in columns:
                            value = item.get(col.name, "")
                            # Convert complex values to string
                            if not isinstance(value, str):
                                value = json.dumps(value) if value is not None else ""
                            row.append(str(value))
                        rows.append(row)

                if not rows:
                    log.warning("No rows found in odata JSON")
                    return None

                return TableBlock(
                    columns=columns,
                    rows=rows,
                    title="Results",
                    download_formats=["csv", "json", "xlsx"]
                )
            else:
                log.warning("Items in odata JSON are not dictionaries")
                return None

        except Exception as e:
            log.error(f"Failed to parse odata JSON: {e}")
            return None

    def _format_odata_dict_to_table(self, data: dict) -> Optional[TableBlock]:
        """Format odata dict (already parsed) into a table block."""
        try:
            import json

            # Check for odata structure with "value" array
            if "value" in data and isinstance(data["value"], list):
                items = data["value"]
            else:
                # If no "value" array, treat the dict itself as a single item
                items = [data]

            if not items:
                log.warning("No items found in odata dict")
                return None

            # Extract columns from the first item, excluding odata metadata fields
            metadata_fields = {"@odata.context", "@odata.metadata", "@odata.type", "@odata.id", "@odata.etag"}
            first_item = items[0]

            if isinstance(first_item, dict):
                columns = []
                for key in first_item.keys():
                    if key not in metadata_fields:
                        columns.append(TableColumn(name=key, type="string"))

                if not columns:
                    log.warning("No columns found in odata dict after filtering metadata")
                    return None

                # Build rows
                rows = []
                for item in items:
                    if isinstance(item, dict):
                        row = []
                        for col in columns:
                            value = item.get(col.name, "")
                            # Convert complex values to string
                            if not isinstance(value, str):
                                value = json.dumps(value) if value is not None else ""
                            row.append(str(value))
                        rows.append(row)

                if not rows:
                    log.warning("No rows found in odata dict")
                    return None

                return TableBlock(
                    columns=columns,
                    rows=rows,
                    title="Results",
                    download_formats=["csv", "json", "xlsx"]
                )
            else:
                log.warning("Items in odata dict are not dictionaries")
                return None

        except Exception as e:
            log.error(f"Failed to format odata dict as table: {e}")
            return None

    def _format_odata_list_to_table(self, data: list) -> Optional[TableBlock]:
        """Format odata list of dicts (already parsed) into a table block."""
        try:
            import json

            if not data:
                log.warning("No items found in odata list")
                return None

            # Extract columns from the first item, excluding odata metadata fields
            metadata_fields = {"@odata.context", "@odata.metadata", "@odata.type", "@odata.id", "@odata.etag"}
            first_item = data[0]

            if isinstance(first_item, dict):
                columns = []
                for key in first_item.keys():
                    if key not in metadata_fields:
                        columns.append(TableColumn(name=key, type="string"))

                if not columns:
                    log.warning("No columns found in odata list after filtering metadata")
                    return None

                # Build rows
                rows = []
                for item in data:
                    if isinstance(item, dict):
                        row = []
                        for col in columns:
                            value = item.get(col.name, "")
                            # Convert complex values to string
                            if not isinstance(value, str):
                                value = json.dumps(value) if value is not None else ""
                            row.append(str(value))
                        rows.append(row)

                if not rows:
                    log.warning("No rows found in odata list")
                    return None

                return TableBlock(
                    columns=columns,
                    rows=rows,
                    title="Results",
                    download_formats=["csv", "json", "xlsx"]
                )
            else:
                log.warning("Items in odata list are not dictionaries")
                return None

        except Exception as e:
            log.error(f"Failed to format odata list as table: {e}")
            return None

    def _build_planning_prompt(self, state: AgentState) -> str:
        """Build the planning prompt with available tools.

        When DB-aware MCP servers are present (``state.schema_context`` is
        non-empty), the prompt includes the cached schema and instructs the
        LLM to generate read-only SQL as part of the plan.
        """
        tool_list = "\n".join([
            f"- {t['tool_name']}: {t.get('description', 'No description')}"
            for t in state.mcp_tools
        ])

        # DB-aware guidance: only injected when schema context is available.
        db_guidance = ""
        if state.schema_context:
            db_guidance = f"""

--- DATABASE CONTEXT ---
The following database schema is available via MCP servers that support
query execution.  Use this schema to write SQL when the user's request
requires data from a database.

{state.schema_context}

IMPORTANT rules for database queries:
- Generate ONLY read-only SELECT / WITH … SELECT statements.
- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or any DDL/DML.
- Identify the correct execute-query tool from the available tools and
  pass the generated SQL as its argument.
- Do NOT call the schema / describe tool yourself; the schema above is
  already cached and optimized for your query.
--- END DATABASE CONTEXT ---"""

        return f"""You have access to the following tools:
{tool_list}
{db_guidance}

Analyze the user's request and determine:
1. What information is needed
2. Which tools should be called (if any)
3. What operations need to be performed on the data
4. What the final answer should look like
5. Whether the user wants the result as a downloadable file (CSV, Excel, PDF,
   JSON, Markdown, or DOCX). If so, plan to call the built-in
   `export_file` tool AFTER any data-fetching tools, passing the data via
   `rows` (for csv/xlsx), `content` (for pdf/docx/md/txt/html), or `data`
   (for json), and a sensible `filename`.
6. Whether the user wants a chart or graph. If so, fetch the data normally.
   The observe step will ask the LLM to decide the chart intent/type/title and
   column mapping, then the backend will build the ChartBlock from tool data.

If tools are needed: briefly state which tools to call and why. Do not include JSON — the system handles tool execution automatically.
If NO tools are needed: answer the user's question directly and completely right now, following every instruction in your system prompt. Do NOT describe what you are about to do — just provide the answer."""

    def _build_chart_prompt(self, mode: str = "llm_chart") -> str:
        """
        chart prompt.

        Modes:
        - llm_chart:
            LLM generates full chart data itself.
        - runtime_chart:
            LLM only decides chart intent/spec.
            Backend builds datasets from tool/MCP results.
        """

        if mode == "runtime_chart":
            return """
    If the user asked for a chart, graph, plot, or visualization, include a
    <chart_spec> block in your response with this exact JSON structure:

    <chart_spec>
    {
    "wants_chart": true,
    "chart_type": "bar",
    "title": "Chart title here",
    "label_key": "column_name_for_labels",
    "value_keys": ["numeric_column_name_1"],
    "x_label": "X axis label",
    "y_label": "Y axis label"
    }
    </chart_spec>

    Supported chart_type values:
    bar, line, pie, doughnut, scatter, area.

    Rules:
    - Do NOT put labels or dataset data inside chart_spec.
    - Choose label_key and value_keys from available data columns.
    - value_keys must contain numeric columns only.
    - If user did not ask for chart, do not include chart_spec.
    - Everything outside <chart_spec> is normal text.
    """

        return (
            "If document context has been provided above, use it to answer first. "
            "Otherwise answer from your own knowledge."
            """

    If the user asks for a chart, graph, plot, or visualization:

    1. Only create a <chart> block when the required data is well-known, stable,
    and you are confident about the values.

    2. If the requested chart needs private, business, live, database, API, or
    user-specific data, and no such data is available, do NOT invent values.

    3. If the chart request is ambiguous or missing the required data source,
    ask a short clarification question instead of generating a chart.


    <chart>
    {
    "chart_type": "bar",
    "title": "Chart title here",
    "labels": ["Label1", "Label2", "Label3"],
    "datasets": [
        {
        "label": "Series name",
        "data": []
        }
    ],
    "x_label": "X axis label",
    "y_label": "Y axis label"
    }
    </chart>

    Supported chart_type values:
    bar, line, pie, doughnut, scatter, area.

    Rules:
    - For pie/doughnut use single dataset.
    - Everything outside <chart> is normal text.
    - Do NOT wrap JSON in markdown code fences.
    """
        )

    def _extract_tool_calls_from_plan(self, state: AgentState) -> None:
        """Extract tool calls from the plan response and populate state.tool_calls."""
        import re

        content = state.response_content

        # Try to find JSON array in the response (more flexible pattern)
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if json_match:
            try:
                import json
                tool_calls_json = json.loads(json_match.group(0))
                for tool_call_data in tool_calls_json:
                    tool_name = tool_call_data.get('tool_name')
                    arguments = tool_call_data.get('arguments', {})

                    # Verify the tool exists
                    tool_exists = any(t['tool_name'] == tool_name for t in state.mcp_tools)
                    if tool_exists:
                        tool_call = ToolCallBlock(
                            tool_name=tool_name,
                            tool_id=str(uuid4()),
                            arguments=arguments,
                            status="pending"
                        )
                        state.tool_calls.append(tool_call)
                        log.info(f"[{state.run_id}] Extracted tool call: {tool_name} with args: {arguments}")
                    else:
                        log.warning(f"[{state.run_id}] Tool not found: {tool_name}")
            except Exception as e:
                log.error(f"[{state.run_id}] Failed to parse tool calls JSON: {e}")
        else:
            log.warning(f"[{state.run_id}] No JSON array found in plan response for tool calls")
            # Try to extract tool mentions as fallback
            for tool in state.mcp_tools:
                if tool['tool_name'].lower() in content.lower():
                    log.info(f"[{state.run_id}] Fallback: Found tool mention: {tool['tool_name']}")
                    tool_call = ToolCallBlock(
                        tool_name=tool['tool_name'],
                        tool_id=str(uuid4()),
                        arguments={},
                        status="pending"
                    )
                    state.tool_calls.append(tool_call)

    async def _load_mcp_tools(self, agent_settings: AIAgentSettings) -> List[Dict]:
        """Load MCP tools for the agent."""
        from ai_agent.services.mcp_service import gather_mcp_tools
        return await gather_mcp_tools(agent_settings, self.db)

    # ------------------------------------------------------------------
    # DB-aware MCP: schema fetching & caching
    # ------------------------------------------------------------------

    async def _load_db_schema_context(
        self,
        state: AgentState,
        db_servers: set,
        user_query: str,
    ) -> None:
        """Fetch, index, and retrieve relevant schema for DB-aware MCP servers.

        Uses the ``schema_engine`` subsystem which implements:
        1. Metadata extraction via the MCP server's schema tool.
        2. LLM-driven table summarization (cached).
        3. BM25 + vector + FK-graph hybrid retrieval.
        4. Token-budgeted context pruning.

        The aggregated context is stored in ``state.schema_context``.
        """
        from ai_agent.schema_engine.orchestrator import get_orchestrator
        from ai_agent.schema_engine.observability.metrics import get_metrics

        metrics = get_metrics()
        context_parts: list[str] = []

        for server_name in db_servers:
            server_tools = [
                t for t in state.mcp_tools
                if t.get("server_name") == server_name
            ]

            try:
                async with metrics.async_timer(
                    "schema_engine.init", server=server_name,
                ):
                    orch = await get_orchestrator(
                        server_name=server_name,
                        mcp_manager=self.mcp_manager,
                        server_tools=server_tools,
                        agent_id=state.agent_id,
                        api_key=state.api_key,
                        model=state.agent_settings.llm_model if state.agent_settings else None,
                        provider=state.agent_settings.llm_provider if state.agent_settings else None,
                    )

                async with metrics.async_timer(
                    "schema_engine.build_context", server=server_name,
                ):
                    ctx = await orch.build_context(user_query)

                log.info(
                    "[%s] Schema engine context for '%s': %d chars (~%d tokens), "
                    "%d tables indexed",
                    state.run_id, server_name, len(ctx), len(ctx) // 4,
                    orch.table_count,
                )

                # Record token reduction for observability
                full_estimate = orch.chunk_count * 200  # rough per-table token avg
                metrics.record_token_reduction(full_estimate, len(ctx) // 4)

                context_parts.append(
                    f"--- Database schema (server: {server_name}) ---\n{ctx}"
                )

            except Exception as e:
                from ai_agent.schema_engine.orchestrator import SchemaEngineError

                if isinstance(e, SchemaEngineError):
                    raise

                log.warning(
                    "[%s] Schema engine failed for '%s': %s",
                    state.run_id, server_name, e,
                    exc_info=True,
                )

        state.schema_context = "\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # SQL guardrail for DB-aware tool execution
    # ------------------------------------------------------------------

    def _apply_sql_guardrail(
        self,
        state: AgentState,
        tool_call: ToolCallBlock,
        tool_def: Dict,
    ) -> Dict[str, Any]:
        """Validate and sanitise any SQL argument before it reaches the MCP
        execute-query tool.  Returns the (possibly cleaned) arguments dict.
        If validation fails the tool_call's status is set to ``"failed"`` and
        a ToolResultBlock error is appended so the observe node sees it.

        Uses the battle-tested ``_validate_select_sql`` and
        ``_guess_sql_arg_key`` helpers from ``services.pipelines``.
        """
        from ai_agent.services.pipelines import (
            _validate_select_sql,
            _guess_sql_arg_key,
        )
        from ai_agent.core.permission_context import get_query_sanitizer

        args = dict(tool_call.arguments)  # shallow copy

        # Identify which argument key carries the SQL string.
        sql_key = _guess_sql_arg_key(tool_def.get("input_schema"))
        if not sql_key:
            # Fallback: scan arguments for a string that looks like SQL.
            for k, v in args.items():
                if isinstance(v, str) and v.strip().upper().startswith(("SELECT", "WITH")):
                    sql_key = k
                    break

        if not sql_key or sql_key not in args:
            # No SQL argument found; nothing to validate.
            return args

        raw_sql = args[sql_key]
        if not isinstance(raw_sql, str):
            return args

        # 1. Read-only validation (strip fences, comments, reject DML/DDL).
        safe_sql = _validate_select_sql(raw_sql)
        if safe_sql is None:
            log.warning(
                "[%s] SQL guardrail BLOCKED non-SELECT query on DB server: %r",
                state.run_id, raw_sql[:200],
            )
            tool_call.status = "failed"
            state.tool_results.append(ToolResultBlock(
                tool_call_id=tool_call.tool_id,
                status="failure",
                result={},
                error=(
                    "SQL guardrail: only read-only SELECT queries are allowed "
                    "against database servers. The generated query was blocked."
                ),
            ))
            return args

        # 2. Additional sanitisation (blocked patterns, multi-statement).
        sanitizer = get_query_sanitizer()
        sanitized_sql, err = sanitizer.sanitize(safe_sql)
        if err:
            log.warning(
                "[%s] SQL sanitizer rejected query: %s — %r",
                state.run_id, err, safe_sql[:200],
            )
            tool_call.status = "failed"
            state.tool_results.append(ToolResultBlock(
                tool_call_id=tool_call.tool_id,
                status="failure",
                result={},
                error=f"SQL sanitizer: {err}",
            ))
            return args

        # 3. Cap unbounded result sets. LLM-generated queries against MCP DB
        # connectors sometimes omit any row limit (e.g. ``SELECT * FROM
        # jobsheet_data WHERE ...``), which can pull megabytes of data
        # through the MCP round trip even though format_response only ever
        # shows the user a 10-row preview. Enforce a hard cap when missing.
        capped_sql = self._enforce_sql_row_limit(sanitized_sql)

        args[sql_key] = capped_sql
        log.info(
            "[%s] SQL guardrail passed (%d chars) for tool '%s'",
            state.run_id, len(capped_sql), tool_call.tool_name,
        )
        return args

    def _enforce_sql_row_limit(self, sql: str) -> str:
        """Append ``LIMIT MAX_SQL_ROW_LIMIT`` to a SELECT/WITH query that has
        no existing LIMIT clause, to bound MCP response size and transfer
        time. Leaves the query untouched if a LIMIT is already present."""
        if re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
            return sql
        return f"{sql} LIMIT {self.MAX_SQL_ROW_LIMIT}"

    # ------------------------------------------------------------------
    # Built-in (non-MCP) tools
    # ------------------------------------------------------------------
    BUILTIN_SERVER = "__builtin__"
    EXPORT_FILE_TOOL = "export_file"

    def _builtin_tools(self) -> List[Dict]:
        """Return built-in tools that the planner/selector can choose.

        These are executed directly by the runtime (see ``_node_execute_tool``)
        rather than through the MCP manager. They share the same dict shape as
        MCP tools so the existing prompt and execution machinery can treat
        them uniformly.
        """
        tools = [
            {
                "server_name": self.BUILTIN_SERVER,
                "server_description": "Built-in runtime tools",
                "tool_name": self.EXPORT_FILE_TOOL,
                "description": (
                    "Generate a downloadable artifact (file) and attach it to "
                    "the response. Use this whenever the user asks for a "
                    "CSV, Excel, PDF, JSON, Markdown, or DOCX export, or when "
                    "presenting a large dataset that benefits from download. "
                    "IMPORTANT: Do NOT pass 'rows', 'data', or 'content' — "
                    "the runtime automatically uses the full dataset from the "
                    "most recent prior tool result. Only specify "
                    "'artifact_type' and 'filename'."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["artifact_type"],
                    "properties": {
                        "artifact_type": {
                            "type": "string",
                            "enum": [
                                "csv", "xlsx", "json", "txt", "md",
                                "html", "xml", "pdf", "docx",
                            ],
                            "description": (
                                "File format. Use 'xlsx' or 'csv' for tabular "
                                "data exports (the runtime auto-fills the data). "
                                "Use 'pdf'/'docx'/'md'/'txt'/'html' for prose. "
                                "Use 'json' only if the user explicitly asked for JSON."
                            ),
                        },
                        "filename": {
                            "type": "string",
                            "description": "Desired filename including extension.",
                        },
                        "rows": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "DO NOT USE — runtime auto-fills the full dataset from the previous tool result. Passing rows here will cause truncation.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Markdown/HTML/text body for prose exports (pdf/docx/md/txt/html).",
                        },
                        "data": {
                            "description": "Free-form payload for json exports.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Short description shown next to the download.",
                        },
                    },
                },
                "requires_query_building": False,
            },
        ]

        return tools

    @staticmethod
    def _is_empty_payload(value: Any) -> bool:
        """Return True if the value looks unset/empty for export purposes."""
        if value is None:
            return True
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            return True
        return False

    @staticmethod
    def _last_data_tool_result(state: AgentState):
        """Return (tool_name, result) for the most recent successful,
        non-``export_file`` tool result, or ``(None, None)`` if there isn't
        one. Shared by every helper that needs "what did we just fetch?"
        (row counting, export payload extraction, default filenames)."""
        id_to_name = {tc.tool_id: tc.tool_name for tc in state.tool_calls}
        for result in reversed(state.tool_results):
            tname = id_to_name.get(getattr(result, "tool_call_id", None))
            if tname == AgentRuntime.EXPORT_FILE_TOOL:
                continue
            if getattr(result, "status", None) != "success":
                continue
            return tname, result
        return None, None

    @staticmethod
    def _default_export_filename(tool_name: Optional[str], artifact_type: str = "xlsx") -> str:
        """Build a readable filename from a tool name, e.g.
        "jira_prismberry_get_issues_by_person_name" -> "issues_by_person_name_2026_08_07.xlsx".
        Used when export_file is forced in code (no LLM available to pick a
        name). Not vendor-specific: just strips a leading "<word>_get/list_"
        style prefix if present, else uses the whole tool name."""
        base = (tool_name or "export").strip()
        base = re.sub(r'^[a-z0-9_]+?_(get|list)_', '', base, flags=re.IGNORECASE) or base
        base = re.sub(r'[^a-zA-Z0-9]+', '_', base).strip('_').lower() or "export"
        date_str = datetime.utcnow().strftime('%Y_%m_%d')
        return f"{base}_{date_str}.{artifact_type}"

    @staticmethod
    def _rows_from_dict(parsed: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Pull a row list out of a dict-shaped tool result without assuming
        any particular vendor's wrapper key (Jira uses "issues", OData uses
        "value", generic REST APIs use "data"/"items"/"results", etc.).
        Instead, scan the dict's values structurally: any non-empty list
        whose items are all dicts is a candidate row set; the longest one
        wins, since real record lists tend to dominate small metadata lists
        (e.g. "warnings": [...]). Returns ``None`` if no such list exists."""
        candidates = [
            value for value in parsed.values()
            if isinstance(value, list) and value
            and all(isinstance(item, dict) for item in value)
        ]
        return max(candidates, key=len) if candidates else None

    def _count_rows_in_last_tool_result(self, state: AgentState) -> Optional[int]:
        """Best-effort row count of the most recent successful, non-builtin
        tool result, used to decide whether ``export_file`` must be forced
        (see ``_node_select_tool``). Returns ``None`` when the result isn't
        a list of rows or nothing usable is available."""
        _, chosen = self._last_data_tool_result(state)
        if chosen is None:
            return None

        raw = chosen.result
        unwrapped: Any = raw
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            first = raw[0]
            if first.get("type") == "text" and isinstance(first.get("text"), str):
                unwrapped = first["text"]

        parsed: Any = unwrapped
        if isinstance(unwrapped, str):
            try:
                parsed = json.loads(unwrapped)
            except Exception:
                return None

        if isinstance(parsed, list):
            return len(parsed)
        if isinstance(parsed, dict):
            rows = self._rows_from_dict(parsed)
            if rows is not None:
                return len(rows)
        return None

    def _extract_last_tool_payload(
        self,
        state: AgentState,
        artifact_type: str,
    ) -> Any:
        """Pull a payload suitable for ``artifact_type`` out of the most
        recent successful, non-builtin tool result. Returns ``None`` if
        nothing usable is available.

        - For tabular formats (csv/xlsx/json): returns a list of dicts when
          possible, falling back to the parsed JSON object.
        - For prose formats (pdf/docx/md/txt/html/xml): returns a markdown
          text representation (rendered from rows if necessary).
        """
        _, chosen = self._last_data_tool_result(state)
        if chosen is None:
            return None

        # MCP results commonly arrive as [{"type": "text", "text": "<json>"}].
        # Unwrap that to the underlying JSON/text.
        raw = chosen.result
        unwrapped: Any = raw
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            first = raw[0]
            if first.get("type") == "text" and isinstance(first.get("text"), str):
                unwrapped = first["text"]

        # Parse JSON text if applicable so downstream generators get
        # structured data.
        parsed: Any = unwrapped
        if isinstance(unwrapped, str):
            try:
                parsed = json.loads(unwrapped)
            except Exception:
                parsed = unwrapped  # keep as plain text

        # Coerce to a list of rows when possible.
        rows: Optional[List[Dict[str, Any]]] = None
        if isinstance(parsed, list):
            rows = [r for r in parsed if isinstance(r, dict)]
        elif isinstance(parsed, dict):
            rows = self._rows_from_dict(parsed)

        if artifact_type in ("csv", "xlsx"):
            return rows if rows is not None else parsed
        if artifact_type == "json":
            return parsed
        # Prose formats: render rows as a markdown table when we have them,
        # else fall back to raw text.
        if rows:
            headers = list(rows[0].keys())
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |")
            return "\n".join(lines)
        if isinstance(parsed, str):
            return parsed
        return json.dumps(parsed, ensure_ascii=False, indent=2, default=str)

    async def _execute_builtin_tool(
        self,
        state: AgentState,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch a built-in tool call. Returns an MCP-shaped result dict.

        Result dict keys mirror what ``mcp_manager.call_tool`` returns:
        ``{"isError": bool, "content": [...]}``.
        """
        if tool_name != self.EXPORT_FILE_TOOL:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown built-in tool: {tool_name}"}]}

        from ai_agent.services.artifact_service import create_artifact

        artifact_type = (arguments.get("artifact_type") or "").lower().strip()
        if not artifact_type:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "'artifact_type' is required."}],
            }

        # Pick the right data payload for the chosen type so the generator
        # receives what it expects.
        if artifact_type in ("csv", "xlsx"):
            data = arguments.get("rows", arguments.get("data", []))
        elif artifact_type == "json":
            data = arguments.get("data", arguments.get("content", {}))
        elif artifact_type in ("pdf", "docx", "md", "txt", "html", "xml"):
            data = arguments.get("content", arguments.get("data", ""))
        else:
            data = arguments.get("data", arguments.get("content"))

        # Auto-fill from the most recent prior tool result when the LLM
        # didn't pass an explicit payload. This is the common case: the
        # agent first fetches data with another tool, then calls
        # ``export_file`` to package it. Forcing the LLM to round-trip the
        # entire dataset through JSON arguments is expensive and fragile.
        if self._is_empty_payload(data):
            recovered = self._extract_last_tool_payload(state, artifact_type)
            if recovered is not None:
                log.info(
                    f"[{state.run_id}] export_file: auto-filled payload "
                    f"from previous tool result for type={artifact_type}"
                )
                data = recovered

        # ── Detect whether data came from a DB-query tool ────────────────
        # For tabular exports we optionally run LLM-based column filtering to
        # strip irrelevant fields from external/API data.  DB query results
        # (requires_query_building=True) are excluded from filtering so every
        # DB column the SQL selected is preserved in full.
        last_tool_name, _ = self._last_data_tool_result(state)
        last_tool_is_db = False
        if last_tool_name:
            for tool in state.mcp_tools:
                if tool.get("tool_name") == last_tool_name:
                    last_tool_is_db = tool.get("requires_query_building", False)
                    break

        # Resolve the user's current message to give the LLM column-filter
        # context about what was actually asked.
        user_message_for_filter = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "user":
                user_message_for_filter = msg.get("content", "")
                break

        try:
            summary = await create_artifact(
                self.db,
                run_id=state.run_id,
                user_id=state.user_id,
                agent_id=state.agent_id,
                session_id=state.session_id,
                conversation_id=state.conversation_id,
                message_id=state.input_message_id,
                artifact_type=artifact_type,
                data=data,
                filename=arguments.get("filename"),
                description=arguments.get("description"),
                # LLM column filtering context
                requires_query_building=last_tool_is_db,
                user_query=user_message_for_filter,
                llm=self.llm,
                litellm_model_string=state.litellm_model_string,
                llm_api_key=state.api_key,
            )
        except ValueError as e:
            log.warning(f"[{state.run_id}] export_file rejected: {e}")
            return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
        except Exception as e:
            log.exception(f"[{state.run_id}] export_file failed")
            return {"isError": True, "content": [{"type": "text", "text": f"Failed to generate artifact: {e}"}]}

        state.artifacts.append(summary)
        log.info(f"[{state.run_id}] Created artifact {summary.id} ({summary.artifact_type}, {summary.size_bytes}B)")

        return {
            "isError": False,
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Created {summary.artifact_type} artifact '{summary.filename}' "
                        f"({summary.size_bytes} bytes). Download URL is attached to the response."
                    ),
                }
            ],
        }

    def _timed(self, node_name: str, fn: Callable) -> Callable:
        """Wrap a node coroutine so its wall-clock duration is recorded in
        ``state.step_timings[node_name]``.  When a node is visited more than
        once (e.g. ``select_tools`` in a multi-tool loop) the times are
        accumulated."""
        async def _wrapper(state_data):
            t0 = time.perf_counter()
            result = await fn(state_data)
            elapsed = time.perf_counter() - t0
            # result may be a dict (LangGraph serialised) or AgentState
            if isinstance(result, AgentState):
                result.step_timings[node_name] = (
                    result.step_timings.get(node_name, 0.0) + elapsed
                )
                result.step_counts[node_name] = (
                    result.step_counts.get(node_name, 0) + 1
                )
                result.step_sequence.append(node_name)
            elif isinstance(result, dict):
                timings = result.get("step_timings") or {}
                timings[node_name] = timings.get(node_name, 0.0) + elapsed
                result["step_timings"] = timings
                counts = result.get("step_counts") or {}
                counts[node_name] = counts.get(node_name, 0) + 1
                result["step_counts"] = counts
                seq = result.get("step_sequence") or []
                seq.append(node_name)
                result["step_sequence"] = seq
            return result
        return _wrapper

    async def run(
        self,
        agent_id: str,
        user_id: int,
        session_id: Optional[int],
        conversation_id: str,
        messages: List[Dict[str, str]],
        mode: RunMode = RunMode.CHAT,
        input_message_id: Optional[int] = None,
        stream: bool = False,
        api_key: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> AgentResponse:
        """Execute an agent run.
        
        Args:
            agent_id: Agent ID
            user_id: User ID
            session_id: Session ID
            conversation_id: Conversation ID
            messages: Conversation messages
            mode: Execution mode
            input_message_id: Input message ID
            stream: Whether to stream response
            api_key: API key for LLM provider
            
        Returns:
            AgentResponse with blocks and metadata
        """
        run_id = str(uuid4())
        self._run_start_time = time.perf_counter()

        # Initialize state
        initial_state = AgentState(
            run_id=run_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            conversation_id=conversation_id,
            input_message_id=input_message_id,
            messages=messages,
            status="running",
            mode=mode,
            api_key=api_key,
            user_email=user_email,
        )
        
        log.info(f"[{run_id}] Starting agent run")
        
        # Create run record
        run_record = AgentRun(
            id=run_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            conversation_id=conversation_id,
            input_message_id=input_message_id,
            mode=mode,
            llm_provider=settings.LLM_PROVIDER,
            llm_model=settings.LLM_MODEL,
            status=RunStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        self.db.add(run_record)
        self.db.commit()

        # Tracked at this scope so the except-block cleanup below can reach
        # them even if the graph raises before the follow-up task is awaited.
        _followups_task = None
        _followups_session = None

        try:
            # Execute the graph
            if stream:
                # Return streaming iterator
                return await self._run_streaming(initial_state, run_record)
            else:
                # Follow-up FAQs only depend on inputs already known at this
                # point (not on the graph's output), so kick them off
                # concurrently with graph execution instead of waiting for
                # the graph to finish first. They're best-effort (failures
                # are logged, never raised), so this overlaps their ~1-2s
                # cost with format_response's much longer LLM call instead
                # of adding it on top of the total request latency. Uses its
                # own DB session (not self.db) since that session is used by
                # graph node execution concurrently and Session objects
                # aren't safe for concurrent use across coroutines.
                from ai_agent.core.db import engine as _db_engine
                from sqlmodel import Session as _Session

                user_question = messages[-1]["content"] if messages else ""
                _followups_session = _Session(_db_engine) if user_question else None
                if user_question:
                    async def _fetch_follow_ups():
                        _t0 = time.perf_counter()
                        from ai_agent.workflows.faq.faq import generate_follow_ups
                        result = await generate_follow_ups(
                            agent_id=agent_id,
                            user_question=user_question,
                            conversation_id=conversation_id,
                            db_session=_followups_session,
                            api_key=api_key,
                        )
                        log.info(
                            "[%s] [LATENCY] generate_follow_ups (own duration, run concurrently with graph): %.1f ms",
                            run_id, (time.perf_counter() - _t0) * 1000,
                        )
                        return result
                    _followups_task = asyncio.create_task(_fetch_follow_ups())

                # Execute synchronously
                final_state = await self._execute_sync(initial_state)

                # Log per-step timing summary
                self._log_timing_summary(run_id, final_state)

                # Build response
                response = self._build_response(final_state, run_record)

                # Collect follow-up FAQs kicked off above
                if _followups_task is not None:
                    _t_wait = time.perf_counter()
                    try:
                        response.follow_up_faqs = await _followups_task
                    except Exception as e:
                        log.warning("Failed to fetch follow-ups in AgentRuntime: %s", e)
                    finally:
                        _followups_session.close()
                        log.info(
                            "[%s] [LATENCY] generate_follow_ups (extra wait added after graph finished): %.1f ms",
                            run_id, (time.perf_counter() - _t_wait) * 1000,
                        )

                # Update run record
                self._update_run_record(run_record, final_state, response)

                return response
                
        except Exception as e:
            log.error(f"[{run_id}] Run failed: {e}")
            run_record.status = RunStatus.FAILED
            run_record.error = str(e)
            run_record.completed_at = datetime.utcnow()
            self.db.commit()
            # Clean up the follow-up task/session if the graph raised
            # before they were awaited/closed on the success path above.
            if _followups_task is not None:
                _followups_task.cancel()
            if _followups_session is not None:
                _followups_session.close()
            raise

    def _log_timing_summary(self, run_id: str, state) -> None:
        """Emit a single INFO log with per-step and total wall-clock timings."""
        total_elapsed = time.perf_counter() - self._run_start_time

        if isinstance(state, AgentState):
            timings: Dict[str, float] = state.step_timings
            counts: Dict[str, int] = state.step_counts
            sequence: List[str] = state.step_sequence
        elif isinstance(state, dict):
            timings = state.get("step_timings") or {}
            counts = state.get("step_counts") or {}
            sequence = state.get("step_sequence") or []
        else:
            timings, counts, sequence = {}, {}, []

        # Canonical step order for display
        step_order = [
            "load_context", "retrieve", "plan",
            "select_tools", "execute_tool", "observe",
            "format_response", "guardrail",
        ]
        lines = []
        for step in step_order:
            if step in timings:
                n = counts.get(step, 1)
                lines.append(f"  {step:<18} {timings[step]*1000:>8.1f} ms  (x{n})")
        # Include any unexpected nodes
        for step, t in timings.items():
            if step not in step_order:
                n = counts.get(step, 1)
                lines.append(f"  {step:<18} {t*1000:>8.1f} ms  (x{n})")

        lines.append(f"  {'TOTAL':<18} {total_elapsed*1000:>8.1f} ms")
        lines.append(f"  Execution sequence: {' -> '.join(sequence)}")

        log.info(
            f"[{run_id}] ===== STEP TIMING SUMMARY =====\n"
            + "\n".join(lines)
            + "\n================================="
        )

    async def _execute_sync(self, state: AgentState) -> AgentState:
        """Execute the graph synchronously."""
        # Run the graph
        result = await self.graph.ainvoke(
            state,
            config={"configurable": {"thread_id": state.run_id}}
        )
        # LangGraph may return dict, convert to AgentState if needed
        if isinstance(result, dict):
            return AgentState(**result)
        return result

    async def _run_streaming(
        self,
        state: AgentState,
        run_record: AgentRun
    ) -> AsyncIterator[StreamingEvent]:
        """Execute with streaming output."""
        # TODO: Implement full streaming support
        # For now, yield events from the graph execution
        
        yield StreamingEvent(
            event_type="run.started",
            run_id=state.run_id,
            data={"message": "Run started"}
        )
        
        final_state = await self._execute_sync(state)
        
        # Yield content blocks
        for i, block in enumerate(final_state.content_blocks):
            yield StreamingEvent(
                event_type="message.delta",
                run_id=state.run_id,
                data={
                    "block_index": i,
                    "block": block.dict()
                }
            )
        
        # Fetch follow-up FAQs for the streaming response
        follow_up_faqs = []
        try:
            from ai_agent.workflows.faq.faq import generate_follow_ups
            user_question = state.messages[-1]["content"] if state.messages else ""
            if user_question:
                follow_up_faqs = await generate_follow_ups(
                    agent_id=state.agent_id,
                    user_question=user_question,
                    conversation_id=state.conversation_id,
                    db_session=self.db,
                    api_key=state.api_key,
                )
        except Exception as e:
            log.warning("Failed to fetch follow-ups in AgentRuntime._run_streaming: %s", e)

        yield StreamingEvent(
            event_type="run.completed",
            run_id=state.run_id,
            data={
                "usage": {
                    "input_tokens": final_state.total_input_tokens,
                    "output_tokens": final_state.total_output_tokens,
                    "cost_usd": final_state.total_cost_usd,
                },
                "follow_up_faqs": follow_up_faqs,
            }
        )

    def _build_response(self, state: Union[AgentState, Dict], run_record: AgentRun) -> AgentResponse:
        """Build the final AgentResponse from state."""
        # Convert dict to AgentState if needed
        if isinstance(state, dict):
            state = AgentState(**state)
        
        usage = UsageInfo(
            input_tokens=state.total_input_tokens,
            output_tokens=state.total_output_tokens,
            total_tokens=state.total_input_tokens + state.total_output_tokens,
            cost_usd=state.total_cost_usd,
        )
        
        metadata = ResponseMetadata(
            run_id=state.run_id,
            message_id=0,  # Updated after message creation
            session_id=str(state.session_id) if state.session_id is not None else None,
            conversation_id=state.conversation_id,
            agent_id=state.agent_id,
            provider=state.agent_settings.llm_provider if state.agent_settings else settings.LLM_PROVIDER,
            model=state.agent_settings.llm_model if state.agent_settings else settings.LLM_MODEL,
            latency_ms=0,
            usage=usage,
            citations=state.citations,
        )
        
        return AgentResponse(
            message_id=0,
            run_id=state.run_id,
            session_id=str(state.session_id) if state.session_id is not None else None,
            conversation_id=state.conversation_id,
            blocks=state.content_blocks,
            artifacts=state.artifacts,
            citations=state.citations,
            tool_calls=state.tool_calls,
            tool_results=state.tool_results,
            usage=usage,
            metadata=metadata,
            is_complete=True,
        )

    def _update_run_record(
        self,
        run_record: AgentRun,
        state: AgentState,
        response: AgentResponse
    ):
        """Update the run record with final results."""
        run_record.status = RunStatus.COMPLETED
        run_record.completed_at = datetime.utcnow()
        run_record.total_steps = state.current_step
        run_record.input_tokens = state.total_input_tokens
        run_record.output_tokens = state.total_output_tokens
        run_record.total_tokens = state.total_input_tokens + state.total_output_tokens
        run_record.cost_usd = state.total_cost_usd
        
        self.db.commit()


# Runtime factory
async def get_agent_runtime(db_session: Session) -> AgentRuntime:
    """Get an AgentRuntime instance."""
    return AgentRuntime(db_session)
