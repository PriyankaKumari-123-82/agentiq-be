import json
import logging
import re

from typing import Optional

from ai_agent.core.config import settings
from ai_agent.core.generate_documents import DocumentGenerator
from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.core.openai_client import create_chat_completion
from ai_agent.core.schema_cache import get_schema_cache, get_schema_selector
from ai_agent.core.permission_context import (
    get_permission_sql_generator,
    get_query_sanitizer,
    create_user_context,
    PermissionLevel,
)
from ai_agent.core.prompt import (
    CONVERSATION_TO_PDF_PROMPT,
    DOWNLOAD_LINK_PROMPT,
    build_mcp_arg_messages,
    build_mcp_data_summary_messages,
    build_mcp_role_tool_picker_messages,
    build_query_builder_messages,
    create_prompt,
)
from ai_agent.helpers.pdf_service import (
    generate_conversation_pdf,
    generate_download_link_response,
    upload_pdf_to_s3,
)
from ai_agent.helpers.schema_formatter import format_schema
from ai_agent.core.rag_service import search_vector_store_for_context
from ai_agent.models import ConversationRole, ConversationType, Message
from ai_agent.services.context import DispatchContext, ResponseModel
from ai_agent.services.mcp_service import _extract_mcp_tool_output_text

log = logging.getLogger(__file__)


# ---------------------------------------------------------------------------
# PDF download pipeline
# ---------------------------------------------------------------------------


async def answer_via_pdf_download(ctx: DispatchContext) -> str:
    """Generate a PDF of the conversation and return a presigned download link
    (as a user-facing response string)."""
    pdf_content = generate_conversation_pdf(
        previous_conversation=ctx.previous_conversation,
        user_message=ctx.chatbot_input.message,
    )
    log.info(f"Generated PDF content for download: {pdf_content}")

    if not pdf_content:
        return "No content available for PDF generation"

    doc_generator = DocumentGenerator()
    filename = f"conversation_{ctx.chatbot_input.conversation_id}.pdf"
    pdf_path = doc_generator.generate_pdf_from_content(
        pdf_content=pdf_content, filename=filename,
    )
    log.info(f"PDF document generated: {pdf_path}")

    if not pdf_path:
        return "Failed to create PDF. Please try again."

    presigned_url = await upload_pdf_to_s3(pdf_path=pdf_path, filename=filename)
    return await generate_download_link_response(
        presigned_url=presigned_url or "", filename=filename,
    )


# ---------------------------------------------------------------------------
# Knowledge-base / RAG pipeline
# ---------------------------------------------------------------------------


def answer_via_knowledge_base(ctx: DispatchContext) -> ResponseModel:
    """RAG flow: retrieve vector-store context, compose prompt, persist the
    user message, call the LLM, persist the assistant message, and return a
    ``ResponseModel`` with the real persisted ``message_id``.

    Unlike the other pipelines this one has DB side-effects, so it returns a
    fully-formed ``ResponseModel`` instead of a plain string."""
    retrieved_context = search_vector_store_for_context(
        session=ctx.session,
        agent_settings=ctx.agent_settings,
        user_message=ctx.chatbot_input.message,
        db_session=ctx.db_session,
    )
    log.info(f"Retrieved context: {retrieved_context}")

    message_prompt = create_prompt(
        custom_prompt=ctx.custom_prompt,
        previous_messages=ctx.messages_to_include,
        web_search_enabled=ctx.web_search_enabled,
        json_prompt=ctx.json_prompt,
        user_question=ctx.chatbot_input.message,
        context=retrieved_context,
    )
    log.info(f"Message prompt: {message_prompt}")

    user_conversation = Message(
        session_id=ctx.session.id,
        agent_id=ctx.chatbot_input.agent_id,
        conversation_id=ctx.chatbot_input.conversation_id,
        role=ConversationRole.USER,
        content=ctx.chatbot_input.message,
        counter=ctx.user_counter,
        conversation_type=ConversationType.TEXT,
        prompt=json.dumps(message_prompt, ensure_ascii=False) if message_prompt else None,
    )
    ctx.db_session.add(user_conversation)
    ctx.db_session.commit()

    previous = ctx.previous_conversation
    temperature = ctx.agent_settings.llm_temperature
    if ctx.chatbot_input.ask_again:
        if previous and previous[-1].get("role") == "assistant":
            previous = previous[:-1]

    system_content = create_chat_completion(
        messages=message_prompt,
        temperature=temperature,
        max_tokens=settings.LLM_MAX_TOKENS,
        web_search_enabled=ctx.web_search_enabled,
        api_key=ctx.api_key,
    )

    system_counter = ctx.user_counter + 1
    system_conversation = Message(
        session_id=ctx.session.id,
        agent_id=ctx.chatbot_input.agent_id,
        conversation_id=ctx.chatbot_input.conversation_id,
        role=ConversationRole.ASSISTANT,
        content=system_content,
        counter=system_counter,
        conversation_type=ConversationType.TEXT,
        prompt=json.dumps(message_prompt, ensure_ascii=False) if message_prompt else None,
    )
    ctx.db_session.add(system_conversation)
    ctx.db_session.commit()
    ctx.db_session.refresh(system_conversation)

    return ResponseModel(
        message_id=system_conversation.id,
        response=system_content,
        session_id=ctx.chatbot_input.session_id,
        conversation_id=ctx.chatbot_input.conversation_id,
    )


# ---------------------------------------------------------------------------
# MCP pipelines
# ---------------------------------------------------------------------------


# SQL keywords that must never appear as the first statement-level token.
# Comparison is done after stripping comments and uppercasing the SQL.
_FORBIDDEN_SQL_LEADERS: tuple[str, ...] = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "MERGE", "REPLACE", "EXEC", "EXECUTE", "CALL",
    "ATTACH", "DETACH", "VACUUM", "ANALYZE", "PRAGMA", "COPY",
)

# Common keys an MCP execute-query tool uses to receive a SQL string.
_SQL_ARG_KEY_CANDIDATES: tuple[str, ...] = ("query", "sql", "statement", "q")


def _validate_select_sql(raw_sql: str) -> str | None:
    """Sanitise an LLM-generated SQL string and return it only when it is a
    single read-only ``SELECT``/``WITH`` statement. Returns ``None`` if the
    query is empty, multi-statement, or contains any destructive verb.

    The cleanup steps (strip markdown fences, strip line/block comments,
    drop a trailing ``;``) are applied before the safety checks; the
    cleaned single-statement SQL is what's returned to the caller so it
    can be passed to the MCP execute-tool.
    """
    if not raw_sql or not raw_sql.strip():
        return None

    # 1. Strip markdown code fences (```sql ... ``` or ``` ... ```) that
    #    LLMs sometimes add despite being told otherwise.
    s = raw_sql.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        s = s[first_nl + 1:] if first_nl != -1 else s.strip("`")
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()

    # 2. Strip ``-- line`` and ``/* block */`` comments.
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i:i + 2] == "--":
            nl = s.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
        elif s[i:i + 2] == "/*":
            end = s.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
        else:
            out.append(s[i])
            i += 1
    cleaned = "".join(out).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()

    # 3. Reject multi-statement input.
    if not cleaned or ";" in cleaned:
        return None

    # 4. First non-whitespace token must be SELECT or WITH.
    upper = cleaned.upper()
    if upper.split(None, 1)[0] not in {"SELECT", "WITH"}:
        return None

    # 5. Defence in depth: forbid destructive verbs anywhere as whole words.
    pattern = r"\b(" + "|".join(_FORBIDDEN_SQL_LEADERS) + r")\b"
    if re.search(pattern, upper):
        return None

    return cleaned


def _guess_sql_arg_key(input_schema: dict | None) -> str | None:
    """Best-effort guess of which property in an MCP tool's ``input_schema``
    expects the SQL string. Returns ``None`` if no obvious match is found."""
    if not isinstance(input_schema, dict):
        return None
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return None
    lowercase_keys = {k.lower(): k for k in properties.keys()}
    for candidate in _SQL_ARG_KEY_CANDIDATES:
        if candidate in lowercase_keys:
            return lowercase_keys[candidate]
    return None


async def build_query_from_mcp_schema(
    server_name: str,
    mcp_tools: list[dict],
    user_question: str,
    user_id: Optional[int] = None,
    tenant_id: Optional[str] = None,
    permission_level: str = "own",
    temperature: float = settings.DEFAULT_LLM_TEMPERATURE,
) -> str:
    """Dynamic-SQL flow for MCP servers with ``requires_query_building=True``.

    Steps:
        1. Pick the schema-returning tool (LLM call).
        2. Extract args for the schema tool (LLM call).
        3. Call the schema tool via the MCP manager.
        4. Build the SQL query from schema + user question (LLM call).
        5. Apply row-level security (permission-aware filtering).
        6. Read-only guardrail check on the generated SQL.
        7. Pick the execute-query tool (LLM call) + extract args (LLM call)
           + call it via the MCP manager.
        8. Summarise the returned rows into natural language (LLM call).

    Args:
        server_name: MCP server to query
        mcp_tools: Available MCP tools
        user_question: Natural language query
        user_id: User ID for permission filtering
        tenant_id: Tenant ID for data isolation
        permission_level: Permission level (none/own/team/department/full)

    Returns a user-facing response string.
    """
    server_tools = [
        {
            "tool_name": t["tool_name"],
            "description": t.get("description"),
            "input_schema": t.get("input_schema") or {},
        }
        for t in mcp_tools
        if t["server_name"] == server_name
    ]
    log.info(
        "MCP query-building flow: server='%s' tools=%d",
        server_name, len(server_tools),
    )

    # 1. Pick the schema-returning tool (only needed if not cached).
    schema_tool_name = None
    schema_tool = None
    
    # Check if we have cached schema first
    schema_cache = get_schema_cache()
    cache_key = f"{server_name}:default"  # Could be parameterized if multiple DBs
    
    cached_schema = schema_cache._local_cache.get(f"schema:{server_name}:default")
    
    if not cached_schema:
        # No cache - need to pick schema tool
        try:
            schema_tool_name = create_chat_completion(
                messages=build_mcp_role_tool_picker_messages("schema", server_tools),
                temperature=0.0,
            ).strip()
            log.info("MCP query-building: schema tool = '%s'", schema_tool_name)
            
            schema_tool = next(
                (t for t in server_tools if t["tool_name"] == schema_tool_name), None
            )
            if schema_tool_name == "NONE" or schema_tool is None:
                return (
                    "The connected database server does not expose a schema tool, "
                    "so I can't build a query dynamically."
                )
        except Exception as e:
            log.error("MCP query-building: schema-tool picker failed: %s", e)
            return "I couldn't find a schema tool on the database server."
    else:
        log.info("MCP query-building: using cached schema (skipping schema tool selection)")

    # 2. Fetch or retrieve cached schema with intelligent table selection.
    async def fetch_schema():
        """Fetch schema from MCP if not cached."""
        if not schema_tool:
            raise ValueError("No schema tool available")
        
        # Extract args for the schema tool (often empty/no required args).
        schema_args: dict = {}
        try:
            schema_arg_messages = build_mcp_arg_messages(
                tool_name=schema_tool["tool_name"],
                description=schema_tool.get("description"),
                input_schema=schema_tool.get("input_schema"),
                user_question=user_question,
            )
            raw_schema_args = create_chat_completion(
                messages=schema_arg_messages, temperature=0.0
            ).strip()
            if raw_schema_args and raw_schema_args != "CANNOT_EXTRACT":
                try:
                    parsed = json.loads(raw_schema_args)
                    if isinstance(parsed, dict):
                        schema_args = parsed
                except Exception:
                    log.warning(
                        "MCP query-building: schema-args JSON parse failed: %r",
                        raw_schema_args,
                    )
        except Exception as e:
            log.error("MCP query-building: schema-args extraction failed: %s", e)
        
        # Call schema tool via MCP
        mcp_mgr = get_mcp_manager()
        schema_result = await mcp_mgr.call_tool(
            server_name, schema_tool["tool_name"], schema_args,
        )
        
        if schema_result.get("isError"):
            raise ValueError("Schema tool returned error")
        
        schema_text = _extract_mcp_tool_output_text(schema_result)
        
        # Parse and format
        try:
            parsed = json.loads(schema_text)
            if isinstance(parsed, dict) and set(parsed.keys()) == {"jsonb_pretty"}:
                parsed = json.loads(parsed["jsonb_pretty"])
            formatted = format_schema(parsed)
            return parsed  # Return parsed structure for caching
        except Exception:
            return schema_text  # Return raw if not JSON
    
    try:
        # Get schema (from cache or fetch)
        schema_entry = await schema_cache.get_or_fetch_schema(
            server_name=server_name,
            database_name="default",  # Could detect from schema_args
            fetch_func=fetch_schema,
        )
        
        # Use intelligent table selection to reduce context size
        schema_selector = get_schema_selector()
        schema_context = schema_selector.build_context_for_query(
            query=user_question,
            schema=schema_entry,
            max_tokens=4000,  # Limit context size
            include_samples=False,  # Disable samples to save tokens
        )
        
        schema_text = schema_context
        log.info(
            "MCP query-building: optimized schema context (%d chars, %d/%d tables)",
            len(schema_text),
            len(schema_selector.select_relevant_tables(user_question, schema_entry)),
            len(schema_entry.tables)
        )
        
    except Exception as e:
        log.error("MCP query-building: schema fetch/optimization failed: %s", e)
        return "Failed to fetch the database schema. Please try again."

    # 4. Query builder LLM call.
    try:
        generated_sql = create_chat_completion(
            messages=build_query_builder_messages(
                user_query=user_question, schema=schema_text,
            ),
            temperature=temperature,
        ).strip()
    except Exception as e:
        log.error("MCP query-building: query builder LLM call failed: %s", e)
        return "I couldn't generate a SQL query for your question."
    log.info("MCP query-building: generated SQL = %r", generated_sql)

    if generated_sql == "UNABLE_TO_GENERATE":
        return (
            "I couldn't build a SQL query for your question from the "
            "available schema. Please rephrase or be more specific."
        )

    # 5. Read-only guardrail (also strips markdown fences and comments).
    safe_sql = _validate_select_sql(generated_sql)
    if safe_sql is None:
        log.warning(
            "MCP query-building: guardrail blocked non-SELECT SQL: %r",
            generated_sql,
        )
        return (
            "Sorry, I can only run read-only SELECT queries against this "
            "database. The generated query was blocked."
        )
    generated_sql = safe_sql

    # 5b. Apply row-level security and sanitize query
    sanitizer = get_query_sanitizer()
    sanitized_sql, error = sanitizer.sanitize(generated_sql)
    if error:
        log.warning(f"MCP query-building: query sanitization failed: {error}")
        return f"Query was blocked for security reasons: {error}"
    generated_sql = sanitized_sql

    # Apply permission-based row-level security
    if user_id:
        try:
            perm_level = PermissionLevel(permission_level)
        except ValueError:
            perm_level = PermissionLevel.OWN
        
        user_ctx = create_user_context(
            user_id=user_id,
            tenant_id=tenant_id,
            permission_level=perm_level,
        )
        
        # Get tables from the schema we already have
        tables_in_query = []
        if 'schema_entry' in locals():
            tables_in_query = list(schema_entry.tables.keys())
        
        perm_generator = get_permission_sql_generator()
        generated_sql = perm_generator.apply_rls_to_query(
            generated_sql,
            user_ctx,
            extracted_tables=tables_in_query,
        )
        log.info(f"MCP query-building: applied RLS for user {user_id}, tenant {tenant_id}")

    # 6. Pick + call the execute-query tool.
    try:
        execute_tool_name = create_chat_completion(
            messages=build_mcp_role_tool_picker_messages("execute", server_tools),
            temperature=0.0,
        ).strip()
    except Exception as e:
        log.error("MCP query-building: execute-tool picker failed: %s", e)
        return "I couldn't find a query-execution tool on the database server."
    log.info("MCP query-building: execute tool = '%s'", execute_tool_name)

    execute_tool = next(
        (t for t in server_tools if t["tool_name"] == execute_tool_name), None
    )
    if execute_tool_name == "NONE" or execute_tool is None:
        return (
            "The connected database server does not expose a query-execution "
            "tool, so I can't run the generated SQL."
        )

    execute_args: dict = {}
    try:
        execute_arg_messages = build_mcp_arg_messages(
            tool_name=execute_tool["tool_name"],
            description=execute_tool.get("description"),
            input_schema=execute_tool.get("input_schema"),
            user_question=f"Execute this SQL query: {generated_sql}",
        )
        raw_execute_args = create_chat_completion(
            messages=execute_arg_messages, temperature=0.0
        ).strip()
        if raw_execute_args and raw_execute_args != "CANNOT_EXTRACT":
            try:
                parsed = json.loads(raw_execute_args)
                if isinstance(parsed, dict):
                    execute_args = parsed
            except Exception:
                log.warning(
                    "MCP query-building: execute-args JSON parse failed: %r",
                    raw_execute_args,
                )
    except Exception as e:
        log.error("MCP query-building: execute-args extraction failed: %s", e)

    # Fallback: if the LLM produced no args but the tool clearly takes a
    # SQL string, inject the generated query under the most likely key.
    if not execute_args:
        sql_key = _guess_sql_arg_key(execute_tool.get("input_schema"))
        if sql_key:
            execute_args = {sql_key: generated_sql}
            log.info(
                "MCP query-building: defaulted execute args to {%r: <sql>}",
                sql_key,
            )

    try:
        execute_result = await mcp_mgr.call_tool(
            server_name, execute_tool["tool_name"], execute_args,
        )
    except Exception as e:
        log.error("MCP query-building: execute tool call failed: %s", e)
        return "Failed to run the SQL query against the database."

    if execute_result.get("isError"):
        log.warning(
            "MCP query-building: execute tool returned an error: %s",
            execute_result,
        )
        return "The database server reported an error while running the query."

    data_text = _extract_mcp_tool_output_text(execute_result)
    log.info("MCP query-building: data fetched (%d chars)", len(data_text))

    # 7. Summarise the data into a human-readable response.
    try:
        summary = create_chat_completion(
            messages=build_mcp_data_summary_messages(
                user_question=user_question,
                sql_query=generated_sql,
                data_text=data_text,
            ),
            temperature=temperature,
        )
    except Exception as e:
        log.error("MCP query-building: summary LLM call failed: %s", e)
        # Degrade gracefully: still return the raw data so the user has
        # something useful.
        return f"Query results:\n{data_text}"

    return summary


async def answer_via_mcp_tool(
    server_name: str,
    mcp_tools: list[dict],
    user_question: str,
    temperature: float = settings.DEFAULT_LLM_TEMPERATURE,
) -> str:
    """Action-style MCP flow for servers with ``requires_query_building=False``.

    Flow:
        1. Tool picker LLM call.
        2. Args extractor LLM call.
        3. ``mcp_mgr.call_tool(server, tool, args)``.
        4. Summary LLM call.
    """
    log.info(
        "MCP action flow (requires_query_building=False) for '%s'", server_name
    )

    # Get tools for this server
    server_tools = [
        {
            "tool_name": t["tool_name"],
            "description": t.get("description", ""),
            "input_schema": t.get("input_schema") or {},
        }
        for t in mcp_tools
        if t["server_name"] == server_name
    ]

    if not server_tools:
        return f"No tools available from server '{server_name}'."

    # 1. Tool picker
    try:
        tool_picker_messages = [
            {"role": "system", "content": "You are a tool selector. Choose the best tool for the user's request."},
            {"role": "user", "content": f"Available tools:\n" + "\n".join([
                f"- {t['tool_name']}: {t['description']}"
                for t in server_tools
            ]) + f"\n\nUser request: {user_question}\n\nWhich tool should be called? Respond with just the tool name, or 'NONE' if no tool is appropriate."}
        ]

        selected_tool_response = create_chat_completion(
            messages=tool_picker_messages,
            temperature=0.0,
            max_tokens=50,
        ).strip()

        log.info(f"MCP action flow: selected tool = '{selected_tool_response}'")

        if selected_tool_response.upper() == "NONE":
            return "No appropriate tool found for this request."

        # Find the selected tool
        selected_tool = None
        for tool in server_tools:
            if tool["tool_name"] == selected_tool_response:
                selected_tool = tool
                break

        if not selected_tool:
            # Fuzzy match
            for tool in server_tools:
                if selected_tool_response.lower() in tool["tool_name"].lower():
                    selected_tool = tool
                    break

        if not selected_tool:
            return f"Selected tool '{selected_tool_response}' not found on server."

    except Exception as e:
        log.error(f"MCP action flow: tool picker failed: {e}")
        return "Failed to select an appropriate tool."

    # 2. Args extractor
    tool_args = {}
    try:
        input_schema = selected_tool.get("input_schema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        if properties:
            args_extraction_messages = [
                {"role": "system", "content": f"Extract arguments for the {selected_tool['tool_name']} tool."},
                {"role": "user", "content": f"Tool schema: {json.dumps(input_schema)}\n\nUser request: {user_question}\n\nExtract the tool arguments as a JSON object. Required fields: {', '.join(required)}"}
            ]

            args_response = create_chat_completion(
                messages=args_extraction_messages,
                temperature=0.0,
                max_tokens=500,
            ).strip()

            try:
                parsed_args = json.loads(args_response)
                if isinstance(parsed_args, dict):
                    tool_args = parsed_args
            except json.JSONDecodeError:
                log.warning(f"MCP action flow: failed to parse args JSON: {args_response[:200]}")
                # Try to extract simple key-value pairs
                tool_args = _extract_simple_args(args_response)

        log.info(f"MCP action flow: extracted args = {tool_args}")

    except Exception as e:
        log.error(f"MCP action flow: args extraction failed: {e}")
        # Continue with empty args, some tools don't need them

    # 3. Execute tool
    try:
        from ai_agent.core.mcp_manager import get_mcp_manager
        mcp_mgr = get_mcp_manager()

        result = await mcp_mgr.call_tool(
            server_name=server_name,
            tool_name=selected_tool["tool_name"],
            args=tool_args,
        )

        if result.get("isError"):
            log.error(f"MCP action flow: tool returned error: {result}")
            return f"The tool reported an error: {result.get('content', 'Unknown error')}"

    except Exception as e:
        log.error(f"MCP action flow: tool execution failed: {e}")
        return f"Failed to execute the tool: {str(e)}"

    # 4. Summarize results
    try:
        data_text = _extract_mcp_tool_output_text(result)

        if len(data_text) < 200:
            # Short result, return directly
            return data_text

        summary_messages = [
            {"role": "system", "content": "Summarize tool output concisely."},
            {"role": "user", "content": f"Tool: {selected_tool['tool_name']}\nOutput:\n{data_text[:2000]}\n\nProvide a brief summary of the results."}
        ]

        summary = create_chat_completion(
            messages=summary_messages,
            temperature=temperature,
            max_tokens=500,
        )

        return summary

    except Exception as e:
        log.error(f"MCP action flow: summary failed: {e}")
        # Return raw data
        return _extract_mcp_tool_output_text(result)


def _extract_simple_args(text: str) -> dict:
    """Extract simple key=value pairs from text as fallback."""
    args = {}
    # Look for patterns like "key: value" or "key = value"
    import re
    pattern = r'["\']?(\w+)["\']?\s*[:=]\s*["\']?([^"\',\n}]+)["\']?'
    matches = re.findall(pattern, text)
    for key, value in matches:
        args[key.strip()] = value.strip()
    return args
