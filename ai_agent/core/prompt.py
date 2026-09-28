import json as _json

# ---------------------------------------------------------------------------
# Intent classifier (top-level router)
# ---------------------------------------------------------------------------
#
# A single LLM call decides which path answers the user's question. The menu is
# built dynamically per agent and contains:
#   * Built-in intents (sql_db, download_template) — gated by agent settings.
#   * One entry per connected MCP server, using the server's human-authored
#     `description` from the DB (admins tune routing by editing descriptions).
#
# The classifier returns the intent name verbatim, or 'none' if nothing fits.
# 'none' is the catch-all fallback — handled by the RAG / general-chat
# pipeline, which uses the agent's knowledge base when available and falls
# back to plain LLM chat otherwise (so greetings and general questions still
# get answered).

# Built-in intent name constants (also used by the dispatcher in message.py).
INTENT_DOWNLOAD_TEMPLATE = "download_template"
INTENT_NONE = "none"

# Stable, agent-agnostic descriptions for built-in intents.
BUILTIN_INTENT_DESCRIPTIONS: dict[str, str] = {
    INTENT_DOWNLOAD_TEMPLATE: (
        "Export the current conversation or response as a downloadable PDF "
        "(only when the user explicitly asks to download/save/export to PDF)."
    ),
}


def build_intent_options(
    is_pdf_export_enabled: bool,
    mcp_servers: list[dict],
) -> list[dict]:
    """Build the dynamic intent menu shown to the classifier.

    Args:
        is_pdf_export_enabled: Whether PDF export of conversations is allowed.
        mcp_servers: List of ``{name, description}`` for connected MCP servers.
            Servers with a missing description fall back to a generic line so
            they remain selectable.

    Returns:
        A list of ``{name, description}`` dicts in router order.

    The knowledge-base / general-chat path is intentionally NOT listed here —
    it is the ``none`` fallback handled by the dispatcher.
    """
    options: list[dict] = []

    if is_pdf_export_enabled:
        options.append({
            "name": INTENT_DOWNLOAD_TEMPLATE,
            "description": BUILTIN_INTENT_DESCRIPTIONS[INTENT_DOWNLOAD_TEMPLATE],
        })

    for srv in mcp_servers:
        name = srv["name"]
        desc = (srv.get("description") or "").strip() or f"Connected MCP server: {name}"
        options.append({"name": name, "description": desc})

    return options


def build_intent_classifier_messages(
    user_question: str,
    intent_options: list[dict],
) -> list[dict]:
    """Single-step intent classifier prompt.

    The LLM picks one intent name (built-in tool name OR an MCP server name)
    based purely on the option descriptions, returning ``'none'`` if nothing
    fits. This is the only routing LLM call — there is no second-stage MCP
    server picker.
    """
    valid_names = [opt["name"] for opt in intent_options]
    option_lines = "\n".join(
        f'- "{opt["name"]}" — {opt["description"]}' for opt in intent_options
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an intent routing classifier.\n\n"

                "Your task is to classify the user's query into EXACTLY ONE intent "
                "from the available options.\n\n"

                "Decision rules:\n"

                "1. Choose a SPECIFIC intent when:\n"
                "- The query clearly matches an option's description.\n"
                "- The query requires structured retrieval (database, APIs, MCP tools, workflows).\n"
                "- The query depends on external, real-time, or proprietary data (RAG).\n\n"

                f"2. Choose '{INTENT_NONE}' when:\n"
                "- The query is general knowledge (facts, concepts, explanations).\n"
                "- The query is conversational (hi, hello, small talk).\n"
                "- The query can be answered without tools or retrieval.\n"
                "- The query does not clearly match any option.\n\n"

                "3. If multiple options seem valid:\n"
                "- Select the MOST specific and directly relevant one.\n"
                "- Prefer tool-based intents ONLY when clearly required.\n\n"

                "Output rules:\n"
                f"- Respond with EXACTLY one of: {' | '.join(valid_names)} | {INTENT_NONE}\n"
                "- No explanation.\n"
                "- No punctuation.\n"
                "- No extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Available options:\n{option_lines}\n\n"
                f"User question: {user_question}\n\n"
                "Which option should answer this question?"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

MCP_ARG_EXTRACTION_PROMPT = {
    "role": "system",
    "content": (
        "You extract arguments for a tool call.\n"
        "Given a tool name, its description, its JSON input schema, and the user's question, "
        "return ONLY a JSON object that satisfies the schema.\n"
        "Rules:\n"
        "- Return strictly valid JSON, no markdown, no backticks, no commentary.\n"
        "- If the schema has no required fields and nothing is inferable, return {}.\n"
        "- If a REQUIRED field cannot be inferred from the user question, "
        "respond with the exact single token: CANNOT_EXTRACT\n"
    ),
}


def build_mcp_arg_messages(
    tool_name: str,
    description: str | None,
    input_schema: dict | None,
    user_question: str,
) -> list[dict]:
    """Messages to ask the LLM to emit arguments for an MCP tool call."""
    schema_str = _json.dumps(input_schema or {}, indent=2)
    desc = (description or "").strip() or "(no description)"
    return [
        MCP_ARG_EXTRACTION_PROMPT,
        {
            "role": "user",
            "content": (
                f"Tool: {tool_name}\n"
                f"Description: {desc}\n"
                f"Input schema (JSON):\n{schema_str}\n\n"
                f"User question: {user_question}\n\n"
                "Return only the JSON arguments object."
            ),
        },
    ]


def build_mcp_summary_messages(
    user_question: str,
    tool_name: str,
    tool_output: str,
    is_error: bool = False,
) -> list[dict]:
    """Messages to ask the LLM to turn a tool call's result into a user-facing reply."""
    system_content = (
        "You are a helpful assistant. Use the tool output below to answer the user's "
        "question concisely and clearly. Do not mention the tool by name unless relevant."
    )
    if is_error:
        system_content += (
            " The tool reported an error; explain what went wrong to the user briefly "
            "and suggest what they could try next."
        )
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Question: {user_question}\n\n"
                f"Tool `{tool_name}` returned:\n{tool_output}"
            ),
        },
    ]

def build_mcp_generic_tool_picker_messages(
    user_question: str,
    tools: list[dict],
) -> list[dict]:
    """Ask the LLM to pick the single best tool on a server for the user's question.

    Used for MCP servers that expose purpose-built actions (Drive, Slack, Jira,
    weather, etc.).

    tools: list of {tool_name, description, input_schema}.
    The LLM must reply with the exact ``tool_name`` or ``NONE``.
    """
    valid_names = [t["tool_name"] for t in tools]
    tool_blocks = []
    for t in tools:
        desc = (t.get("description") or "").strip() or "(no description)"
        tool_blocks.append(f"- {t['tool_name']}: {desc}")
    tools_text = "\n".join(tool_blocks)

    return [
        {
            "role": "system",
            "content": (
                "You pick a single MCP tool that best answers the user's question.\n"
                "Rules:\n"
                f"- Respond with EXACTLY one of: {' | '.join(valid_names)} | NONE\n"
                "- No punctuation, no explanation, no markdown.\n"
                "- Pick NONE if no tool fits the question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Available tools:\n{tools_text}\n\n"
                f"User question: {user_question}\n\n"
                "Which tool should be called?"
            ),
        },
    ]


def build_mcp_role_tool_picker_messages(
    role: str,
    tools: list[dict],
) -> list[dict]:
    """Ask the LLM to pick which tool on a server performs a given role.

    role: Either ``"schema"`` (a tool that returns DB schema / tables / columns)
          or ``"execute"`` (a tool that runs an arbitrary SQL query).
    tools: list of {tool_name, description, input_schema}.

    The LLM must reply with the exact ``tool_name`` or ``NONE``.
    """
    valid_names = [t["tool_name"] for t in tools]
    tool_blocks = []
    for t in tools:
        desc = (t.get("description") or "").strip() or "(no description)"
        tool_blocks.append(f"- {t['tool_name']}: {desc}")
    tools_text = "\n".join(tool_blocks)

    if role == "schema":
        role_instructions = (
            "You must pick the tool that returns the database SCHEMA "
            "(table definitions, columns, types, or a tables listing). "
            "It must NOT be a tool that runs arbitrary SQL."
        )
    elif role == "execute":
        role_instructions = (
            "You must pick the tool that EXECUTES an arbitrary SQL query "
            "supplied by the caller (typically takes a 'query' / 'sql' string). "
            "It must NOT be a tool that only returns schema or domain-specific data."
        )
    else:
        raise ValueError(f"Unknown role: {role!r}")

    return [
        {
            "role": "system",
            "content": (
                f"You select an MCP tool for a specific role: {role}.\n"
                f"{role_instructions}\n"
                "Rules:\n"
                f"- Respond with EXACTLY one of: {' | '.join(valid_names)} | NONE\n"
                "- No punctuation, no explanation, no markdown.\n"
                "- Pick NONE if no tool fits the role."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Available tools:\n{tools_text}\n\n"
                f"Which tool matches the '{role}' role?"
            ),
        },
    ]


def build_query_builder_messages(
    user_query: str,
    schema: str,
    ) -> list[dict]:

        system_content = f"""
    You are an expert PostgreSQL SQL generator.

    Generate a single valid PostgreSQL SELECT query using ONLY the provided schema.

    Rules:
    - Return ONLY raw SQL.
    - No markdown or explanations.
    - Only SELECT queries are allowed.
    - Never use INSERT, UPDATE, DELETE, DROP, ALTER, or DDL.
    - Use only tables and columns present in the schema.
    - Do not invent columns or tables.
    - Always end the query with a semicolon.
    - Use table aliases when joins are used.
    - Qualify columns in multi-table queries.
    - Avoid sensitive columns like password/token unless explicitly requested.

    Joins:
    - Prefer explicit foreign keys like:
    camera_id:uuid FK->cameras.id
    - Use the FK relationship for joins whenever available.
    - If FK is missing, infer joins using <table>_id = table.id patterns.
    - Use the simplest valid join path.

    Soft Deletes:
    - If a table contains is_deleted, filter with:
    is_deleted = false

    Limits:
    - Apply LIMIT 100 unless the user explicitly asks for all results.
    - Do not apply LIMIT to aggregate queries.

    If the schema does not contain enough information to answer safely, return exactly:
    UNABLE_TO_GENERATE

    Schema:
    {schema}
    """

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]



def build_mcp_data_summary_messages(
    user_question: str,
    sql_query: str,
    data_text: str,
) -> list[dict]:
    """Messages for the final MCP-flow LLM call: turn the rows returned by
    the execute-query tool into a human-readable answer to the user.

    The MCP execute tool returns text already (often pre-formatted as a
    table or JSON), so the LLM is told to read it as-is and write a clear
    natural-language summary, citing concrete numbers where useful.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful data analyst. The user asked a question, "
                "we ran a SQL query against an MCP database server, and got "
                "back the data shown below. Answer the user's question in a "
                "clear, concise natural-language response.\n\n"
                "Rules:\n"
                "- Do NOT invent data; rely only on the rows provided.\n"
                "- If the data is empty, say so plainly.\n"
                "- Cite concrete numbers / names where useful.\n"
                "- Do NOT include the raw SQL or repeat the question verbatim."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {user_question}\n\n"
                f"SQL executed:\n{sql_query}\n\n"
                f"Data returned by the server:\n{data_text}"
            ),
        },
    ]


default_prompt = {
    "role": "system",
    "content": "You are an assistant the user can ask any question."
        "Answer in your usual style: clear, practical guidance with headings, "
        "bullet points, and a short summary at the end."
    }

reference_material_disable_web_search = {
    "role": "system",
    "content": "Additional rules for answering:\n"
        "1. Use ONLY the Reference material passed to you.\n"
        "2. Do NOT use outside knowledge, memory, or assumptions.\n"
        "3. If the reference material contains URLs or references, always include them in the answer."
    }

reference_material_enable_web_search = {
    "role": "system",
    "content": "1. If relevant reference documents are provided, use them to support your answer. "
        "2. If they are not relevant, ignore them. Never mention 'retrieved documents' directly—just weave the information naturally into the response."
        "3. If the reference material contains URLs or references, always include them in the answer."
}


retrieved_docs = {"role": "system", "content": "CONTEXT:\n\n"}


DOWNLOAD_LINK_PROMPT = {
    "role": "system",
    "content": (
        "You are a document generation assistant. Your task is to provide a download link for the generated content.\n\n"
        "You will be given context that contains a download link. Extract this information and return it in JSON format.\n\n"
        "Your response must be a JSON object with exactly these three keys:\n"
        "- message: A descriptive message about the download (e.g., 'Your document is ready for download')\n"
        "- link: The download URL from the context\n"
        "- filename: The suggested filename for the download (e.g., 'response.pdf')\n\n"
        "Important rules:\n"
        "- Only return a JSON object, no other text\n"
        "- Use the exact link provided in the context\n"
        "- Provide a meaningful filename based on the content\n"
        "- Keep the message brief and user-friendly\n"
    )
}

CONVERSATION_TO_PDF_PROMPT = {
    "role": "system",
    "content": (
        "You are a document generation assistant. Your task is to create a well-formatted markdown document from a conversation for PDF conversion.\n\n"
        "You will be given:\n"
        "- Previous conversation history between user and assistant\n"
        "- The user's current question/request\n\n"
        "Your task:\n"
        "1. Analyze conversation to understand the user's intent\n"
        "2. Create comprehensive markdown content that addresses their needs\n"
        "3. Format content professionally with proper headings, structure, and styling\n\n"
        "Formatting requirements:\n"
        "- Use proper markdown headings (# ## ###)\n"
        "- Include bullet points or numbered lists where appropriate\n"
        "- Add a title that reflects the content\n"
        "- Ensure content flows logically and is complete\n"
        "- Include any relevant data, code examples, or explanations from the conversation\n\n"
        "Special instructions:\n"
        "- If the user asked for a PDF of the last 4-5 messages, focus on those recent exchanges\n"
        "- If the user asked for a PDF of the entire conversation, include all relevant parts\n"
        "- If the user asked to summarize the conversation, create a concise summary of key points discussed\n"
        "- If the user asked for a specific topic summary, focus on that topic from the conversation\n"
        "- Make content self-contained and comprehensive\n"
        "- Do not include this instruction text in your response\n"
        "- Return ONLY markdown content, no explanations or meta-commentary\n\n"
        "The markdown content will be converted to PDF, so ensure it is properly formatted for document generation."
    )
}


def create_prompt(custom_prompt: dict | str = None, previous_messages: list = None,
                 web_search_enabled: bool = False, json_prompt: bool = False, 
                 user_question: str = None, context: str = None):
    """
    Create a prompt for chat completion API.
    
    Args:
        custom_prompt: Optional custom prompt to use instead of the default.
                      Can be a dict, JSON string, or regular string.
        previous_messages: Optional list of previous messages in the conversation
        web_search_enabled: Whether web search is enabled for this conversation
        json_prompt: Whether to enforce JSON response format (handled by the API)
        user_question: The current user's question to be included in the prompt
        context: Optional retrieved context from vector store
        
    Returns:
        list: List of message dictionaries for chat completion API
    """
    prompt = []
    
    # Add default or custom system prompt
    if custom_prompt is None:
        prompt.append(default_prompt)
    else:
        if json_prompt and isinstance(custom_prompt, str):
            import json
            parsed_prompt = json.loads(custom_prompt)
            prompt.append(parsed_prompt)
        else:
            prompt.append({
                "role": "system",
                "content": str(custom_prompt)
            })
    
    # Add web search reference material
    if web_search_enabled:
        prompt.append(reference_material_enable_web_search)
    else:
        prompt.append(reference_material_disable_web_search)

    # Add retrieved documents context before conversation so the model has it before the question
    if context and context.strip():
        prompt.append({"role": "system", "content": f"CONTEXT:\n\n{context}"})

    # Add conversation history if provided
    if previous_messages:
        prompt.extend(previous_messages)

    # Add current user question if provided
    if user_question and user_question.strip():
        prompt.append({
            "role": "user",
            "content": user_question.strip()
        })

    return prompt

