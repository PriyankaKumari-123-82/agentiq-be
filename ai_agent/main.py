import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import API routers
from ai_agent.core.config import settings
from ai_agent.api.v1.message import router as message_router
from ai_agent.api.v1.session import router as session_router
from ai_agent.api.v1.user import router as user_router
from ai_agent.api.v1.agent import router as agent_router
from ai_agent.api.v1.mcp import router as mcp_router
from ai_agent.api.v1.document import router as document_router
from ai_agent.api.v1.stats import router as stats_router
from ai_agent.api.v1.voice import router as voice_router
from ai_agent.api.v1.streaming import router as streaming_router
from ai_agent.api.v1.artifacts import router as artifacts_router
from ai_agent.api.v1.usage import router as usage_router
from ai_agent.api.v1.sdk import router as sdk_router
from ai_agent.api.v1.email import router as email_router
from ai_agent.api.v1.organization import router as organization_router
from ai_agent.api.v1.analytics.token import router as analytics_token_router
from ai_agent.api.v1.token.token import router as token_usage_router
from ai_agent.api.v1.governance.token import router as governance_token_router
from ai_agent.api.v1.roles import router as roles_router
from ai_agent.api.v1.members import router as members_router
from ai_agent.api.v1.invitations import router as invitations_router
from ai_agent.api.v1.monitor.monitor_type import router as monitor_types_router
from ai_agent.api.v1.monitor.monitor import router as monitors_router
from ai_agent.api.v1.query_history import router as query_history_router
from ai_agent.api.v1.pluggy import router as pluggy_router
from ai_agent.api.v1.webhook import router as webhook_router
from ai_agent.api.v1.bajaj_dashboard import router as bajaj_dashboard_router
from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.core.llm_gateway import get_llm_gateway
from ai_agent.core.guardrails import get_guardrails_service
from ai_agent.schema_engine.orchestrator import prewarm_all_db_servers

# Force the schema-engine's SQLAlchemy models to register (and fully resolve
# their string-based relationships) at startup, single-threaded, before any
# request thread can lazy-load an unrelated relationship mid-import and
# permanently poison SQLAlchemy's mapper configuration for the process.
import ai_agent.schema_engine.store.pg_models  # noqa: F401


log = logging.getLogger(__name__)



logging.basicConfig(level=logging.DEBUG)
project_version = "0.1.0"
project_name = "AI Agent API"

@asynccontextmanager
async def lifespan(application: FastAPI) -> None:
    global openai_client

    # Startup: initialise MCP manager singleton
    mcp_manager = get_mcp_manager()
    log.info("MCP Manager initialised (connections are created on-demand per agent).")
    
    # Startup: initialise LLM Gateway
    llm_gateway = get_llm_gateway()
    log.info(f"LLM Gateway initialised with provider: {settings.LLM_PROVIDER}")
    
    # Startup: initialise RAG service
    from ai_agent.core.rag_service import get_rag_service
    rag_service = get_rag_service()
    log.info("RAG Service initialised with LlamaIndex")

    # Startup: initialise Milvus connection
    from ai_agent.core.vector_db import DocumentStore
    DocumentStore()._ensure_connected()
    log.info("Milvus connection initialised")

    # Startup: pre-load the Guardrails service (Presidio + spaCy NLP model).
    # This is a heavy one-time load (~2-12s); doing it here at boot instead
    # of lazily on the first request keeps that cost off the first user's
    # request latency.
    get_guardrails_service()

    # Startup: pre-warm the schema engine for every DB-aware MCP server
    # already linked to an agent. Same rationale as the guardrails preload
    # above — see prewarm_all_db_servers's docstring for details.
    await prewarm_all_db_servers(mcp_manager)

    yield

    # Shutdown: disconnect all MCP servers
    await mcp_manager.disconnect_all()
    log.info("All MCP servers disconnected.")

app = FastAPI(
    title=project_name,
    description="API for managing AI agents, conversations, and user interactions",
    version=project_version,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

# Include API routers
app.include_router(message_router, prefix=settings.API_V1_STR)
app.include_router(session_router, prefix=settings.API_V1_STR)
app.include_router(bajaj_dashboard_router, prefix=settings.API_V1_STR)
app.include_router(stats_router, prefix=settings.API_V1_STR)
app.include_router(document_router, prefix=settings.API_V1_STR)
app.include_router(user_router, prefix=settings.API_V1_STR)
app.include_router(agent_router, prefix=settings.API_V1_STR)
app.include_router(mcp_router, prefix=settings.API_V1_STR)
app.include_router(voice_router, prefix=settings.API_V1_STR)
app.include_router(streaming_router, prefix=settings.API_V1_STR)
app.include_router(artifacts_router, prefix=settings.API_V1_STR)
app.include_router(usage_router, prefix=settings.API_V1_STR)
app.include_router(sdk_router, prefix=settings.API_V1_STR)
app.include_router(email_router, prefix=settings.API_V1_STR)
app.include_router(organization_router, prefix=settings.API_V1_STR)
app.include_router(analytics_token_router, prefix=settings.API_V1_STR)
app.include_router(token_usage_router, prefix=settings.API_V1_STR)
app.include_router(governance_token_router, prefix=settings.API_V1_STR)
app.include_router(roles_router, prefix=settings.API_V1_STR)
app.include_router(members_router, prefix=settings.API_V1_STR)
app.include_router(invitations_router, prefix=settings.API_V1_STR)
app.include_router(monitor_types_router, prefix=settings.API_V1_STR)
app.include_router(monitors_router, prefix=settings.API_V1_STR)
app.include_router(query_history_router, prefix=settings.API_V1_STR)
app.include_router(pluggy_router, prefix=f"{settings.API_V1_STR}/pluggy")
app.include_router(webhook_router, prefix=f"{settings.API_V1_STR}/webhook")
