# Agent IQ Platform Enhancement - Implementation Summary

This document summarizes the major enhancements made to the Agent IQ platform to transform it into a modern, enterprise-grade agent platform.

## Overview of Changes

### 1. Multi-LLM Support via LiteLLM
**File:** `ai_agent/core/llm_gateway.py`

- Unified gateway supporting OpenAI, Anthropic, Google, Mistral, Cohere, Groq, Azure
- Automatic fallback handling
- Token counting without API calls
- Support for streaming responses
- Cost estimation per request

**Configuration:**
```python
# In agent settings or config
LLM_PROVIDER="openai"  # or "anthropic", "google", "mistral", etc.
LLM_MODEL="gpt-4o"     # or "claude-3-5-sonnet", "gemini-1.5-pro", etc.
```

### 2. Agent Runtime with LangGraph
**File:** `ai_agent/core/agent_runtime.py`

- State-machine-based agent execution
- Multi-step planning and tool calling
- Execution limits (max_steps, max_tool_calls, timeout)
- Persistent run tracking with `agent_runs` table
- Step-by-step execution tracking with `agent_run_steps` table

**Modes:**
- `chat`: Standard conversational mode
- `tool_use`: Single-step tool execution
- `planner`: Plan-then-execute mode
- `autonomous`: Multi-step agentic loop

### 3. Advanced RAG with LlamaIndex
**File:** `ai_agent/core/rag_service.py`

- Hybrid search (vector + keyword)
- Cohere reranking support
- Citation tracking with source documents
- Query rewriting for better retrieval
- Automatic text extraction from PDFs, DOCX, HTML

**Features:**
```python
# The RAG service now provides:
- Citations with relevance scores
- Page/section references
- Source document metadata
- Hybrid retrieval with configurable weights
```

### 4. Streaming Responses
**File:** `ai_agent/api/v1/streaming.py`

- Server-Sent Events (SSE) endpoint
- Real-time progress updates
- Streaming content blocks
- Tool call progress events
- Citation discovery events

**Endpoint:**
```
POST /api/v1/stream/message
Content-Type: application/json

{
  "conversation_id": "abc123",
  "session_id": "xyz789",
  "message": "Analyze Q3 sales data",
  "mode": "tool_use",
  "stream_citations": true,
  "stream_tool_calls": true
}
```

### 5. Block-Based Response Format
**File:** `ai_agent/core/response_schemas.py`

Responses now include structured blocks:

```json
{
  "message_id": 123,
  "run_id": "uuid",
  "blocks": [
    {"type": "markdown", "content": "Here are the Q3 sales results:"},
    {
      "type": "table",
      "content": {
        "columns": [{"name": "Product", "type": "string"}, {"name": "Revenue", "type": "number"}],
        "rows": [["Product A", 150000], ["Product B", 230000]]
      }
    },
    {"type": "citation", "content": {"source_name": "sales_report.pdf", "page": 5}}
  ],
  "citations": [...],
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 800,
    "cost_usd": 0.015
  },
  "artifacts": [...]
}
```

### 6. Usage Tracking & Quotas
**Files:** `ai_agent/core/usage_tracker.py`, `ai_agent/api/v1/usage.py`

**Features:**
- Per-request token/cost tracking in `llm_usage_events` table
- Usage quotas with period limits (daily/weekly/monthly)
- Per-user and per-agent quotas
- Cost estimation API
- Usage analytics

**API Endpoints:**
```
GET /api/v1/usage/user/{user_id}/summary?days=30
GET /api/v1/usage/user/{user_id}/events
GET /api/v1/usage/agent/{agent_id}/summary
GET /api/v1/usage/run/{run_id}
POST /api/v1/usage/quotas/{user_id}
GET /api/v1/usage/quotas/{user_id}
GET /api/v1/usage/cost-estimate?provider=openai&model=gpt-4o&input_tokens=1000&output_tokens=500
```

### 7. File Artifacts
**File:** `ai_agent/api/v1/artifacts.py`

Generated outputs can be saved as artifacts:
- PDF, CSV, XLSX, JSON, TXT, MD, HTML
- Presigned download URLs
- Automatic expiration (30 days default)
- Size tracking

**API Endpoints:**
```
POST /api/v1/artifacts/generate/{run_id}
GET /api/v1/artifacts/run/{run_id}
GET /api/v1/artifacts/conversation/{conversation_id}
GET /api/v1/artifacts/{artifact_id}
DELETE /api/v1/artifacts/{artifact_id}
```

### 8. Background Document Processing
**Files:** `ai_agent/tasks/document_tasks.py`, `ai_agent/tasks/cleanup.py`

- Celery-based async document ingestion
- Progress tracking with `document_ingestion_jobs` table
- Chunk-level processing status
- Automatic retry for failed jobs
- Cleanup tasks for expired artifacts

**Configuration:**
```bash
# Start Celery worker
celery -A ai_agent.core.celery_config worker -l info -Q documents

# Start Celery beat for scheduled tasks
celery -A ai_agent.core.celery_config beat -l info
```

### 9. Guardrails & Safety
**File:** `ai_agent/core/guardrails.py`

- PII detection with Presidio (or regex fallback)
- Prompt injection detection
- Content moderation
- SQL injection prevention for database tools
- Tool permission validation

**Configuration:**
```python
ENABLE_PII_DETECTION=True
ENABLE_PROMPT_INJECTION_DETECTION=True
ENABLE_CONTENT_MODERATION=True
```

### 10. Langfuse Observability
**File:** `ai_agent/core/langfuse_integration.py`

- Automatic tracing of LLM calls
- Agent run tracking
- Prompt management
- Custom scoring for feedback

**Configuration:**
```python
LANGFUSE_PUBLIC_KEY="pk-..."
LANGFUSE_SECRET_KEY="sk-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 11. MCP Action Flow Implementation
**File:** `ai_agent/services/pipelines.py`

The previously stubbed MCP action flow is now fully implemented:
- Tool selection via LLM
- Argument extraction from user input
- Tool execution via MCP manager
- Result summarization

**Flow:**
1. Parse available tools from MCP servers
2. LLM selects appropriate tool
3. LLM extracts arguments from user query
4. Execute tool via MCP
5. Summarize results for user

### 12. Database Schema Extensions
**Migration:** `ai_agent/alembic/versions/2026_05_11_0000_add_agent_runtime_models.py`

New tables added:
- `agent_runs`: Complete execution runs
- `agent_run_steps`: Step-by-step execution
- `llm_usage_events`: Detailed usage tracking
- `artifacts`: Generated file outputs
- `audit_logs`: Security audit trail
- `document_ingestion_jobs`: Background processing status
- `usage_quotas`: User/agent quotas

### 13. New API Endpoints

**Message v2 (AgentRuntime):**
```
POST /api/v1/message/send-message-v2
Query params: mode, user_id
Returns: Block-based response with usage info
```

**Streaming:**
```
POST /api/v1/stream/message
Returns: SSE stream with events
```

## Migration Steps

### 1. Install Dependencies
```bash
pip install -e ".[dev]"
```

New dependencies:
- langgraph>=0.2.0
- langfuse>=2.0.0
- llama-index>=0.12.0
- litellm>=1.59.0
- celery>=5.3.0
- presidio-analyzer>=2.2.0
- redis>=5.0.0

### 2. Run Database Migration
```bash
cd ai_agent
alembic upgrade head
```

### 3. Configure Environment Variables
```bash
# Add to .env
REDIS_URL=redis://localhost:6379/0
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
ENABLE_PII_DETECTION=true
```

### 4. Start Celery Workers
```bash
# Terminal 1: Worker for document processing
celery -A ai_agent.core.celery_config worker -l info -Q documents -n documents@%h

# Terminal 2: Worker for cleanup tasks
celery -A ai_agent.core.celery_config worker -l info -Q cleanup -n cleanup@%h

# Terminal 3: Beat scheduler
celery -A ai_agent.core.celery_config beat -l info
```

### 5. Start Application
```bash
uvicorn ai_agent.main:app --reload
```

## Usage Examples

### Using the New AgentRuntime Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/message/send-message-v2?mode=tool_use&user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "session_id": "sess_456",
    "message": "Query all customers from the database",
    "agent_id": "agent_789"
  }'
```

### Streaming Response

```bash
curl -N -H "Accept: text/event-stream" \
  -X POST "http://localhost:8000/api/v1/stream/message?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "session_id": "sess_456",
    "message": "Generate a sales report",
    "mode": "autonomous",
    "stream_citations": true,
    "stream_tool_calls": true
  }'
```

### Setting Usage Quotas

```bash
curl -X POST "http://localhost:8000/api/v1/usage/quotas/1" \
  -H "Content-Type: application/json" \
  -d '{
    "quota_type": "tokens",
    "limit": 100000,
    "period": "monthly",
    "action_on_exceed": "block"
  }'
```

### Generating an Artifact

```bash
curl -X POST "http://localhost:8000/api/v1/artifacts/generate/{run_id}?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_type": "csv",
    "data": {
      "rows": [
        {"customer": "Acme Corp", "revenue": 150000},
        {"customer": "Globex", "revenue": 230000}
      ]
    },
    "filename": "sales_report.csv"
  }'
```

## Architecture Changes

### Before
```
User Request -> Intent Router -> Single Pipeline -> Response
```

### After
```
User Request -> Guardrails Check -> Quota Check -> AgentRuntime -> 
  [Planning -> Retrieval -> Tool Selection -> Tool Execution -> 
   Observation -> Response Formatting -> Guardrails] -> 
Block-Based Response with Usage/Citations/Artifacts
```

## Key Improvements

1. **Multi-Provider Support**: No longer locked into OpenAI
2. **Agentic Execution**: Multi-step planning and tool use
3. **Observability**: Full tracing with Langfuse
4. **Cost Control**: Usage tracking and quotas
5. **Safety**: Guardrails for PII, injection, moderation
6. **Scalability**: Background processing with Celery
7. **Rich Responses**: Tables, citations, artifacts
8. **Streaming**: Real-time response delivery

## Backwards Compatibility

The original `/send-message` endpoint continues to work as before.
New features are available via:
- `/send-message-v2` - Block-based AgentRuntime
- `/stream/message` - Streaming SSE
- All new `/usage`, `/artifacts`, `/stream` endpoints

## Next Steps / Future Enhancements

1. **Voice Integration**: Refactor voice endpoint to use AgentRuntime
2. **Multi-Agent**: Add supervisor/worker agent patterns
3. **Workflow Builder**: Visual workflow editor
4. **Evaluation Framework**: Automated testing with Ragas/DeepEval
5. **Memory**: Long-term conversation memory
6. **Embeddable SDK**: JavaScript/React SDK for external integration

## Support

For issues or questions:
1. Check logs with `DEBUG=true`
2. Review Langfuse traces
3. Verify quota limits in usage dashboard
4. Check Celery worker status
