import asyncio
import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

_CONNECT_TIMEOUT_S = 30.0


def _task_is_being_cancelled() -> bool:
    """True when the *current* task has genuinely been cancelled.

    Distinguishes a real cancellation (client disconnect, shutdown) from a
    CancelledError leaking out of a library's internal task group.
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0

log = logging.getLogger(__name__)


# Monkey-patch mcp streamable_http to log raw response on JSON parse failure
def _install_mcp_response_logger() -> None:
    try:
        from mcp.client import streamable_http as _sh
        from mcp.types import JSONRPCMessage
        from mcp.shared.message import SessionMessage

        async def _patched(self, response, read_stream_writer, is_initialization=False, **kwargs):
            try:
                content = await response.aread()
            except Exception as exc:
                log.error("MCP: failed to read response body: %s", exc)
                await read_stream_writer.send(exc)
                return

            log.info(
                "MCP raw JSON response (status=%s, url=%s, len=%d): %r",
                getattr(response, "status_code", "?"),
                str(getattr(response, "url", "<unknown>")),
                len(content) if content else 0,
                content[:2000] if content else b"",
            )

            try:
                from pydantic import TypeAdapter
                message = TypeAdapter(JSONRPCMessage).validate_json(content)
                if is_initialization:
                    self._maybe_extract_protocol_version_from_message(message)
                session_message = SessionMessage(message)
                await read_stream_writer.send(session_message)
            except Exception as exc:
                log.error(
                    "MCP server returned invalid JSON-RPC response. raw=%r | error: %s",
                    content,
                    exc,
                )
                await read_stream_writer.send(exc)

        _sh.StreamableHTTPTransport._handle_json_response = _patched
        log.info("Installed MCP raw-response logger")
    except Exception as exc:
        log.warning(f"Could not install MCP raw-response logger: {exc}")


_install_mcp_response_logger()


class ServerStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Parsed config for a single MCP server (SSE transport)."""
    name: str
    url: str


@dataclass
class MCPConnection:
    """Holds the live objects for a connected MCP server."""
    config: MCPServerConfig
    status: ServerStatus = ServerStatus.DISCONNECTED
    session: Optional[ClientSession] = None
    _transport_cm: Optional[Any] = None
    _session_cm: Optional[Any] = None
    _read_stream: Optional[Any] = None
    _write_stream: Optional[Any] = None
    error_message: Optional[str] = None


class MCPManager:
    """Singleton manager that maintains live MCP server connections.

    Server configs are provided at runtime (from agent settings stored in DB),
    not from a static config file.
    """

    _instance: Optional["MCPManager"] = None

    def __new__(cls) -> "MCPManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._connections: Dict[str, MCPConnection] = {}

    # ------------------------------------------------------------------
    # Helpers to build MCPServerConfig from a user-supplied dict
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_server_dict(entry: Dict[str, Any]) -> MCPServerConfig:
        """Convert a caller-supplied dict to an MCPServerConfig."""
        name = entry.get("name") or "unnamed"
        url = entry.get("url")
        if not url:
            raise ValueError(f"MCP server '{name}' requires 'url'.")
        return MCPServerConfig(name=name, url=url)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_connected(self, server_name: str) -> bool:
        conn = self._connections.get(server_name)
        return conn is not None and conn.status == ServerStatus.CONNECTED

    # ------------------------------------------------------------------
    # Test connection (ephemeral — does not keep the connection alive)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Schema structure validation (used during connection testing)
    # ------------------------------------------------------------------

    _SCHEMA_TOOL_KEYWORDS = frozenset({
        "get_schema", "get-schema", "schema", "tables", "list_tables",
        "get_tables", "get-tables", "introspect", "describe",
    })

    @staticmethod
    def _validate_schema_structure(raw_result: Any) -> Optional[str]:
        """Validate that an MCP get_schema tool result is in nested format.

        Expected shape::

            [
              {
                "table_name": "...",
                "columns": [
                  {"column_name": "...", "data_type": "...", ...},
                  ...
                ]
              },
              ...
            ]

        Returns ``None`` if valid, or a descriptive error string if not.
        """
        import json as _json

        # Parse: handle both raw MCP content list and already-parsed objects.
        parsed = raw_result
        if isinstance(raw_result, dict) and "content" in raw_result:
            parts = [
                item.get("text", "")
                for item in (raw_result.get("content") or [])
                if isinstance(item, dict)
            ]
            parsed = "\n".join(p for p in parts if p)

        if isinstance(parsed, str):
            try:
                parsed = _json.loads(parsed.strip())
            except (_json.JSONDecodeError, ValueError):
                return "Schema response could not be parsed as JSON."

        if not isinstance(parsed, list) or len(parsed) == 0:
            return "Schema response is not a non-empty list."

        first = parsed[0]
        if not isinstance(first, dict):
            return "Schema response items must be objects (dicts)."

        if "table_name" not in first:
            return (
                "Schema format not proper: Missing 'table_name' key in first item — "
                "response does not match expected structure."
            )

        if "columns" not in first:
            has_column_name = "column_name" in first or "columnName" in first
            if has_column_name:
                return (
                    "Schema format not proper: Missing 'columns' key — response appears "
                    "to be in flat (one-row-per-column) format instead of nested "
                    "(one-row-per-table with columns array)."
                )
            return (
                "Schema format not proper: Missing 'columns' key in table object — "
                "expected each table entry to have a 'columns' array."
            )

        columns = first["columns"]
        if isinstance(columns, str):
            try:
                columns = json.loads(columns)
            except json.JSONDecodeError:
                return (
                    "Schema format not proper: 'columns' is a string but could not "
                    "be parsed as JSON."
                )
        if not isinstance(columns, list) or len(columns) == 0:
            return (
                "Schema format not proper: 'columns' must be a non-empty list "
                f"but got {type(columns).__name__}."
            )

        first_col = columns[0]
        if not isinstance(first_col, dict):
            return "Schema format not proper: Column entries must be objects (dicts)."

        missing = [k for k in ("column_name", "data_type") if k not in first_col]
        if missing:
            return (
                f"Schema format not proper: Column object is missing required key(s): "
                f"{missing}. Got keys: {list(first_col.keys())}."
            )

        return None  # valid

    async def test_connection(
        self,
        config_dict: Dict[str, Any],
        *,
        requires_query_building: bool = False,
    ) -> Dict[str, Any]:
        """Spawn a temporary MCP server, do the handshake, list tools, optionally
        validate get_schema output, then tear everything down.

        Schema structure validation is only performed when ``requires_query_building``
        is ``True`` (i.e. the connector is a generic DB that needs SQL generation).

        Returns ``{"success": True, "tools": [...]}`` on success or
        ``{"success": False, "message": "..."}`` on failure."""
        try:
            cfg = self._parse_server_dict(config_dict)
        except Exception as exc:
            return {"success": False, "message": str(exc)}

        try:
            async with streamable_http_client(cfg.url) as transport_streams:
                read_stream, write_stream = transport_streams[0], transport_streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = [
                        {"name": t.name, "description": t.description}
                        for t in result.tools
                    ]

                    # Schema structure validation — only for DB connectors that
                    # require query building (i.e. generic DB + SQL generation).
                    # Purpose-built tool servers don't expose get_schema, so skip.
                    if requires_query_building:
                        schema_tool_name = next(
                            (
                                t.name for t in result.tools
                                if any(kw in t.name.lower() for kw in self._SCHEMA_TOOL_KEYWORDS)
                            ),
                            None,
                        )
                        if schema_tool_name:
                            try:
                                log.info(
                                    "MCP test '%s': validating schema structure via tool '%s'",
                                    cfg.name, schema_tool_name,
                                )
                                schema_result = await session.call_tool(schema_tool_name, arguments={})
                                contents = []
                                for item in schema_result.content:
                                    contents.append({
                                        "type": item.type,
                                        "text": getattr(item, "text", None),
                                    })
                                validation_error = self._validate_schema_structure(
                                    {"content": contents}
                                )
                                if validation_error:
                                    log.warning(
                                        "MCP test '%s': schema validation failed — %s",
                                        cfg.name, validation_error,
                                    )
                                    return {"success": False, "message": validation_error}
                                log.info(
                                    "MCP test '%s': schema structure valid.", cfg.name
                                )
                            except Exception as schema_exc:
                                log.warning(
                                    "MCP test '%s': schema validation call failed: %s",
                                    cfg.name, schema_exc,
                                )
                                return {
                                    "success": False,
                                    "message": (
                                        f"Schema validation failed: could not call "
                                        f"'{schema_tool_name}': {schema_exc}"
                                    ),
                                }

            log.info(
                "MCP server '%s' test successful — %d tool(s): %s",
                cfg.name, len(tools), [t["name"] for t in tools],
            )
            return {"success": True, "tools": tools}

        except BaseException as exc:
            # anyio TaskGroup wraps errors in an ExceptionGroup; unwrap to surface
            # the real cause (e.g. httpx.UnsupportedProtocol, FileNotFoundError).
            inner = exc
            while hasattr(inner, "exceptions") and getattr(inner, "exceptions", None):
                inner = inner.exceptions[0]
            log.error("MCP test_connection failed for '%s': %s", cfg.name, inner)
            return {"success": False, "message": str(inner)}

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    async def connect(self, server_name: str, config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Connect to an MCP server.

        If the server is already connected, returns existing tools.
        Otherwise ``config_dict`` (the dict stored in the agent's
        ``mcp_servers`` JSON column) is required to know *how* to
        spawn the process.
        """
        # Already connected?
        if self.is_connected(server_name):
            tools = await self.list_tools(server_name)
            return {"status": "already_connected", "tools": tools}

        if config_dict is None:
            raise ValueError(
                f"MCP server '{server_name}' is not connected and no config was provided."
            )

        cfg = self._parse_server_dict(config_dict)
        conn = MCPConnection(config=cfg, status=ServerStatus.CONNECTING)
        self._connections[server_name] = conn

        retry_count = 0
        max_retries = 1

        while retry_count <= max_retries:
            async def _handshake() -> None:
                """Open the transport and complete initialize().

                Runs as its own task so that a cancel scope inside the MCP SDK
                cancels *this* task rather than the caller's. Without that
                isolation the leaked CancelledError is indistinguishable from a
                genuine cancellation of the request.
                """
                async with asyncio.timeout(_CONNECT_TIMEOUT_S):
                    transport_cm = streamable_http_client(cfg.url)
                    transport_streams = await transport_cm.__aenter__()
                    read_stream, write_stream = transport_streams[0], transport_streams[1]
                    conn._transport_cm = transport_cm
                    conn._read_stream = read_stream
                    conn._write_stream = write_stream

                    session_cm = ClientSession(read_stream, write_stream)
                    session: ClientSession = await session_cm.__aenter__()
                    conn._session_cm = session_cm
                    conn.session = session

                    await session.initialize()

            try:
                await asyncio.ensure_future(_handshake())

                conn.status = ServerStatus.CONNECTED
                conn.error_message = None
                log.info("Connected to MCP server '%s'.", server_name)

                tools = await self.list_tools(server_name)
                return {"status": "connected", "tools": tools}

            except asyncio.CancelledError:

                if _task_is_being_cancelled():
                    raise
                log.error(
                    "MCP server '%s' cancelled mid-handshake (attempt %d) — "
                    "treating as a connection failure",
                    server_name, retry_count + 1,
                )
                await self._cleanup_connection(conn)
                retry_count += 1

            except Exception as exc:
                log.error(
                    "Failed to connect to MCP server '%s' (attempt %d): %s",
                    server_name, retry_count + 1, exc,
                )
                await self._cleanup_connection(conn)
                retry_count += 1

        conn.status = ServerStatus.ERROR
        conn.error_message = f"Failed after {max_retries + 1} attempt(s)"
        return {"status": "error", "error": conn.error_message}

    async def disconnect(self, server_name: str) -> Dict[str, Any]:
        """Kill the MCP server process and remove from registry."""
        conn = self._connections.pop(server_name, None)
        if conn is None:
            return {"status": "not_connected"}
        await self._cleanup_connection(conn)
        log.info("Disconnected from MCP server '%s'.", server_name)
        return {"status": "disconnected"}

    async def disconnect_all(self) -> None:
        """Gracefully disconnect every active server."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    # ------------------------------------------------------------------
    # Tool operations
    # ------------------------------------------------------------------

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List all tools advertised by a connected MCP server."""
        conn = self._connections.get(server_name)
        if not conn or conn.status != ServerStatus.CONNECTED or not conn.session:
            raise RuntimeError(f"MCP server '{server_name}' is not connected.")

        result = await conn.session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
            })
        return tools

    async def call_tool(
        self, server_name: str, tool_name: str, args: Dict[str, Any]
    ) -> Any:
        """Invoke a tool on a connected MCP server and return the result.

        Retries once with a fresh connection if the underlying transport
        raises a validation/protocol error (e.g. the server returns an
        invalid/empty JSON-RPC response on a stale session).
        """
        conn = self._connections.get(server_name)
        if not conn or conn.status != ServerStatus.CONNECTED or not conn.session:
            raise RuntimeError(f"MCP server '{server_name}' is not connected.")

        log.info(
            "Calling MCP tool '%s' on server '%s' with args=%r",
            tool_name, server_name, args,
        )

        async def _do_call():
            return await conn.session.call_tool(tool_name, arguments=args)

        try:
            result = await _do_call()
        except Exception as exc:
            log.warning(
                "MCP call_tool failed on '%s' (server '%s'): %s. Reconnecting and retrying once...",
                tool_name, server_name, exc,
            )
            # Save config and reconnect with a fresh session
            cfg_dict = {"name": conn.config.name, "url": conn.config.url}
            await self._cleanup_connection(conn)
            self._connections.pop(server_name, None)
            reconnect_result = await self.connect(server_name, cfg_dict)
            if reconnect_result.get("status") not in ("connected", "already_connected"):
                raise RuntimeError(
                    f"Failed to reconnect to MCP server '{server_name}' after error: {exc}"
                ) from exc
            conn = self._connections.get(server_name)
            if not conn or not conn.session:
                raise RuntimeError(
                    f"MCP server '{server_name}' is not connected after reconnect"
                ) from exc
            result = await conn.session.call_tool(tool_name, arguments=args)

        contents = []
        for item in result.content:
            contents.append({
                "type": item.type,
                "text": getattr(item, "text", None),
                "data": getattr(item, "data", None),
                "mimeType": getattr(item, "mimeType", None),
            })
        return {
            "isError": getattr(result, "isError", getattr(result, "is_error", False)),
            "content": contents,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cleanup_connection(self, conn: MCPConnection) -> None:
        """Tear down streams and session for a connection, ignoring errors."""
        if conn._session_cm is not None:
            try:
                await conn._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            conn._session_cm = None
            conn.session = None

        if conn._transport_cm is not None:
            try:
                await conn._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            conn._transport_cm = None

        conn.status = ServerStatus.DISCONNECTED


# Convenience accessor
def get_mcp_manager() -> MCPManager:
    return MCPManager()
