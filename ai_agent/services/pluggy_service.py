import time
import logging
import base64
import json
import asyncio
from typing import Dict, Any, Optional
from typing import Tuple

from urllib.parse import urlencode, urljoin

import httpx

from ai_agent.core.config import settings

logger = logging.getLogger(__name__)

# Module-level token cache
_cached_token: Optional[str] = None
_cached_refresh_token: Optional[str] = None
_token_expires_at: float = 0
_token_lock = asyncio.Lock()

def _get_jwt_exp(token: str) -> float:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return 0
        payload_b64 = parts[1]
        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
        return float(payload.get('exp', 0))
    except Exception:
        return 0

def _extract_tokens(data: dict) -> Tuple[Optional[str], Optional[str]]:
    token = data.get("data", {}).get("accessToken") or data.get("accessToken")
    refresh = data.get("data", {}).get("refreshToken") or data.get("refreshToken")
    return token, refresh


async def get_pluggy_token() -> str:
    """Authenticate with Pluggy and return a cached JWT token."""
    global _cached_token, _cached_refresh_token, _token_expires_at
    
    # Check if token is still valid (give 60 seconds buffer)
    if _cached_token and _token_expires_at > time.time() + 60:
        return _cached_token

    async with _token_lock:
        # Double-check inside lock
        if _cached_token and _token_expires_at > time.time() + 60:
            return _cached_token

        # Attempt to refresh token if we have a refresh token
        if _cached_refresh_token:
            logger.info("Refreshing Pluggy token...")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{settings.PLUGGY_BASE_URL}/auth/refresh-token",
                        params={"refreshToken": _cached_refresh_token}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    new_token, new_refresh = _extract_tokens(data)
                    
                    if new_token:
                        _cached_token = new_token
                        if new_refresh:
                            _cached_refresh_token = new_refresh
                            
                        exp = _get_jwt_exp(new_token)
                        _token_expires_at = exp if exp > 0 else time.time() + (15 * 60)
                        logger.info("Pluggy token refresh successful.")
                        return _cached_token
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 401, 403):
                    logger.warning(f"Pluggy refresh token invalid/expired (status {e.response.status_code}), falling back to login.")
                    _cached_token = None
                    _cached_refresh_token = None
                else:
                    logger.warning(f"Failed to refresh Pluggy token due to server error (status {e.response.status_code}), falling back to login.")
                    _cached_token = None
            except Exception as e:
                logger.warning(f"Failed to refresh Pluggy token (network/unknown error), falling back to login: {e}")
                _cached_token = None

        if not settings.PLUGGY_USERNAME or not settings.PLUGGY_PASSWORD:
            raise ValueError("Pluggy credentials are not configured.")

        logger.info("Logging in to Pluggy to acquire new token...")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.PLUGGY_BASE_URL}/auth/login",
                json={
                    "username": settings.PLUGGY_USERNAME,
                    "password": settings.PLUGGY_PASSWORD
                }
            )
            response.raise_for_status()
            data = response.json()
            
            token, refresh_token = _extract_tokens(data)
            
            if not token:
                raise ValueError("Failed to retrieve token from Pluggy login response")
                
            _cached_token = token
            _cached_refresh_token = refresh_token
            
            exp = _get_jwt_exp(token)
            _token_expires_at = exp if exp > 0 else time.time() + (15 * 60)
            
            logger.info("Pluggy login successful.")
            return _cached_token



async def create_tenant(org_name: str, org_id: int) -> Optional[str]:
    """Create a new tenant in Pluggy for the given organization."""
    try:
        token = await get_pluggy_token()
        
        # We auto-generate a pseudo-password since it's required but won't be used
        # We use a placeholder email based on org name
        import uuid
        pseudo_email = f"admin_{org_id}@prismberry.local"
        
        payload = {
            "firstName": "Admin",
            "lastName": "User",
            "companyName": org_name,
            "companyEmail": pseudo_email,
            "companyAddress": "N/A",
            "companyNumber": "N/A",
            "email": pseudo_email,
            "password": "Password!123",
            "country": "US"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.PLUGGY_BASE_URL}/tenant/create",
                json=payload,
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            data = response.json()
            # The UUID of the created tenant is in data.id
            tenant_data = data.get("data", {})
            return tenant_data.get("id") or data.get("id")
    except Exception as e:
        logger.warning(f"Failed to create Pluggy tenant for org {org_id} ({org_name}): {e}")
        return None


async def update_tenant(pluggy_tenant_id: str, org_name: str) -> None:
    """Update an existing tenant in Pluggy."""
    try:
        token = await get_pluggy_token()
        
        payload = {
            "name": org_name
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{settings.PLUGGY_BASE_URL}/tenant/{pluggy_tenant_id}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to update Pluggy tenant {pluggy_tenant_id}: {e}")


async def get_connectors(page: int = 1, size: int = 100) -> list:
    """Fetch global available connectors."""
    try:
        token = await get_pluggy_token()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.PLUGGY_BASE_URL}/connector/list/operation_count",
                params={"sortDirection": "ASC", "scope": "ALL"},
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            
            inner_data = data.get("data")
            if isinstance(inner_data, list):
                return inner_data
            elif isinstance(inner_data, dict):
                return inner_data.get("items") or []
            
            return data.get("items") or []
    except Exception as e:
        logger.error(f"Failed to fetch global connectors: {e}")
        return []


async def get_personalized_connectors(pluggy_tenant_id: str, page: int = 1, size: int = 100) -> list:
    """Fetch personalized (connected) connectors for a specific tenant."""
    try:
        token = await get_pluggy_token()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.PLUGGY_BASE_URL}/connector/list-personalized/{pluggy_tenant_id}",
                params={"pageNumber": page, "pageSize": size},
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            
            inner_data = data.get("data")
            if isinstance(inner_data, list):
                return inner_data
            elif isinstance(inner_data, dict):
                return inner_data.get("items") or []
                
            return data.get("items") or []
    except Exception as e:
        logger.error(f"Failed to fetch personalized connectors for tenant {pluggy_tenant_id}: {e}")
        return []


async def get_connector_mcp_url(pluggy_tenant_id: str, connector_key: str) -> Optional[str]:
    """Fetch the MCP server URL for a connected Pluggy connector."""
    try:
        token = await get_pluggy_token()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.PLUGGY_BASE_URL}/mcp/consumer-config",
                params={"connector": connector_key},
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-TENANT-ID": pluggy_tenant_id
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("serverUrl")
    except Exception as e:
        logger.error(f"Failed to fetch MCP URL for connector {connector_key}: {e}")
        return None


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def build_oauth_url(
    pluggy_tenant_id: str,
    connector_id: str,
    redirect_url: str,
    is_test: bool = False,
) -> str:
    """Build the Pluggy OAuth connect URL for a given connector.

    The frontend opens this URL in a popup/new tab. After the user completes
    Microsoft login, Pluggy redirects back to ``redirect_url``.
    """
    if "?" not in redirect_url:
        redirect_url = f"{redirect_url.rstrip('/')}?oauth=ms_teams"

    params = urlencode({
        "tenantId": pluggy_tenant_id,
        "redirectURL": redirect_url,
        "connectorId": connector_id,
        "isTest": str(is_test).lower(),
    })
    base = settings.PLUGGY_BASE_URL.rstrip("/")
    url = f"{base}/oauth/connect/MS-TEAMS?{params}"
    logger.info(f"Generated Pluggy OAuth URL: {url}")
    return url
