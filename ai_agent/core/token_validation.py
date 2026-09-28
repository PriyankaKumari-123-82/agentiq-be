import logging

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ai_agent.core.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def decode_token_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate Bearer token with Keycloak and return decoded JSON payload.

    Use this as a FastAPI dependency in any router:
        from ai_agent.core.token_validation import decode_token_payload

        @router.get("/some-endpoint")
        async def my_endpoint(payload: dict = Depends(decode_token_payload)):
            ...
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await validate_token(credentials.credentials)


async def validate_token(token: str) -> dict:
    """Validate a Bearer token by calling the Keycloak userinfo endpoint.

    Args:
        token: Bearer access token to validate.

    Returns:
        Decoded JSON from the identity provider (e.g., sub, email, name, roles).

    Raises:
        HTTPException 401 – token is invalid or expired.
        HTTPException 503 – identity provider is unreachable.
    """
    userinfo_url = (
        f"{settings.KEYCLOAK_SERVER_URL}/realms/{settings.KEYCLOAK_REALM}"
        "/protocol/openid-connect/userinfo"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(userinfo_url, headers=headers)
            if resp.status_code == 401:
                www_auth = resp.headers.get("WWW-Authenticate", "Bearer")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired access token.",
                    headers={"WWW-Authenticate": www_auth},
                )
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        logger.error("Identity provider unreachable during token validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        logger.error("Unexpected error during token validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )
