import base64
import json as _json
import logging
import time

import httpx
from fastapi import HTTPException, status

from ai_agent.core.config import settings
from ai_agent.core.email_service import email_service

log = logging.getLogger(__name__)


def _get_jwt_payload(token: str) -> dict:
    """Decode a JWT payload without verifying the signature (read-only claim inspection)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        padding = "=" * (4 - len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(parts[1] + padding)
        return _json.loads(decoded)
    except Exception:
        return {}


async def _get_service_token() -> str:
    """Obtain a service-level access token via client_credentials grant."""
    token_url = (
        f"{settings.KEYCLOAK_SERVER_URL}/realms/{settings.KEYCLOAK_REALM}"
        "/protocol/openid-connect/token"
    )
    data = {
        "grant_type": "client_credentials",
        "client_id": settings.KEYCLOAK_BACKEND_CLIENT_ID,
        "client_secret": settings.KEYCLOAK_BACKEND_CLIENT_SECRET,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(token_url, data=data)
            if not resp.is_success:
                log.error("Keycloak service token failed [%s]: %s", resp.status_code, resp.text)
                resp.raise_for_status()
            return resp.json()["access_token"]
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when fetching service token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error fetching service token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def create_identity_user(
    email: str,
    password: str,
    first_name: str,
    last_name: str | None = None,
    email_verified: bool = False,
) -> str:
    """Register a user with the identity provider.

    By default creates the user with emailVerified=False and sends a VERIFY_EMAIL action
    email. Pass `email_verified=True` to skip that step entirely — used when accepting an
    organization invite, where clicking the unique emailed invite link already proves the
    invitee controls the inbox, making a second verification email redundant.

    Raises:
        HTTPException 409 - email already registered.
        HTTPException 503 - identity provider is unreachable or returns an unexpected error.
    """
    token = await _get_service_token()
    base = f"{settings.KEYCLOAK_SERVER_URL}/admin/realms/{settings.KEYCLOAK_REALM}"
    payload: dict = {
        "username": email,
        "email": email,
        "emailVerified": email_verified,
        "enabled": True,
        "requiredActions": [] if email_verified else ["VERIFY_EMAIL"],
        "credentials": [
            {"type": "password", "value": password, "temporary": False}
        ],
    }
    payload["firstName"] = first_name
    if last_name:
        payload["lastName"] = last_name

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{base}/users",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 409:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered.",
                )
            if not resp.is_success:
                log.error("Keycloak user creation failed [%s]: %s", resp.status_code, resp.text)
                resp.raise_for_status()
            location = resp.headers.get("Location", "")
            user_id = location.rstrip("/").split("/")[-1]
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Could not retrieve user ID from identity provider.",
                )

            if not email_verified:
                verify_resp = await client.put(
                    f"{base}/users/{user_id}/send-verify-email",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if not verify_resp.is_success:
                    log.error(
                        "Failed to send verification email for %s [%s]: %s",
                        email, verify_resp.status_code, verify_resp.text,
                    )

            return user_id
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when creating user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error creating user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def update_identity_user(
    email: str,
    first_name: str | None = None,
    last_name: str | None = None,
) -> None:
    """Update a user's profile attributes (firstName/lastName) in the identity provider.

    Looks up the user by email via the Admin API and applies the provided fields.
    Only non-None fields are updated.

    Raises:
        HTTPException 404 – user not found in the identity provider.
        HTTPException 503 – identity provider is unreachable.
    """
    admin_token = await _get_service_token()
    base = f"{settings.KEYCLOAK_SERVER_URL}/admin/realms/{settings.KEYCLOAK_REALM}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            lookup = await client.get(
                f"{base}/users",
                params={"email": email, "exact": "true"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            lookup.raise_for_status()
            users = lookup.json()
            if not users:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found in identity provider.",
                )
            user_id = users[0]["id"]

            # Only the name fields are sent. Email/username are intentionally
            # excluded from the payload so they can never be modified here.
            update_fields: dict = {}
            if first_name is not None:
                update_fields["firstName"] = first_name
            if last_name is not None:
                update_fields["lastName"] = last_name

            if not update_fields:
                return

            update_resp = await client.put(
                f"{base}/users/{user_id}",
                json=update_fields,
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )
            update_resp.raise_for_status()
            log.info("Updated identity profile for %s", email)
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable during profile update: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error during profile update: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def verify_email_by_token(token: str) -> bool:
    """Verify a user's email using the Keycloak action token JWT from the verification email.

    Parses the sub claim (user ID) from the JWT, then sets emailVerified=True via Admin API.

    Returns True on success, False if the token is invalid or expired.

    Raises:
        HTTPException 503 – identity provider is unreachable.
    """
    payload = _get_jwt_payload(token)
    if not payload:
        log.warning("Email verification: could not parse JWT")
        return False

    exp = payload.get("exp", 0)
    if exp and exp < time.time():
        log.warning("Email verification token expired (exp=%s)", exp)
        return False

    user_id = payload.get("sub")
    if not user_id:
        log.warning("Email verification token missing sub claim")
        return False

    admin_token = await _get_service_token()
    base = f"{settings.KEYCLOAK_SERVER_URL}/admin/realms/{settings.KEYCLOAK_REALM}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            user_resp = await client.get(
                f"{base}/users/{user_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            if user_resp.status_code == 404:
                log.warning("Email verification: user not found %s", user_id)
                return False
            user_resp.raise_for_status()
            user_data = user_resp.json()

            user_data["emailVerified"] = True
            user_data["requiredActions"] = []

            update_resp = await client.put(
                f"{base}/users/{user_id}",
                json=user_data,
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )
            update_resp.raise_for_status()
            log.info("Email verified for user %s", user_id)
            return True

    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable during email verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error during email verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def authenticate_user(
    email: str,
    password: str,
) -> dict:
    """Authenticate a user with the identity provider and return tokens.

    Args:
        email: User's email address
        password: User's password

    Returns:
        Dictionary containing access_token, refresh_token, expires_in, etc.

    Raises:
        HTTPException 401 - invalid credentials.
        HTTPException 503 - identity provider is unreachable.
    """
    token_url = (
        f"{settings.KEYCLOAK_SERVER_URL}/realms/{settings.KEYCLOAK_REALM}"
        "/protocol/openid-connect/token"
    )
    data = {
        "grant_type": "password",
        "client_id": settings.KEYCLOAK_BACKEND_CLIENT_ID,
        "client_secret": settings.KEYCLOAK_BACKEND_CLIENT_SECRET,
        "username": email,
        "password": password,
        "scope": "openid email profile",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(token_url, data=data)
            if resp.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password.",
                )
            if not resp.is_success:
                body = resp.json() if resp.content else {}
                error = body.get("error", "")
                error_description = body.get("error_description", "")
                log.error(
                    "Keycloak auth failed [%s] for %s: %s - %s",
                    resp.status_code, email, error, error_description,
                )
                if error == "invalid_grant" and "not fully set up" in error_description:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Email not verified. Please check your inbox and verify your email before logging in.",
                    )
                resp.raise_for_status()

            tokens = resp.json()
            # Enforce email verification via the JWT claim rather than Keycloak's
            # required-action mechanism, which can be unreliable for admin-initiated
            # verification emails.
            payload = _get_jwt_payload(tokens.get("access_token", ""))
            if not payload.get("email_verified", False):
                log.warning("Login blocked for %s — email not verified (JWT claim)", email)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Email not verified. Please check your inbox and verify your email before logging in.",
                )
            return tokens
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when authenticating user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error authenticating user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def refresh_access_token(refresh_token: str) -> dict:
    """Exchange a refresh token for a new token pair via the identity provider.

    Args:
        refresh_token: Refresh token previously issued alongside an access token.

    Returns:
        Dictionary containing access_token, refresh_token, expires_in, etc.

    Raises:
        HTTPException 401 – refresh token is invalid or expired.
        HTTPException 503 – identity provider is unreachable.
    """
    token_url = (
        f"{settings.KEYCLOAK_SERVER_URL}/realms/{settings.KEYCLOAK_REALM}"
        "/protocol/openid-connect/token"
    )
    data = {
        "grant_type": "refresh_token",
        "client_id": settings.KEYCLOAK_BACKEND_CLIENT_ID,
        "client_secret": settings.KEYCLOAK_BACKEND_CLIENT_SECRET,
        "refresh_token": refresh_token,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(token_url, data=data)
            if resp.status_code == 401 or resp.status_code == 400:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired. Please log in again.",
                )
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when refreshing token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error refreshing token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def get_user_profile(access_token: str) -> dict:
    """Fetch user profile from identity provider using access token.

    Args:
        access_token: Valid access token from login

    Returns:
        Dictionary containing user profile information (sub, email, given_name, etc.)

    Raises:
        HTTPException 401 – invalid or expired token.
        HTTPException 503 – identity provider is unreachable.
    """
    userinfo_url = (
        f"{settings.KEYCLOAK_SERVER_URL}/realms/{settings.KEYCLOAK_REALM}"
        "/protocol/openid-connect/userinfo"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired access token.",
                )
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when fetching user profile: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error fetching user profile: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def reset_password_by_token(token: str, new_password: str) -> bool:
    """Reset a user's password using the Keycloak action token from the reset email.

    Decodes the JWT action token to get the user ID, then sets the new password
    via the Admin API and clears the UPDATE_PASSWORD required action.

    Returns True on success, False if the token is invalid or expired.
    Raises HTTPException 503 if the identity provider is unreachable.
    """
    payload = _get_jwt_payload(token)
    if not payload:
        log.warning("Password reset: could not parse JWT token")
        return False

    exp = payload.get("exp", 0)
    if exp and exp < time.time():
        log.warning("Password reset token expired (exp=%s)", exp)
        return False

    user_id = payload.get("sub")
    if not user_id:
        log.warning("Password reset token missing sub claim")
        return False

    admin_token = await _get_service_token()
    base = f"{settings.KEYCLOAK_SERVER_URL}/admin/realms/{settings.KEYCLOAK_REALM}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Set the new password directly via Admin API
            reset_resp = await client.put(
                f"{base}/users/{user_id}/reset-password",
                json={"type": "password", "value": new_password, "temporary": False},
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )
            if reset_resp.status_code == 404:
                log.warning("Password reset: user not found %s", user_id)
                return False
            reset_resp.raise_for_status()

            # Clear the UPDATE_PASSWORD required action so login isn't blocked
            user_resp = await client.get(
                f"{base}/users/{user_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            user_resp.raise_for_status()
            user_data = user_resp.json()
            user_data["requiredActions"] = [
                a for a in user_data.get("requiredActions", [])
                if a != "UPDATE_PASSWORD"
            ]
            await client.put(
                f"{base}/users/{user_id}",
                json=user_data,
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

            log.info("Password reset successfully for user %s", user_id)

            # Send password-changed confirmation email
            user_email = user_data.get("email", "")
            first_name = user_data.get("firstName") or user_data.get("username") or "there"
            if user_email:
                try:
                    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:40px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06);">
        <tr><td style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);padding:32px 40px;text-align:center;">
          <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">AgentIQ</h1>
        </td></tr>
        <tr><td style="padding:40px;">
          <h2 style="margin:0 0 16px;color:#1e293b;font-size:20px;">Your password has been updated</h2>
          <p style="margin:0 0 12px;color:#475569;font-size:15px;">Dear {first_name},</p>
          <p style="margin:0 0 24px;color:#475569;font-size:15px;">Your <strong>AgentIQ</strong> account password was successfully changed.</p>
          <p style="margin:0 0 12px;color:#475569;font-size:15px;">If you made this change, you can safely ignore this email.</p>
          <p style="margin:0;color:#475569;font-size:15px;">If you did <strong>not</strong> make this change, please reset your password immediately and contact our support team.</p>
        </td></tr>
        <tr><td style="padding:24px 40px;border-top:1px solid #e2e8f0;text-align:center;">
          <p style="margin:0;color:#94a3b8;font-size:13px;">AgentIQ &mdash; All rights reserved.</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
                    text_body = (
                        f"Your AgentIQ password has been updated\n\n"
                        f"Dear {first_name},\n\n"
                        f"Your AgentIQ account password was successfully changed.\n\n"
                        f"If you made this change, you can safely ignore this email.\n\n"
                        f"If you did not make this change, please reset your password immediately "
                        f"and contact our support team.\n\nAgentIQ"
                    )
                    email_service.send_html_email(
                        to_email=user_email,
                        subject="Your AgentIQ password has been updated",
                        html_body=html_body,
                        text_body=text_body,
                    )
                except Exception as email_exc:
                    # Non-fatal: log but don't fail the reset
                    log.error("Failed to send password-changed notification to %s: %s", user_email, email_exc)

            return True

    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable during password reset: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error during password reset: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def send_reset_email(email: str) -> None:
    """Send a password-reset email to the user via the identity provider.

    The identity provider will email the user a link that lets them set a new
    password (new-password + confirm-password are enforced by the provider UI).

    Raises:
        HTTPException 404 – no user with this email in the identity provider.
        HTTPException 503 – identity provider is unreachable.
    """
    token = await _get_service_token()
    base = (
        f"{settings.KEYCLOAK_SERVER_URL}/admin/realms/{settings.KEYCLOAK_REALM}"
    )

    # 1. Look up the Keycloak user by email
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            search_resp = await client.get(
                f"{base}/users",
                params={"email": email, "exact": "true"},
                headers={"Authorization": f"Bearer {token}"},
            )
            search_resp.raise_for_status()
            users = search_resp.json()
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when searching user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error searching user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )

    if not users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    keycloak_user_id = users[0]["id"]

    # 2. Trigger the UPDATE_PASSWORD action email
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.put(
                f"{base}/users/{keycloak_user_id}/execute-actions-email",
                json=["UPDATE_PASSWORD"],
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when sending reset email: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error sending reset email: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )


async def reset_user_password(
    email: str,
    old_password: str,
    new_password: str,
) -> None:
    """Reset a user's password in the identity provider.

    Steps:
        1. Validate the old password via a login (password grant) call.
        2. Look up the Keycloak user by email via the Admin API.
        3. Update the password via PUT /users/{id}/reset-password.

    Raises:
        HTTPException 401 – old password is incorrect.
        HTTPException 404 – user not found in the identity provider.
        HTTPException 503 – identity provider is unreachable.
    """
    # 1. Validate old password
    try:
        await authenticate_user(email=email, password=old_password)
    except HTTPException:
        raise

    # 2. Get admin token and find Keycloak user
    token = await _get_service_token()
    base = (
        f"{settings.KEYCLOAK_SERVER_URL}/admin/realms/{settings.KEYCLOAK_REALM}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            search_resp = await client.get(
                f"{base}/users",
                params={"email": email, "exact": "true"},
                headers={"Authorization": f"Bearer {token}"},
            )
            search_resp.raise_for_status()
            users = search_resp.json()
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when searching user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error searching user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )

    if not users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    keycloak_user_id = users[0]["id"]

    # 3. Update password via Admin API
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.put(
                f"{base}/users/{keycloak_user_id}/reset-password",
                json={
                    "type": "password",
                    "value": new_password,
                    "temporary": False,
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
    except httpx.ConnectError as exc:
        log.error("Identity provider unreachable when resetting password: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )
    except Exception as exc:
        log.error("Unexpected error resetting password: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider error. Please try again later.",
        )

async def exchange_authorization_code(code: str) -> dict:
    token_url = (
        f"{settings.KEYCLOAK_SERVER_URL}/realms/"
        f"{settings.KEYCLOAK_REALM}"
        "/protocol/openid-connect/token"
    )

    data = {
        "grant_type": "authorization_code",
        "client_id": settings.KEYCLOAK_BACKEND_CLIENT_ID,
        "client_secret": settings.KEYCLOAK_BACKEND_CLIENT_SECRET,
        "code": code,
        "redirect_uri": settings.SSO_CALLBACK_URL,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(token_url, data=data)

            if not response.is_success:
                log.error(
                    "Keycloak authorization-code exchange failed [%s]: %s",
                    response.status_code,
                    response.text,
                )
                response.raise_for_status()

            return response.json()

    except httpx.ConnectError as exc:
        log.error(
            "Identity provider unreachable during authorization-code exchange: %s",
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider is unreachable. Please try again later.",
        )