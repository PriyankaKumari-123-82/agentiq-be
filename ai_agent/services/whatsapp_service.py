import logging
from typing import Optional
import httpx

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


async def send_whatsapp_text_message(
    phone_number: str,
    text_content: str,
    channel_id: Optional[str] = None,
    recipient_name: str = "WhatsApp User",
) -> bool:
    """
    Send an outbound text message to a WhatsApp recipient via Pluggy / Gallabox.
    """
    pluggy_url = settings.PLUGGY_UNIFIED_API_URL
    headers = {
        "accept": "application/json, text/plain, */*",
        "X-API-Key": settings.PLUGGY_API_KEY,
        "content-type": "application/json",
    }

    outbound_payload = {
        "unifiedOperation": "Send Text Message",
        "connectorId": settings.GALLABOX_CONNECTOR_ID,
        "connector": "GALLABOX",
        "payload": {
            "apiSecret": settings.GALLABOX_API_SECRET,
            "apiKey": settings.GALLABOX_API_KEY,
            "channelId": channel_id or settings.GALLABOX_CHANNEL_ID,
            "channelType": "whatsapp",
            "recipient": {
                "name": recipient_name,
                "phone": phone_number,
            },
            "whatsapp": {
                "type": "text",
                "text": {
                    "body": text_content,
                    "previewUrl": False,
                },
            },
        },
        "tenantId": settings.GALLABOX_TENANT_ID,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(pluggy_url, headers=headers, json=outbound_payload)
            if resp.status_code >= 400:
                log.error(
                    f"Failed to send WhatsApp message to '{phone_number}': {resp.status_code} - {resp.text}"
                )
                return False
            else:
                log.info(f"Successfully sent WhatsApp message to '{phone_number}'")
                return True
    except Exception as e:
        log.error(f"Error sending WhatsApp message to '{phone_number}': {e}")
        return False


async def send_whatsapp_template_message(
    phone_number: str,
    recipient_name: str = "WhatsApp User",
    agent_name: str = "",
) -> bool:
    """
    Send an approved WhatsApp template message via the Pluggy unified API.

    Routes through Pluggy (unifiedOperation = "Send WhatsApp Template") rather
    than calling Gallabox directly.  The template name is loaded from
    ``settings.GALLABOX_WELCOME_TEMPLATE_NAME``.

    Returns True if Pluggy accepted the request (HTTP 2xx), False otherwise.
    The caller must NOT treat a False return as a hard failure — this is a
    best-effort operation (e.g. welcome message after number registration).
    """
    headers = {
        "accept": "application/json, text/plain, */*",
        "X-API-Key": settings.PLUGGY_API_KEY,
        "content-type": "application/json",
    }

    payload = {
        "unifiedOperation": "Send WhatsApp Template",
        "connectorId": settings.GALLABOX_CONNECTOR_ID,
        "connector": "GALLABOX",
        "payload": {
            "channelId": settings.GALLABOX_CHANNEL_ID,
            "channelType": "whatsapp",
            "recipient": {
                "name": recipient_name,
                "phone": phone_number,
            },
            "whatsapp": {
                "type": "template",
                "template": {
                    "templateName": settings.GALLABOX_WELCOME_TEMPLATE_NAME,
                    "bodyValues": {
                        "agent_name": agent_name,
                    },
                },
            },
        },
        "tenantId": settings.GALLABOX_TENANT_ID,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(settings.PLUGGY_UNIFIED_API_URL, headers=headers, json=payload)
            if resp.status_code >= 400:
                log.error(
                    "Failed to send WhatsApp template to '%s': %d - %s",
                    phone_number,
                    resp.status_code,
                    resp.text,
                )
                return False
            log.info(
                "Successfully dispatched WhatsApp template '%s' to '%s' (status %d)",
                settings.GALLABOX_WELCOME_TEMPLATE_NAME,
                phone_number,
                resp.status_code,
            )
            return True
    except Exception as e:
        log.error(
            "Error sending WhatsApp template to '%s': %s",
            phone_number,
            e,
        )
        return False
