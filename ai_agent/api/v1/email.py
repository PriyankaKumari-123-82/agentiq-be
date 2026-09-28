import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session

from ai_agent.core import db
from ai_agent.core.aws_s3 import s3_client
from ai_agent.core.config import settings
from ai_agent.core.email_service import email_service
from ai_agent.models import Artifact

log = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["email"])


class SendEmailRequest(BaseModel):
    artifact_id: str = Field(description="Artifact ID to attach")
    recipient: str = Field(description="Recipient email address")
    subject: str = Field(default="", description="Email subject")
    body: str = Field(default="", description="Email body text")


class SendEmailResponse(BaseModel):
    success: bool
    message: str
    recipient: Optional[str] = None
    artifact_filename: Optional[str] = None


@router.post("/send", response_model=SendEmailResponse)
async def send_email(
    request: SendEmailRequest,
    user_id: int = Query(..., description="User ID"),
    db_session: Session = Depends(db.get_session),
):
    """Send an email with an artifact attachment after user approval."""

    # Fetch artifact from DB
    artifact = db_session.get(Artifact, request.artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.user_id != user_id:
        raise HTTPException(status_code=403, detail="Artifact does not belong to user")

    # Fetch file bytes from storage
    try:
        response = s3_client.s3_client.get_object(Bucket=s3_client.bucket_name, Key=artifact.storage_key)
        file_bytes = response["Body"].read()
    except Exception as e:
        log.exception("Failed to fetch artifact %s from storage", request.artifact_id)
        raise HTTPException(status_code=500, detail=f"Could not fetch artifact from storage: {e}")

    # Send email with attachment
    try:
        email_service.send_email_with_bytes_attachment(
            to_email=request.recipient,
            subject=request.subject or f"Attachment: {artifact.filename}",
            body=request.body or "Please find the attached file.",
            attachment_bytes=file_bytes,
            attachment_filename=artifact.filename,
            mime_type=artifact.mime_type,
        )
    except Exception as e:
        log.exception("Failed to send email to %s", request.recipient)
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

    log.info("Email sent to %s with artifact %s", request.recipient, artifact.filename)
    return SendEmailResponse(
        success=True,
        message=f"Email sent successfully to {request.recipient}",
        recipient=request.recipient,
        artifact_filename=artifact.filename,
    )
