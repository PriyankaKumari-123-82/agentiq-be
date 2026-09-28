import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via AWS SES."""

    def __init__(self):
        self.from_email = settings.SES_FROM_EMAIL
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client(
                "ses",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
            )
        return self._client

    def _build_raw_message(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachment_bytes: Optional[bytes] = None,
        attachment_filename: Optional[str] = None,
        mime_type: str = "application/octet-stream",
    ) -> bytes:
        msg = MIMEMultipart()
        msg["From"] = self.from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if attachment_bytes and attachment_filename:
            main_type, sub_type = (mime_type.split("/", 1) + ["octet-stream"])[:2]
            part = MIMEBase(main_type, sub_type)
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{attachment_filename}"',
            )
            msg.attach(part)

        return msg.as_bytes()

    def send_email_with_bytes_attachment(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachment_bytes: Optional[bytes] = None,
        attachment_filename: Optional[str] = None,
        mime_type: str = "application/octet-stream",
    ) -> bool:
        """Send an email via AWS SES with an optional bytes attachment."""
        if not self.from_email:
            raise RuntimeError("SES_FROM_EMAIL is not configured")

        try:
            raw_bytes = self._build_raw_message(
                to_email=to_email,
                subject=subject,
                body=body,
                attachment_bytes=attachment_bytes,
                attachment_filename=attachment_filename,
                mime_type=mime_type,
            )
            self._get_client().send_raw_email(
                Source=self.from_email,
                Destinations=[to_email],
                RawMessage={"Data": raw_bytes},
            )
            log.info("SES email sent to %s (attachment=%s)", to_email, attachment_filename)
            return True

        except ClientError as e:
            msg = e.response["Error"]["Message"]
            log.error("SES ClientError sending to %s: %s", to_email, msg)
            raise RuntimeError(f"SES error: {msg}") from e
        except Exception as e:
            log.error("Failed to send SES email to %s: %s", to_email, e)
            raise

    def send_html_email(self, to_email: str, subject: str, html_body: str, text_body: str) -> bool:
        """Send an HTML email via AWS SES with a plain-text fallback."""
        if not self.from_email:
            raise RuntimeError("SES_FROM_EMAIL is not configured")

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_email
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            self._get_client().send_raw_email(
                Source=self.from_email,
                Destinations=[to_email],
                RawMessage={"Data": msg.as_bytes()},
            )
            log.info("SES HTML email sent to %s subject=%r", to_email, subject)
            return True

        except ClientError as e:
            msg_err = e.response["Error"]["Message"]
            log.error("SES ClientError sending HTML email to %s: %s", to_email, msg_err)
            raise RuntimeError(f"SES error: {msg_err}") from e
        except Exception as e:
            log.error("Failed to send SES HTML email to %s: %s", to_email, e)
            raise


# Global email service instance
email_service = EmailService()
