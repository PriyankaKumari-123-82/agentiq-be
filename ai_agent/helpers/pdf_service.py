"""PDF generation, S3 upload, and download-link helpers used by the
PDF-export pipeline (and the conversation-transcript endpoint).

No HTTP, no DB writes.
"""
import io
import logging

from ai_agent.core.aws_s3 import s3_client
from ai_agent.core.config import settings
from ai_agent.core.openai_client import create_chat_completion
from ai_agent.core.prompt import CONVERSATION_TO_PDF_PROMPT, DOWNLOAD_LINK_PROMPT

log = logging.getLogger(__file__)


async def generate_download_link_response(presigned_url: str, filename: str) -> str:
    """Format a user-facing JSON response describing a download link.

    Uses ``DOWNLOAD_LINK_PROMPT`` to render the message.
    """
    context = f"Download link: {presigned_url}\nFilename: {filename}"
    messages = [
        DOWNLOAD_LINK_PROMPT,
        {"role": "user", "content": context},
    ]
    response = create_chat_completion(
        messages=messages,
        model=settings.LLM_MODEL,
        temperature=0.3,
    )
    return response


async def upload_pdf_to_s3(pdf_path: str, filename: str) -> str | None:
    """Upload a local PDF to S3 under ``conversations/<filename>`` and
    return a 7-day presigned URL. Returns ``None`` on failure."""
    try:
        s3_key = f"conversations/{filename}"
        with open(pdf_path, "rb") as pdf_file:
            file_content = pdf_file.read()
        file_obj = io.BytesIO(file_content)

        await s3_client.upload_fileobj(
            file_obj=file_obj,
            object_name=s3_key,
            content_type="application/pdf",
        )

        presigned_url = s3_client.generate_presigned_url(
            object_name=s3_key,
            expiration=604800,  # 7 days
        )

        log.info(f"PDF uploaded to S3: {s3_key}")
        log.info(f"Presigned URL generated: {presigned_url}")
        return presigned_url
    except Exception as e:
        log.error(f"Failed to upload PDF to S3: {e}")
        return None


def generate_conversation_pdf(
    previous_conversation: list,
    user_message: str,
) -> str | None:
    """Render the conversation history + current user message as Markdown
    suitable for PDF conversion. Returns ``None`` on failure."""
    try:
        conversation_context = []
        for msg in previous_conversation:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
            conversation_context.append(f"{role.title()}: {content}")
        conversation_context.append(f"User: {user_message}")

        pdf_prompt_messages = [
            CONVERSATION_TO_PDF_PROMPT,
            {
                "role": "user",
                "content": (
                    "Conversation history:\n"
                    + "\n".join(conversation_context)
                    + f"\n\nCurrent request: {user_message}"
                ),
            },
        ]
        markdown_content = create_chat_completion(
            messages=pdf_prompt_messages,
            temperature=0.7,
        )
        return markdown_content
    except Exception as e:
        log.error(f"Error generating conversation PDF: {e}")
        return None
