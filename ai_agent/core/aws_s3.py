import os
import io
import asyncio
import logging
from typing import BinaryIO, Optional
import boto3
from botocore.exceptions import ClientError
from .config import settings

log = logging.getLogger(__name__)

class S3Client:
    """AWS S3 client for file operations."""
    
    def __init__(self):
        """Initialize the S3 client with credentials from settings."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.AWS_S3_BUCKET_NAME
    
    async def upload_fileobj(
        self,
        file_obj: BinaryIO,
        object_name: str,
        content_type: Optional[str] = None
    ) -> None:
        """
        Upload a file-like object to S3.
        
        Args:
            file_obj: File-like object to upload
            object_name: S3 object name (path in the bucket)
            content_type: Optional MIME type of the file
            
        Raises:
            Exception: If upload fails
        """
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
            
        # Ensure the file pointer is at the start
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        # Perform the upload
        self.s3_client.upload_fileobj(
            file_obj,
            self.bucket_name,
            object_name,
            ExtraArgs=extra_args
        )
    
    def delete_file(self, object_name: str) -> None:
        """
        Delete a file from S3.
        
        Args:
            object_name: S3 object name (path in the bucket)
        """
        self.s3_client.delete_object(
            Bucket=self.bucket_name,
            Key=object_name
        )
    
    def generate_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for a file in S3.
        
        Args:
            object_name: S3 object name (path in the bucket). Can be either just the key or in format 'bucket_name/key'
            expiration: Time in seconds for the presigned URL to remain valid
            
        Returns:
            str: Presigned URL as a string
        """
        # Handle case where object_name is in format 'bucket_name/key'
        if '/' in object_name and object_name.startswith(f"{self.bucket_name}/"):
            object_name = object_name[len(self.bucket_name) + 1:]  # Remove the bucket name prefix
            
        response = self.s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': self.bucket_name,
                'Key': object_name
            },
            ExpiresIn=expiration
        )
        return response

    def get_presigned_url_from_uri(self, uri: Optional[str], expiration: int = 3600) -> Optional[str]:
        """Convert an internal s3:// URI to a presigned URL."""
        if not uri:
            return None
        if uri.startswith("s3://"):
            key = uri[5:]
            return self.generate_presigned_url(key, expiration)
        return uri

    async def read_object(self, object_name: str) -> bytes:
        """
        Read file content from S3 as bytes.
        """
        if '/' in object_name and object_name.startswith(f"{self.bucket_name}/"):
            object_name = object_name[len(self.bucket_name) + 1:]
            
        file_obj = io.BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, object_name, file_obj)
        file_obj.seek(0)
        return file_obj.read()

# Create a singleton instance
s3_client = S3Client()
