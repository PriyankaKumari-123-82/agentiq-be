import os
import mimetypes
from minio import Minio
from typing import Optional, Dict, Any, List
from pathlib import Path
from fastapi import Depends
from .config import settings

class MinioClient:
    def __init__(self):
        """
        Initialize MinIO client
        """
        # Initialize MinIO client with settings
        self.client = Minio(
            endpoint=settings.MINIO_HOST,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.bucket_name = settings.MINIO_BUCKET
        # Set up local document directory
        self.document_dir = Path(settings.DOC_DIR)
        self.document_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the MinIO bucket exists, create it if it doesn't"""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def download_file(self, object_name: str) -> str:
        """
        Download a file from MinIO to the local document directory
        
        Args:
            object_name: Name of the object in the MinIO bucket
            
        Returns:
            str: Path to the downloaded file
        """
        # Create the full path for the downloaded file
        local_path = self.document_dir / object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file from MinIO
        self.client.fget_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=str(local_path)
        )
        
        return str(local_path)

    def upload_file(self, file_path: Path, 
                   object_name: str = None, 
                   metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Upload a file to MinIO
        
        Args:
            file_path: Path to the local file to upload
            object_name: Name to give the object (defaults to the file's name)
            metadata: Optional metadata to attach to the object
            
        Returns:
            Dict containing upload details including etag and version_id
        """
        # Ensure the bucket exists before uploading
        self._ensure_bucket_exists()
        
        # Use the provided object name or default to the file's relative path
        if object_name is None:
            # Get the relative path from the document directory
            object_name = str(file_path.relative_to(self.document_dir))
        
        # Ensure forward slashes for MinIO object names
        object_name = object_name.replace('\\', '/')
        
        # Determine content type from file extension
        content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        # Upload the file to MinIO
        result = self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=str(file_path),
            content_type=content_type,
            metadata=metadata or {}
        )
        
        # Return upload details
        return {
            'bucket_name': self.bucket_name,
            'object_name': object_name,
            'etag': result.etag,
            'version_id': result.version_id,
            'url': f"{settings.MINIO_HOST}/{self.bucket_name}/{object_name}"
        }

    def get_file_type(self, file_path: str) -> str:
        """
        Determine the MIME type of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: MIME type of the file
        """
        # Guess the MIME type based on file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'

    def delete_local_file(self, file_path: str) -> bool:
        """
        Delete a file from the local document directory
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            bool: True if deletion was successful
        """
        path = Path(file_path)
        # Check if file exists before attempting to delete
        if path.exists():
            path.unlink()
            return True
        return False
            
    def presigned_get_object(self, object_name: str, 
                           expires: int = 3600,
                           response_headers: Dict[str, str] = None) -> str:
        """
        Generate a pre-signed URL for getting an object from MinIO
        
        Args:
            object_name: Name of the object
            expires: Expiration time in seconds (default: 1 hour)
            response_headers: Optional headers to override in the response
            
        Returns:
            str: Pre-signed URL for the object
        """
        # Generate a pre-signed URL for HTTP GET operations
        return self.client.presigned_get_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            expires=expires,
            response_headers=response_headers or {}
        )

    def get_object_metadata(self, object_name: str) -> Dict[str, Any]:
        """
        Get metadata for an object in MinIO
        
        Args:
            object_name: Name of the object
            
        Returns:
            Dict containing object metadata including size, content type, etc.
        """
        # Get object statistics and metadata
        stat = self.client.stat_object(self.bucket_name, object_name)
        return {
            'size': stat.size,
            'content_type': stat.content_type,
            'etag': stat.etag,
            'last_modified': stat.last_modified,
            'metadata': stat.metadata,
            'url': f"{settings.MINIO_HOST}/{self.bucket_name}/{object_name}"
        }

    def delete_object(self, object_name: str) -> None:
        """
        Delete an object from MinIO
        
        Args:
            object_name: Name of the object to delete
        """
        self.client.remove_object(self.bucket_name, object_name)

def get_minio_client() -> MinioClient:
    """
    Dependency function to get a MinIO client instance.
    
    Returns:
        MinioClient: An instance of the MinIO client
    """
    return MinioClient()
