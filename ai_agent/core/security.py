"""Security utilities for handling sensitive data like API keys."""
import os
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Handles encryption and decryption of sensitive data.
    
    Uses PBKDF2 for key derivation and Fernet for symmetric encryption.
    The encryption key is derived from a secret key and a salt.
    """
    
    def __init__(self, secret_key: str, salt: Optional[bytes] = None):
        """Initialize the encryption manager.
        
        Args:
            secret_key: A secret key used for encryption/decryption.
                       In production, this should come from environment variables.
            salt: Optional salt for key derivation. If None, a new one is generated.
        """
        self.secret_key = secret_key.encode()
        self.salt = salt or os.urandom(16)
        self.fernet = self._get_fernet()
    
    def _get_fernet(self) -> Fernet:
        """Create a Fernet instance with a key derived from the secret key and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.secret_key))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt a string."""
        if not data:
            return ""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return f"{base64.b64encode(self.salt).decode()}:{encrypted.decode()}"
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Failed to encrypt data") from e
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt a string encrypted with this class."""
        if not encrypted_data:
            return ""
            
        try:
            if ":" not in encrypted_data:
                raise ValueError("Invalid encrypted data format")
                
            salt_part, encrypted_part = encrypted_data.split(":", 1)
            salt = base64.b64decode(salt_part)
            
            # Recreate the Fernet instance with the stored salt
            temp_manager = EncryptionManager(
                self.secret_key.decode(),
                salt=salt
            )
            
            return temp_manager.fernet.decrypt(encrypted_part.encode()).decode()
        except (InvalidToken, ValueError) as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data. The data may be corrupted or the key may be incorrect.") from e

# Global instance - in production, SECRET_KEY should come from environment variables
SECRET_KEY = os.getenv("ENCRYPTION_SECRET_KEY", "dev-secret-key-change-in-production")
encryption_manager = EncryptionManager(SECRET_KEY)

def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance."""
    return encryption_manager
