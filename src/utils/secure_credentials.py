"""Secure credential management for sensitive API keys and secrets.

This module provides secure storage and retrieval of credentials using
encryption and best practices for credential management.
"""

import base64
import os
import json
import warnings
from pathlib import Path
from typing import Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import structlog

logger = structlog.get_logger(__name__)


class SecureCredentialManager:
    """Manages secure storage and retrieval of credentials."""
    
    def __init__(self, credential_dir: Optional[Path] = None):
        """Initialize credential manager.
        
        Args:
            credential_dir: Directory to store encrypted credentials (default: ~/.arc-credentials)
        """
        self.credential_dir = credential_dir or Path.home() / ".arc-credentials"
        self.credential_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Secure the directory permissions (Unix-like systems)
        if os.name != 'nt':
            os.chmod(self.credential_dir, 0o700)
        
        self._key_file = self.credential_dir / ".key"
        self._credentials_file = self.credential_dir / "credentials.enc"
        self._cipher = self._get_or_create_cipher()
        
    def _get_or_create_cipher(self) -> Fernet:
        """Get or create encryption cipher.
        
        Returns:
            Fernet cipher instance
        """
        if self._key_file.exists():
            # Load existing key
            with open(self._key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new key
            key = self._generate_key()
            
            # Save key with restricted permissions
            with open(self._key_file, 'wb') as f:
                f.write(key)
            
            # Secure the key file
            if os.name != 'nt':
                os.chmod(self._key_file, 0o600)
                
        return Fernet(key)
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key.
        
        Returns:
            Encryption key bytes
        """
        # Use system-specific entropy source
        salt = os.urandom(16)
        
        # Store salt for key derivation
        salt_file = self.credential_dir / ".salt"
        with open(salt_file, 'wb') as f:
            f.write(salt)
        
        if os.name != 'nt':
            os.chmod(salt_file, 0o600)
            
        # Derive key from machine-specific data
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # Use combination of hostname and user as password
        password = f"{os.environ.get('USER', 'default')}-{os.uname().nodename if hasattr(os, 'uname') else 'windows'}".encode()
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        return key
    
    def store_credential(self, key: str, value: str, metadata: Optional[dict[str, Any]] = None) -> bool:
        """Store a credential securely.
        
        Args:
            key: Credential key (e.g., 'WANDB_API_KEY')
            value: Credential value (the secret)
            metadata: Optional metadata about the credential
            
        Returns:
            True if stored successfully
        """
        try:
            # Load existing credentials
            credentials = self._load_credentials()
            
            # Add new credential
            credentials[key] = {
                'value': self._cipher.encrypt(value.encode()).decode(),
                'metadata': metadata or {},
                'stored_at': os.environ.get('CURRENT_TIME', 'unknown')
            }
            
            # Save credentials
            self._save_credentials(credentials)
            
            logger.info("credential_stored", key=key)
            return True
            
        except Exception as e:
            logger.error("credential_store_failed", key=key, error=str(e))
            return False
    
    def retrieve_credential(self, key: str) -> Optional[str]:
        """Retrieve a credential securely.
        
        Args:
            key: Credential key to retrieve
            
        Returns:
            Decrypted credential value or None if not found
        """
        try:
            credentials = self._load_credentials()
            
            if key not in credentials:
                logger.warning("credential_not_found", key=key)
                return None
                
            encrypted_value = credentials[key]['value']
            decrypted_value = self._cipher.decrypt(encrypted_value.encode()).decode()
            
            logger.info("credential_retrieved", key=key)
            return decrypted_value
            
        except Exception as e:
            logger.error("credential_retrieve_failed", key=key, error=str(e))
            return None
    
    def _load_credentials(self) -> dict[str, Any]:
        """Load encrypted credentials from disk.
        
        Returns:
            Dictionary of credentials
        """
        if not self._credentials_file.exists():
            return {}
            
        try:
            with open(self._credentials_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_credentials(self, credentials: dict[str, Any]):
        """Save encrypted credentials to disk.
        
        Args:
            credentials: Credentials dictionary to save
        """
        with open(self._credentials_file, 'w') as f:
            json.dump(credentials, f, indent=2)
            
        # Secure the credentials file
        if os.name != 'nt':
            os.chmod(self._credentials_file, 0o600)
    
    def delete_credential(self, key: str) -> bool:
        """Delete a stored credential.
        
        Args:
            key: Credential key to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            credentials = self._load_credentials()
            
            if key in credentials:
                del credentials[key]
                self._save_credentials(credentials)
                logger.info("credential_deleted", key=key)
                return True
            else:
                logger.warning("credential_not_found_for_deletion", key=key)
                return False
                
        except Exception as e:
            logger.error("credential_delete_failed", key=key, error=str(e))
            return False
    
    def validate_credential(self, key: str, validator_func: Optional[callable] = None) -> bool:
        """Validate a stored credential.
        
        Args:
            key: Credential key to validate
            validator_func: Optional function to validate the credential value
            
        Returns:
            True if credential exists and is valid
        """
        value = self.retrieve_credential(key)
        
        if value is None:
            return False
            
        if validator_func:
            try:
                return validator_func(value)
            except Exception as e:
                logger.error("credential_validation_failed", key=key, error=str(e))
                return False
                
        return True
    
    def get_credential_with_fallback(self, key: str, env_var: Optional[str] = None) -> Optional[str]:
        """Get credential from secure storage with environment variable fallback.
        
        Args:
            key: Credential key to retrieve
            env_var: Environment variable name to check as fallback
            
        Returns:
            Credential value or None
        """
        # Try secure storage first
        value = self.retrieve_credential(key)
        if value:
            return value
            
        # Fall back to environment variable
        if env_var:
            value = os.environ.get(env_var)
            if value:
                logger.warning("credential_from_env", key=key, env_var=env_var)
                # Optionally store it securely for next time
                self.store_credential(key, value, {'source': 'environment'})
                return value
                
        return None


# Global instance
_credential_manager: Optional[SecureCredentialManager] = None


def get_credential_manager() -> SecureCredentialManager:
    """Get the global credential manager instance.
    
    Returns:
        SecureCredentialManager instance
    """
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = SecureCredentialManager()
    return _credential_manager


def setup_wandb_credential() -> bool:
    """Interactive setup for W&B API key.
    
    Returns:
        True if setup successful
    """
    manager = get_credential_manager()
    
    # Check if already stored
    if manager.validate_credential('WANDB_API_KEY'):
        logger.info("wandb_credential_already_configured")
        return True
        
    # Check environment variable
    env_key = os.environ.get('WANDB_API_KEY')
    if env_key:
        if manager.store_credential('WANDB_API_KEY', env_key, {'source': 'environment'}):
            logger.info("wandb_credential_stored_from_env")
            return True
            
    # If running in non-interactive environment, fail
    if not os.isatty(0):
        logger.error("wandb_credential_setup_requires_interaction")
        return False
        
    print("\n=== W&B API Key Setup ===")
    print("Your API key will be encrypted and stored securely.")
    print("Get your API key from: https://wandb.ai/authorize")
    print("Note: The key will not be displayed as you type.\n")
    
    import getpass
    api_key = getpass.getpass("Enter your W&B API key: ")
    
    if not api_key:
        logger.error("wandb_credential_setup_cancelled")
        return False
        
    # Validate key format (basic check)
    if len(api_key) < 20:
        logger.error("wandb_credential_invalid_format")
        return False
        
    # Store the credential
    if manager.store_credential('WANDB_API_KEY', api_key, {'source': 'interactive_setup'}):
        print("\n✓ W&B API key stored securely!")
        logger.info("wandb_credential_setup_complete")
        return True
    else:
        print("\n✗ Failed to store W&B API key.")
        return False