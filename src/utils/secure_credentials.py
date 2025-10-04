"""Secure credential management for sensitive API keys and secrets.

This module provides secure storage and retrieval of credentials using
encryption and best practices for credential management.
"""

import base64
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = structlog.get_logger(__name__)


class SecureCredentialManager:
    """Manages secure storage and retrieval of credentials."""

    def __init__(self, credential_dir: Path | None = None):
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

    def store_credential(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> bool:
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
                'stored_at': datetime.now().isoformat()
            }

            # Save credentials
            self._save_credentials(credentials)

            logger.info("credential_stored", key=key)
            return True

        except Exception as e:
            logger.error("credential_store_failed", key=key, error=str(e))
            return False

    def retrieve_credential(self, key: str) -> str | None:
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
            with open(self._credentials_file) as f:
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

    def validate_credential(self, key: str, validator_func: Callable[[str], bool] | None = None) -> bool:
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

    def get_credential_with_fallback(self, key: str, env_var: str | None = None) -> str | None:
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
_credential_manager: SecureCredentialManager | None = None


def get_credential_manager() -> SecureCredentialManager:
    """Get the global credential manager instance.

    Returns:
        SecureCredentialManager instance
    """
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = SecureCredentialManager()
    return _credential_manager


class PlatformCredentialManager:
    """Manages platform-specific credentials for rotation automation."""

    def __init__(self, credential_manager: SecureCredentialManager = None):
        self.credential_manager = credential_manager or get_credential_manager()
        self.logger = logging.getLogger('platform_credentials')

        # Platform-specific credential keys
        self.platform_credentials = {
            'kaggle': {
                'username': 'KAGGLE_USERNAME',
                'key': 'KAGGLE_KEY',
                'api_token': 'KAGGLE_API_TOKEN'
            },
            'colab': {
                'service_account': 'GOOGLE_SERVICE_ACCOUNT_KEY',
                'project_id': 'GOOGLE_CLOUD_PROJECT',
                'credentials_json': 'GOOGLE_APPLICATION_CREDENTIALS'
            },
            'paperspace': {
                'api_key': 'PAPERSPACE_API_KEY',
                'api_token': 'PAPERSPACE_API_TOKEN',
                'project_id': 'PAPERSPACE_PROJECT_ID'
            },
            'gcs': {
                'service_account_key': 'GCS_SERVICE_ACCOUNT_KEY',
                'bucket_name': 'GCS_BUCKET_NAME',
                'project_id': 'GCS_PROJECT_ID'
            },
            'email': {
                'smtp_server': 'EMAIL_SMTP_SERVER',
                'smtp_port': 'EMAIL_SMTP_PORT',
                'username': 'EMAIL_USERNAME',
                'password': 'EMAIL_PASSWORD',
                'from_address': 'EMAIL_FROM_ADDRESS'
            },
            'wandb': {
                'api_key': 'WANDB_API_KEY',
                'entity': 'WANDB_ENTITY',
                'project': 'WANDB_PROJECT'
            }
        }

    def setup_platform_credentials(self) -> dict[str, bool]:
        """Set up all platform credentials from environment variables.

        Returns:
            Dictionary mapping platform names to setup success status
        """
        results = {}

        for platform, credentials in self.platform_credentials.items():
            platform_result = True

            for cred_name, env_var in credentials.items():
                env_value = os.environ.get(env_var)
                if env_value:
                    key = f"{platform}_{cred_name}"
                    success = self.credential_manager.store_credential(
                        key,
                        env_value,
                        {
                            'platform': platform,
                            'env_var': env_var,
                            'setup_time': datetime.now().isoformat()
                        }
                    )
                    if not success:
                        platform_result = False
                        self.logger.error(f"Failed to store {platform} {cred_name}")
                else:
                    self.logger.warning(f"Missing environment variable: {env_var}")
                    platform_result = False

            results[platform] = platform_result

        return results

    def get_platform_credential(self, platform: str, credential_name: str) -> str | None:
        """Get a specific platform credential.

        Args:
            platform: Platform name (kaggle, colab, paperspace, gcs, email, wandb)
            credential_name: Credential name within the platform

        Returns:
            Credential value or None if not found
        """
        if platform not in self.platform_credentials:
            self.logger.error(f"Unknown platform: {platform}")
            return None

        if credential_name not in self.platform_credentials[platform]:
            self.logger.error(f"Unknown credential {credential_name} for platform {platform}")
            return None

        key = f"{platform}_{credential_name}"
        env_var = self.platform_credentials[platform][credential_name]

        return self.credential_manager.get_credential_with_fallback(key, env_var)

    def get_all_platform_credentials(self, platform: str) -> dict[str, str | None]:
        """Get all credentials for a specific platform.

        Args:
            platform: Platform name

        Returns:
            Dictionary mapping credential names to values
        """
        if platform not in self.platform_credentials:
            return {}

        result = {}
        for cred_name in self.platform_credentials[platform]:
            result[cred_name] = self.get_platform_credential(platform, cred_name)

        return result

    def validate_platform_credentials(self, platform: str) -> bool:
        """Validate that all required credentials exist for a platform.

        Args:
            platform: Platform name to validate

        Returns:
            True if all credentials are present and valid
        """
        if platform not in self.platform_credentials:
            return False

        credentials = self.get_all_platform_credentials(platform)

        # Check if all required credentials are present
        for cred_name in self.platform_credentials[platform]:
            if not credentials.get(cred_name):
                self.logger.error(f"Missing credential {cred_name} for platform {platform}")
                return False

        return True

    def validate_all_platforms(self) -> dict[str, bool]:
        """Validate credentials for all platforms.

        Returns:
            Dictionary mapping platform names to validation results
        """
        results = {}
        for platform in self.platform_credentials:
            results[platform] = self.validate_platform_credentials(platform)
        return results

    def update_platform_credential(self, platform: str, credential_name: str, value: str) -> bool:
        """Update a specific platform credential.

        Args:
            platform: Platform name
            credential_name: Credential name
            value: New credential value

        Returns:
            True if updated successfully
        """
        if platform not in self.platform_credentials:
            self.logger.error(f"Unknown platform: {platform}")
            return False

        if credential_name not in self.platform_credentials[platform]:
            self.logger.error(f"Unknown credential {credential_name} for platform {platform}")
            return False

        key = f"{platform}_{credential_name}"
        return self.credential_manager.store_credential(
            key,
            value,
            {
                'platform': platform,
                'credential_name': credential_name,
                'updated_time': datetime.now().isoformat()
            }
        )

    def get_platform_config(self, platform: str) -> dict[str, Any]:
        """Get complete platform configuration including credentials.

        Args:
            platform: Platform name

        Returns:
            Platform configuration dictionary
        """
        credentials = self.get_all_platform_credentials(platform)

        # Platform-specific configurations
        configs = {
            'kaggle': {
                'api_base': 'https://www.kaggle.com/api/v1',
                'gpu_hours_weekly': 30,
                'session_timeout_hours': 12,
                'max_concurrent_kernels': 2
            },
            'colab': {
                'api_base': 'https://colab.research.google.com',
                'gpu_hours_daily': 12,
                'session_timeout_hours': 12,
                'max_concurrent_sessions': 1
            },
            'paperspace': {
                'api_base': 'https://api.paperspace.io',
                'gpu_hours_daily': 6,
                'session_timeout_hours': 6,
                'max_concurrent_machines': 1
            },
            'gcs': {
                'storage_limit_gb': 5,
                'checkpoint_retention_days': 7,
                'max_concurrent_uploads': 3
            },
            'email': {
                'rate_limit_per_hour': 10,
                'retry_attempts': 3,
                'timeout_seconds': 30
            },
            'wandb': {
                'storage_limit_gb': 100,
                'api_base': 'https://api.wandb.ai'
            }
        }

        config = configs.get(platform, {})
        config['credentials'] = credentials
        config['platform'] = platform

        return config


# Singleton instance
_platform_credential_manager = None


def get_platform_credential_manager() -> PlatformCredentialManager:
    """Get singleton platform credential manager instance."""
    global _platform_credential_manager
    if _platform_credential_manager is None:
        _platform_credential_manager = PlatformCredentialManager()
    return _platform_credential_manager


def setup_platform_credentials_from_env() -> dict[str, bool]:
    """Convenience function to set up all platform credentials from environment."""
    manager = get_platform_credential_manager()
    return manager.setup_platform_credentials()


def validate_all_platform_credentials() -> dict[str, bool]:
    """Convenience function to validate all platform credentials."""
    manager = get_platform_credential_manager()
    return manager.validate_all_platforms()


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
