"""Security infrastructure for authentication and authorization.

This module provides authentication middleware, password hashing utilities,
and security-related helper functions for the ARC Prize 2025 evaluation system.
"""

from typing import Any

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from src.utils.jwt_auth import JWTManager, get_jwt_manager

logger = structlog.get_logger(__name__)

# Password hashing configuration
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # Cost factor 12 for secure hashing
)

# HTTP Bearer security scheme
security = HTTPBearer()


class AuthenticationError(HTTPException):
    """Custom exception for authentication failures."""

    def __init__(self, detail: str = "Authentication failed", headers: dict[str, Any] | None = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers=headers or {"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(HTTPException):
    """Custom exception for authorization failures."""

    def __init__(self, detail: str = "Access denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password to hash

    Returns:
        Hashed password string
    """
    try:
        hashed = pwd_context.hash(password)
        logger.debug("password_hashed", length=len(password))
        return hashed
    except Exception as e:
        logger.error("password_hashing_failed", error=str(e))
        raise RuntimeError("Failed to hash password") from e


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Previously hashed password

    Returns:
        True if password is valid, False otherwise
    """
    try:
        is_valid = pwd_context.verify(plain_password, hashed_password)
        logger.debug("password_verification", is_valid=is_valid)
        return is_valid
    except Exception as e:
        logger.error("password_verification_failed", error=str(e))
        # Return False instead of raising to prevent timing attacks
        return False


def validate_password_strength(password: str) -> dict[str, Any]:
    """Validate password meets security requirements.

    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number

    Args:
        password: Password to validate

    Returns:
        Dictionary with validation results and feedback
    """
    results = {
        "is_valid": True,
        "errors": [],
        "suggestions": []
    }

    if len(password) < 8:
        results["is_valid"] = False
        results["errors"].append("Password must be at least 8 characters long")
        results["suggestions"].append("Use a longer password with mixed characters")

    if not any(c.isupper() for c in password):
        results["is_valid"] = False
        results["errors"].append("Password must contain at least one uppercase letter")
        results["suggestions"].append("Add uppercase letters (A-Z)")

    if not any(c.islower() for c in password):
        results["is_valid"] = False
        results["errors"].append("Password must contain at least one lowercase letter")
        results["suggestions"].append("Add lowercase letters (a-z)")

    if not any(c.isdigit() for c in password):
        results["is_valid"] = False
        results["errors"].append("Password must contain at least one number")
        results["suggestions"].append("Add numbers (0-9)")

    # Optional: Check for common weak passwords
    weak_passwords = [
        "password", "123456", "password123", "admin", "qwerty",
        "letmein", "welcome", "monkey", "1234567890"
    ]

    if password.lower() in weak_passwords:
        results["is_valid"] = False
        results["errors"].append("Password is too common and easily guessable")
        results["suggestions"].append("Use a unique password that's not commonly used")

    logger.debug("password_validation", is_valid=results["is_valid"], error_count=len(results["errors"]))

    return results


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
) -> str:
    """Extract and validate current user from JWT token.

    Args:
        credentials: HTTP Bearer credentials from request
        jwt_manager: JWT manager instance

    Returns:
        User ID from validated token

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        user_id = jwt_manager.get_current_user(credentials)
        logger.debug("user_authenticated", user_id=user_id)
        return user_id
    except HTTPException as e:
        logger.warning("authentication_failed", detail=e.detail, status_code=e.status_code)
        raise AuthenticationError(detail=e.detail) from e
    except Exception as e:
        logger.error("authentication_error", error=str(e), exc_info=True)
        raise AuthenticationError(detail="Authentication failed") from None


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
) -> str | None:
    """Extract current user from JWT token if provided (optional authentication).

    Args:
        credentials: Optional HTTP Bearer credentials from request
        jwt_manager: JWT manager instance

    Returns:
        User ID from validated token, or None if no valid token provided
    """
    if not credentials:
        return None

    try:
        user_id = jwt_manager.get_current_user(credentials)
        logger.debug("optional_user_authenticated", user_id=user_id)
        return user_id
    except HTTPException:
        logger.debug("optional_authentication_failed")
        return None
    except Exception as e:
        logger.error("optional_authentication_error", error=str(e))
        return None


def create_access_token(user_id: str, additional_claims: dict[str, Any] | None = None) -> str:
    """Create a JWT access token for a user.

    Args:
        user_id: User ID to encode in token
        additional_claims: Optional additional claims to include

    Returns:
        Encoded JWT access token
    """
    jwt_manager = get_jwt_manager()
    return jwt_manager.create_access_token(user_id, additional_claims)


def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token for a user.

    Args:
        user_id: User ID to encode in token

    Returns:
        Encoded JWT refresh token
    """
    jwt_manager = get_jwt_manager()
    return jwt_manager.create_refresh_token(user_id)


def verify_refresh_token(token: str) -> str:
    """Verify and decode a refresh token.

    Args:
        token: Refresh token to verify

    Returns:
        User ID from verified token

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        jwt_manager = get_jwt_manager()
        payload = jwt_manager.verify_token(token, "refresh")
        user_id = payload.get("sub")

        if not user_id:
            raise AuthenticationError(detail="Invalid refresh token")

        logger.debug("refresh_token_verified", user_id=user_id)
        return user_id
    except HTTPException as e:
        logger.warning("refresh_token_verification_failed", detail=e.detail)
        raise AuthenticationError(detail=e.detail) from e
    except Exception as e:
        logger.error("refresh_token_verification_error", error=str(e))
        raise AuthenticationError(detail="Invalid refresh token") from None


class SecurityConfig:
    """Security configuration settings."""

    # Rate limiting settings for authentication endpoints
    LOGIN_RATE_LIMIT_PER_MINUTE = 5
    LOGIN_RATE_LIMIT_PER_HOUR = 20

    # Account lockout settings
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

    # Session settings
    MAX_CONCURRENT_SESSIONS = 3

    # Token settings
    ACCESS_TOKEN_EXPIRE_MINUTES = 15
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    # Password settings
    MIN_PASSWORD_LENGTH = 8
    PASSWORD_HISTORY_COUNT = 5
    PASSWORD_EXPIRY_DAYS = 90


def get_security_headers() -> dict[str, str]:
    """Get security headers to add to responses.

    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }
