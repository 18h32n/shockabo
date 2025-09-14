"""JWT Authentication utilities for securing API and WebSocket connections.

This module provides JWT token generation, validation, and middleware
for authenticating users in REST APIs and WebSocket connections.
"""

import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import structlog
from fastapi import HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = structlog.get_logger(__name__)


class JWTConfig:
    """Configuration for JWT authentication."""

    def __init__(self):
        """Initialize JWT configuration."""
        # Generate or load secret key
        self.secret_key = os.environ.get("JWT_SECRET_KEY")
        if not self.secret_key:
            # Generate a secure random key if not set
            self.secret_key = secrets.token_urlsafe(32)
            logger.warning("jwt_secret_generated", message="Using generated JWT secret. Set JWT_SECRET_KEY in production.")

        self.algorithm = os.environ.get("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
        self.refresh_token_expire_days = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.issuer = os.environ.get("JWT_ISSUER", "arc-evaluation-system")
        self.audience = os.environ.get("JWT_AUDIENCE", "arc-evaluation-api")


class JWTManager:
    """Manages JWT token operations."""

    def __init__(self, config: JWTConfig | None = None):
        """Initialize JWT manager.

        Args:
            config: JWT configuration (uses defaults if not provided)
        """
        self.config = config or JWTConfig()
        self.security = HTTPBearer()

    def create_access_token(
        self,
        subject: str,
        additional_claims: dict[str, Any] | None = None,
        expires_delta: timedelta | None = None
    ) -> str:
        """Create a JWT access token.

        Args:
            subject: The subject of the token (usually user ID)
            additional_claims: Additional claims to include in the token
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=self.config.access_token_expire_minutes)

        claims = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(UTC),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "access",
        }

        if additional_claims:
            claims.update(additional_claims)

        token = jwt.encode(claims, self.config.secret_key, algorithm=self.config.algorithm)

        logger.info("access_token_created", subject=subject, expires_at=expire.isoformat())
        return token

    def create_refresh_token(self, subject: str) -> str:
        """Create a JWT refresh token.

        Args:
            subject: The subject of the token (usually user ID)

        Returns:
            Encoded JWT refresh token
        """
        expire = datetime.now(UTC) + timedelta(days=self.config.refresh_token_expire_days)

        claims = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(UTC),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "refresh",
            "jti": secrets.token_urlsafe(16),  # Unique token ID for revocation
        }

        token = jwt.encode(claims, self.config.secret_key, algorithm=self.config.algorithm)

        logger.info("refresh_token_created", subject=subject, expires_at=expire.isoformat())
        return token

    def verify_token(self, token: str, token_type: str = "access") -> dict[str, Any]:
        """Verify and decode a JWT token.

        Args:
            token: The JWT token to verify
            token_type: Expected token type ("access" or "refresh")

        Returns:
            Decoded token claims

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience,
                options={"verify_exp": True}
            )

            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}"
                )

            logger.debug("token_verified", subject=payload.get("sub"), token_type=token_type)
            return payload

        except jwt.ExpiredSignatureError as e:
            logger.warning("token_expired", token_type=token_type)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            ) from e
        except jwt.InvalidTokenError as e:
            logger.warning("token_invalid", token_type=token_type, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            ) from e

    def get_current_user(self, credentials: HTTPAuthorizationCredentials) -> str:
        """Extract current user from JWT token.

        Args:
            credentials: HTTP Bearer credentials

        Returns:
            User ID from token

        Raises:
            HTTPException: If token is invalid
        """
        token = credentials.credentials
        payload = self.verify_token(token, "access")
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims"
            )

        return user_id

    async def authenticate_websocket(self, websocket: WebSocket) -> str | None:
        """Authenticate a WebSocket connection.

        Args:
            websocket: The WebSocket connection to authenticate

        Returns:
            User ID if authenticated, None otherwise
        """
        try:
            # Get token from query parameters or headers
            token = websocket.query_params.get("token")

            if not token:
                # Try to get from Authorization header
                auth_header = websocket.headers.get("authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header.split(" ")[1]

            if not token:
                logger.warning("websocket_auth_failed", reason="No token provided")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
                return None

            # Verify token
            payload = self.verify_token(token, "access")
            user_id = payload.get("sub")

            if not user_id:
                logger.warning("websocket_auth_failed", reason="Invalid token claims")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid authentication")
                return None

            logger.info("websocket_authenticated", user_id=user_id)
            return user_id

        except HTTPException as e:
            logger.warning("websocket_auth_failed", reason="Token verification failed")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
            return None
        except Exception as e:
            logger.error("websocket_auth_error", error=str(e), exc_info=True)
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Authentication error")
            return None

    def create_api_key(self, user_id: str, name: str, expires_days: int = 365) -> str:
        """Create a long-lived API key for service authentication.

        Args:
            user_id: User ID associated with the API key
            name: Name/description for the API key
            expires_days: Number of days until expiration

        Returns:
            API key token
        """
        expire = datetime.now(UTC) + timedelta(days=expires_days)

        claims = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.now(UTC),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "api_key",
            "name": name,
            "jti": secrets.token_urlsafe(16),
        }

        token = jwt.encode(claims, self.config.secret_key, algorithm=self.config.algorithm)

        logger.info(
            "api_key_created",
            user_id=user_id,
            name=name,
            expires_at=expire.isoformat()
        )
        return token


# Global JWT manager instance
_jwt_manager: JWTManager | None = None


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager instance.

    Returns:
        JWTManager instance
    """
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager
 
