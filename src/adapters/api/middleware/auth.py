"""Authentication middleware for FastAPI with JWT token validation.

This module provides middleware for authenticating requests using JWT tokens
and managing authentication state across the application.
"""

import time
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.infrastructure.security import get_security_headers
from src.utils.jwt_auth import JWTManager, get_jwt_manager

logger = structlog.get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT-based authentication with configurable endpoint protection."""

    def __init__(
        self,
        app: Callable,
        protected_patterns: set[str] | None = None,
        excluded_patterns: set[str] | None = None,
        jwt_manager: JWTManager | None = None
    ):
        """Initialize authentication middleware.
        
        Args:
            app: FastAPI application instance
            protected_patterns: URL patterns that require authentication
            excluded_patterns: URL patterns to exclude from authentication
            jwt_manager: JWT manager instance (optional, will create if None)
        """
        super().__init__(app)
        self.jwt_manager = jwt_manager or get_jwt_manager()

        # Default protected patterns - endpoints requiring authentication
        self.protected_patterns = protected_patterns or {
            "/api/v1/evaluation/submit",
            "/api/v1/evaluation/results",
            "/api/v1/evaluation/history",
            "/api/v1/dashboard/metrics",
            "/api/v1/dashboard/experiments",
        }

        # Patterns to exclude from authentication (public endpoints)
        self.excluded_patterns = excluded_patterns or {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/refresh",
            "/auth/register",  # If user registration is added
        }

        logger.info(
            "auth_middleware_initialized",
            protected_patterns=len(self.protected_patterns),
            excluded_patterns=len(self.excluded_patterns)
        )

    def is_protected_endpoint(self, path: str) -> bool:
        """Check if an endpoint requires authentication.
        
        Args:
            path: Request path to check
            
        Returns:
            True if endpoint requires authentication
        """
        # First check exclusions (public endpoints)
        for pattern in self.excluded_patterns:
            if path.startswith(pattern):
                return False

        # Then check protected patterns
        for pattern in self.protected_patterns:
            if path.startswith(pattern):
                return True

        # Default to not protected for unknown endpoints
        return False

    def extract_token(self, request: Request) -> str | None:
        """Extract JWT token from request headers or query parameters.
        
        Args:
            request: FastAPI request object
            
        Returns:
            JWT token string if found, None otherwise
        """
        # Check Authorization header (preferred method)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]

        # Check query parameter (for WebSocket connections)
        token_param = request.query_params.get("token")
        if token_param:
            return token_param

        # Check cookie (alternative method)
        cookie_token = request.cookies.get("access_token")
        if cookie_token:
            return cookie_token

        return None

    async def authenticate_request(self, request: Request) -> dict[str, Any] | None:
        """Authenticate a request using JWT token.
        
        Args:
            request: FastAPI request object
            
        Returns:
            User info dictionary if authenticated, None otherwise
        """
        token = self.extract_token(request)
        if not token:
            logger.debug("no_token_found", path=request.url.path)
            return None

        try:
            # Verify the token
            payload = self.jwt_manager.verify_token(token, "access")
            user_id = payload.get("sub")

            if not user_id:
                logger.warning("invalid_token_claims", path=request.url.path)
                return None

            # Prepare user info
            user_info = {
                "user_id": user_id,
                "token_type": payload.get("type"),
                "issued_at": payload.get("iat"),
                "expires_at": payload.get("exp"),
                "additional_claims": {
                    k: v for k, v in payload.items()
                    if k not in ["sub", "type", "iat", "exp", "iss", "aud"]
                }
            }

            logger.debug("token_authenticated", user_id=user_id, path=request.url.path)
            return user_info

        except Exception as e:
            logger.warning(
                "token_authentication_failed",
                path=request.url.path,
                error=str(e)
            )
            return None

    def create_authentication_error_response(
        self,
        request: Request,
        message: str = "Authentication required",
        error_code: str = "authentication_required"
    ) -> JSONResponse:
        """Create a standardized authentication error response.
        
        Args:
            request: FastAPI request object
            message: Error message
            error_code: Error code for client handling
            
        Returns:
            JSON error response
        """
        security_headers = get_security_headers()
        security_headers["WWW-Authenticate"] = "Bearer"

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": error_code,
                "message": message,
                "path": request.url.path,
                "timestamp": time.time()
            },
            headers=security_headers
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with authentication middleware.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        request_path = request.url.path

        # Check if this endpoint requires authentication
        if not self.is_protected_endpoint(request_path):
            # Public endpoint - process without authentication
            response = await call_next(request)

            # Add security headers to all responses
            security_headers = get_security_headers()
            for key, value in security_headers.items():
                response.headers[key] = value

            return response

        # Protected endpoint - authentication required
        logger.debug("authenticating_protected_endpoint", path=request_path)

        # Authenticate the request
        user_info = await self.authenticate_request(request)

        if not user_info:
            # Authentication failed
            processing_time = time.time() - start_time
            logger.warning(
                "authentication_required",
                path=request_path,
                method=request.method,
                processing_time_ms=processing_time * 1000,
                client_ip=getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            )

            return self.create_authentication_error_response(request)

        # Authentication successful - add user info to request state
        request.state.user = user_info
        request.state.user_id = user_info["user_id"]

        try:
            # Process the authenticated request
            response = await call_next(request)

            # Add security headers
            security_headers = get_security_headers()
            for key, value in security_headers.items():
                response.headers[key] = value

            # Log successful authenticated request
            processing_time = time.time() - start_time
            logger.info(
                "authenticated_request_completed",
                user_id=user_info["user_id"],
                path=request_path,
                method=request.method,
                status_code=response.status_code,
                processing_time_ms=processing_time * 1000
            )

            return response

        except Exception as e:
            # Log authenticated request failure
            processing_time = time.time() - start_time
            logger.error(
                "authenticated_request_failed",
                user_id=user_info["user_id"],
                path=request_path,
                method=request.method,
                processing_time_ms=processing_time * 1000,
                error=str(e),
                exc_info=True
            )
            raise


class WebSocketAuthenticationMixin:
    """Mixin for WebSocket authentication using JWT tokens."""

    @staticmethod
    async def authenticate_websocket(websocket, jwt_manager: JWTManager | None = None) -> str | None:
        """Authenticate a WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
            jwt_manager: JWT manager instance
            
        Returns:
            User ID if authenticated, None otherwise
        """
        if not jwt_manager:
            jwt_manager = get_jwt_manager()

        return await jwt_manager.authenticate_websocket(websocket)


def setup_authentication_middleware(
    app,
    protected_patterns: set[str] | None = None,
    excluded_patterns: set[str] | None = None
):
    """Setup authentication middleware for FastAPI application.
    
    Args:
        app: FastAPI application instance
        protected_patterns: Custom protected URL patterns
        excluded_patterns: Custom excluded URL patterns
    """
    auth_middleware = AuthenticationMiddleware(
        app,
        protected_patterns=protected_patterns,
        excluded_patterns=excluded_patterns
    )

    app.add_middleware(AuthenticationMiddleware,
                      protected_patterns=protected_patterns,
                      excluded_patterns=excluded_patterns)

    logger.info(
        "authentication_middleware_configured",
        protected_count=len(auth_middleware.protected_patterns),
        excluded_count=len(auth_middleware.excluded_patterns)
    )
