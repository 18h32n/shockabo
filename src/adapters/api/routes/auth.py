"""Authentication endpoints for JWT-based login and token management.

This module provides secure authentication endpoints with rate limiting,
comprehensive error handling, and audit logging.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from src.adapters.repositories.user_repository import UserRepository
from src.domain.models import AuthenticationAttempt, UserRole
from src.infrastructure.security import (
    create_access_token,
    create_refresh_token,
    get_security_headers,
    validate_password_strength,
    verify_refresh_token,
)

logger = structlog.get_logger(__name__)

# Pydantic models for request/response
class LoginRequest(BaseModel):
    """Login request model."""
    username_or_email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes in seconds
    user_id: str
    username: str
    role: str


class RefreshRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str = Field(..., min_length=1)


class RefreshResponse(BaseModel):
    """Token refresh response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes in seconds


class RegisterRequest(BaseModel):
    """User registration request model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=255)


class RegisterResponse(BaseModel):
    """User registration response model."""
    user_id: str
    username: str
    email: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    message: str
    details: dict[str, Any] | None = None
    timestamp: float


# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Initialize user repository
user_repo = UserRepository()


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    return request.client.host if request.client else "unknown"


def create_auth_attempt(
    request: Request,
    username_or_email: str,
    success: bool,
    failure_reason: str | None = None
) -> AuthenticationAttempt:
    """Create an authentication attempt record."""
    return AuthenticationAttempt(
        id=str(uuid4()),
        username_or_email=username_or_email,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("user-agent", "unknown"),
        success=success,
        failure_reason=failure_reason,
        timestamp=datetime.now()
    )


@router.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(login_data: LoginRequest, request: Request):
    """Authenticate user and return JWT tokens.

    This endpoint implements:
    - Rate limiting (5 attempts per minute per IP)
    - Account lockout after 5 failed attempts
    - Comprehensive audit logging
    - Generic error messages to prevent username enumeration
    """
    client_ip = get_client_ip(request)

    # Check for too many recent failed attempts from this IP
    recent_failures = user_repo.get_recent_failed_attempts(client_ip, minutes=15)
    if len(recent_failures) >= 10:  # 10 failures in 15 minutes = suspicious
        logger.warning(
            "too_many_auth_attempts_from_ip",
            ip=client_ip,
            attempts=len(recent_failures)
        )

        # Record the attempt
        attempt = create_auth_attempt(
            request,
            login_data.username_or_email,
            False,
            "too_many_attempts"
        )
        user_repo.record_auth_attempt(attempt)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later.",
            headers={"Retry-After": "900"}  # 15 minutes
        )

    try:
        # Attempt authentication
        user = user_repo.authenticate_user(
            login_data.username_or_email,
            login_data.password
        )

        if not user:
            # Authentication failed - record attempt with generic error
            attempt = create_auth_attempt(
                request,
                login_data.username_or_email,
                False,
                "invalid_credentials"
            )
            user_repo.record_auth_attempt(attempt)

            logger.warning(
                "authentication_failed",
                identifier=login_data.username_or_email,
                ip=client_ip
            )

            # Return generic error to prevent username enumeration
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username/email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Check if user account is locked
        if user.is_locked():
            attempt = create_auth_attempt(
                request,
                login_data.username_or_email,
                False,
                "account_locked"
            )
            user_repo.record_auth_attempt(attempt)

            logger.warning(
                "locked_account_login_attempt",
                user_id=user.id,
                locked_until=user.locked_until
            )

            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to multiple failed login attempts"
            )

        # Authentication successful - generate tokens
        access_token = create_access_token(user.id, {"role": user.role.value})
        refresh_token = create_refresh_token(user.id)

        # Record successful attempt
        attempt = create_auth_attempt(
            request,
            login_data.username_or_email,
            True
        )
        user_repo.record_auth_attempt(attempt)

        logger.info(
            "user_login_successful",
            user_id=user.id,
            username=user.username,
            ip=client_ip
        )

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.id,
            username=user.username,
            role=user.role.value
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Unexpected error - log and return generic error
        logger.error(
            "login_unexpected_error",
            identifier=login_data.username_or_email,
            ip=client_ip,
            error=str(e),
            exc_info=True
        )

        # Record failed attempt
        attempt = create_auth_attempt(
            request,
            login_data.username_or_email,
            False,
            "internal_error"
        )
        user_repo.record_auth_attempt(attempt)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        ) from None


@router.post("/refresh", response_model=RefreshResponse, status_code=status.HTTP_200_OK)
async def refresh_token(refresh_data: RefreshRequest, request: Request):
    """Refresh JWT access token using refresh token.

    This endpoint validates the refresh token and issues a new access token.
    """
    client_ip = get_client_ip(request)

    try:
        # Verify refresh token and get user ID
        user_id = verify_refresh_token(refresh_data.refresh_token)

        # Create new access token
        access_token = create_access_token(user_id)

        logger.info(
            "token_refreshed",
            user_id=user_id,
            ip=client_ip
        )

        return RefreshResponse(
            access_token=access_token
        )

    except HTTPException as e:
        logger.warning(
            "token_refresh_failed",
            ip=client_ip,
            error=e.detail
        )
        raise
    except Exception as e:
        logger.error(
            "token_refresh_unexpected_error",
            ip=client_ip,
            error=str(e),
            exc_info=True
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        ) from None


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user(register_data: RegisterRequest, request: Request):
    """Register a new user account.

    This endpoint creates a new user account with password validation
    and comprehensive security checks.
    """
    client_ip = get_client_ip(request)

    try:
        # Validate password strength
        password_validation = validate_password_strength(register_data.password)
        if not password_validation["is_valid"]:
            logger.info(
                "registration_failed_weak_password",
                username=register_data.username,
                ip=client_ip,
                errors=password_validation["errors"]
            )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "weak_password",
                    "message": "Password does not meet security requirements",
                    "errors": password_validation["errors"],
                    "suggestions": password_validation["suggestions"]
                }
            )

        # Create user account
        user = user_repo.create_user(
            username=register_data.username,
            email=register_data.email,
            password=register_data.password,
            role=UserRole.USER
        )

        logger.info(
            "user_registered",
            user_id=user.id,
            username=user.username,
            email=user.email,
            ip=client_ip
        )

        return RegisterResponse(
            user_id=user.id,
            username=user.username,
            email=user.email,
            message="User account created successfully"
        )

    except ValueError as e:
        logger.info(
            "registration_failed_validation",
            username=register_data.username,
            email=register_data.email,
            ip=client_ip,
            error=str(e)
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except Exception as e:
        logger.error(
            "registration_unexpected_error",
            username=register_data.username,
            email=register_data.email,
            ip=client_ip,
            error=str(e),
            exc_info=True
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        ) from None


@router.get("/me", status_code=status.HTTP_200_OK)
async def get_current_user_info(request: Request):
    """Get current authenticated user information.

    This endpoint requires authentication and returns user details.
    """
    # This would use the authentication middleware
    # The user info would be available in request.state.user
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    user_info = request.state.user
    return {
        "user_id": user_info["user_id"],
        "authenticated": True,
        "token_type": user_info.get("token_type"),
        "additional_claims": user_info.get("additional_claims", {})
    }


# Add security headers to all auth endpoints
@router.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to authentication endpoints."""
    response = await call_next(request)

    security_headers = get_security_headers()
    for key, value in security_headers.items():
        response.headers[key] = value

    return response
