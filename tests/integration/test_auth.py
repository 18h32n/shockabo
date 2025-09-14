"""Integration tests for authentication framework.

This module provides comprehensive tests for the JWT-based authentication system,
including login/logout flows, token management, rate limiting, and security features.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.adapters.api.app import app
from src.adapters.repositories.user_repository import UserRepository
from src.domain.models import UserRole, AccountStatus
from src.infrastructure.security import hash_password, validate_password_strength
from src.utils.jwt_auth import get_jwt_manager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def user_repo():
    """Create test user repository with in-memory database."""
    return UserRepository(":memory:")


@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123",
        "role": UserRole.USER
    }


@pytest.fixture
def created_test_user(user_repo, test_user_data):
    """Create a test user in the repository."""
    return user_repo.create_user(
        username=test_user_data["username"],
        email=test_user_data["email"],
        password=test_user_data["password"],
        role=test_user_data["role"]
    )


class TestUserRegistration:
    """Test user registration functionality."""
    
    def test_register_user_success(self, client):
        """Test successful user registration."""
        registration_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "NewPass123"
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == registration_data["username"]
        assert data["email"] == registration_data["email"]
        assert "user_id" in data
        assert data["message"] == "User account created successfully"
    
    def test_register_duplicate_username(self, client):
        """Test registration with duplicate username."""
        registration_data = {
            "username": "testuser",
            "email": "test1@example.com",
            "password": "TestPass123"
        }
        
        # First registration
        response1 = client.post("/auth/register", json=registration_data)
        assert response1.status_code == 201
        
        # Duplicate username with different email
        registration_data["email"] = "test2@example.com"
        response2 = client.post("/auth/register", json=registration_data)
        
        assert response2.status_code == 400
        assert "Username already exists" in response2.json()["detail"]
    
    def test_register_duplicate_email(self, client):
        """Test registration with duplicate email."""
        registration_data = {
            "username": "testuser1",
            "email": "test@example.com",
            "password": "TestPass123"
        }
        
        # First registration
        response1 = client.post("/auth/register", json=registration_data)
        assert response1.status_code == 201
        
        # Duplicate email with different username
        registration_data["username"] = "testuser2"
        response2 = client.post("/auth/register", json=registration_data)
        
        assert response2.status_code == 400
        assert "Email already exists" in response2.json()["detail"]
    
    def test_register_weak_password(self, client):
        """Test registration with weak password."""
        registration_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "weak"  # Too short, no uppercase, no number
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "weak_password" in data["detail"]["error"]
        assert "errors" in data["detail"]
        assert "suggestions" in data["detail"]
    
    def test_register_invalid_email(self, client):
        """Test registration with invalid email format."""
        registration_data = {
            "username": "testuser",
            "email": "invalid-email",
            "password": "TestPass123"
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 422  # Validation error


class TestUserLogin:
    """Test user login functionality."""
    
    def test_login_success_with_username(self, client, created_test_user):
        """Test successful login with username."""
        login_data = {
            "username_or_email": "testuser",
            "password": "TestPass123"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 900  # 15 minutes
        assert data["username"] == "testuser"
        assert data["role"] == "user"
    
    def test_login_success_with_email(self, client, created_test_user):
        """Test successful login with email."""
        login_data = {
            "username_or_email": "test@example.com",
            "password": "TestPass123"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["username"] == "testuser"
    
    def test_login_invalid_username(self, client):
        """Test login with invalid username."""
        login_data = {
            "username_or_email": "nonexistent",
            "password": "TestPass123"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Invalid username/email or password" in response.json()["detail"]
    
    def test_login_invalid_password(self, client, created_test_user):
        """Test login with invalid password."""
        login_data = {
            "username_or_email": "testuser",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Invalid username/email or password" in response.json()["detail"]
    
    def test_login_account_lockout(self, client, user_repo, test_user_data):
        """Test account lockout after multiple failed attempts."""
        # Create user
        user = user_repo.create_user(**test_user_data)
        
        login_data = {
            "username_or_email": "testuser",
            "password": "wrongpassword"
        }
        
        # Make 5 failed login attempts
        for i in range(5):
            response = client.post("/auth/login", json=login_data)
            assert response.status_code == 401
        
        # 6th attempt should result in account lockout
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 423
        assert "Account is temporarily locked" in response.json()["detail"]
        
        # Even correct password should be locked
        login_data["password"] = "TestPass123"
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 423


class TestTokenRefresh:
    """Test token refresh functionality."""
    
    def test_refresh_token_success(self, client, created_test_user):
        """Test successful token refresh."""
        # First login to get refresh token
        login_data = {
            "username_or_email": "testuser",
            "password": "TestPass123"
        }
        
        login_response = client.post("/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh the token
        refresh_data = {"refresh_token": refresh_token}
        refresh_response = client.post("/auth/refresh", json=refresh_data)
        
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 900
    
    def test_refresh_invalid_token(self, client):
        """Test refresh with invalid token."""
        refresh_data = {"refresh_token": "invalid-token"}
        response = client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401
        assert "Invalid token" in response.json()["detail"]
    
    def test_refresh_access_token_as_refresh(self, client, created_test_user):
        """Test using access token as refresh token."""
        # Login to get access token
        login_data = {
            "username_or_email": "testuser",
            "password": "TestPass123"
        }
        
        login_response = client.post("/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Try to use access token as refresh token
        refresh_data = {"refresh_token": access_token}
        response = client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401
        assert "Invalid token type" in response.json()["detail"]


class TestProtectedEndpoints:
    """Test protected endpoint access."""
    
    def test_access_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get("/auth/me")
        
        assert response.status_code == 401
        assert "authentication_required" in response.json()["error"]
    
    def test_access_protected_endpoint_with_valid_token(self, client, created_test_user):
        """Test accessing protected endpoint with valid token."""
        # Login to get access token
        login_data = {
            "username_or_email": "testuser",
            "password": "TestPass123"
        }
        
        login_response = client.post("/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is True
        assert "user_id" in data
    
    def test_access_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/auth/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_access_protected_endpoint_with_expired_token(self, client):
        """Test accessing protected endpoint with expired token."""
        jwt_manager = get_jwt_manager()
        
        # Create expired token (expired 1 hour ago)
        expired_token = jwt_manager.create_access_token(
            "test-user",
            expires_delta=timedelta(hours=-1)
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/auth/me", headers=headers)
        
        assert response.status_code == 401
        assert "Token has expired" in response.json()["detail"]


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_login_rate_limiting(self, client):
        """Test rate limiting on login endpoint."""
        login_data = {
            "username_or_email": "testuser",
            "password": "wrongpassword"
        }
        
        # Make rapid login attempts
        responses = []
        for i in range(15):  # Exceed the limit of 10 in 15 minutes
            response = client.post("/auth/login", json=login_data)
            responses.append(response)
        
        # At some point, we should get rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Expected rate limiting to kick in"


class TestSecurityHeaders:
    """Test security headers are applied."""
    
    def test_auth_endpoints_have_security_headers(self, client):
        """Test that auth endpoints include security headers."""
        response = client.post("/auth/login", json={
            "username_or_email": "test",
            "password": "test"
        })
        
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
    
    def test_protected_endpoints_have_security_headers(self, client):
        """Test that protected endpoints include security headers."""
        response = client.get("/auth/me")
        
        # Should have security headers even on auth failure
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers


class TestPasswordSecurity:
    """Test password security functions."""
    
    def test_password_hashing_and_verification(self):
        """Test password hashing and verification."""
        from src.infrastructure.security import hash_password, verify_password
        
        password = "TestPass123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong", hashed) is False
    
    def test_password_strength_validation(self):
        """Test password strength validation."""
        # Strong password
        result = validate_password_strength("StrongPass123")
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        
        # Weak password
        result = validate_password_strength("weak")
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert any("8 characters" in error for error in result["errors"])
        assert any("uppercase" in error for error in result["errors"])
        assert any("number" in error for error in result["errors"])
    
    def test_common_password_rejection(self):
        """Test that common passwords are rejected."""
        common_passwords = ["password", "123456", "password123"]
        
        for password in common_passwords:
            result = validate_password_strength(password)
            assert result["is_valid"] is False
            assert any("common" in error.lower() for error in result["errors"])


class TestJWTTokens:
    """Test JWT token functionality."""
    
    def test_jwt_token_creation_and_validation(self):
        """Test JWT token creation and validation."""
        jwt_manager = get_jwt_manager()
        
        # Create access token
        user_id = "test-user-123"
        token = jwt_manager.create_access_token(user_id)
        
        # Validate token
        payload = jwt_manager.verify_token(token, "access")
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
    
    def test_jwt_refresh_token_creation(self):
        """Test JWT refresh token creation."""
        jwt_manager = get_jwt_manager()
        
        user_id = "test-user-123"
        token = jwt_manager.create_refresh_token(user_id)
        
        payload = jwt_manager.verify_token(token, "refresh")
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "jti" in payload  # Unique token ID
    
    def test_jwt_token_expiration(self):
        """Test JWT token expiration."""
        jwt_manager = get_jwt_manager()
        
        # Create token that expires in 1 second
        user_id = "test-user-123"
        token = jwt_manager.create_access_token(
            user_id, 
            expires_delta=timedelta(seconds=1)
        )
        
        # Token should be valid immediately
        payload = jwt_manager.verify_token(token, "access")
        assert payload["sub"] == user_id
        
        # Wait for expiration
        time.sleep(2)
        
        # Token should now be expired
        with pytest.raises(Exception) as exc_info:
            jwt_manager.verify_token(token, "access")
        
        assert "expired" in str(exc_info.value).lower()


class TestAuditLogging:
    """Test authentication audit logging."""
    
    @patch('src.adapters.api.routes.auth.logger')
    def test_successful_login_logging(self, mock_logger, client, created_test_user):
        """Test that successful logins are logged."""
        login_data = {
            "username_or_email": "testuser",
            "password": "TestPass123"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        # Verify logging was called
        mock_logger.info.assert_called()
        log_calls = [call.args for call in mock_logger.info.call_args_list]
        assert any("user_login_successful" in str(call) for call in log_calls)
    
    @patch('src.adapters.api.routes.auth.logger')
    def test_failed_login_logging(self, mock_logger, client):
        """Test that failed logins are logged."""
        login_data = {
            "username_or_email": "nonexistent",
            "password": "wrongpass"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
        
        # Verify warning logging was called
        mock_logger.warning.assert_called()
        log_calls = [call.args for call in mock_logger.warning.call_args_list]
        assert any("authentication_failed" in str(call) for call in log_calls)


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/auth/login",
            data="invalid-json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        # Missing password
        response = client.post("/auth/login", json={
            "username_or_email": "test"
        })
        
        assert response.status_code == 422
    
    def test_empty_request_body(self, client):
        """Test handling of empty request body."""
        response = client.post("/auth/login", json={})
        
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])