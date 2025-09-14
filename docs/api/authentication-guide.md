# Authentication Framework Documentation

## Overview

The ARC Prize 2025 evaluation system implements a comprehensive JWT-based authentication framework designed for security, performance, and ease of use. This document provides complete implementation details, API usage examples, and security considerations.

## Architecture

### Components

1. **JWT Token Management** (`src/utils/jwt_auth.py`)
   - Token generation and validation
   - Support for access and refresh tokens
   - Configurable expiration policies

2. **Security Infrastructure** (`src/infrastructure/security.py`)
   - Password hashing with bcrypt
   - Password strength validation
   - Authentication dependencies for FastAPI

3. **Authentication Middleware** (`src/adapters/api/middleware/auth.py`)
   - Automatic token validation for protected endpoints
   - Configurable endpoint protection patterns
   - Security headers injection

4. **User Management** (`src/adapters/repositories/user_repository.py`)
   - User account creation and management
   - Service account support
   - Authentication attempt tracking

5. **Authentication Endpoints** (`src/adapters/api/routes/auth.py`)
   - Login, registration, and token refresh
   - Rate limiting and abuse protection
   - Comprehensive audit logging

## Security Features

### Token Strategy
- **Access Tokens**: 15-minute expiration, stateless JWT
- **Refresh Tokens**: 7-day expiration, unique token IDs
- **Algorithm**: HS256 (configurable to RS256 for production)
- **Claims**: User ID, role, issuer, audience validation

### Password Security
- **Hashing**: bcrypt with cost factor 12
- **Requirements**: Minimum 8 characters, uppercase, lowercase, number
- **Validation**: Real-time strength checking with feedback
- **History**: Prevents password reuse (configurable)

### Rate Limiting
- **Login Attempts**: 5 per minute per IP address
- **Account Lockout**: 15 minutes after 5 failed attempts
- **Global Protection**: 10 failures per IP in 15 minutes = blocked

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000
- Referrer-Policy: strict-origin-when-cross-origin

## API Endpoints

### POST /auth/login

Authenticate user and receive JWT tokens.

**Request:**
```json
{
  "username_or_email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 900,
  "user_id": "uuid-string",
  "username": "johndoe",
  "role": "user"
}
```

**Error Responses:**
- **401 Unauthorized**: Invalid credentials
- **423 Locked**: Account temporarily locked
- **429 Too Many Requests**: Rate limit exceeded

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username_or_email": "user@example.com",
    "password": "SecurePass123"
  }'
```

### POST /auth/refresh

Refresh access token using refresh token.

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 900
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your-refresh-token-here"
  }'
```

### POST /auth/register

Register new user account.

**Request:**
```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123"
}
```

**Response (201 Created):**
```json
{
  "user_id": "uuid-string",
  "username": "johndoe",
  "email": "john@example.com",
  "message": "User account created successfully"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePass123"
  }'
```

### GET /auth/me

Get current authenticated user information.

**Headers:**
```
Authorization: Bearer your-access-token-here
```

**Response (200 OK):**
```json
{
  "user_id": "uuid-string",
  "authenticated": true,
  "token_type": "access",
  "additional_claims": {
    "role": "user"
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer your-access-token-here"
```

## Protected Endpoints

The following endpoint patterns require authentication:

- `/api/v1/evaluation/submit` - Task submission
- `/api/v1/evaluation/results` - Results retrieval
- `/api/v1/evaluation/history` - User history
- `/api/v1/dashboard/metrics` - Dashboard metrics
- `/api/v1/dashboard/experiments` - Experiment management

**Usage Example:**
```bash
# First, login to get tokens
LOGIN_RESPONSE=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username_or_email": "user@example.com", "password": "SecurePass123"}')

# Extract access token
ACCESS_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

# Use token for protected endpoint
curl -X POST "http://localhost:8000/api/v1/evaluation/submit" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "example-task", "solution": [[1,2],[3,4]]}'
```

## Environment Configuration

Required environment variables:

```bash
# JWT Configuration
JWT_SECRET_KEY=your-256-bit-secret-key-here-replace-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_ISSUER=arc-evaluation-system
JWT_AUDIENCE=arc-evaluation-api

# Rate Limiting for Auth Endpoints
RATE_LIMIT_AUTH_RPM=20
RATE_LIMIT_AUTH_RPH=200
```

### Production Security Considerations

1. **JWT Secret Key**: Generate cryptographically secure 256-bit key
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Algorithm Choice**: Consider RS256 for multi-service environments
   - Requires public/private key pair
   - Better for distributed systems
   - More secure token verification

3. **HTTPS Only**: Never transmit tokens over HTTP in production

4. **Token Storage**: 
   - Store access tokens in memory only
   - Store refresh tokens in secure HTTP-only cookies
   - Never store tokens in localStorage in browsers

## Integration Examples

### Python Client

```python
import requests
from typing import Optional

class ARCClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
    
    def login(self, username_or_email: str, password: str) -> bool:
        """Login and store tokens."""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={
                "username_or_email": username_or_email,
                "password": password
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            return True
        
        return False
    
    def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            return False
        
        response = requests.post(
            f"{self.base_url}/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            return True
        
        return False
    
    def get_headers(self) -> dict:
        """Get headers with authentication."""
        if not self.access_token:
            raise ValueError("Not authenticated. Call login() first.")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def submit_task(self, task_id: str, solution: list) -> dict:
        """Submit task solution."""
        response = requests.post(
            f"{self.base_url}/api/v1/evaluation/submit",
            headers=self.get_headers(),
            json={"task_id": task_id, "solution": solution}
        )
        
        if response.status_code == 401:
            # Token might be expired, try to refresh
            if self.refresh_access_token():
                response = requests.post(
                    f"{self.base_url}/api/v1/evaluation/submit",
                    headers=self.get_headers(),
                    json={"task_id": task_id, "solution": solution}
                )
        
        return response.json()

# Usage
client = ARCClient()
if client.login("user@example.com", "SecurePass123"):
    result = client.submit_task("task-123", [[1, 2], [3, 4]])
    print(result)
```

### JavaScript Client

```javascript
class ARCClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.accessToken = null;
        this.refreshToken = null;
    }
    
    async login(usernameOrEmail, password) {
        const response = await fetch(`${this.baseUrl}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username_or_email: usernameOrEmail,
                password: password
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            this.accessToken = data.access_token;
            this.refreshToken = data.refresh_token;
            return true;
        }
        
        return false;
    }
    
    async refreshAccessToken() {
        if (!this.refreshToken) return false;
        
        const response = await fetch(`${this.baseUrl}/auth/refresh`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                refresh_token: this.refreshToken
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            this.accessToken = data.access_token;
            return true;
        }
        
        return false;
    }
    
    getHeaders() {
        if (!this.accessToken) {
            throw new Error('Not authenticated. Call login() first.');
        }
        
        return {
            'Authorization': `Bearer ${this.accessToken}`,
            'Content-Type': 'application/json'
        };
    }
    
    async submitTask(taskId, solution) {
        let response = await fetch(`${this.baseUrl}/api/v1/evaluation/submit`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                task_id: taskId,
                solution: solution
            })
        });
        
        if (response.status === 401) {
            // Token might be expired, try to refresh
            if (await this.refreshAccessToken()) {
                response = await fetch(`${this.baseUrl}/api/v1/evaluation/submit`, {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify({
                        task_id: taskId,
                        solution: solution
                    })
                });
            }
        }
        
        return await response.json();
    }
}

// Usage
const client = new ARCClient();
if (await client.login('user@example.com', 'SecurePass123')) {
    const result = await client.submitTask('task-123', [[1, 2], [3, 4]]);
    console.log(result);
}
```

## Error Handling

### Common Error Codes

| Status | Error Code | Description | Action |
|--------|------------|-------------|--------|
| 401 | `authentication_required` | No token provided | Login first |
| 401 | `invalid_token` | Token invalid/expired | Refresh or re-login |
| 423 | `account_locked` | Account temporarily locked | Wait or contact support |
| 429 | `rate_limit_exceeded` | Too many requests | Wait and retry |
| 400 | `weak_password` | Password doesn't meet requirements | Use stronger password |

### Error Response Format

```json
{
  "error": "authentication_required",
  "message": "Authentication required to access this resource",
  "path": "/api/v1/evaluation/submit",
  "timestamp": 1693123456.789
}
```

## Monitoring and Logging

The authentication system provides comprehensive logging for:

- Successful/failed login attempts
- Token refresh operations
- Account lockouts and security events
- Rate limiting violations
- Password policy violations

Log entries include:
- User ID (when available)
- IP address
- User agent
- Timestamp
- Action result
- Error details

Example log entry:
```json
{
  "event": "user_login_successful",
  "user_id": "uuid-string",
  "username": "johndoe",
  "ip": "192.168.1.100",
  "timestamp": "2023-08-27T10:30:00Z",
  "level": "info"
}
```

## Testing

See `tests/integration/test_auth.py` for comprehensive integration tests covering:
- Login/logout flows
- Token refresh mechanisms
- Protected endpoint access
- Rate limiting behavior
- Error handling scenarios
- Security header validation

Run tests with:
```bash
pytest tests/integration/test_auth.py -v
```