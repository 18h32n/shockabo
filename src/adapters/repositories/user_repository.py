"""User repository for authentication and user management.

This module provides data access layer for user accounts, service accounts,
and authentication tracking using SQLite database.
"""

import sqlite3
from datetime import datetime, timedelta
from uuid import uuid4

import structlog

from src.domain.models import AccountStatus, AuthenticationAttempt, ServiceAccount, User, UserRole
from src.infrastructure.security import hash_password, verify_password

logger = structlog.get_logger(__name__)


class UserRepository:
    """Repository for user account management."""

    def __init__(self, db_path: str = "./data/arc_prize.db"):
        """Initialize user repository with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database tables for user management."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    last_login_at TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    api_key_hash TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    last_used_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id TEXT PRIMARY KEY,
                    username_or_email TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_attempts_username ON auth_attempts(username_or_email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip ON auth_attempts(ip_address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_attempts_timestamp ON auth_attempts(timestamp)")

            conn.commit()

    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> User:
        """Create a new user account.
        
        Args:
            username: Unique username
            email: Unique email address
            password: Plain text password (will be hashed)
            role: User role
            
        Returns:
            Created User object
            
        Raises:
            ValueError: If username or email already exists
        """
        user_id = str(uuid4())
        password_hash = hash_password(password)
        now = datetime.now()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (id, username, email, password_hash, role, status, 
                                     created_at, updated_at, failed_login_attempts, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, '{}')
                """, (user_id, username, email, password_hash, role.value,
                      AccountStatus.ACTIVE.value, now, now))
                conn.commit()

                logger.info("user_created", user_id=user_id, username=username, role=role.value)

                return User(
                    id=user_id,
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    role=role,
                    status=AccountStatus.ACTIVE,
                    created_at=now,
                    updated_at=now,
                    failed_login_attempts=0,
                    metadata={}
                )

        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            else:
                raise ValueError("User creation failed") from e

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User object if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM users WHERE username = ?
            """, (username,))

            row = cursor.fetchone()
            if not row:
                return None

            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                role=UserRole(row["role"]),
                status=AccountStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                last_login_at=datetime.fromisoformat(row["last_login_at"]) if row["last_login_at"] else None,
                failed_login_attempts=row["failed_login_attempts"],
                locked_until=datetime.fromisoformat(row["locked_until"]) if row["locked_until"] else None,
                metadata={}  # Could parse JSON from metadata field
            )

    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email.
        
        Args:
            email: Email to search for
            
        Returns:
            User object if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM users WHERE email = ?
            """, (email,))

            row = cursor.fetchone()
            if not row:
                return None

            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                role=UserRole(row["role"]),
                status=AccountStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                last_login_at=datetime.fromisoformat(row["last_login_at"]) if row["last_login_at"] else None,
                failed_login_attempts=row["failed_login_attempts"],
                locked_until=datetime.fromisoformat(row["locked_until"]) if row["locked_until"] else None,
                metadata={}
            )

    def authenticate_user(self, username_or_email: str, password: str) -> User | None:
        """Authenticate a user with username/email and password.
        
        Args:
            username_or_email: Username or email address
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        # Try to find user by username first, then email
        user = self.get_user_by_username(username_or_email)
        if not user:
            user = self.get_user_by_email(username_or_email)

        if not user:
            logger.debug("user_not_found", identifier=username_or_email)
            return None

        # Check if account is active and not locked
        if not user.is_active():
            logger.warning("inactive_user_login_attempt", user_id=user.id, status=user.status.value)
            return None

        # Verify password
        if not verify_password(password, user.password_hash):
            logger.warning("invalid_password_attempt", user_id=user.id)
            self._record_failed_login(user.id)
            return None

        # Authentication successful - reset failed attempts and update last login
        self._record_successful_login(user.id)

        # Return updated user object
        user.last_login_at = datetime.now()
        user.failed_login_attempts = 0
        user.locked_until = None

        logger.info("user_authenticated", user_id=user.id, username=user.username)
        return user

    def _record_failed_login(self, user_id: str):
        """Record a failed login attempt and potentially lock account."""
        with sqlite3.connect(self.db_path) as conn:
            # Increment failed attempts
            conn.execute("""
                UPDATE users 
                SET failed_login_attempts = failed_login_attempts + 1,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.now(), user_id))

            # Check if account should be locked (5 failed attempts)
            cursor = conn.execute("""
                SELECT failed_login_attempts FROM users WHERE id = ?
            """, (user_id,))

            row = cursor.fetchone()
            if row and row[0] >= 5:
                # Lock account for 15 minutes
                locked_until = datetime.now() + timedelta(minutes=15)
                conn.execute("""
                    UPDATE users 
                    SET locked_until = ?, status = ?, updated_at = ?
                    WHERE id = ?
                """, (locked_until, AccountStatus.LOCKED.value, datetime.now(), user_id))

                logger.warning("user_account_locked", user_id=user_id, locked_until=locked_until)

            conn.commit()

    def _record_successful_login(self, user_id: str):
        """Record a successful login and reset failed attempts."""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now()
            conn.execute("""
                UPDATE users 
                SET last_login_at = ?, 
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    status = ?,
                    updated_at = ?
                WHERE id = ?
            """, (now, AccountStatus.ACTIVE.value, now, user_id))
            conn.commit()

    def record_auth_attempt(self, attempt: AuthenticationAttempt):
        """Record an authentication attempt for security monitoring.
        
        Args:
            attempt: AuthenticationAttempt object to record
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO auth_attempts (id, username_or_email, ip_address, user_agent,
                                         success, failure_reason, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (attempt.id, attempt.username_or_email, attempt.ip_address,
                  attempt.user_agent, attempt.success, attempt.failure_reason,
                  attempt.timestamp, "{}"))
            conn.commit()

    def get_recent_failed_attempts(self, ip_address: str, minutes: int = 15) -> list[AuthenticationAttempt]:
        """Get recent failed authentication attempts from an IP.
        
        Args:
            ip_address: IP address to check
            minutes: Number of minutes to look back
            
        Returns:
            List of recent failed attempts
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM auth_attempts 
                WHERE ip_address = ? AND success = 0 AND timestamp > ?
                ORDER BY timestamp DESC
            """, (ip_address, cutoff))

            attempts = []
            for row in cursor.fetchall():
                attempts.append(AuthenticationAttempt(
                    id=row["id"],
                    username_or_email=row["username_or_email"],
                    ip_address=row["ip_address"],
                    user_agent=row["user_agent"] or "",
                    success=bool(row["success"]),
                    failure_reason=row["failure_reason"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata={}
                ))

            return attempts

    def create_service_account(self, name: str, description: str, permissions: list[str]) -> ServiceAccount:
        """Create a new service account.
        
        Args:
            name: Service account name
            description: Description of the service account
            permissions: List of permissions
            
        Returns:
            Created ServiceAccount object
        """
        service_id = str(uuid4())
        api_key = str(uuid4())  # Generate API key
        api_key_hash = hash_password(api_key)  # Hash the API key
        now = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO service_accounts (id, name, description, api_key_hash,
                                            permissions, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')
            """, (service_id, name, description, api_key_hash,
                  ",".join(permissions), AccountStatus.ACTIVE.value, now, now))
            conn.commit()

            logger.info("service_account_created", service_id=service_id, name=name)

            service_account = ServiceAccount(
                id=service_id,
                name=name,
                description=description,
                api_key_hash=api_key_hash,
                permissions=permissions,
                status=AccountStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                metadata={"api_key": api_key}  # Return API key once for client
            )

            return service_account
