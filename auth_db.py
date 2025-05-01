import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
import streamlit as st
import os

# Helper function to determine the database path
def get_auth_db_path():
    """Helper function to determine the database path"""
    if os.path.exists('/app'):
        # Docker environment
        db_dir = '/app/data'
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, 'robocasting_auth.db')
    else:
        # Local development environment
        db_dir = 'data'
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, 'robocasting_auth.db')


def init_auth_db():
    """Initialize the authentication database and create tables if they don't exist"""
    # Get database path
    auth_db = get_auth_db_path()
    
    # Create the database directory if needed
    os.makedirs(os.path.dirname(auth_db), exist_ok=True)

    conn = sqlite3.connect(auth_db)
    c = conn.cursor()

    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        is_admin BOOLEAN NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create rate limiting table
    c.execute('''
    CREATE TABLE IF NOT EXISTS login_attempts (
        ip_address TEXT PRIMARY KEY,
        failed_attempts INTEGER NOT NULL,
        last_attempt TIMESTAMP NOT NULL
    )
    ''')

    conn.commit()
    conn.close()

    # Initialize default admin if no users exist
    if not user_exists("admin"):
        create_user("admin", "robocasting", is_admin=True)
        return True

    return False


def hash_password(password):
    """Create a hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password, is_admin=False):
    """Create a new user in the database"""
    if user_exists(username):
        return False

    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute('''
    INSERT INTO users (username, password_hash, is_admin)
    VALUES (?, ?, ?)
    ''', (username, hash_password(password), is_admin))

    conn.commit()
    conn.close()
    return True


def user_exists(username):
    """Check if a user exists in the database"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
    count = c.fetchone()[0]

    conn.close()
    return count > 0


def verify_user(username, password):
    """Verify a user's credentials"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()

    conn.close()

    if not result:
        return False

    stored_hash = result[0]
    return stored_hash == hash_password(password)


def is_admin(username):
    """Check if a user is an admin"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
    result = c.fetchone()

    conn.close()

    if not result:
        return False

    return bool(result[0])


def get_client_ip():
    """Get the client's IP address (or a substitute in local development)"""
    # In local development, Streamlit doesn't provide the actual client IP
    # So we'll use a placeholder in that case
    try:
        # Try to get IP from Streamlit's runtime config
        return st.runtime.get_instance().get_client_ip()
    except:
        # For local development, use a placeholder value
        return "127.0.0.1"


def is_ip_blocked(ip, max_attempts=3, block_duration=10):
    """Check if an IP is currently blocked"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute("SELECT failed_attempts, last_attempt FROM login_attempts WHERE ip_address = ?", (ip,))
    result = c.fetchone()

    conn.close()

    # If IP not in records, it's not blocked
    if not result:
        return False, 0

    failed_attempts, last_attempt_str = result
    last_attempt = datetime.fromisoformat(last_attempt_str)

    # Check if the IP should still be blocked
    if failed_attempts >= max_attempts:
        # Calculate when the block expires
        block_expires = last_attempt + timedelta(minutes=block_duration)

        # If current time is before expiration, the IP is still blocked
        if datetime.now() < block_expires:
            # Return True (blocked) and the remaining time
            remaining = (block_expires - datetime.now()).total_seconds() / 60
            return True, round(remaining)
        else:
            # Block has expired, reset attempts
            reset_attempts(ip)
            return False, 0

    return False, 0


def record_failed_attempt(ip, max_attempts=3, block_duration=10):
    """Record a failed login attempt for an IP"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    # Check if IP exists in the database
    c.execute("SELECT failed_attempts, last_attempt FROM login_attempts WHERE ip_address = ?", (ip,))
    result = c.fetchone()

    now = datetime.now().isoformat()

    if result:
        failed_attempts, last_attempt_str = result
        last_attempt = datetime.fromisoformat(last_attempt_str)

        # Check if we need to reset attempts due to time expiration
        if datetime.now() > last_attempt + timedelta(minutes=block_duration):
            failed_attempts = 0

        # Increment failed attempts
        failed_attempts += 1

        c.execute('''
        UPDATE login_attempts 
        SET failed_attempts = ?, last_attempt = ? 
        WHERE ip_address = ?
        ''', (failed_attempts, now, ip))
    else:
        # First attempt for this IP
        failed_attempts = 1
        c.execute('''
        INSERT INTO login_attempts (ip_address, failed_attempts, last_attempt)
        VALUES (?, ?, ?)
        ''', (ip, failed_attempts, now))

    conn.commit()
    conn.close()

    # Return the number of attempts remaining before block
    return max_attempts - failed_attempts


def reset_attempts(ip):
    """Reset the failed attempts for an IP after successful login"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute('''
    UPDATE login_attempts 
    SET failed_attempts = 0, last_attempt = ? 
    WHERE ip_address = ?
    ''', (datetime.now().isoformat(), ip))

    conn.commit()
    conn.close()


def list_users():
    """List all users in the database"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute("SELECT username, is_admin, created_at FROM users")
    users = c.fetchall()

    conn.close()

    return users


def delete_user(username):
    """Delete a user from the database"""
    if username == "admin":
        return False  # Don't allow deleting the main admin

    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute("DELETE FROM users WHERE username = ?", (username,))

    rows_affected = c.rowcount
    conn.commit()
    conn.close()

    return rows_affected > 0


def change_password(username, new_password):
    """Change a user's password"""
    conn = sqlite3.connect(get_auth_db_path())
    c = conn.cursor()

    c.execute('''
    UPDATE users 
    SET password_hash = ?
    WHERE username = ?
    ''', (hash_password(new_password), username))

    rows_affected = c.rowcount
    conn.commit()
    conn.close()

    return rows_affected > 0