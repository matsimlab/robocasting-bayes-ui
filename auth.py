import streamlit as st
import hashlib
import os
from auth_db import (init_auth_db, verify_user, create_user, is_admin,
                     get_client_ip, is_ip_blocked, record_failed_attempt,
                     reset_attempts, list_users, delete_user, change_password)


def login_page():
    """Show the login page and handle authentication"""
    # Initialize the database and create default admin if needed
    first_time = init_auth_db()

    # Set up session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.is_admin = False

    if not st.session_state.authenticated:
        st.title("🔒 Robocasting Login")

        # Get the client's IP
        client_ip = get_client_ip()

        # Check if the IP is blocked
        blocked, remaining_minutes = is_ip_blocked(client_ip)

        if blocked:
            st.error(f"Too many failed login attempts. Please try again in {remaining_minutes} minutes.")
            return False

        if first_time:
            st.warning(
                "First-time setup: Default credentials created (admin/robocasting). Please change them after logging in!")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.is_admin = is_admin(username)
                # Reset failed attempts on successful login
                reset_attempts(client_ip)
                st.rerun()
            else:
                # Record the failed attempt
                attempts_remaining = record_failed_attempt(client_ip)
                if attempts_remaining > 0:
                    st.error(f"Invalid username or password. {attempts_remaining} attempts remaining before lockout.")
                else:
                    st.error("Too many failed login attempts. Your IP has been temporarily blocked.")

        return False

    return True


def logout():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.is_admin = False
    st.rerun()


def show_logout_button():
    """Display a logout button in the sidebar"""
    # Initialize session state variables if they don't exist
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if st.session_state.authenticated:
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state.username}**")
            if st.button("Logout"):
                logout()


def show_user_management():
    """Display user management after navigation if user is authenticated"""
    if not st.session_state.authenticated:
        return

    # Add a separator between app content and user management
    st.sidebar.markdown("---")

    # User management options
    with st.sidebar:
        # Change password option for all users
        with st.expander("Change Password", expanded=False):
            change_password_form()

        # Admin panel only for admin users
        if st.session_state.is_admin:
            with st.expander("User Management", expanded=False):
                admin_panel()


def admin_panel():
    """Admin panel for user management"""
    st.subheader("User Management")

    # User creation form
    with st.form("create_user_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        is_admin_user = st.checkbox("Admin User")
        submit = st.form_submit_button("Create User")

        if submit and new_username and new_password:
            if create_user(new_username, new_password, is_admin_user):
                st.success(f"User '{new_username}' created successfully")
            else:
                st.error(f"User '{new_username}' already exists")

    # List existing users
    users = list_users()
    if users:
        st.subheader("Existing Users")
        for user, is_admin_flag, created_at in users:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{user}**" + (" (Admin)" if is_admin_flag else ""))
            with col2:
                st.write(f"Created: {created_at[:10]}")
            with col3:
                # Don't allow deleting own account or the main admin
                if user != "admin" and user != st.session_state.username:
                    if st.button("Delete", key=f"delete_{user}"):
                        if delete_user(user):
                            st.success(f"User '{user}' deleted")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete user '{user}'")


def change_password_form():
    """Form for changing a user's password"""
    with st.form("change_password_form"):
        st.subheader("Change Password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Change Password")

        if submit:
            if not new_password:
                st.error("Password cannot be empty")
            elif new_password != confirm_password:
                st.error("Passwords don't match")
            else:
                if change_password(st.session_state.username, new_password):
                    st.success("Password changed successfully")
                else:
                    st.error("Failed to change password")