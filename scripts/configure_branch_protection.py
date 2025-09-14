#!/usr/bin/env python3
"""
Script to configure GitHub branch protection rules for ARC Prize 2025 repository.
This script sets up branch protection rules to enforce code quality and review processes.
"""

import json
import os
import requests
import sys
from typing import Dict, Any


def get_github_token() -> str:
    """Get GitHub token from environment variable."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise ValueError(
            "GITHUB_TOKEN environment variable is required. "
            "Generate a token at https://github.com/settings/tokens"
        )
    return token


def configure_branch_protection(
    owner: str, 
    repo: str, 
    branch: str = "main",
    token: str = None
) -> bool:
    """
    Configure branch protection rules for the specified branch.
    
    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        branch: Branch name to protect (default: main)
        token: GitHub API token
        
    Returns:
        True if configuration was successful, False otherwise
    """
    if not token:
        token = get_github_token()
    
    # Branch protection configuration
    protection_config = {
        "required_status_checks": {
            "strict": True,
            "checks": [
                {"context": "test"},
                {"context": "security"},
                {"context": "docker"}
            ]
        },
        "enforce_admins": False,  # Allow admins to bypass in emergencies
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True,
            "required_approving_review_count": 1,
            "require_last_push_approval": True
        },
        "restrictions": None,  # No user/team restrictions
        "required_linear_history": False,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": True
    }
    
    # GitHub API endpoint
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}/protection"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.put(url, headers=headers, json=protection_config)
        
        if response.status_code == 200:
            print(f"✅ Successfully configured branch protection for {owner}/{repo}:{branch}")
            return True
        elif response.status_code == 403:
            print(f"❌ Insufficient permissions to configure branch protection")
            print(f"   Ensure the GitHub token has 'repo' permissions")
            return False
        else:
            print(f"❌ Failed to configure branch protection: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error configuring branch protection: {e}")
        return False


def main():
    """Main function to run branch protection configuration."""
    # Get repository information from environment or command line
    if len(sys.argv) >= 3:
        owner = sys.argv[1]
        repo = sys.argv[2]
        branch = sys.argv[3] if len(sys.argv) > 3 else "main"
    else:
        # Try to get from GitHub Actions environment
        github_repository = os.getenv('GITHUB_REPOSITORY', '')
        if github_repository:
            owner, repo = github_repository.split('/')
            branch = os.getenv('GITHUB_REF_NAME', 'main')
        else:
            print("Usage: python configure_branch_protection.py <owner> <repo> [branch]")
            print("Or set GITHUB_REPOSITORY environment variable")
            sys.exit(1)
    
    print(f"Configuring branch protection for {owner}/{repo}:{branch}")
    
    try:
        success = configure_branch_protection(owner, repo, branch)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()