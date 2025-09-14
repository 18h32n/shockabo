"""
HuggingFace authentication configuration utilities.

Provides simple token management for HuggingFace model access.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_hf_token() -> str | None:
    """
    Get HuggingFace token from environment variables or .env file.

    Returns:
        HuggingFace token if available, None otherwise
    """
    # Try environment variable first
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    if token:
        logger.info("Found HuggingFace token in environment variables")
        return token

    # Try .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HUGGINGFACE_TOKEN=") or line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        if token and token != "your_token_here":
                            logger.info("Found HuggingFace token in .env file")
                            return token
        except Exception as e:
            logger.warning(f"Error reading .env file: {e}")

    logger.warning("No HuggingFace token found. Public models will be used.")
    return None


def setup_hf_auth() -> bool:
    """
    Set up HuggingFace authentication if token is available.

    Returns:
        True if authentication was set up, False otherwise
    """
    token = get_hf_token()

    if token:
        # Set the token in environment for transformers library
        os.environ["HF_TOKEN"] = token

        try:
            # Try to login using huggingface_hub if available
            from huggingface_hub import login
            login(token=token, write_permission=False)
            logger.info("Successfully authenticated with HuggingFace")
            return True
        except ImportError:
            logger.warning("huggingface_hub not available, but token is set in environment")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with HuggingFace: {e}")
            return False

    return False


def get_model_access_info(model_name: str) -> dict:
    """
    Get information about model access requirements.

    Args:
        model_name: Name of the model to check

    Returns:
        Dictionary with access information
    """
    # Known models that require authentication
    auth_required_models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
    ]

    # Public models that work without authentication
    public_models = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "distilbert-base-uncased",
        "microsoft/DialoGPT-medium",
    ]

    requires_auth = any(auth_model in model_name for auth_model in auth_required_models)
    is_public = any(public_model in model_name for public_model in public_models)

    return {
        "model_name": model_name,
        "requires_authentication": requires_auth,
        "is_public": is_public,
        "token_available": get_hf_token() is not None,
        "can_access": (not requires_auth) or get_hf_token() is not None
    }


def create_env_template() -> None:
    """Create a template .env file if it doesn't exist."""
    env_file = Path(".env")
    template_file = Path(".env.example")

    template_content = """# HuggingFace Authentication
# Get your token from https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=your_token_here

# Optional: Specify cache directory
HF_HOME=./cache/huggingface
"""

    if not template_file.exists():
        with open(template_file, 'w') as f:
            f.write(template_content)
        logger.info("Created .env.example template file")

    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(template_content)
        logger.info("Created .env file - please add your HuggingFace token")


def suggest_public_model(model_name: str) -> str | None:
    """
    Suggest a public model alternative if the requested model requires auth.

    Args:
        model_name: Name of the requested model

    Returns:
        Suggested public model name, or None if no good alternative
    """
    # Model suggestions mapping
    suggestions = {
        "meta-llama/Llama-3.2-1B": "gpt2",  # Similar size for testing
        "meta-llama/Llama-3.2-3B": "gpt2-medium",  # Slightly larger
        "meta-llama/Llama-2-7b-hf": "gpt2-large",  # Larger public model
    }

    for auth_model, public_alternative in suggestions.items():
        if auth_model in model_name:
            return public_alternative

    return "gpt2"  # Default fallback


if __name__ == "__main__":
    # Demo script
    create_env_template()

    print("HuggingFace Authentication Setup")
    print("=" * 40)

    token_available = get_hf_token() is not None
    print(f"Token available: {'Yes' if token_available else 'No'}")

    if setup_hf_auth():
        print("Authentication: SUCCESS")
    else:
        print("Authentication: Not set up (using public models)")

    # Test some models
    test_models = [
        "meta-llama/Llama-3.2-1B",
        "gpt2",
        "microsoft/DialoGPT-medium"
    ]

    print("\nModel Access Check:")
    for model in test_models:
        info = get_model_access_info(model)
        status = "✓ Can access" if info["can_access"] else "✗ Cannot access"
        print(f"  {model}: {status}")
        if not info["can_access"] and info["requires_authentication"]:
            suggestion = suggest_public_model(model)
            print(f"    → Suggested alternative: {suggestion}")
