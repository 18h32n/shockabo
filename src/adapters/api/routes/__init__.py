"""API route handlers for ARC Prize 2025 competition."""

from fastapi import APIRouter

from . import evaluation


def create_api_router() -> APIRouter:
    """Create the main API router with all route modules.

    Returns:
        APIRouter: Configured API router with all endpoints
    """
    # Create main API router
    api_router = APIRouter(prefix="/api/v1")

    # Include route modules
    api_router.include_router(evaluation.router, tags=["evaluation"])

    return api_router
