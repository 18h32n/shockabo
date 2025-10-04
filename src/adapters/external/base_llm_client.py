"""Base class for LLM provider clients."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from src.infrastructure.components.budget_controller import (
    BudgetController,
    BudgetExceededException,
)
from src.utils.secure_credentials import get_credential_manager

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM provider implementations."""

    def __init__(
        self,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        budget_controller: BudgetController | None = None
    ):
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None
        self.budget_controller = budget_controller

        # Get API key from secure credentials if not provided
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._get_api_key_from_credentials()

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass

    @abstractmethod
    def _get_api_key_from_credentials(self) -> str:
        """Get API key from secure credentials."""
        pass

    @abstractmethod
    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare request payload for the specific provider."""
        pass

    @abstractmethod
    async def _parse_response(
        self,
        response_data: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Parse provider-specific response format."""
        pass

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> tuple[str, int, int]:
        """Generate response from LLM. Returns (response, input_tokens, output_tokens)."""
        await self._ensure_session()

        # Prepare request
        request_data = await self._prepare_request(
            prompt, max_tokens, temperature, **kwargs
        )

        # Check budget before making request
        if self.budget_controller:
            estimated_cost = self._estimate_request_cost(len(prompt), max_tokens)
            if not self.budget_controller.can_afford_request(self.get_name(), estimated_cost):
                raise BudgetExceededException(
                    f"Request to {self.get_name()} would exceed budget. "
                    f"Estimated cost: ${estimated_cost:.4f}"
                )

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)

                # Make request
                headers = self._get_headers()

                async with self._session.post(
                    self.api_endpoint,
                    json=request_data,
                    headers=headers
                ) as response:
                    response_data = await response.json()

                    if response.status != 200:
                        error_msg = response_data.get("error", {}).get("message", str(response_data))
                        raise Exception(f"API error ({response.status}): {error_msg}")

                    # Parse response
                    response_text, input_tokens, output_tokens = await self._parse_response(response_data)

                    # Track actual cost after successful response
                    if self.budget_controller:
                        actual_cost = self._calculate_actual_cost(input_tokens, output_tokens)
                        self.budget_controller.track_usage(
                            self.get_name(),
                            actual_cost,
                            input_tokens,
                            output_tokens
                        )

                    return response_text, input_tokens, output_tokens

            except TimeoutError:
                last_error = Exception(f"Request timed out after {self.timeout}s")
                logger.warning(f"Timeout on attempt {attempt + 1} for {self.get_name()}")
            except Exception as e:
                last_error = e
                logger.warning(f"Error on attempt {attempt + 1} for {self.get_name()}: {e}")

        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")

    def _get_headers(self) -> dict[str, str]:
        """Get common headers for API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    @abstractmethod
    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        pass

    @abstractmethod
    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class OpenRouterClient(BaseLLMClient):
    """Base client for OpenRouter-based models."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        api_endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        **kwargs
    ):
        self.model_id = model_id
        super().__init__(api_key, api_endpoint, **kwargs)

    def _get_api_key_from_credentials(self) -> str:
        """Get OpenRouter API key from credentials."""
        cred_manager = get_credential_manager()
        return cred_manager.get_credential("OPENROUTER_API_KEY")

    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare OpenRouter request."""
        return {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0)
        }

    async def _parse_response(
        self,
        response_data: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Parse OpenRouter response."""
        # Extract generated text
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")

        response_text = choices[0]["message"]["content"]

        # Extract token usage
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return response_text, input_tokens, output_tokens
