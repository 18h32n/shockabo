"""Gemini 2.5 Flash client implementation."""

import logging
from typing import Any

from src.infrastructure.components.budget_controller import BudgetController
from src.utils.secure_credentials import get_credential_manager

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Client for Gemini 2.5 Flash model via Google AI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gemini-2.5-flash",
        api_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        budget_controller: BudgetController | None = None,
        **kwargs
    ):
        self.model_id = model_id
        kwargs['budget_controller'] = budget_controller
        super().__init__(api_key, api_endpoint, **kwargs)
        self.system_prompt = """You are Gemini 2.5 Flash, an advanced AI optimized for code generation and pattern recognition in ARC (Abstraction and Reasoning Corpus) tasks.

Your expertise includes:
1. Rapid pattern analysis and recognition
2. Efficient Python code generation
3. Grid-based transformations and manipulations
4. Logical reasoning for complex puzzles
5. Concise yet comprehensive solutions

Key requirements:
- Generate clean, efficient Python code using numpy
- Focus on identifying transformation rules and patterns
- Include clear, descriptive comments
- Optimize for both speed and accuracy
- Keep solutions minimal but complete

Output format:
```python
def solve(input_grid):
    # Clear explanation of the transformation logic
    return output_grid
```"""

    def get_name(self) -> str:
        """Get provider name."""
        return "Gemini 2.5 Flash"

    def _get_api_key_from_credentials(self) -> str:
        """Get Google AI API key from credentials."""
        cred_manager = get_credential_manager()
        return cred_manager.get_credential("GOOGLE_AI_API_KEY")

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Google AI API requests."""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare Gemini-specific request."""
        # Format prompt with system instructions
        formatted_prompt = f"{self.system_prompt}\n\nTask:\n{prompt}"

        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": formatted_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topK": kwargs.get("top_k", 40),
                "topP": kwargs.get("top_p", 0.95),
                "maxOutputTokens": max_tokens,
                "stopSequences": kwargs.get("stop", ["```\n\n", "\n\n\n"]),
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }

    async def _parse_response(
        self,
        response_data: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Parse Gemini response format."""
        # Extract generated text
        candidates = response_data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in response")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise ValueError("No parts in response content")

        response_text = parts[0].get("text", "")

        # Extract token usage
        usage_metadata = response_data.get("usageMetadata", {})
        input_tokens = usage_metadata.get("promptTokenCount", 0)
        output_tokens = usage_metadata.get("candidatesTokenCount", 0)

        return response_text, input_tokens, output_tokens

    async def generate_program(
        self,
        task_description: str,
        examples: str,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> tuple[str, int, int]:
        """Generate a program for an ARC task using Gemini's rapid analysis."""
        prompt = f"""Analyze this ARC task and generate an optimal Python solution.

Task Description:
{task_description}

Examples:
{examples}

Use your pattern recognition capabilities to identify the transformation rule and implement an efficient solution."""

        return await self.generate(prompt, max_tokens, temperature)

    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        # Gemini 2.5 Flash costs $0.31 per million input tokens, $2.62 per million output tokens
        # Estimate tokens as ~4 chars per token
        estimated_input_tokens = prompt_length / 4
        input_cost = (estimated_input_tokens / 1_000_000) * 0.31
        output_cost = (max_tokens / 1_000_000) * 2.62
        return input_cost + output_cost

    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        # Gemini 2.5 Flash: $0.31/M input, $2.62/M output
        input_cost = (input_tokens / 1_000_000) * 0.31
        output_cost = (output_tokens / 1_000_000) * 2.62
        return input_cost + output_cost
