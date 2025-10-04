"""Qwen2.5-Coder client implementation."""

import logging
from typing import Any

from src.infrastructure.components.budget_controller import BudgetController

from .base_llm_client import OpenRouterClient

logger = logging.getLogger(__name__)


class QwenClient(OpenRouterClient):
    """Client for Qwen2.5-Coder model via OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "qwen/qwen-2.5-coder-32b-instruct",
        budget_controller: BudgetController | None = None,
        **kwargs
    ):
        kwargs['budget_controller'] = budget_controller
        super().__init__(model_id, api_key, **kwargs)
        self.system_prompt = """You are Qwen, a code generation expert specializing in ARC (Abstraction and Reasoning Corpus) tasks.

Your goal is to generate Python programs that solve ARC puzzles by analyzing input-output patterns.

Key requirements:
1. Generate clean, efficient Python code
2. Use numpy for grid operations
3. Include clear comments explaining the logic
4. Focus on pattern recognition and transformation rules
5. Keep solutions concise but complete

Output format:
```python
def solve(input_grid):
    # Your solution here
    return output_grid
```"""

    def get_name(self) -> str:
        """Get provider name."""
        return "Qwen2.5-Coder"

    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        # Qwen costs $0.15 per million tokens (both input and output)
        # Estimate tokens as ~4 chars per token
        estimated_input_tokens = prompt_length / 4
        estimated_total_tokens = estimated_input_tokens + max_tokens
        cost_per_token = 0.15 / 1_000_000
        return estimated_total_tokens * cost_per_token

    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        # Qwen: $0.15/M tokens for both input and output
        total_tokens = input_tokens + output_tokens
        cost_per_token = 0.15 / 1_000_000
        return total_tokens * cost_per_token

    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare Qwen-specific request with system prompt."""
        # Format prompt for code generation
        formatted_prompt = f"{self.system_prompt}\n\nTask:\n{prompt}"

        # Use base class method with formatted prompt
        request_data = await super()._prepare_request(
            formatted_prompt,
            max_tokens,
            temperature,
            **kwargs
        )

        # Add Qwen-specific parameters
        request_data.update({
            "top_k": kwargs.get("top_k", 40),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stop": ["```\n\n", "\n\n\n"],  # Stop at end of code block
        })

        return request_data

    async def generate_program(
        self,
        task_description: str,
        examples: str,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> tuple[str, int, int]:
        """Generate a program for an ARC task."""
        prompt = f"""Given the following ARC task, generate a Python function that solves it.

Task Description:
{task_description}

Examples:
{examples}

Generate a clean, efficient solution that handles the pattern transformation."""

        return await self.generate(prompt, max_tokens, temperature)
