"""GPT-5 client implementation."""

import logging
from typing import Any

from src.infrastructure.components.budget_controller import BudgetController
from src.utils.secure_credentials import get_credential_manager

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GPT5Client(BaseLLMClient):
    """Client for GPT-5 model via OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gpt-5",
        api_endpoint: str = "https://api.openai.com/v1/chat/completions",
        budget_controller: BudgetController | None = None,
        **kwargs
    ):
        self.model_id = model_id
        kwargs['budget_controller'] = budget_controller
        super().__init__(api_key, api_endpoint, **kwargs)
        self.system_prompt = """You are GPT-5, OpenAI's most advanced language model, with exceptional capabilities in abstract reasoning and code generation for ARC (Abstraction and Reasoning Corpus) tasks.

Your advanced capabilities include:
1. Superior pattern recognition and abstraction
2. Multi-step reasoning and planning
3. Advanced algorithmic problem solving
4. Grid transformation expertise
5. Optimization of complex logical sequences

Key requirements:
1. Generate highly efficient, elegant Python code
2. Leverage numpy for optimal grid operations
3. Implement sophisticated pattern analysis
4. Provide detailed reasoning in comments
5. Optimize for both correctness and performance
6. Consider edge cases and robustness

Output format:
```python
def solve(input_grid):
    # Detailed multi-step reasoning process
    # explaining the transformation logic
    return output_grid
```"""

    def get_name(self) -> str:
        """Get provider name."""
        return "GPT-5"

    def _get_api_key_from_credentials(self) -> str:
        """Get OpenAI API key from credentials."""
        cred_manager = get_credential_manager()
        return cred_manager.get_credential("OPENAI_API_KEY")

    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare GPT-5-specific request."""
        return {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "stop": kwargs.get("stop", ["```\n\n", "\n\n\n"]),
            # GPT-5 specific parameters for enhanced reasoning
            "reasoning_effort": kwargs.get("reasoning_effort", "high"),
            "response_format": {"type": "text"}
        }

    async def _parse_response(
        self,
        response_data: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Parse OpenAI response format."""
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

    async def generate_program(
        self,
        task_description: str,
        examples: str,
        max_tokens: int = 8192,
        temperature: float = 0.2
    ) -> tuple[str, int, int]:
        """Generate a program for an ARC task using GPT-5's advanced reasoning."""
        prompt = f"""Analyze this ARC task with your advanced reasoning capabilities and generate an optimal solution.

Task Description:
{task_description}

Examples:
{examples}

Apply multi-step reasoning to:
1. Identify all possible patterns and transformation rules
2. Consider edge cases and variations
3. Generate the most robust and efficient solution
4. Validate your approach against the examples

Generate a comprehensive Python solution with detailed reasoning."""

        return await self.generate(prompt, max_tokens, temperature)

    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        # GPT-5 costs $1.25 per million input tokens, $10.00 per million output tokens
        # Estimate tokens as ~4 chars per token
        estimated_input_tokens = prompt_length / 4
        input_cost = (estimated_input_tokens / 1_000_000) * 1.25
        output_cost = (max_tokens / 1_000_000) * 10.00
        return input_cost + output_cost

    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        # GPT-5: $1.25/M input, $10.00/M output
        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 10.00
        return input_cost + output_cost
