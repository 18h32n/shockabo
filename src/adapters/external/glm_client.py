"""GLM-4.5 client implementation."""

import logging
from typing import Any

from src.infrastructure.components.budget_controller import BudgetController

from .base_llm_client import OpenRouterClient

logger = logging.getLogger(__name__)


class GLMClient(OpenRouterClient):
    """Client for GLM-4.5 model via OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "zhipuai/glm-4-plus",
        budget_controller: BudgetController | None = None,
        **kwargs
    ):
        kwargs['budget_controller'] = budget_controller
        super().__init__(model_id, api_key, **kwargs)
        self.system_prompt = """You are GLM-4.5, a large language model developed by Zhipu AI, specialized in logical reasoning and code generation for ARC (Abstraction and Reasoning Corpus) tasks.

Your strengths include:
1. Strong logical reasoning and pattern analysis
2. Efficient algorithmic thinking
3. Grid-based problem solving
4. Mathematical pattern recognition
5. Systematic approach to complex puzzles

Key requirements:
1. Generate clean, well-structured Python code
2. Use numpy for efficient grid operations
3. Implement clear, logical transformation rules
4. Provide comprehensive comments explaining the reasoning
5. Focus on correctness and efficiency

Output format:
```python
def solve(input_grid):
    # Step-by-step logical explanation
    # of the transformation process
    return output_grid
```"""

    def get_name(self) -> str:
        """Get provider name."""
        return "GLM-4.5"

    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare GLM-specific request with system prompt."""
        # Format prompt with GLM's preferred structure
        formatted_prompt = f"{self.system_prompt}\n\n任务 (Task):\n{prompt}"

        # Use base class method with formatted prompt
        request_data = await super()._prepare_request(
            formatted_prompt,
            max_tokens,
            temperature,
            **kwargs
        )

        # Add GLM-specific parameters for better reasoning
        request_data.update({
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            "do_sample": kwargs.get("do_sample", True),
            "stop": ["```\n\n", "\n\n\n", "任务完成", "Task completed"],
        })

        return request_data

    async def generate_program(
        self,
        task_description: str,
        examples: str,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> tuple[str, int, int]:
        """Generate a program for an ARC task using GLM's logical reasoning."""
        prompt = f"""请分析以下ARC任务并生成Python解决方案 (Please analyze the following ARC task and generate a Python solution):

任务描述 (Task Description):
{task_description}

示例 (Examples):
{examples}

请运用你的逻辑推理能力，识别模式转换规则，并实现一个高效准确的解决方案。
(Please use your logical reasoning capabilities to identify pattern transformation rules and implement an efficient and accurate solution.)"""

        return await self.generate(prompt, max_tokens, temperature)

    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        # GLM-4.5 costs $0.59 per million input tokens, $2.19 per million output tokens
        # Estimate tokens as ~4 chars per token
        estimated_input_tokens = prompt_length / 4
        input_cost = (estimated_input_tokens / 1_000_000) * 0.59
        output_cost = (max_tokens / 1_000_000) * 2.19
        return input_cost + output_cost

    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        # GLM-4.5: $0.59/M input, $2.19/M output
        input_cost = (input_tokens / 1_000_000) * 0.59
        output_cost = (output_tokens / 1_000_000) * 2.19
        return input_cost + output_cost
