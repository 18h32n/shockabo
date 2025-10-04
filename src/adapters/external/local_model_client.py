"""Falcon Mamba 7B local model client implementation."""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any

from src.infrastructure.components.budget_controller import BudgetController
from src.utils.secure_credentials import get_credential_manager

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class LocalModelClient(BaseLLMClient):
    """Client for Falcon Mamba 7B running locally via Ollama or similar."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "falcon-mamba:7b",
        api_endpoint: str = "http://localhost:11434/api/generate",
        budget_controller: BudgetController | None = None,
        **kwargs
    ):
        self.model_id = model_id
        self.local_mode = kwargs.get("local_mode", "ollama")  # ollama, transformers, or custom
        kwargs['budget_controller'] = budget_controller
        super().__init__(api_key, api_endpoint, **kwargs)
        self.system_prompt = """You are Falcon Mamba 7B, a state-space model optimized for efficient sequence processing and code generation in ARC (Abstraction and Reasoning Corpus) tasks.

Your specialized capabilities:
1. Efficient sequential pattern processing
2. Memory-efficient grid transformations
3. Fast inference for iterative problem solving
4. Lightweight but accurate code generation
5. Local processing with privacy preservation

Key requirements:
1. Generate concise, efficient Python code
2. Use numpy for grid operations
3. Focus on clear, direct transformation logic
4. Minimize computational complexity
5. Optimize for local execution speed

Output format:
```python
def solve(input_grid):
    # Efficient transformation logic
    return output_grid
```"""

    def get_name(self) -> str:
        """Get provider name."""
        return "Falcon Mamba 7B (Local)"

    def _get_api_key_from_credentials(self) -> str:
        """Get API key from credentials (may not be needed for local models)."""
        try:
            cred_manager = get_credential_manager()
            return cred_manager.get_credential("LOCAL_MODEL_API_KEY", required=False) or ""
        except Exception:
            return ""

    def _get_headers(self) -> dict[str, str]:
        """Get headers for local model API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _prepare_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> dict[str, Any]:
        """Prepare request for local model."""
        # Format prompt with system instructions
        formatted_prompt = f"{self.system_prompt}\n\nTask:\n{prompt}"

        if self.local_mode == "ollama":
            return {
                "model": self.model_id,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_k": kwargs.get("top_k", 40),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": max_tokens,
                    "stop": kwargs.get("stop", ["```\n\n", "\n\n\n", "<|end|>"])
                }
            }
        else:
            # Generic local API format
            return {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": kwargs.get("top_k", 40),
                "top_p": kwargs.get("top_p", 0.9),
                "stop": kwargs.get("stop", ["```\n\n", "\n\n\n"])
            }

    async def _parse_response(
        self,
        response_data: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Parse local model response format."""
        if self.local_mode == "ollama":
            # Ollama response format
            response_text = response_data.get("response", "")

            # Ollama doesn't always provide token counts, estimate them
            input_tokens = len(response_data.get("context", [])) or 0
            output_tokens = len(response_text.split()) * 1.3  # Rough estimation

            return response_text, int(input_tokens), int(output_tokens)
        else:
            # Generic format similar to OpenAI
            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            response_text = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")

            # Extract token usage if available
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            return response_text, input_tokens, output_tokens

    async def _run_local_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> tuple[str, int, int]:
        """Run inference using local subprocess (fallback method)."""
        try:
            # Create temporary files for input/output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                input_data = {
                    "model": self.model_id,
                    "prompt": f"{self.system_prompt}\n\nTask:\n{prompt}",
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                json.dump(input_data, f)
                input_file = f.name

            # Run local model via subprocess (example for custom setup)
            cmd = [
                "python", "-m", "transformers.models.mamba.modeling_mamba",
                "--model", self.model_id,
                "--input", input_file,
                "--output", f"{input_file}.out"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )

            if result.returncode != 0:
                raise Exception(f"Local model execution failed: {result.stderr}")

            # Read output
            with open(f"{input_file}.out") as f:
                output_data = json.load(f)

            response_text = output_data.get("response", "")
            input_tokens = output_data.get("input_tokens", 0)
            output_tokens = output_data.get("output_tokens", 0)

            # Cleanup
            os.unlink(input_file)
            os.unlink(f"{input_file}.out")

            return response_text, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            raise Exception(f"Local model inference error: {e}") from e

    async def generate_program(
        self,
        task_description: str,
        examples: str,
        max_tokens: int = 2048,
        temperature: float = 0.5
    ) -> tuple[str, int, int]:
        """Generate a program for an ARC task using local Falcon Mamba model."""
        prompt = f"""Analyze this ARC task and generate an efficient local solution.

Task Description:
{task_description}

Examples:
{examples}

Generate a lightweight, efficient Python solution optimized for local execution and quick inference."""

        return await self.generate(prompt, max_tokens, temperature)

    async def check_model_availability(self) -> bool:
        """Check if the local model is available and running."""
        try:
            await self._ensure_session()

            # Test with a simple request
            test_data = {"model": self.model_id, "prompt": "Hello", "stream": False}
            headers = self._get_headers()

            async with self._session.post(
                self.api_endpoint,
                json=test_data,
                headers=headers
            ) as response:
                return response.status == 200

        except Exception as e:
            logger.warning(f"Local model availability check failed: {e}")
            return False

    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        # Local model has no cost
        return 0.0

    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        # Local model has no cost
        return 0.0
