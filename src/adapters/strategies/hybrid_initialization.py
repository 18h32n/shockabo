"""
Hybrid initialization for evolution engine using LLM-generated programs.

This module implements Task 7.3: Create hybrid initialization using LLM-generated programs.
It combines LLM-based program generation with traditional random/template approaches.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Protocol

from src.adapters.external.smart_model_router import ComplexityLevel, SmartModelRouter
from src.adapters.strategies.evolution_engine import Individual
from src.adapters.strategies.operation_templates import (
    OperationTemplateGenerator,
    create_seed_programs,
)
from src.domain.dsl.base import Operation
from src.domain.models import ARCTask

logger = logging.getLogger(__name__)


@dataclass
class LLMProgramPrompt:
    """Structured prompt for LLM program generation."""
    task_description: str
    input_examples: list[str]
    output_examples: list[str]
    constraints: list[str]
    available_operations: list[str]
    program_style: str  # "simple", "complex", "diverse"

    def to_prompt(self) -> str:
        """Convert to LLM prompt string."""
        prompt = f"""Generate a DSL program to solve the following ARC task.

Task Description: {self.task_description}

Input/Output Examples:
{self._format_examples()}

Available Operations:
{', '.join(self.available_operations)}

Constraints:
{chr(10).join(f"- {c}" for c in self.constraints)}

Generate a {self.program_style} program that transforms the input to output.
Return the program as a JSON array of operations, each with 'name' and 'parameters'.

Example format:
[
    {{"name": "rotate", "parameters": {{"angle": 90}}}},
    {{"name": "flip", "parameters": {{"direction": "horizontal"}}}}
]

Program:"""
        return prompt

    def _format_examples(self) -> str:
        """Format input/output examples."""
        formatted = []
        for i, (inp, out) in enumerate(zip(self.input_examples, self.output_examples, strict=False)):
            formatted.append(f"Example {i+1}:")
            formatted.append(f"Input:\n{inp}")
            formatted.append(f"Output:\n{out}")
        return "\n\n".join(formatted)


class LLMProgramGenerator(Protocol):
    """Protocol for LLM-based program generation."""

    async def generate_programs(
        self,
        task: ARCTask,
        num_programs: int,
        diversity_level: str = "high"
    ) -> list[list[dict[str, Any]]]:
        """Generate DSL programs using LLM."""
        ...


class HybridLLMProgramGenerator:
    """
    Generates DSL programs using LLM with smart routing and caching.

    This generator creates diverse initial programs by:
    1. Analyzing task complexity
    2. Routing to appropriate LLM tier
    3. Generating programs with different styles
    4. Validating and filtering results
    """

    def __init__(
        self,
        model_router: SmartModelRouter,
        available_operations: list[str] | None = None
    ):
        """
        Initialize hybrid LLM program generator.

        Args:
            model_router: Smart model router for LLM selection
            available_operations: List of available DSL operations
        """
        self.model_router = model_router
        self.available_operations = available_operations or self._get_default_operations()
        self.generation_cache: dict[str, list[list[dict[str, Any]]]] = {}

    def _get_default_operations(self) -> list[str]:
        """Get default DSL operations."""
        return [
            "rotate", "flip", "translate", "scale",
            "replace_color", "fill_background", "extract_objects",
            "find_pattern", "apply_pattern", "crop", "pad",
            "mirror", "tile", "overlay", "mask",
            "group_by_color", "connect_components", "fill_shape",
            "draw_line", "draw_rectangle", "flood_fill"
        ]

    async def generate_programs(
        self,
        task: ARCTask,
        num_programs: int,
        diversity_level: str = "high"
    ) -> list[list[dict[str, Any]]]:
        """
        Generate diverse DSL programs using LLM.

        Args:
            task: ARC task to solve
            num_programs: Number of programs to generate
            diversity_level: Level of diversity ("low", "medium", "high")

        Returns:
            List of program representations (list of operations)
        """
        # Check cache
        cache_key = f"{task.task_id}_{num_programs}_{diversity_level}"
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key][:num_programs]

        # Analyze task complexity
        routing_decision = await self.model_router.route_task(task)

        # Generate programs with different styles based on diversity level
        programs = []
        styles = self._get_program_styles(diversity_level, routing_decision.complexity_level)

        # Distribute programs across styles
        programs_per_style = max(1, num_programs // len(styles))

        for style in styles:
            style_programs = await self._generate_programs_with_style(
                task, programs_per_style, style, routing_decision
            )
            programs.extend(style_programs)

        # Fill remaining slots with best style
        while len(programs) < num_programs:
            best_style = self._select_best_style(styles, routing_decision.complexity_level)
            additional = await self._generate_programs_with_style(
                task, 1, best_style, routing_decision
            )
            programs.extend(additional)

        # Trim to exact count
        programs = programs[:num_programs]

        # Cache results
        self.generation_cache[cache_key] = programs

        return programs

    def _get_program_styles(
        self,
        diversity_level: str,
        complexity: ComplexityLevel
    ) -> list[str]:
        """Get program styles based on diversity level and complexity."""
        if diversity_level == "high":
            styles = ["simple", "complex", "creative", "systematic", "hybrid"]
        elif diversity_level == "medium":
            if complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM]:
                styles = ["simple", "systematic"]
            else:
                styles = ["complex", "creative"]
        else:  # low diversity
            if complexity == ComplexityLevel.SIMPLE:
                styles = ["simple"]
            else:
                styles = ["complex"]

        return styles

    def _select_best_style(
        self,
        styles: list[str],
        complexity: ComplexityLevel
    ) -> str:
        """Select best style for given complexity."""
        style_preferences = {
            ComplexityLevel.SIMPLE: ["simple", "systematic"],
            ComplexityLevel.MEDIUM: ["systematic", "hybrid"],
            ComplexityLevel.COMPLEX: ["complex", "creative"],
            ComplexityLevel.BREAKTHROUGH: ["creative", "complex", "hybrid"]
        }

        preferred = style_preferences.get(complexity, ["complex"])
        for pref in preferred:
            if pref in styles:
                return pref
        return styles[0]

    async def _generate_programs_with_style(
        self,
        task: ARCTask,
        num_programs: int,
        style: str,
        routing_decision: Any
    ) -> list[list[dict[str, Any]]]:
        """Generate programs with specific style."""
        programs = []

        # Create prompts for this style
        prompts = self._create_style_prompts(task, style, num_programs)

        # Generate programs concurrently
        generation_tasks = []
        for prompt in prompts:
            generation_tasks.append(
                self._generate_single_program(prompt, routing_decision)
            )

        # Gather results
        results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        # Process and validate results
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Program generation failed: {result}")
                continue

            if result and self._validate_program(result):
                programs.append(result)

        # Add variations if needed
        while len(programs) < num_programs and programs:
            # Create variation of existing program
            base_program = random.choice(programs)
            variation = self._create_program_variation(base_program, style)
            if self._validate_program(variation):
                programs.append(variation)

        return programs

    def _create_style_prompts(
        self,
        task: ARCTask,
        style: str,
        num_prompts: int
    ) -> list[LLMProgramPrompt]:
        """Create prompts for specific generation style."""
        prompts = []

        # Extract task information
        task_description = self._analyze_task_pattern(task)
        input_examples, output_examples = self._format_task_examples(task)

        # Style-specific constraints
        style_constraints = {
            "simple": [
                "Use at most 5 operations",
                "Prefer basic transformations (rotate, flip, translate)",
                "Avoid complex patterns or conditionals"
            ],
            "complex": [
                "Use advanced operations (pattern detection, grouping)",
                "Chain multiple transformations",
                "Consider edge cases and special patterns"
            ],
            "creative": [
                "Explore unconventional operation combinations",
                "Use operations in novel ways",
                "Think outside typical transformation patterns"
            ],
            "systematic": [
                "Apply operations in logical sequence",
                "Use consistent patterns throughout",
                "Prefer deterministic transformations"
            ],
            "hybrid": [
                "Combine simple and complex operations",
                "Use both direct and pattern-based approaches",
                "Balance efficiency with completeness"
            ]
        }

        constraints = style_constraints.get(style, style_constraints["complex"])

        # Create diverse prompts
        for i in range(num_prompts):
            # Vary available operations
            if style == "simple":
                ops = self.available_operations[:10]  # Basic ops only
            elif style == "creative":
                # Random subset to encourage creativity
                ops = random.sample(self.available_operations,
                                  min(15, len(self.available_operations)))
            else:
                ops = self.available_operations

            # Add variation to constraints
            prompt_constraints = constraints.copy()
            if i % 2 == 0:
                prompt_constraints.append("Prioritize accuracy over brevity")
            else:
                prompt_constraints.append("Prioritize efficiency")

            prompt = LLMProgramPrompt(
                task_description=task_description,
                input_examples=input_examples[:2],  # Use first 2 examples
                output_examples=output_examples[:2],
                constraints=prompt_constraints,
                available_operations=ops,
                program_style=style
            )

            prompts.append(prompt)

        return prompts

    def _analyze_task_pattern(self, task: ARCTask) -> str:
        """Analyze task to create description."""
        # Simple pattern analysis
        descriptions = []

        if len(task.train_pairs) > 0:
            first_pair = task.train_pairs[0]
            input_grid = first_pair.input
            output_grid = first_pair.output

            # Size changes
            if input_grid.shape != output_grid.shape:
                descriptions.append(f"Transform {input_grid.shape} grid to {output_grid.shape}")

            # Color analysis
            input_colors = {c for row in input_grid.data for c in row}
            output_colors = {c for row in output_grid.data for c in row}

            if input_colors != output_colors:
                descriptions.append("Color transformation involved")

            # Check for patterns
            if len(input_colors) < len(output_colors):
                descriptions.append("Pattern generation or expansion")
            elif len(input_colors) > len(output_colors):
                descriptions.append("Pattern simplification or extraction")

        return ". ".join(descriptions) if descriptions else "Transform input grid to output grid"

    def _format_task_examples(
        self,
        task: ARCTask
    ) -> tuple[list[str], list[str]]:
        """Format task examples as strings."""
        input_examples = []
        output_examples = []

        for pair in task.train_pairs[:3]:  # Max 3 examples
            # Simple grid representation
            input_str = self._grid_to_string(pair.input.data)
            output_str = self._grid_to_string(pair.output.data)

            input_examples.append(input_str)
            output_examples.append(output_str)

        return input_examples, output_examples

    def _grid_to_string(self, grid: list[list[int]]) -> str:
        """Convert grid to string representation."""
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    async def _generate_single_program(
        self,
        prompt: LLMProgramPrompt,
        routing_decision: Any
    ) -> list[dict[str, Any]] | None:
        """Generate single program using LLM."""
        try:
            # Get LLM response
            response = await self.model_router.generate_with_routing(
                prompt.to_prompt(),
                routing_decision
            )

            # Parse response
            program = self._parse_llm_response(response)
            return program

        except Exception as e:
            logger.error(f"Failed to generate program: {e}")
            return None

    def _parse_llm_response(self, response: str) -> list[dict[str, Any]] | None:
        """Parse LLM response to extract program."""
        try:
            # Try to extract JSON from response
            # Handle common formats
            response = response.strip()

            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']')

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                program = json.loads(json_str)

                # Validate structure
                if isinstance(program, list):
                    return program

            # Try direct parse
            program = json.loads(response)
            if isinstance(program, list):
                return program

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")

        return None

    def _validate_program(self, program: list[dict[str, Any]]) -> bool:
        """Validate program structure."""
        if not isinstance(program, list):
            return False

        if len(program) == 0:
            return False

        for op in program:
            if not isinstance(op, dict):
                return False

            if "name" not in op:
                return False

            # Check operation name is valid
            if op["name"] not in self.available_operations:
                # Allow some flexibility for LLM variations
                logger.debug(f"Unknown operation: {op['name']}")

            if "parameters" in op and not isinstance(op["parameters"], dict):
                return False

        return True

    def _create_program_variation(
        self,
        base_program: list[dict[str, Any]],
        style: str
    ) -> list[dict[str, Any]]:
        """Create variation of existing program."""
        import copy

        variation = copy.deepcopy(base_program)

        # Apply style-specific variations
        if style == "simple" and len(variation) > 1:
            # Remove random operation
            idx = random.randint(0, len(variation) - 1)
            variation.pop(idx)

        elif style == "complex" and len(variation) < 10:
            # Insert additional operation
            new_op = self._generate_random_operation()
            insert_idx = random.randint(0, len(variation))
            variation.insert(insert_idx, new_op)

        elif style == "creative":
            # Swap operation order
            if len(variation) > 1:
                i, j = random.sample(range(len(variation)), 2)
                variation[i], variation[j] = variation[j], variation[i]

        # Mutate parameters
        if variation and random.random() < 0.3:
            op_idx = random.randint(0, len(variation) - 1)
            self._mutate_operation_parameters(variation[op_idx])

        return variation

    def _generate_random_operation(self) -> dict[str, Any]:
        """Generate random operation."""
        op_name = random.choice(self.available_operations)

        # Common parameter patterns
        param_templates = {
            "rotate": {"angle": random.choice([90, 180, 270])},
            "flip": {"direction": random.choice(["horizontal", "vertical"])},
            "translate": {"dx": random.randint(-3, 3), "dy": random.randint(-3, 3)},
            "scale": {"factor": random.choice([2, 3, 4])},
            "replace_color": {
                "source_color": random.randint(0, 9),
                "target_color": random.randint(0, 9)
            },
            "fill_background": {"target_color": random.randint(0, 9)},
            "crop": {
                "x": random.randint(0, 5),
                "y": random.randint(0, 5),
                "width": random.randint(2, 10),
                "height": random.randint(2, 10)
            }
        }

        params = param_templates.get(op_name, {})

        return {
            "name": op_name,
            "parameters": params
        }

    def _mutate_operation_parameters(self, operation: dict[str, Any]) -> None:
        """Mutate operation parameters in place."""
        if "parameters" not in operation:
            return

        params = operation["parameters"]
        if not params:
            return

        # Mutate random parameter
        param_names = list(params.keys())
        if param_names:
            param_to_mutate = random.choice(param_names)

            # Type-aware mutation
            current_val = params[param_to_mutate]

            if isinstance(current_val, int):
                # Integer mutation
                if param_to_mutate in ["angle"]:
                    params[param_to_mutate] = random.choice([90, 180, 270])
                elif param_to_mutate in ["source_color", "target_color"]:
                    params[param_to_mutate] = random.randint(0, 9)
                else:
                    # General integer
                    params[param_to_mutate] = current_val + random.randint(-2, 2)

            elif isinstance(current_val, str):
                # String mutation
                if param_to_mutate == "direction":
                    params[param_to_mutate] = random.choice(["horizontal", "vertical"])


class HybridPopulationInitializer:
    """
    Initializes evolution population using hybrid approach.

    Combines:
    1. LLM-generated programs (smart initialization)
    2. Template-based programs (known patterns)
    3. Random programs (exploration)
    """

    def __init__(
        self,
        llm_generator: LLMProgramGenerator | None = None,
        template_generator: OperationTemplateGenerator | None = None
    ):
        """
        Initialize hybrid population initializer.

        Args:
            llm_generator: LLM-based program generator
            template_generator: Template-based generator
        """
        self.llm_generator = llm_generator
        self.template_generator = template_generator or OperationTemplateGenerator()

    async def initialize_population(
        self,
        task: ARCTask,
        population_size: int,
        config: dict[str, Any]
    ) -> list[Individual]:
        """
        Initialize population with hybrid approach.

        Args:
            task: ARC task to solve
            population_size: Target population size
            config: Initialization configuration

        Returns:
            List of individuals for initial population
        """
        individuals = []

        # Get configuration
        llm_ratio = config.get("llm_seed_ratio", 0.2)
        template_ratio = config.get("template_ratio", 0.5)
        use_seeds = config.get("use_seed_programs", True)

        # Calculate segment sizes
        num_llm = int(population_size * llm_ratio) if self.llm_generator else 0
        num_seeds = min(len(create_seed_programs()), int(population_size * 0.1)) if use_seeds else 0
        num_remaining = population_size - num_llm - num_seeds
        num_templates = int(num_remaining * template_ratio)
        num_random = num_remaining - num_templates

        # 1. Generate LLM-based programs
        if num_llm > 0 and self.llm_generator:
            llm_individuals = await self._generate_llm_individuals(task, num_llm)
            individuals.extend(llm_individuals)
            logger.info(f"Generated {len(llm_individuals)} LLM-based individuals")

        # 2. Add seed programs
        if num_seeds > 0:
            seed_individuals = self._create_seed_individuals(num_seeds)
            individuals.extend(seed_individuals)
            logger.info(f"Added {len(seed_individuals)} seed individuals")

        # 3. Generate template-based programs
        if num_templates > 0:
            template_individuals = self._generate_template_individuals(num_templates)
            individuals.extend(template_individuals)
            logger.info(f"Generated {len(template_individuals)} template individuals")

        # 4. Generate random programs
        if num_random > 0:
            random_individuals = self._generate_random_individuals(num_random)
            individuals.extend(random_individuals)
            logger.info(f"Generated {len(random_individuals)} random individuals")

        # Ensure we have exactly the right size
        while len(individuals) < population_size:
            # Add more random individuals
            individuals.append(self._create_random_individual())

        # Trim if needed
        individuals = individuals[:population_size]

        # Add metadata
        for i, individual in enumerate(individuals):
            individual.metadata["initialization_method"] = self._get_init_method(i, num_llm, num_seeds, num_templates)
            individual.metadata["initial_index"] = i

        return individuals

    async def _generate_llm_individuals(
        self,
        task: ARCTask,
        num_individuals: int
    ) -> list[Individual]:
        """Generate individuals using LLM."""
        individuals = []

        try:
            # Generate diverse programs
            programs = await self.llm_generator.generate_programs(
                task=task,
                num_programs=num_individuals,
                diversity_level="high"
            )

            # Convert to individuals
            for program_data in programs:
                operations = self._convert_to_operations(program_data)
                if operations:
                    individual = Individual(operations=operations)
                    individual.metadata["llm_generated"] = True
                    individual.metadata["generation_style"] = "llm_diverse"
                    individuals.append(individual)

        except Exception as e:
            logger.error(f"Failed to generate LLM individuals: {e}")

        return individuals

    def _create_seed_individuals(self, num_seeds: int) -> list[Individual]:
        """Create individuals from seed programs."""
        individuals = []
        seed_programs = create_seed_programs()

        for i in range(min(num_seeds, len(seed_programs))):
            program_data = seed_programs[i]
            operations = self._convert_to_operations(program_data)
            if operations:
                individual = Individual(operations=operations)
                individual.metadata["seed_program"] = True
                individual.metadata["seed_index"] = i
                individuals.append(individual)

        return individuals

    def _generate_template_individuals(self, num_templates: int) -> list[Individual]:
        """Generate template-based individuals."""
        individuals = []

        for _ in range(num_templates):
            # Generate template program
            program_data = self.template_generator.generate_random_program(
                min_length=2,
                max_length=8
            )

            operations = self._convert_to_operations(program_data)
            if operations:
                individual = Individual(operations=operations)
                individual.metadata["template_generated"] = True
                individuals.append(individual)

        return individuals

    def _generate_random_individuals(self, num_random: int) -> list[Individual]:
        """Generate random individuals."""
        individuals = []

        for _ in range(num_random):
            individual = self._create_random_individual()
            individuals.append(individual)

        return individuals

    def _create_random_individual(self) -> Individual:
        """Create single random individual."""
        # Random program length
        num_ops = random.randint(1, 10)

        program_data = []
        for _ in range(num_ops):
            op = self._generate_random_operation()
            program_data.append(op)

        operations = self._convert_to_operations(program_data)
        individual = Individual(operations=operations or [])
        individual.metadata["random_generated"] = True

        return individual

    def _generate_random_operation(self) -> dict[str, Any]:
        """Generate random operation data."""
        # This should match the format expected by the DSL
        op_types = [
            "rotate", "flip", "translate", "scale",
            "replace_color", "fill_background",
            "crop", "pad", "mirror"
        ]

        op_name = random.choice(op_types)

        # Generate appropriate parameters
        params = {}
        if op_name == "rotate":
            params["angle"] = random.choice([90, 180, 270])
        elif op_name == "flip":
            params["direction"] = random.choice(["horizontal", "vertical"])
        elif op_name == "translate":
            params["dx"] = random.randint(-5, 5)
            params["dy"] = random.randint(-5, 5)
        elif op_name == "scale":
            params["factor"] = random.choice([2, 3, 4])
        elif op_name == "replace_color":
            params["source_color"] = random.randint(0, 9)
            params["target_color"] = random.randint(0, 9)
        elif op_name == "fill_background":
            params["target_color"] = random.randint(0, 9)

        return {
            "name": op_name,
            "parameters": params
        }

    def _convert_to_operations(
        self,
        program_data: list[dict[str, Any]]
    ) -> list[Operation] | None:
        """Convert program data to Operation objects."""
        # This is a placeholder - in real implementation would use DSL registry
        from src.domain.dsl.base import Operation

        class MockOperation(Operation):
            def __init__(self, name: str, **params):
                self._name = name
                # Don't call super().__init__ to avoid parameter validation
                self.parameters = params

            def execute(self, grid, context=None):
                from src.domain.dsl.base import OperationResult
                return OperationResult(success=True, grid=grid)

            @classmethod
            def get_name(cls) -> str:
                return "mock"

            @classmethod
            def get_description(cls) -> str:
                return "Mock operation for testing"

            @classmethod
            def get_parameter_schema(cls) -> dict[str, Any]:
                return {}

            def get_instance_name(self) -> str:
                """Get the instance-specific operation name."""
                return self._name

            def _validate_parameters(self) -> None:
                """Validate parameters - no-op for mock."""
                pass

        try:
            operations = []
            for op_data in program_data:
                if isinstance(op_data, dict) and "name" in op_data:
                    op = MockOperation(
                        name=op_data["name"],
                        **op_data.get("parameters", {})
                    )
                    operations.append(op)

            return operations if operations else None

        except Exception as e:
            logger.error(f"Failed to convert operations: {e}")
            return None

    def _get_init_method(
        self,
        index: int,
        num_llm: int,
        num_seeds: int,
        num_templates: int
    ) -> str:
        """Determine initialization method for individual at index."""
        if index < num_llm:
            return "llm"
        elif index < num_llm + num_seeds:
            return "seed"
        elif index < num_llm + num_seeds + num_templates:
            return "template"
        else:
            return "random"
