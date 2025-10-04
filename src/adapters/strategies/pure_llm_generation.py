"""
Pure LLM generation strategy for evolution offspring.

Generates new programs entirely via LLM synthesis without genetic operators.
Used by bandit controller for high-quality but expensive program generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from src.adapters.strategies.evolution_engine import Individual
from src.adapters.strategies.hybrid_initialization import (
    HybridLLMProgramGenerator,
    LLMProgramPrompt,
)
from src.domain.models import ARCTask

logger = logging.getLogger(__name__)


class PureLLMGenerationStrategy:
    """
    Pure LLM-based offspring generation for evolution.

    Uses LLM to generate entirely new programs based on parent characteristics
    and task requirements, without traditional genetic operators.
    """

    def __init__(self, smart_model_router: Any | None = None):
        """
        Initialize pure LLM generation strategy.

        Args:
            smart_model_router: Router for LLM selection (optional)
        """
        self.smart_model_router = smart_model_router
        self.llm_generator = (
            HybridLLMProgramGenerator(smart_model_router) if smart_model_router else None
        )

    async def generate_offspring(
        self, parent1: Individual, parent2: Individual, task: ARCTask
    ) -> list[Individual]:
        """
        Generate offspring using pure LLM synthesis.

        Analyzes parent programs and generates new variations using LLM,
        inspired by parent characteristics but not directly derived from them.

        Args:
            parent1: First parent individual
            parent2: Second parent individual
            task: ARC task being solved

        Returns:
            List of 2 offspring individuals
        """
        if not self.llm_generator:
            logger.warning("No LLM generator available, returning clones")
            return [self._clone_individual(parent1), self._clone_individual(parent2)]

        try:
            parent_analysis = self._analyze_parents(parent1, parent2)

            self._create_synthesis_prompt(task, parent_analysis)

            programs = await self.llm_generator.generate_programs(
                task, num_programs=2, diversity_level="medium"
            )

            offspring = []
            for program_ops in programs[:2]:
                individual = Individual(operations=program_ops, fitness=0.0)
                individual.metadata["generation_strategy"] = "pure_llm"
                individual.metadata["parent_influence"] = parent_analysis
                offspring.append(individual)

            if len(offspring) < 2:
                offspring.append(self._clone_individual(parent1))

            return offspring[:2]

        except Exception as e:
            logger.error(f"LLM generation failed: {e}, falling back to clones")
            return [self._clone_individual(parent1), self._clone_individual(parent2)]

    def _analyze_parents(self, parent1: Individual, parent2: Individual) -> dict[str, Any]:
        """
        Analyze parent characteristics for LLM guidance.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Dictionary of parent characteristics
        """
        analysis = {
            "parent1_length": len(parent1.operations),
            "parent2_length": len(parent2.operations),
            "parent1_fitness": parent1.fitness,
            "parent2_fitness": parent2.fitness,
            "parent1_operations": [op.get_name() if hasattr(op, 'get_name') else "unknown" for op in parent1.operations],
            "parent2_operations": [op.get_name() if hasattr(op, 'get_name') else "unknown" for op in parent2.operations],
            "common_operations": self._find_common_operations(parent1, parent2),
            "best_parent_ops": (
                parent1.operations if parent1.fitness > parent2.fitness else parent2.operations
            ),
        }
        return analysis

    def _find_common_operations(self, parent1: Individual, parent2: Individual) -> list[str]:
        """Find operations present in both parents."""
        ops1 = {op.get_name() if hasattr(op, 'get_name') else str(type(op).__name__) for op in parent1.operations}
        ops2 = {op.get_name() if hasattr(op, 'get_name') else str(type(op).__name__) for op in parent2.operations}
        return list(ops1 & ops2)

    def _create_synthesis_prompt(
        self, task: ARCTask, parent_analysis: dict[str, Any]
    ) -> LLMProgramPrompt:
        """
        Create LLM prompt for offspring synthesis.

        Args:
            task: ARC task
            parent_analysis: Parent characteristics

        Returns:
            Structured LLM prompt
        """
        input_examples = [self._grid_to_string(example.input) for example in task.train[:2]]
        output_examples = [self._grid_to_string(example.output) for example in task.train[:2]]

        available_ops = [
            "rotate",
            "flip",
            "translate",
            "crop",
            "pad",
            "color_replace",
            "color_map",
            "flood_fill",
            "pattern_match",
            "overlay",
            "extract_region",
        ]

        constraints = [
            f"Generate programs similar in style to parents (lengths: {parent_analysis['parent1_length']}, {parent_analysis['parent2_length']})",
            f"Consider using operations: {', '.join(parent_analysis['common_operations'][:5])}",
            f"Target fitness: {max(parent_analysis['parent1_fitness'], parent_analysis['parent2_fitness']):.2f}+",
            "Prefer innovative variations over exact parent reproduction",
        ]

        return LLMProgramPrompt(
            task_description=f"Transform input grids to output grids (Task: {task.task_id})",
            input_examples=input_examples,
            output_examples=output_examples,
            constraints=constraints,
            available_operations=available_ops,
            program_style="diverse",
        )

    def _grid_to_string(self, grid: list[list[int]]) -> str:
        """Convert grid to string representation."""
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    def _clone_individual(self, individual: Individual) -> Individual:
        """Create a clone of an individual."""
        clone = Individual(operations=individual.operations.copy(), fitness=individual.fitness)
        clone.metadata = individual.metadata.copy()
        clone.metadata["generation_strategy"] = "clone"
        return clone
