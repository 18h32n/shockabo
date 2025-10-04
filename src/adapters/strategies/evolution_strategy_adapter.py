"""
Evolution strategy adapter for integration with evaluation framework.

This adapter wraps the EvolutionEngine to provide a standardized interface
compatible with the evaluation service, returning proper EvaluationResult objects.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.adapters.strategies.evolution_engine import EvolutionEngine, Individual
from src.adapters.strategies.sandbox_executor import SandboxExecutor
from src.domain.evaluation_models import StrategyType
from src.domain.models import ARCTask
from src.domain.services.dsl_engine import DSLEngine
from src.domain.services.evaluation_service import EvaluationResult
from src.domain.services.strategy_interface import BaseStrategy
from src.infrastructure.config import GeneticAlgorithmConfig

logger = structlog.get_logger(__name__)


@dataclass
class EvolutionSubmission:
    """Represents a submission from the evolution strategy."""

    individual: Individual
    predicted_output: list[list[int]]
    confidence: float
    generation: int
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EvolutionStrategyAdapter(BaseStrategy):
    """
    Adapter to integrate EvolutionEngine with the evaluation framework.

    Provides standardized interface for task solving and result formatting
    compatible with EvaluationService expectations.
    """

    def __init__(
        self,
        config: GeneticAlgorithmConfig,
        dsl_engine: DSLEngine,
        evaluation_service,
        sandbox_executor: SandboxExecutor | None = None
    ):
        """
        Initialize the evolution strategy adapter.

        Args:
            config: Genetic algorithm configuration
            dsl_engine: DSL engine for program execution
            evaluation_service: Evaluation service for result formatting
            sandbox_executor: Optional sandbox for safe execution
        """
        super().__init__(evaluation_service)
        self.config = config
        self.dsl_engine = dsl_engine
        self.sandbox_executor = sandbox_executor or SandboxExecutor()
        self.evolution_engine = None
        self.logger = structlog.get_logger(__name__).bind(strategy="evolution")

    async def solve_task(
        self,
        task: ARCTask,
        max_attempts: int = 2,
        experiment_name: str | None = None
    ) -> EvaluationResult:
        """
        Solve an ARC task using evolutionary search.

        Args:
            task: The ARC task to solve
            max_attempts: Maximum number of solution attempts (default 2)
            experiment_name: Optional experiment name for tracking

        Returns:
            EvaluationResult compatible with evaluation framework
        """
        start_time = time.time()

        # Initialize evolution engine for this task
        self.evolution_engine = EvolutionEngine(
            config=self.config,
            dsl_engine=self.dsl_engine,
            experiment_orchestrator=None  # Will be set if needed
        )

        try:
            # Run evolution
            best_individual, evolution_stats = await self.evolution_engine.evolve(
                task=task,
                experiment_name=experiment_name or f"evolution_{task.task_id}"
            )

            # Get top N individuals for multiple attempts
            top_individuals = self._get_top_individuals(
                self.evolution_engine.all_individuals_history,
                max_attempts
            )

            # Execute programs and prepare submissions
            submissions = []
            for idx, individual in enumerate(top_individuals[:max_attempts]):
                submission = await self._execute_individual(task, individual, idx + 1)
                if submission:
                    submissions.append(submission)

            # Convert to evaluation format
            predictions = [
                (sub.predicted_output, sub.confidence)
                for sub in submissions
            ]

            # Prepare evolution-specific metadata
            metadata = {
                "strategy_type": StrategyType.EVOLUTION.value,
                "total_programs_generated": evolution_stats.get("total_programs_generated", 0),
                "generations_run": evolution_stats.get("generations", 0),
                "final_best_fitness": evolution_stats.get("best_fitness", 0.0),
                "convergence_generation": evolution_stats.get("convergence_generation"),
                "diversity_metrics": evolution_stats.get("final_diversity_metrics", {}),
                "mutation_success_rate": evolution_stats.get("mutation_success_rate", 0.0),
                "crossover_success_rate": evolution_stats.get("crossover_success_rate", 0.0),
                "evolution_time_seconds": time.time() - start_time,
                "genealogy_depth": evolution_stats.get("max_genealogy_depth", 0),
                "successful_mutations": evolution_stats.get("successful_mutations", []),
                "program_lengths": {
                    "min": evolution_stats.get("min_program_length", 0),
                    "max": evolution_stats.get("max_program_length", 0),
                    "avg": evolution_stats.get("avg_program_length", 0)
                }
            }

            # Create unified submission format (Task 5.4)
            submission_data = self.create_submission_format(
                task_id=task.task_id,
                predictions=[sub.predicted_output for sub in submissions],
                confidence_scores=[sub.confidence for sub in submissions],
                metadata=metadata
            )

            # Add submission format to metadata
            metadata["submission_format"] = submission_data

            # Use evaluation service to create proper result
            result = self.evaluation_service.evaluate_task_with_attempts(
                task=task,
                predictions=predictions,
                strategy_used="evolution",
                metadata=metadata
            )

            self.logger.info(
                "evolution_task_completed",
                task_id=task.task_id,
                num_attempts=len(submissions),
                best_accuracy=result.final_accuracy,
                total_programs=metadata["total_programs_generated"],
                evolution_time=metadata["evolution_time_seconds"]
            )

            return result

        except Exception as e:
            self.logger.error(
                "evolution_strategy_error",
                task_id=task.task_id,
                error=str(e),
                exc_info=True
            )

            # Return empty result on error
            return EvaluationResult(
                task_id=task.task_id,
                strategy_used="evolution",
                attempts=[],
                metadata={
                    "error": str(e),
                    "strategy_type": StrategyType.EVOLUTION.value
                }
            )

    async def _execute_individual(
        self,
        task: ARCTask,
        individual: Individual,
        attempt_number: int
    ) -> EvolutionSubmission | None:
        """
        Execute an individual's program and create a submission.

        Args:
            task: The ARC task
            individual: The individual to execute
            attempt_number: The attempt number (1 or 2)

        Returns:
            EvolutionSubmission or None if execution fails
        """
        try:
            start_time = time.time()

            # Execute program in sandbox
            result = self.sandbox_executor.execute_operations(
                operations=individual.operations,
                input_grid=task.test_input,
                timeout=1.0,  # 1 second timeout
                memory_limit_mb=100
            )

            if result.success and result.output is not None:
                execution_time = (time.time() - start_time) * 1000

                # Calculate confidence based on fitness and other factors
                confidence = self._calculate_confidence(
                    individual,
                    result.execution_time,
                    attempt_number
                )

                return EvolutionSubmission(
                    individual=individual,
                    predicted_output=result.output,
                    confidence=confidence,
                    generation=individual.metadata.get("generation", 0),
                    execution_time_ms=execution_time,
                    metadata={
                        "attempt": attempt_number,
                        "program_length": individual.program_length(),
                        "age": individual.age,
                        "parent_ids": list(individual.parent_ids),
                        "species_id": individual.species_id,
                        "novelty_score": individual.novelty_score,
                        "mutation_history": individual.metadata.get("mutation_history", []),
                        "execution_metrics": {
                            "sandbox_time": result.execution_time,
                            "memory_used_mb": result.memory_used_mb,
                            "operations_executed": result.operations_executed
                        }
                    }
                )

            return None

        except Exception as e:
            self.logger.warning(
                "individual_execution_error",
                task_id=task.task_id,
                individual_id=individual.id,
                error=str(e)
            )
            return None

    def _get_top_individuals(
        self,
        all_individuals: list[Individual],
        n: int
    ) -> list[Individual]:
        """
        Get top N unique individuals by fitness.

        Args:
            all_individuals: All individuals from evolution
            n: Number of top individuals to return

        Returns:
            List of top unique individuals
        """
        # Sort by fitness (descending) and remove duplicates
        seen_programs = set()
        unique_individuals = []

        sorted_individuals = sorted(
            all_individuals,
            key=lambda x: x.fitness,
            reverse=True
        )

        for individual in sorted_individuals:
            program_hash = individual.id  # Already a hash of operations
            if program_hash not in seen_programs:
                seen_programs.add(program_hash)
                unique_individuals.append(individual)
                if len(unique_individuals) >= n:
                    break

        return unique_individuals

    def _calculate_confidence(
        self,
        individual: Individual,
        execution_time: float,
        attempt_number: int
    ) -> float:
        """
        Calculate confidence score for a solution.

        Args:
            individual: The individual solution
            execution_time: Execution time in seconds
            attempt_number: Which attempt this is

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from fitness
        base_confidence = individual.fitness

        # Adjust for execution time (prefer faster programs)
        time_penalty = min(execution_time / 1.0, 0.1)  # Max 10% penalty

        # Adjust for program complexity
        length_penalty = min(individual.program_length() / 100, 0.1)  # Max 10% penalty

        # Adjust for attempt number (slightly lower confidence for 2nd attempt)
        attempt_penalty = 0.05 if attempt_number > 1 else 0.0

        # Bonus for high novelty score
        novelty_bonus = 0.0
        if individual.novelty_score is not None and individual.novelty_score > 0.8:
            novelty_bonus = 0.05

        # Calculate final confidence
        confidence = base_confidence - time_penalty - length_penalty - attempt_penalty + novelty_bonus

        return max(0.0, min(1.0, confidence))

    def get_strategy_type(self) -> StrategyType:
        """
        Get the strategy type enum.

        Returns:
            StrategyType enum value
        """
        return StrategyType.EVOLUTION

    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get information about the evolution strategy.

        Returns:
            Dictionary with strategy information
        """
        return {
            "name": "Evolution Strategy",
            "type": StrategyType.EVOLUTION.value,
            "version": "2.5.0",
            "capabilities": {
                "multi_attempt": True,
                "max_attempts": 2,
                "parallel_evaluation": True,
                "genealogy_tracking": True,
                "diversity_preservation": True,
                "hybrid_initialization": self.config.initialization.hybrid_ratio > 0
            },
            "configuration": {
                "population_size": self.config.population.size,
                "max_generations": self.config.convergence.max_generations,
                "target_programs": 500,
                "time_limit_seconds": 300,
                "parallelization": {
                    "workers": self.config.parallelization.workers,
                    "batch_size": self.config.parallelization.batch_size,
                    "gpu_enabled": self.config.parallelization.gpu_acceleration
                },
                "operators": {
                    "crossover_rate": self.config.genetic_operators.crossover.rate,
                    "mutation_rate": self.config.genetic_operators.mutation.base_rate,
                    "selection_method": self.config.selection.method,
                    "elitism_rate": self.config.selection.elitism_rate
                }
            }
        }
