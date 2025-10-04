"""
Program Synthesis Strategy Adapter for ARC tasks.

This module provides integration with the DSL-based program synthesis approach,
implementing automatic program generation, execution, and optimization for ARC
task solving using the domain-specific language.
"""

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from src.domain.dsl.base import DSLProgram, OperationResult
from src.domain.dsl.color import (
    ColorFilterOperation,
    ColorInvertOperation,
    ColorMapOperation,
    ColorReplaceOperation,
)
from src.domain.dsl.composition import (
    ConcatenateOperation,
    ExtractRegionOperation,
    OverlayOperation,
    ResizeOperation,
)

# Import available operations for program generation
from src.domain.dsl.geometric import (
    CropOperation,
    FlipOperation,
    PadOperation,
    RotateOperation,
    TranslateOperation,
)
from src.domain.dsl.pattern import (
    FloodFillOperation,
    PatternFillOperation,
    PatternMatchOperation,
    PatternReplaceOperation,
)
from src.domain.dsl.serialization import DSLProgramSerializer
from src.domain.dsl.types import Grid, get_grid_dimensions
from src.domain.models import (
    ARCTask,
    ARCTaskSolution,
    ResourceUsage,
    StrategyType,
)
from src.domain.services.dsl_engine import DSLEngine, DSLEngineBuilder

# Import voting system for ensemble compatibility
try:
    from src.utils.ttt_voting import HybridVoter, create_prediction_candidate
    VOTING_SYSTEM_AVAILABLE = True
except ImportError:
    VOTING_SYSTEM_AVAILABLE = False
    logger.warning("Voting system not available - ensemble compatibility limited")


class ProgramGenerationStrategy(Enum):
    """Strategy for generating DSL programs."""
    TEMPLATE_BASED = "template"
    SEARCH_BASED = "search"
    LEARNING_BASED = "learning"
    HYBRID = "hybrid"


@dataclass
class ProgramSynthesisConfig:
    """Configuration for Program Synthesis Strategy."""

    # DSL Engine configuration
    execution_timeout: float = 5.0  # 5 seconds max per program execution
    memory_limit_mb: int = 100
    enable_caching: bool = True

    # Program generation configuration
    generation_strategy: ProgramGenerationStrategy = ProgramGenerationStrategy.HYBRID
    max_program_length: int = 10  # Maximum operations per program
    max_generation_attempts: int = 50  # Maximum programs to try
    beam_search_width: int = 5  # For search-based generation

    # Template-based generation
    use_common_templates: bool = True
    template_mutation_rate: float = 0.3

    # Search-based generation
    max_search_depth: int = 8
    random_seed: int | None = 42

    # Learning-based generation (placeholder for future ML models)
    use_pattern_learning: bool = False
    pattern_db_path: Path | None = None

    # Evolution-based generation
    use_evolution: bool = True
    evolution_config_path: Path | None = Path("configs/strategies/evolution.yaml")
    max_evolution_time: float = 30.0  # Max time for evolution per task

    # Performance optimization
    early_stopping_threshold: float = 0.95  # Stop if confidence > 95%
    parallel_execution: bool = False  # For future parallel evaluation

    # Resource constraints
    max_total_time: float = 60.0  # Maximum total time for synthesis
    cache_successful_programs: bool = True


class ProgramTemplate:
    """Template for common ARC transformation patterns."""

    def __init__(self, name: str, operations: list[dict[str, Any]],
                 description: str = "", priority: int = 1):
        """
        Initialize a program template.

        Args:
            name: Template identifier
            operations: List of operation specifications
            description: Human-readable description
            priority: Template priority (higher = more important)
        """
        self.name = name
        self.operations = operations
        self.description = description
        self.priority = priority

    def generate_variants(self, mutation_rate: float = 0.3) -> list[DSLProgram]:
        """Generate template variants by mutating parameters."""
        variants = [DSLProgram(operations=self.operations.copy())]

        if random.random() < mutation_rate:
            # Generate parameter mutations
            for _ in range(3):  # Generate up to 3 variants
                mutated_ops = []
                for op_spec in self.operations:
                    mutated_op = op_spec.copy()
                    # Add parameter mutations based on operation type
                    if op_spec.get("name") == "rotate":
                        mutated_op["parameters"] = {"angle": random.choice([90, 180, 270])}
                    elif op_spec.get("name") == "color_replace":
                        mutated_op["parameters"] = {
                            "from_color": random.randint(0, 9),
                            "to_color": random.randint(0, 9)
                        }
                    mutated_ops.append(mutated_op)

                variants.append(DSLProgram(operations=mutated_ops))

        return variants


class ProgramSynthesisAdapter:
    """Program Synthesis Strategy adapter for ARC tasks."""

    def __init__(self, config: ProgramSynthesisConfig | None = None):
        """Initialize Program Synthesis adapter with configuration."""
        self.config = config or ProgramSynthesisConfig()

        # Initialize DSL engine
        self.dsl_engine = self._create_dsl_engine()

        # Initialize program serializer for caching
        self.serializer = DSLProgramSerializer()

        # Program generation components
        self.templates = self._create_common_templates()
        self.successful_programs: dict[str, DSLProgram] = {}

        # Performance tracking
        self.generation_stats = {
            "programs_generated": 0,
            "programs_executed": 0,
            "successful_programs": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0
        }

        # Random seed for reproducibility
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def _create_dsl_engine(self) -> DSLEngine:
        """Create and configure DSL engine with all operations."""
        engine = (DSLEngineBuilder()
                 .with_timeout(self.config.execution_timeout)
                 .with_memory_limit(self.config.memory_limit_mb)
                 .with_operations(
                     # Geometric operations
                     RotateOperation, FlipOperation, TranslateOperation,
                     CropOperation, PadOperation,
                     # Color operations
                     ColorMapOperation, ColorFilterOperation, ColorReplaceOperation,
                     ColorInvertOperation,
                     # Pattern operations
                     PatternFillOperation, PatternMatchOperation, PatternReplaceOperation,
                     FloodFillOperation,
                     # Composition operations
                     OverlayOperation, ExtractRegionOperation, ConcatenateOperation,
                     ResizeOperation
                 )
                 .build())

        return engine

    def _create_common_templates(self) -> list[ProgramTemplate]:
        """Create library of common ARC transformation templates."""
        templates = [
            # Simple transformations
            ProgramTemplate(
                name="rotate_90",
                operations=[{"name": "rotate", "parameters": {"angle": 90}}],
                description="Simple 90-degree rotation",
                priority=3
            ),

            ProgramTemplate(
                name="flip_horizontal",
                operations=[{"name": "flip", "parameters": {"direction": 0}}],
                description="Horizontal flip/mirror",
                priority=3
            ),

            ProgramTemplate(
                name="color_swap",
                operations=[{
                    "name": "color_replace",
                    "parameters": {"from_color": 1, "to_color": 2}
                }],
                description="Simple color replacement",
                priority=2
            ),

            # Complex patterns
            ProgramTemplate(
                name="rotate_and_overlay",
                operations=[
                    {"name": "rotate", "parameters": {"angle": 90}},
                    {"name": "overlay", "parameters": {"blend_mode": "add"}}
                ],
                description="Rotate and overlay transformation",
                priority=2
            ),

            ProgramTemplate(
                name="color_invert_and_flip",
                operations=[
                    {"name": "color_invert", "parameters": {}},
                    {"name": "flip", "parameters": {"direction": 1}}
                ],
                description="Invert colors then flip vertically",
                priority=1
            ),

            # Pattern-based transformations
            ProgramTemplate(
                name="flood_fill_pattern",
                operations=[{
                    "name": "flood_fill",
                    "parameters": {"start_position": [0, 0], "new_color": 1}
                }],
                description="Flood fill from corner",
                priority=2
            )
        ]

        return sorted(templates, key=lambda t: t.priority, reverse=True)

    def solve(self, task: ARCTask) -> ARCTaskSolution:
        """
        Solve ARC task using program synthesis with DSL.

        Args:
            task: ARC task to solve

        Returns:
            Solution with predictions and metadata
        """
        start_time = datetime.now()
        synthesis_start = time.time()

        logger.info(f"Starting program synthesis for task: {task.task_id}")

        try:
            # Analyze task to understand transformation patterns
            analysis = self._analyze_task(task)

            # Generate candidate programs
            candidate_programs = self._generate_programs(task, analysis)

            # Evaluate programs and select best
            best_program, evaluation_results = self._evaluate_programs(
                candidate_programs, task
            )

            if best_program is None:
                logger.warning(f"No successful program found for task: {task.task_id}")
                return self._create_failure_solution(task, start_time)

            # Execute best program on test input
            final_result = self._execute_program_on_test(best_program, task)

            synthesis_time = time.time() - synthesis_start

            # Prepare solution metadata
            metadata = {
                "program_synthesis": True,
                "generation_strategy": self.config.generation_strategy.value,
                "best_program": best_program.to_dict(),
                "programs_evaluated": len(candidate_programs),
                "synthesis_time": synthesis_time,
                "task_analysis": analysis,
                "evaluation_results": evaluation_results,
                "performance_target_met": synthesis_time < self.config.max_total_time,
                **self.generation_stats
            }

            # Calculate confidence based on training accuracy
            confidence = self._calculate_confidence(evaluation_results)

            return ARCTaskSolution(
                task_id=task.task_id,
                predictions=[final_result.grid] if final_result.success else [task.test_input],
                strategy_used=StrategyType.PROGRAM_SYNTHESIS,
                confidence_score=confidence,
                metadata=metadata,
                resource_usage=self._calculate_resource_usage(task, start_time, synthesis_time)
            )

        except Exception as e:
            logger.error(f"Program synthesis failed for task {task.task_id}: {e}")
            return self._create_failure_solution(task, start_time, error=str(e))

    def _analyze_task(self, task: ARCTask) -> dict[str, Any]:
        """Analyze task to identify transformation patterns."""
        analysis = {
            "grid_dimensions": [],
            "color_changes": [],
            "spatial_changes": [],
            "pattern_complexity": "low"
        }

        # Analyze training examples
        for _i, example in enumerate(task.train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            # Grid dimension analysis
            input_dims = get_grid_dimensions(input_grid)
            output_dims = get_grid_dimensions(output_grid)
            analysis["grid_dimensions"].append({
                "input": input_dims,
                "output": output_dims,
                "size_changed": input_dims != output_dims
            })

            # Basic color analysis
            input_colors = set()
            output_colors = set()
            for row in input_grid:
                input_colors.update(row)
            for row in output_grid:
                output_colors.update(row)

            analysis["color_changes"].append({
                "input_colors": sorted(input_colors),
                "output_colors": sorted(output_colors),
                "colors_added": list(output_colors - input_colors),
                "colors_removed": list(input_colors - output_colors)
            })

        # Determine pattern complexity
        if len(task.train_examples) > 3:
            analysis["pattern_complexity"] = "high"
        elif any(dim["size_changed"] for dim in analysis["grid_dimensions"]):
            analysis["pattern_complexity"] = "medium"

        return analysis

    def _generate_programs(self, task: ARCTask, analysis: dict[str, Any]) -> list[DSLProgram]:
        """Generate candidate programs using configured strategy."""
        programs = []

        if self.config.generation_strategy in [
            ProgramGenerationStrategy.TEMPLATE_BASED,
            ProgramGenerationStrategy.HYBRID
        ]:
            programs.extend(self._generate_template_based_programs(task, analysis))

        if self.config.generation_strategy in [
            ProgramGenerationStrategy.SEARCH_BASED,
            ProgramGenerationStrategy.HYBRID
        ]:
            programs.extend(self._generate_search_based_programs(task, analysis))

        if self.config.generation_strategy in [
            ProgramGenerationStrategy.LEARNING_BASED,
            ProgramGenerationStrategy.HYBRID
        ]:
            programs.extend(self._generate_learning_based_programs(task, analysis))

        # Limit total programs and remove duplicates
        programs = list({self.serializer.create_program_fingerprint(p.operations): p for p in programs}.values())
        return programs[:self.config.max_generation_attempts]

    def _generate_template_based_programs(self, task: ARCTask, analysis: dict[str, Any]) -> list[DSLProgram]:
        """Generate programs using template matching and mutation."""
        programs = []

        # Prioritize templates based on task characteristics
        relevant_templates = self._select_relevant_templates(analysis)

        for template in relevant_templates:
            # Generate template variants
            variants = template.generate_variants(self.config.template_mutation_rate)

            for variant in variants:
                # Adapt template based on task analysis
                adapted_program = self._adapt_template_to_task(variant, task, analysis)
                if adapted_program:
                    programs.append(adapted_program)

                    # Add template combinations
                    if len(programs) < self.config.max_generation_attempts // 3:
                        combined_programs = self._generate_template_combinations(
                            adapted_program, relevant_templates, task, analysis
                        )
                        programs.extend(combined_programs[:3])  # Limit combinations per template

                if len(programs) >= self.config.max_generation_attempts // 2:
                    break

            if len(programs) >= self.config.max_generation_attempts // 2:
                break

        self.generation_stats["programs_generated"] += len(programs)
        return programs

    def _generate_search_based_programs(self, task: ARCTask, analysis: dict[str, Any]) -> list[DSLProgram]:
        """Generate programs using search-based exploration."""
        programs = []
        available_operations = self.dsl_engine.get_registered_operations()

        # Check if we should use evolution
        if self.config.generation_strategy == ProgramGenerationStrategy.SEARCH_BASED and self.config.use_evolution:
            # Use genetic algorithm for program generation
            evolved_programs = self._generate_evolved_programs(task, analysis)
            programs.extend(evolved_programs)
        else:
            # Weighted selection based on analysis
            operation_weights = self._calculate_operation_weights(analysis, available_operations)

            # Generate random programs with weighted selection
            for _ in range(self.config.max_generation_attempts // 3):
                program_length = random.randint(1, min(self.config.max_program_length, 5))
                operations = []

                for _ in range(program_length):
                    # Weighted random selection of operations
                    op_name = random.choices(
                        available_operations,
                        weights=[operation_weights.get(op, 1.0) for op in available_operations]
                    )[0]

                    # Generate context-aware parameters
                    parameters = self._generate_operation_parameters(op_name, task)

                    operations.append({
                        "name": op_name,
                        "parameters": parameters
                    })

                programs.append(DSLProgram(operations=operations))

        # Generate beam search programs
        if self.config.beam_search_width > 0:
            beam_programs = self._generate_beam_search_programs(task, analysis, available_operations)
            programs.extend(beam_programs)

        self.generation_stats["programs_generated"] += len(programs)
        return programs

    def _generate_learning_based_programs(self, task: ARCTask, analysis: dict[str, Any]) -> list[DSLProgram]:
        """Generate programs using learned patterns (placeholder for future ML)."""
        programs = []

        # Placeholder for future machine learning-based program generation
        # This could include:
        # - Pattern recognition from successful programs
        # - Neural program synthesis
        # - Reinforcement learning-based program search

        logger.info("Learning-based program generation not yet implemented")
        return programs

    async def _generate_evolved_programs(self, task: ARCTask, analysis: dict[str, Any]) -> list[DSLProgram]:
        """Generate programs using genetic algorithm evolution."""
        programs = []

        try:
            # Import evolution components
            from src.adapters.strategies.evolution_engine import EvolutionEngine
            from src.infrastructure.config import GeneticAlgorithmConfig

            # Load evolution config
            evolution_config = GeneticAlgorithmConfig()

            # Override some settings for integration
            evolution_config.population.size = 100  # Smaller population for speed
            evolution_config.convergence.max_generations = 20
            evolution_config.performance.generation_timeout = 5

            # Create evolution engine
            evolution_engine = EvolutionEngine(
                config=evolution_config,
                dsl_engine=self.dsl_engine
            )

            # Run evolution (with timeout)
            import asyncio
            start_time = time.time()

            try:
                best_individual, stats = await asyncio.wait_for(
                    evolution_engine.evolve(task),
                    timeout=self.config.max_evolution_time
                )

                # Convert best individual to DSLProgram
                if best_individual and best_individual.fitness > 0.5:
                    operations = []
                    for op in best_individual.operations:
                        operations.append({
                            "name": op.get_name(),
                            "parameters": op.parameters
                        })

                    programs.append(DSLProgram(operations=operations))
                    logger.info(f"Evolution found program with fitness {best_individual.fitness}")

                # Also add some elite individuals
                elite = evolution_engine.population.get_elite(5)
                for ind in elite[1:]:  # Skip best (already added)
                    if ind.fitness > 0.3:
                        operations = []
                        for op in ind.operations:
                            operations.append({
                                "name": op.get_name(),
                                "parameters": op.parameters
                            })
                        programs.append(DSLProgram(operations=operations))

            except TimeoutError:
                logger.warning(f"Evolution timeout after {time.time() - start_time:.1f}s")

            finally:
                # Clean up
                evolution_engine.cleanup()

        except Exception as e:
            logger.error(f"Evolution failed: {e}")

        return programs

    def _adapt_template_to_task(self, program: DSLProgram, task: ARCTask, analysis: dict[str, Any]) -> DSLProgram | None:
        """Adapt a template program to the specific task characteristics."""
        adapted_operations = []

        for op_spec in program.operations:
            adapted_op = op_spec.copy()

            # Adapt based on analysis
            if op_spec["name"] == "color_replace" and analysis.get("color_changes"):
                # Use actual color changes observed in training
                color_change = analysis["color_changes"][0]
                if color_change["colors_removed"] and color_change["colors_added"]:
                    adapted_op["parameters"] = {
                        "from_color": color_change["colors_removed"][0],
                        "to_color": color_change["colors_added"][0]
                    }

            elif op_spec["name"] == "crop" and analysis.get("grid_dimensions"):
                # Adapt crop based on typical output size
                dim_info = analysis["grid_dimensions"][0]
                if dim_info["size_changed"]:
                    output_dims = dim_info["output"]
                    adapted_op["parameters"] = {
                        "region": {
                            "top_left": [0, 0],
                            "bottom_right": [output_dims[0] - 1, output_dims[1] - 1]
                        }
                    }

            adapted_operations.append(adapted_op)

        return DSLProgram(operations=adapted_operations)

    def _generate_operation_parameters(self, op_name: str, task: ARCTask) -> dict[str, Any]:
        """Generate reasonable parameters for an operation based on task."""
        if op_name == "rotate":
            return {"angle": random.choice([90, 180, 270])}

        elif op_name == "flip":
            return {"direction": random.randint(0, 3)}

        elif op_name == "color_replace":
            return {
                "from_color": random.randint(0, 9),
                "to_color": random.randint(0, 9)
            }

        elif op_name == "translate":
            return {
                "row_offset": random.randint(-3, 3),
                "col_offset": random.randint(-3, 3)
            }

        elif op_name == "flood_fill":
            # Use corner positions commonly
            positions = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
            return {
                "start_position": list(random.choice(positions)),
                "new_color": random.randint(0, 9)
            }

        # Default empty parameters
        return {}

    def _evaluate_programs(self, programs: list[DSLProgram], task: ARCTask) -> tuple[DSLProgram | None, dict[str, Any]]:
        """Evaluate programs on training examples and select best."""
        best_program = None
        best_score = 0.0
        evaluation_results = {
            "program_scores": [],
            "execution_times": [],
            "errors": []
        }

        for i, program in enumerate(programs):
            try:
                score, exec_time = self._evaluate_single_program(program, task)

                evaluation_results["program_scores"].append(score)
                evaluation_results["execution_times"].append(exec_time)

                if score > best_score:
                    best_score = score
                    best_program = program

                # Early stopping if we find a perfect program
                if score >= self.config.early_stopping_threshold:
                    logger.info(f"Early stopping: found program with score {score:.3f}")
                    break

            except Exception as e:
                logger.debug(f"Program {i} evaluation failed: {e}")
                evaluation_results["errors"].append(str(e))
                evaluation_results["program_scores"].append(0.0)
                evaluation_results["execution_times"].append(0.0)

        evaluation_results["best_score"] = best_score
        evaluation_results["programs_evaluated"] = len(programs)

        return best_program, evaluation_results

    def _evaluate_single_program(self, program: DSLProgram, task: ARCTask) -> tuple[float, float]:
        """Evaluate a single program on all training examples."""
        total_score = 0.0
        execution_times = []

        for example in task.train_examples:
            input_grid = example["input"]
            expected_output = example["output"]

            # Execute program on training input
            start_time = time.time()
            result = self.dsl_engine.execute_program(program, input_grid)
            exec_time = time.time() - start_time

            execution_times.append(exec_time)
            self.generation_stats["programs_executed"] += 1

            if result.success:
                # Calculate similarity score
                score = self._calculate_grid_similarity(result.grid, expected_output)
                total_score += score
            else:
                # Penalize failed executions
                total_score += 0.0

        # Average score across all training examples
        avg_score = total_score / len(task.train_examples) if task.train_examples else 0.0
        avg_exec_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        return avg_score, avg_exec_time

    def _calculate_grid_similarity(self, grid1: Grid, grid2: Grid) -> float:
        """Calculate similarity score between two grids (0.0 to 1.0)."""
        if not grid1 or not grid2:
            return 0.0

        # Check dimensions first
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return 0.0

        # Count matching cells
        total_cells = len(grid1) * len(grid1[0])
        matching_cells = 0

        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                if grid1[r][c] == grid2[r][c]:
                    matching_cells += 1

        return matching_cells / total_cells if total_cells > 0 else 0.0

    def _execute_program_on_test(self, program: DSLProgram, task: ARCTask) -> OperationResult:
        """Execute the best program on the test input."""
        try:
            result = self.dsl_engine.execute_program(program, task.test_input)

            if result.success:
                self.generation_stats["successful_programs"] += 1

                # Cache successful program
                if self.config.cache_successful_programs:
                    program_hash = self.serializer.create_program_fingerprint(program.operations)
                    self.successful_programs[program_hash] = program

            return result

        except Exception as e:
            logger.error(f"Failed to execute program on test input: {e}")
            return OperationResult(
                success=False,
                grid=task.test_input,
                error_message=str(e)
            )

    def _calculate_confidence(self, evaluation_results: dict[str, Any]) -> float:
        """Calculate confidence score based on evaluation results."""
        best_score = evaluation_results.get("best_score", 0.0)

        # Confidence is based on training accuracy
        confidence = min(best_score, 1.0)

        # Adjust for execution reliability
        errors = evaluation_results.get("errors", [])
        if errors:
            error_rate = len(errors) / evaluation_results.get("programs_evaluated", 1)
            confidence *= (1.0 - error_rate * 0.5)  # Penalize unreliable execution

        return max(0.0, min(1.0, confidence))

    def _calculate_resource_usage(self, task: ARCTask, start_time: datetime, synthesis_time: float) -> ResourceUsage:
        """Calculate resource usage for the synthesis process."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Estimate memory usage (simplified)
        estimated_memory = (
            len(task.train_examples) * 50 +  # Training examples
            self.generation_stats["programs_generated"] * 10 +  # Generated programs
            100  # Base overhead
        )

        return ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            cpu_seconds=duration,
            memory_mb=estimated_memory / 1024,  # Convert to MB
            gpu_memory_mb=None,  # DSL doesn't use GPU
            api_calls={},
            total_tokens=0,
            estimated_cost=0.0,
            timestamp=datetime.now(),
        )

    def _create_failure_solution(self, task: ARCTask, start_time: datetime, error: str = None) -> ARCTaskSolution:
        """Create a failure solution when synthesis doesn't succeed."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        metadata = {
            "program_synthesis": True,
            "success": False,
            "error": error or "No successful program found",
            "generation_strategy": self.config.generation_strategy.value,
            **self.generation_stats
        }

        return ARCTaskSolution(
            task_id=task.task_id,
            predictions=[task.test_input],  # Return input as fallback
            strategy_used=StrategyType.PROGRAM_SYNTHESIS,
            confidence_score=0.0,
            metadata=metadata,
            resource_usage=self._calculate_resource_usage(task, start_time, duration)
        )

    def _select_relevant_templates(self, analysis: dict[str, Any]) -> list[ProgramTemplate]:
        """Select templates most relevant to the task characteristics."""
        relevant_templates = self.templates.copy()

        # Prioritize based on analysis
        if analysis.get("pattern_complexity") == "low":
            # Prefer simple templates for low complexity
            relevant_templates = [t for t in relevant_templates if len(t.operations) <= 2]
            relevant_templates.extend([t for t in self.templates if len(t.operations) > 2])

        # Check for color changes
        color_changes = analysis.get("color_changes", [])
        has_color_changes = any(
            cc["colors_added"] or cc["colors_removed"] for cc in color_changes
        )
        if has_color_changes:
            # Prioritize color-related templates
            color_templates = [t for t in relevant_templates if any(
                "color" in op.get("name", "") for op in t.operations
            )]
            other_templates = [t for t in relevant_templates if t not in color_templates]
            relevant_templates = color_templates + other_templates

        # Check for size changes
        size_changes = analysis.get("grid_dimensions", [])
        has_size_changes = any(dim["size_changed"] for dim in size_changes)
        if has_size_changes:
            # Prioritize geometric templates
            geo_templates = [t for t in relevant_templates if any(
                op.get("name") in ["rotate", "flip", "crop", "pad", "translate"]
                for op in t.operations
            )]
            other_templates = [t for t in relevant_templates if t not in geo_templates]
            relevant_templates = geo_templates + other_templates

        return relevant_templates

    def _generate_template_combinations(self, base_program: DSLProgram, templates: list[ProgramTemplate],
                                      task: ARCTask, analysis: dict[str, Any]) -> list[DSLProgram]:
        """Generate combinations of templates with the base program."""
        combinations = []

        for template in templates[:3]:  # Limit to avoid explosion
            if len(template.operations) + len(base_program.operations) <= self.config.max_program_length:
                # Try appending template to base
                combined_ops = base_program.operations + template.operations
                combinations.append(DSLProgram(operations=combined_ops))

                # Try prepending template to base
                combined_ops = template.operations + base_program.operations
                combinations.append(DSLProgram(operations=combined_ops))

        return combinations[:5]  # Limit combinations per template

    def _calculate_operation_weights(self, analysis: dict[str, Any], operations: list[str]) -> dict[str, float]:
        """Calculate weights for operations based on task analysis."""
        weights = {}

        for op_name in operations:
            base_weight = 1.0

            # Weight based on analysis
            if "color" in op_name:
                color_changes = analysis.get("color_changes", [])
                if any(cc["colors_added"] or cc["colors_removed"] for cc in color_changes):
                    base_weight *= 2.0

            if op_name in ["rotate", "flip", "translate"]:
                # Geometric operations are generally useful
                base_weight *= 1.5

            if op_name in ["crop", "pad", "resize"]:
                size_changes = analysis.get("grid_dimensions", [])
                if any(dim["size_changed"] for dim in size_changes):
                    base_weight *= 2.0

            # Pattern operations for complex tasks
            if op_name.startswith("pattern") or "fill" in op_name:
                if analysis.get("pattern_complexity") in ["medium", "high"]:
                    base_weight *= 1.3

            weights[op_name] = base_weight

        return weights

    def _generate_beam_search_programs(self, task: ARCTask, analysis: dict[str, Any],
                                     available_operations: list[str]) -> list[DSLProgram]:
        """Generate programs using beam search."""
        beam = [DSLProgram(operations=[])]  # Start with empty program

        for _depth in range(min(self.config.max_search_depth, self.config.max_program_length)):
            new_candidates = []

            for program in beam:
                # Generate extensions for each program in beam
                for op_name in available_operations[:10]:  # Limit branching factor
                    if len(program.operations) < self.config.max_program_length:
                        parameters = self._generate_operation_parameters(op_name, task)
                        new_op = {"name": op_name, "parameters": parameters}

                        extended_program = DSLProgram(
                            operations=program.operations + [new_op]
                        )
                        new_candidates.append(extended_program)

            # Evaluate candidates and keep top K
            if new_candidates:
                candidate_scores = []
                for candidate in new_candidates:
                    try:
                        score, _ = self._evaluate_single_program(candidate, task)
                        candidate_scores.append((score, candidate))
                    except Exception:
                        candidate_scores.append((0.0, candidate))

                # Keep best candidates
                candidate_scores.sort(key=lambda x: x[0], reverse=True)
                beam = [prog for _, prog in candidate_scores[:self.config.beam_search_width]]

            # Early termination if we find good programs
            if any(score > 0.8 for score, _ in candidate_scores[:3]):
                break

        return beam

    def get_synthesis_stats(self) -> dict[str, Any]:
        """Get comprehensive program synthesis statistics."""
        dsl_stats = self.dsl_engine.get_execution_stats()

        return {
            "generation_stats": self.generation_stats.copy(),
            "dsl_execution_stats": {
                "total_execution_time": dsl_stats.total_execution_time,
                "operation_count": dsl_stats.operation_count,
                "cache_hits": dsl_stats.cache_hits,
                "cache_misses": dsl_stats.cache_misses,
                "peak_memory_mb": dsl_stats.peak_memory_mb,
                "slow_operations": dsl_stats.slow_operations
            },
            "config": {
                "generation_strategy": self.config.generation_strategy.value,
                "max_program_length": self.config.max_program_length,
                "max_generation_attempts": self.config.max_generation_attempts,
                "execution_timeout": self.config.execution_timeout
            },
            "successful_programs_cached": len(self.successful_programs),
            "templates_available": len(self.templates)
        }

    def generate_ensemble_candidates(self, task: ARCTask, num_candidates: int = 5) -> list[Any]:
        """
        Generate multiple prediction candidates for ensemble voting.

        This method creates multiple program synthesis attempts with different
        configurations to provide diverse candidates for ensemble voting.

        Args:
            task: ARC task to solve
            num_candidates: Number of candidates to generate

        Returns:
            List of PredictionCandidate objects compatible with voting system
        """
        if not VOTING_SYSTEM_AVAILABLE:
            logger.warning("Voting system not available - generating single solution")
            solution = self.solve(task)
            return [solution.predictions[0]] if solution.predictions else []

        candidates = []

        # Generate candidates with different strategies and configurations
        self._generate_strategy_candidates()

        original_config = self.config

        for i in range(num_candidates):
            try:
                # Vary configuration for diversity
                varied_config = self._create_varied_config(original_config, i)
                temp_adapter = ProgramSynthesisAdapter(varied_config)

                # Solve with this configuration
                solution = temp_adapter.solve(task)

                if solution.predictions and len(solution.predictions) > 0:
                    # Create prediction candidate
                    candidate = create_prediction_candidate(
                        prediction=solution.predictions[0],
                        confidence=solution.confidence_score,
                        source_type="program_synthesis",
                        augmentation_type=f"config_variant_{i}",
                        original=(i == 0),  # First candidate is "original"
                        strategy=varied_config.generation_strategy.value,
                        synthesis_time=solution.metadata.get("synthesis_time", 0.0),
                        programs_evaluated=solution.metadata.get("programs_evaluated", 0)
                    )
                    candidates.append(candidate)

            except Exception as e:
                logger.debug(f"Failed to generate candidate {i}: {e}")
                continue

        logger.info(f"Generated {len(candidates)} ensemble candidates for task {task.task_id}")
        return candidates

    def _create_varied_config(self, base_config: ProgramSynthesisConfig, variant_index: int) -> ProgramSynthesisConfig:
        """Create a varied configuration for ensemble diversity."""
        strategies = [
            ProgramGenerationStrategy.TEMPLATE_BASED,
            ProgramGenerationStrategy.SEARCH_BASED,
            ProgramGenerationStrategy.HYBRID
        ]

        # Create varied config
        varied_config = ProgramSynthesisConfig(
            generation_strategy=strategies[variant_index % len(strategies)],
            execution_timeout=base_config.execution_timeout * (0.8 + variant_index * 0.1),
            max_program_length=min(base_config.max_program_length + variant_index, 15),
            max_generation_attempts=max(base_config.max_generation_attempts - variant_index * 5, 10),
            beam_search_width=max(base_config.beam_search_width + variant_index, 2),
            template_mutation_rate=min(base_config.template_mutation_rate + variant_index * 0.1, 0.8),
            random_seed=base_config.random_seed + variant_index if base_config.random_seed else None,
            early_stopping_threshold=max(base_config.early_stopping_threshold - variant_index * 0.05, 0.7),
            cache_successful_programs=base_config.cache_successful_programs,
            max_total_time=base_config.max_total_time
        )

        return varied_config

    def solve_with_ensemble_voting(self, task: ARCTask, num_candidates: int = 5) -> ARCTaskSolution:
        """
        Solve task using ensemble voting across multiple program synthesis attempts.

        Args:
            task: ARC task to solve
            num_candidates: Number of candidate solutions to generate

        Returns:
            ARCTaskSolution with ensemble-voted result
        """
        if not VOTING_SYSTEM_AVAILABLE:
            logger.warning("Voting system not available - using single solution")
            return self.solve(task)

        start_time = datetime.now()
        synthesis_start = time.time()

        logger.info(f"Starting ensemble voting for task: {task.task_id}")

        try:
            # Generate multiple candidates
            candidates = self.generate_ensemble_candidates(task, num_candidates)

            if not candidates:
                logger.warning("No candidates generated - falling back to single solution")
                return self.solve(task)

            # Use hybrid voter to select best prediction
            voter = HybridVoter()
            voting_result = voter.vote_all_predictions(
                candidates,
                fallback_prediction=task.test_input
            )

            synthesis_time = time.time() - synthesis_start

            # Calculate ensemble confidence
            ensemble_confidence = voting_result.confidence_score

            # Prepare ensemble metadata
            metadata = {
                "program_synthesis": True,
                "ensemble_voting": True,
                "generation_strategy": "ensemble",
                "candidates_generated": len(candidates),
                "voting_method": voting_result.voting_method,
                "agreement_ratio": voting_result.agreement_ratio,
                "vote_distribution": voting_result.vote_distribution,
                "synthesis_time": synthesis_time,
                "voting_metadata": voting_result.metadata,
                "performance_target_met": synthesis_time < self.config.max_total_time,
                **self.generation_stats
            }

            return ARCTaskSolution(
                task_id=task.task_id,
                predictions=[voting_result.best_prediction],
                strategy_used=StrategyType.PROGRAM_SYNTHESIS,
                confidence_score=ensemble_confidence,
                metadata=metadata,
                resource_usage=self._calculate_resource_usage(task, start_time, synthesis_time)
            )

        except Exception as e:
            logger.error(f"Ensemble voting failed for task {task.task_id}: {e}")
            # Fallback to regular solve
            return self.solve(task)

    def verify_ensemble_compatibility(self) -> dict[str, Any]:
        """
        Verify compatibility with ensemble voting system.

        Returns:
            Dictionary containing compatibility status and details
        """
        compatibility_report = {
            "voting_system_available": VOTING_SYSTEM_AVAILABLE,
            "can_generate_candidates": True,
            "supports_confidence_scores": True,
            "supports_metadata_for_voting": True,
            "prediction_format_compatible": True,
            "ensemble_methods_available": [],
            "compatibility_issues": []
        }

        if VOTING_SYSTEM_AVAILABLE:
            # Test candidate generation
            try:
                test_task = ARCTask(
                    task_id="compatibility_test",
                    task_source="test",
                    train_examples=[{"input": [[1]], "output": [[2]]}],
                    test_input=[[1]]
                )

                candidates = self.generate_ensemble_candidates(test_task, 2)
                if not candidates:
                    compatibility_report["compatibility_issues"].append(
                        "Failed to generate prediction candidates"
                    )
                    compatibility_report["can_generate_candidates"] = False

                compatibility_report["ensemble_methods_available"].extend([
                    "generate_ensemble_candidates",
                    "solve_with_ensemble_voting"
                ])

            except Exception as e:
                compatibility_report["compatibility_issues"].append(
                    f"Candidate generation test failed: {e}"
                )
                compatibility_report["can_generate_candidates"] = False
        else:
            compatibility_report["compatibility_issues"].append(
                "Voting system (ttt_voting.py) not available"
            )
            compatibility_report["can_generate_candidates"] = False

        # Check result format compatibility
        try:
            test_solution = ARCTaskSolution(
                task_id="test",
                predictions=[[[1, 2], [3, 4]]],
                strategy_used=StrategyType.PROGRAM_SYNTHESIS,
                confidence_score=0.8,
                metadata={"test": True}
            )

            # Verify required fields exist
            required_fields = ["task_id", "predictions", "strategy_used", "confidence_score"]
            for field in required_fields:
                if not hasattr(test_solution, field):
                    compatibility_report["compatibility_issues"].append(
                        f"Missing required field: {field}"
                    )

        except Exception as e:
            compatibility_report["compatibility_issues"].append(
                f"Solution format test failed: {e}"
            )
            compatibility_report["prediction_format_compatible"] = False

        # Overall compatibility status
        compatibility_report["is_fully_compatible"] = (
            len(compatibility_report["compatibility_issues"]) == 0
        )

        return compatibility_report

    def cleanup(self) -> None:
        """Clean up resources and clear caches."""
        logger.info("Starting program synthesis adapter cleanup")

        # Clear caches
        self.successful_programs.clear()
        if hasattr(self.dsl_engine, 'clear_cache'):
            self.dsl_engine.clear_cache()

        # Log final statistics
        final_stats = self.get_synthesis_stats()
        logger.info(f"Program synthesis session stats: {final_stats['generation_stats']}")

        logger.info("Program synthesis adapter cleanup complete")

    @contextmanager
    def _synthesis_timeout_context(self):
        """Context manager to enforce total synthesis timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Program synthesis exceeded {self.config.max_total_time}s limit")

        # Set up timeout signal (Unix systems only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.max_total_time))

            try:
                yield
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # For Windows or systems without SIGALRM
            yield
