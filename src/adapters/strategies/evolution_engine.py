"""
Evolution engine for genetic algorithm-based program synthesis.

This module implements a flexible genetic programming system to evolve DSL programs
that solve ARC tasks. It supports population management, crossover/mutation operators,
fitness evaluation, diversity preservation, and parallel execution.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import diskcache
import numpy as np
import psutil
import yaml

logger = logging.getLogger(__name__)

from src.domain.dsl.base import Operation
from src.domain.dsl.types import Grid
from src.domain.models import ARCTask, StrategyType
from src.domain.services.dsl_engine import DSLEngine
from src.domain.services.experiment_orchestrator import (
    ExperimentConfig,
    ExperimentOrchestrator,
    ExperimentPriority,
    ExperimentProgress,
    ExperimentResources,
)
from src.infrastructure.config import GeneticAlgorithmConfig


def load_evolution_config(config_path: str | Path | None = None) -> GeneticAlgorithmConfig:
    """Load evolution configuration from YAML file."""
    if config_path is None:
        config_path = Path("configs/strategies/evolution.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Evolution config not found: {config_path}")

    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    # Create config object and populate from yaml
    config = GeneticAlgorithmConfig()

    # Map YAML structure to config object
    if 'population' in yaml_config:
        for key, value in yaml_config['population'].items():
            setattr(config.population, key, value)

    if 'genetic_operators' in yaml_config:
        ops = yaml_config['genetic_operators']
        if 'crossover' in ops:
            for key, value in ops['crossover'].items():
                setattr(config.genetic_operators.crossover, key, value)
        if 'mutation' in ops:
            for key, value in ops['mutation'].items():
                setattr(config.genetic_operators.mutation, key, value)

    if 'fitness' in yaml_config:
        for key, value in yaml_config['fitness'].items():
            setattr(config.fitness, key, value)

    if 'diversity' in yaml_config:
        for key, value in yaml_config['diversity'].items():
            setattr(config.diversity, key, value)

    if 'parallelization' in yaml_config:
        for key, value in yaml_config['parallelization'].items():
            setattr(config.parallelization, key, value)

    if 'convergence' in yaml_config:
        for key, value in yaml_config['convergence'].items():
            setattr(config.convergence, key, value)

    if 'performance' in yaml_config:
        for key, value in yaml_config['performance'].items():
            setattr(config.performance, key, value)

    if 'reproducibility' in yaml_config:
        for key, value in yaml_config['reproducibility'].items():
            setattr(config.reproducibility, key, value)

    # Add platform overrides, island model, and novelty search
    if 'platform_overrides' in yaml_config:
        config.platform_overrides = yaml_config['platform_overrides']

    if 'island_model' in yaml_config:
        config.island_model = yaml_config['island_model']

    if 'novelty_search' in yaml_config:
        config.novelty_search = yaml_config['novelty_search']

    return config


class SelectionMethod(Enum):
    """Selection methods for breeding."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"


class DiversityMethod(Enum):
    """Methods for maintaining diversity."""
    FITNESS_SHARING = "fitness_sharing"
    SPECIATION = "speciation"
    NOVELTY_SEARCH = "novelty"
    CROWDING = "crowding"


@dataclass
class Individual:
    """
    Represents an individual in the genetic algorithm population.

    Attributes:
        operations: List of DSL operations forming the program
        fitness: Fitness score (0.0 to 1.0, where 1.0 is perfect)
        age: Number of generations this individual has existed
        parent_ids: Set of parent individual IDs (for genealogy tracking)
        created_at: Timestamp when individual was created
        metadata: Additional information about the individual
        cached_execution: Optional cached execution results
        species_id: Optional species identifier for niching
        novelty_score: Optional novelty score for novelty search
    """
    operations: list[Operation]
    fitness: float = 0.0
    age: int = 0
    parent_ids: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    cached_execution: dict[str, Grid] | None = None
    species_id: int | None = None
    novelty_score: float | None = None

    def __post_init__(self):
        """Initialize ID after creation."""
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID based on operations."""
        op_str = "".join([f"{op.get_name()}{op.parameters}" for op in self.operations])
        return hashlib.md5(op_str.encode()).hexdigest()[:12]

    def program_length(self) -> int:
        """Get the number of operations in the program."""
        return len(self.operations)

    def increment_age(self) -> None:
        """Increment the age of this individual."""
        self.age += 1

    def is_elite(self, elite_threshold: float = 0.9) -> bool:
        """Check if this individual is elite (high fitness)."""
        return self.fitness >= elite_threshold


@dataclass
class Population:
    """
    Manages a population of individuals for genetic algorithm.

    Attributes:
        individuals: List of individuals in the population
        generation: Current generation number
        species: Optional species information for niching
        diversity_metrics: Tracked diversity metrics
        best_individual: Best individual found so far
        best_fitness_history: History of best fitness per generation
    """
    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0
    species: dict[int, list[Individual]] = field(default_factory=dict)
    diversity_metrics: dict[str, float] = field(default_factory=dict)
    best_individual: Individual | None = None
    best_fitness_history: list[float] = field(default_factory=list)

    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)
        self._update_best_individual(individual)

    def _update_best_individual(self, individual: Individual) -> None:
        """Update best individual if new one is better."""
        if self.best_individual is None or individual.fitness > self.best_individual.fitness:
            self.best_individual = individual

    def remove_individual(self, individual: Individual) -> None:
        """Remove an individual from the population."""
        self.individuals.remove(individual)

    def get_elite(self, elite_size: int) -> list[Individual]:
        """Get the top elite_size individuals by fitness."""
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_individuals[:elite_size]

    def size(self) -> int:
        """Get population size."""
        return len(self.individuals)

    def average_fitness(self) -> float:
        """Calculate average fitness of population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    def fitness_variance(self) -> float:
        """Calculate variance of fitness in population."""
        if not self.individuals:
            return 0.0
        avg = self.average_fitness()
        return sum((ind.fitness - avg) ** 2 for ind in self.individuals) / len(self.individuals)

    def update_generation_stats(self) -> None:
        """Update statistics for current generation."""
        if self.best_individual:
            self.best_fitness_history.append(self.best_individual.fitness)

        # Calculate diversity metrics
        unique_programs = len({ind.id for ind in self.individuals})
        self.diversity_metrics['unique_programs'] = unique_programs / self.size()
        self.diversity_metrics['fitness_variance'] = self.fitness_variance()
        self.diversity_metrics['average_age'] = sum(ind.age for ind in self.individuals) / self.size()

    def increment_generation(self) -> None:
        """Move to next generation."""
        self.generation += 1
        for individual in self.individuals:
            individual.increment_age()


class GeneticOperator(ABC):
    """Abstract base class for genetic operators."""

    @abstractmethod
    def apply(self, *individuals: Individual) -> list[Individual]:
        """Apply genetic operator to individuals."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get operator name."""
        pass


class DiversityMetric(ABC):
    """Abstract base class for diversity metrics."""

    @abstractmethod
    def calculate(self, population: Population) -> float:
        """Calculate diversity metric for population."""
        pass

    @abstractmethod
    def apply_pressure(self, individual: Individual, population: Population) -> float:
        """Apply diversity pressure to fitness score."""
        pass


class FitnessEvaluator:
    """
    Evaluates fitness of individuals based on ARC task performance.

    Supports multi-objective fitness, caching, and early termination.
    """

    def __init__(
        self,
        task: ARCTask,
        dsl_engine: DSLEngine,
        weights: dict[str, float] | None = None,
        cache_enabled: bool = True,
        early_termination_threshold: float = 0.95,
        cache_dir: str = ".fitness_cache",
        program_timeout: float = 1.0,
        memory_limit_mb: int = 100
    ):
        """
        Initialize fitness evaluator.

        Args:
            task: ARC task to evaluate against
            dsl_engine: DSL engine for executing programs
            weights: Weights for different fitness components
            cache_enabled: Whether to cache fitness scores
            early_termination_threshold: Stop if fitness exceeds this
            cache_dir: Directory for disk cache
            program_timeout: Timeout in seconds for each program execution
            memory_limit_mb: Memory limit in MB for each program execution
        """
        self.task = task
        self.dsl_engine = dsl_engine
        self.weights = weights or {
            'grid_similarity': 0.65,
            'program_length': 0.15,
            'complexity': 0.15,
            'execution_time': 0.05
        }
        self.cache_enabled = cache_enabled
        self.early_termination_threshold = early_termination_threshold
        self.program_timeout = program_timeout
        self.memory_limit_mb = memory_limit_mb

        # Initialize disk cache for persistent storage
        if cache_enabled:
            self._cache = diskcache.Cache(
                cache_dir,
                size_limit=1024 * 1024 * 1024,  # 1GB cache limit
                eviction_policy='least-recently-used',
                statistics=True
            )
            # Also keep a small in-memory cache for fast access
            self._memory_cache: dict[str, float] = {}
            self._memory_cache_limit = 1000
        else:
            self._cache = None
            self._memory_cache = {}

    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual.

        Returns:
            Fitness score between 0.0 and 1.0
        """
        # Check memory cache first
        if self.cache_enabled and individual.id in self._memory_cache:
            return self._memory_cache[individual.id]

        # Check disk cache
        if self.cache_enabled and self._cache:
            try:
                cached_fitness = self._cache.get(individual.id)
                if cached_fitness is not None:
                    # Store in memory cache for faster access
                    self._update_memory_cache(individual.id, cached_fitness)
                    return cached_fitness
            except Exception:
                # Continue if cache access fails
                pass

        start_time = time.time()

        try:
            # Execute program on training examples
            total_similarity = 0.0
            execution_results = {}

            for i, example in enumerate(self.task.train_examples):
                input_grid = example['input']
                expected_output = example['output']

                # Execute program
                result = self.dsl_engine.execute_program(
                    individual.operations,
                    input_grid
                )

                if result['success']:
                    output_grid = result['output']
                    execution_results[f'train_{i}'] = output_grid

                    # Store execution result for co-evolution (Task 7.4)
                    if i == 0:  # Store first example result
                        individual.metadata['execution_result'] = output_grid

                    # Calculate similarity
                    similarity = self._calculate_grid_similarity(output_grid, expected_output)
                    total_similarity += similarity

                    # Early termination if perfect
                    if similarity >= self.early_termination_threshold:
                        individual.fitness = similarity
                        if self.cache_enabled:
                            self._cache[individual.id] = similarity
                        return similarity
                else:
                    # Failed execution
                    continue

            # Calculate average similarity across training examples
            if self.task.train_examples:
                avg_similarity = total_similarity / len(self.task.train_examples)
            else:
                avg_similarity = 0.0

            # Calculate other fitness components
            execution_time = time.time() - start_time
            program_length_penalty = 1.0 / (1.0 + individual.program_length() / 10.0)
            complexity_penalty = self._calculate_complexity_penalty(individual)
            time_penalty = 1.0 / (1.0 + execution_time)

            # Combine fitness components
            fitness = (
                self.weights['grid_similarity'] * avg_similarity +
                self.weights['program_length'] * program_length_penalty +
                self.weights['complexity'] * complexity_penalty +
                self.weights['execution_time'] * time_penalty
            )

            # Cache results
            individual.cached_execution = execution_results
            individual.fitness = fitness
            if self.cache_enabled:
                self._update_caches(individual.id, fitness)

            return fitness

        except (TimeoutError, MemoryError) as e:
            # Handle resource limit errors
            individual.metadata['evaluation_error'] = f"Resource limit exceeded: {str(e)}"
            individual.metadata['exceeded_limits'] = True
            individual.metadata['error_type'] = 'timeout' if isinstance(e, TimeoutError) else 'memory'
            fitness = 0.0
            if self.cache_enabled:
                self._update_caches(individual.id, fitness)
            return fitness
        except Exception as e:
            # Handle other evaluation errors gracefully
            individual.metadata['evaluation_error'] = str(e)
            individual.metadata['error_type'] = 'execution'
            fitness = 0.0
            if self.cache_enabled:
                self._update_caches(individual.id, fitness)
            return fitness

    def _calculate_grid_similarity(self, output: Grid, expected: Grid) -> float:
        """
        Calculate similarity between two grids with partial credit.

        Implements multiple scoring mechanisms:
        - Exact match ratio (pixel-wise accuracy)
        - Shape similarity bonus
        - Pattern matching partial credit
        - Color distribution similarity
        """
        output_array = np.array(output)
        expected_array = np.array(expected)

        # Base score components
        exact_match_weight = 0.6
        shape_weight = 0.15
        pattern_weight = 0.15
        color_weight = 0.1

        # 1. Exact match ratio (primary metric)
        if output_array.shape == expected_array.shape:
            matches = (output_array == expected_array).sum()
            total = output_array.size
            exact_match_score = matches / total if total > 0 else 0.0
        else:
            # Partial credit for shape mismatch
            exact_match_score = 0.0

        # 2. Shape similarity bonus
        shape_score = self._calculate_shape_similarity(output_array.shape, expected_array.shape)

        # 3. Pattern matching partial credit
        pattern_score = self._calculate_pattern_similarity(output_array, expected_array)

        # 4. Color distribution similarity
        color_score = self._calculate_color_similarity(output_array, expected_array)

        # Combine scores with weights
        total_score = (
            exact_match_weight * exact_match_score +
            shape_weight * shape_score +
            pattern_weight * pattern_score +
            color_weight * color_score
        )

        # Bonus for perfect match
        if exact_match_score == 1.0:
            total_score = 1.0

        return min(1.0, total_score)

    def _calculate_complexity_penalty(self, individual: Individual) -> float:
        """
        Calculate complexity penalty for the program.

        Considers:
        - Operation diversity (variety of operations used)
        - Control flow complexity (loops, conditionals)
        - Parameter complexity (numeric values, ranges)
        """
        operations = individual.operations

        # 1. Operation diversity score
        unique_ops = len({op.get_name() for op in operations})
        total_ops = len(operations)
        diversity_score = unique_ops / total_ops if total_ops > 0 else 0.0

        # 2. Control flow complexity
        control_flow_ops = {'repeat', 'conditional', 'iterate', 'map', 'filter'}
        control_flow_count = sum(1 for op in operations if op.get_name() in control_flow_ops)
        control_flow_ratio = control_flow_count / total_ops if total_ops > 0 else 0.0

        # 3. Parameter complexity
        param_complexity = 0.0
        for op in operations:
            if hasattr(op, 'parameters') and op.parameters:
                # More parameters = more complex
                param_count = len(op.parameters) if isinstance(op.parameters, list | dict) else 1
                param_complexity += param_count

        avg_param_complexity = param_complexity / total_ops if total_ops > 0 else 0.0
        normalized_param_complexity = 1.0 / (1.0 + avg_param_complexity / 3.0)  # Normalize

        # Combine scores (lower complexity is better)
        complexity_score = (
            0.4 * diversity_score +  # Reward diverse operations
            0.3 * (1.0 - control_flow_ratio) +  # Penalize too many control flows
            0.3 * normalized_param_complexity  # Prefer simpler parameters
        )

        return complexity_score

    def _calculate_shape_similarity(self, shape1: tuple, shape2: tuple) -> float:
        """Calculate similarity between two shapes."""
        if shape1 == shape2:
            return 1.0

        # Calculate dimensional similarity
        height_ratio = min(shape1[0], shape2[0]) / max(shape1[0], shape2[0])
        width_ratio = min(shape1[1], shape2[1]) / max(shape1[1], shape2[1])

        # Average of dimension ratios
        return (height_ratio + width_ratio) / 2.0

    def _calculate_pattern_similarity(self, output: np.ndarray, expected: np.ndarray) -> float:
        """
        Calculate pattern similarity using local feature matching.

        This provides partial credit for solutions that have correct local patterns
        but might be shifted, scaled, or have minor errors.
        """
        # If shapes don't match, resize for comparison
        if output.shape != expected.shape:
            # Resize to smaller shape
            min_height = min(output.shape[0], expected.shape[0])
            min_width = min(output.shape[1], expected.shape[1])
            output_cropped = output[:min_height, :min_width]
            expected_cropped = expected[:min_height, :min_width]
        else:
            output_cropped = output
            expected_cropped = expected

        # Calculate local pattern matches using sliding windows
        window_size = 3
        pattern_matches = 0
        total_windows = 0

        if output_cropped.shape[0] >= window_size and output_cropped.shape[1] >= window_size:
            for i in range(output_cropped.shape[0] - window_size + 1):
                for j in range(output_cropped.shape[1] - window_size + 1):
                    output_window = output_cropped[i:i+window_size, j:j+window_size]
                    expected_window = expected_cropped[i:i+window_size, j:j+window_size]

                    if np.array_equal(output_window, expected_window):
                        pattern_matches += 1
                    total_windows += 1

            return pattern_matches / total_windows if total_windows > 0 else 0.0
        else:
            # Fallback for small grids
            return 1.0 if np.array_equal(output_cropped, expected_cropped) else 0.0

    def _calculate_color_similarity(self, output: np.ndarray, expected: np.ndarray) -> float:
        """Calculate similarity of color distributions."""
        # Get unique colors and their counts
        output_colors, output_counts = np.unique(output, return_counts=True)
        expected_colors, expected_counts = np.unique(expected, return_counts=True)

        # Normalize counts to probabilities
        output_probs = output_counts / output_counts.sum()
        expected_probs = expected_counts / expected_counts.sum()

        # Calculate similarity score based on color distribution overlap
        similarity = 0.0
        for color in range(10):  # ARC uses colors 0-9
            p1 = output_probs[output_colors == color]
            p2 = expected_probs[expected_colors == color]
            p1 = p1[0] if len(p1) > 0 else 0.0
            p2 = p2[0] if len(p2) > 0 else 0.0
            # Use min as similarity measure
            similarity += min(p1, p2)

        return similarity

    def _update_memory_cache(self, key: str, value: float) -> None:
        """Update memory cache with LRU eviction."""
        if len(self._memory_cache) >= self._memory_cache_limit:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        self._memory_cache[key] = value

    def _update_caches(self, key: str, value: float) -> None:
        """Update both memory and disk caches."""
        # Update memory cache
        self._update_memory_cache(key, value)

        # Update disk cache
        if self._cache:
            try:
                self._cache.set(key, value)
            except Exception:
                # Ignore cache write errors
                pass

    def clear_cache(self) -> None:
        """Clear fitness caches."""
        self._memory_cache.clear()
        if self._cache:
            try:
                self._cache.clear()
            except Exception:
                pass

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory_cache_size': len(self._memory_cache),
            'memory_cache_limit': self._memory_cache_limit
        }

        if self._cache:
            try:
                stats.update({
                    'disk_cache_size': len(self._cache),
                    'disk_cache_volume': self._cache.volume(),
                    'disk_cache_hits': self._cache.stats()[0],
                    'disk_cache_misses': self._cache.stats()[1]
                })
            except Exception:
                pass

        return stats


class ConvergenceTracker:
    """
    Tracks convergence criteria for evolution termination.

    Monitors fitness improvement, generation count, and stagnation.
    """

    def __init__(self, config: GeneticAlgorithmConfig):
        """Initialize convergence tracker."""
        self.config = config
        self.generations_since_improvement = 0
        self.last_best_fitness = 0.0
        self.start_time = time.time()
        self.min_programs_target = 500  # Minimum programs to generate

    def has_converged(self, population: Population, total_programs: int = 0) -> bool:
        """Check if evolution has converged."""
        # Don't converge until we've generated minimum programs
        if total_programs < self.min_programs_target:
            # But still respect time limit
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 300:  # 5 minute limit
                return True
            return False

        # Check generation limit
        if population.generation >= self.config.convergence.max_generations:
            return True

        # Check stagnation
        current_best = population.best_individual.fitness if population.best_individual else 0.0
        fitness_improvement = current_best - self.last_best_fitness

        if fitness_improvement < self.config.convergence.min_fitness_improvement:
            self.generations_since_improvement += 1
        else:
            self.generations_since_improvement = 0
            self.last_best_fitness = current_best

        if self.generations_since_improvement >= self.config.convergence.stagnation_patience:
            return True

        # Check time limit (5 minutes = 300 seconds)
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 300:  # 5 minute limit
            return True

        return False


class EvolutionEngine:
    """
    Main evolution engine for genetic algorithm-based program synthesis.

    Implements the complete genetic algorithm pipeline including initialization,
    selection, crossover, mutation, evaluation, and convergence detection.
    """

    def __init__(
        self,
        config: GeneticAlgorithmConfig,
        dsl_engine: DSLEngine,
        operation_templates: list[list[Operation]] | None = None,
        experiment_orchestrator: ExperimentOrchestrator | None = None,
        smart_model_router: Any | None = None,
        bandit_controller: Any | None = None,
        task_feature_extractor: Any | None = None
    ):
        """
        Initialize evolution engine.

        Args:
            config: Genetic algorithm configuration
            dsl_engine: DSL engine for program execution
            operation_templates: Optional templates for seeding population
            experiment_orchestrator: Optional orchestrator for experiment tracking
            smart_model_router: Optional smart model router for LLM-based initialization (Task 7.3)
            bandit_controller: Optional bandit controller for strategy selection (Task 8)
            task_feature_extractor: Optional feature extractor for contextual bandits (Task 8)
        """
        self.config = config
        self.dsl_engine = dsl_engine
        self.operation_templates = operation_templates or []
        self.population = Population()
        self.fitness_evaluator: FitnessEvaluator | None = None
        self.convergence_tracker = ConvergenceTracker(config)
        self.experiment_orchestrator = experiment_orchestrator
        self.smart_model_router = smart_model_router
        self.bandit_controller = bandit_controller
        self.task_feature_extractor = task_feature_extractor

        # Apply platform-specific configuration overrides (Task 8.3)
        self._apply_platform_overrides()

        # Initialize random seed for reproducibility
        if config.reproducibility.seed is not None:
            random.seed(config.reproducibility.seed)
            np.random.seed(config.reproducibility.seed)
            # Store seed for logging
            self.random_seed = config.reproducibility.seed
        else:
            # Generate and store seed for this run
            self.random_seed = random.randint(0, 2**32 - 1)
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Initialize visualizer for genealogy tracking
        from src.adapters.strategies.evolution_visualization import EvolutionVisualizer
        self.visualizer = EvolutionVisualizer()

        # Initialize genetic operators
        self._init_operators()

        # Initialize parallel processing
        if config.parallelization.workers > 0:
            self.executor = ProcessPoolExecutor(max_workers=config.parallelization.workers)
        else:
            self.executor = None  # No parallel processing

        # Experiment tracking
        self.experiment_id: str | None = None
        self.total_programs_generated = 0
        self.generation_times: list[float] = []

        # Memory management
        self.memory_limit_mb = config.performance.memory_limit
        self.last_memory_check = time.time()
        self.memory_cleanup_frequency = 5  # Check every 5 generations

        # Operator success tracking for Task 5.3
        self.mutation_success_count = 0
        self.mutation_total_count = 0
        self.crossover_success_count = 0
        self.crossover_total_count = 0
        self.operator_success_by_type = {}

        # Performance monitoring
        self.performance_metrics = {
            "evaluation_times": [],
            "generation_times": [],
            "memory_usage": [],
            "programs_per_second": [],
            "cache_hit_rate": 0.0,
            "evaluation_success_rate": 0.0
        }

        # Track all individuals across generations for export
        self.all_individuals_history: list[Individual] = []

        # Co-evolution support (Task 7.4)
        self.coevolution_enabled = False
        self.coevolution_engine = None

        # Initialize checkpoint manager for Task 6.3
        self._checkpoint_manager = None
        if config.reproducibility.checkpoint_enabled:
            try:
                from src.utils.checkpoint_manager import CheckpointManager
                checkpoint_dir = Path(config.reproducibility.checkpoint_dir)
                self._checkpoint_manager = CheckpointManager(str(checkpoint_dir))
            except ImportError as e:
                print(f"Warning: Checkpoint manager unavailable: {e}")
                self._checkpoint_manager = None

        # Initialize generation strategies for Task 8
        self._init_generation_strategies()

    def _init_operators(self) -> None:
        """Initialize genetic operators based on configuration."""
        from src.adapters.strategies.diversity_mechanisms import (
            FitnessSharing,
            HybridDiversity,
            NoveltySearch,
            Speciation,
        )
        from src.adapters.strategies.genetic_operators import (
            AdaptiveMutation,
            InsertDeleteMutation,
            OperationReplacementMutation,
            ParameterMutation,
            ReorderMutation,
            SinglePointCrossover,
            SubtreeCrossover,
            UniformCrossover,
        )

        # Initialize crossover operators
        self.crossover_operators = {
            "single_point": SinglePointCrossover(),
            "uniform": UniformCrossover(),
            "subtree": SubtreeCrossover()
        }

        # Initialize mutation operators
        self.mutation_operators = {
            "operation_replace": OperationReplacementMutation(),
            "parameter_mutate": ParameterMutation(),
            "insert_delete": InsertDeleteMutation(),
            "reorder": ReorderMutation(),
            "adaptive": AdaptiveMutation()
        }

        # Initialize enhanced adaptive mutation for Task 7.2
        if self.config.genetic_operators.mutation.adaptive:
            from src.adapters.strategies.adaptive_evolution import (
                AdaptiveMutationController,
                EnhancedAdaptiveMutation,
                create_adaptive_mutation_schedule,
            )

            # Create mutation schedule based on config
            schedule = create_adaptive_mutation_schedule(
                strategy=getattr(self.config.genetic_operators.mutation, 'adaptive_strategy', 'fitness_based'),
                base_rate=self.config.genetic_operators.mutation.base_rate,
                max_rate=self.config.genetic_operators.mutation.max_rate
            )

            # Create controller
            self.adaptive_controller = AdaptiveMutationController(schedule)

            # Create enhanced mutation with base operators
            base_operators = [
                self.mutation_operators["operation_replace"],
                self.mutation_operators["parameter_mutate"],
                self.mutation_operators["insert_delete"],
                self.mutation_operators["reorder"]
            ]

            self.enhanced_adaptive_mutation = EnhancedAdaptiveMutation(
                controller=self.adaptive_controller,
                base_operators=base_operators
            )

            # Replace adaptive mutation with enhanced version
            self.mutation_operators["adaptive"] = self.enhanced_adaptive_mutation
        else:
            self.adaptive_controller = None
            self.enhanced_adaptive_mutation = None

        # Initialize diversity mechanism
        diversity_method = self.config.diversity.method
        if diversity_method == "fitness_sharing":
            self.diversity_mechanism = FitnessSharing(
                niche_radius=self.config.diversity.niche_radius
            )
        elif diversity_method == "speciation":
            self.diversity_mechanism = Speciation(
                compatibility_threshold=self.config.diversity.species_threshold
            )
        elif diversity_method == "novelty":
            self.diversity_mechanism = NoveltySearch()
        else:
            self.diversity_mechanism = HybridDiversity()

        # Initialize novelty search for Task 7.5
        self.novelty_search_enabled = False
        self.novelty_search_engine = None

        # Check if novelty search is configured
        if hasattr(self.config, 'novelty_search') and self.config.novelty_search.get('enabled', False):
            from src.adapters.strategies.novelty_search import (
                NoveltySearch as EnhancedNoveltySearch,
            )
            self.novelty_search_enabled = True
            self.novelty_search_engine = EnhancedNoveltySearch(
                archive_size=self.config.novelty_search.get('archive_size', 2000),
                min_novelty_distance=self.config.novelty_search.get('min_distance', 0.1),
                novelty_weight=self.config.novelty_search.get('novelty_weight', 0.5),
                k_neighbors=self.config.novelty_search.get('k_neighbors', 15)
            )

    def _apply_platform_overrides(self) -> None:
        """Apply platform-specific configuration overrides (Task 8.3)."""
        # Import platform detector
        from src.infrastructure.components.platform_detector import Platform, get_platform_detector

        # Detect current platform
        detector = get_platform_detector()
        platform_info = detector.detect_platform()
        current_platform = platform_info.platform

        # Check for platform overrides in config
        if hasattr(self.config, 'platform_overrides') and self.config.platform_overrides:
            platform_overrides = self.config.platform_overrides

            # Map platform enum to string key
            platform_key_map = {
                Platform.KAGGLE: 'kaggle',
                Platform.COLAB: 'colab',
                Platform.PAPERSPACE: 'paperspace',
                Platform.LOCAL: 'local'
            }

            platform_key = platform_key_map.get(current_platform)
            if platform_key and platform_key in platform_overrides:
                override_config = platform_overrides[platform_key]

                # Apply parallelization overrides
                if 'parallelization' in override_config:
                    para_overrides = override_config['parallelization']
                    if 'workers' in para_overrides:
                        self.config.parallelization.workers = para_overrides['workers']
                    if 'batch_size' in para_overrides:
                        self.config.parallelization.batch_size = para_overrides['batch_size']
                    if 'gpu_acceleration' in para_overrides:
                        self.config.parallelization.gpu_acceleration = para_overrides['gpu_acceleration']
                    if 'gpu_batch_size' in para_overrides:
                        self.config.parallelization.gpu_batch_size = para_overrides['gpu_batch_size']

                # Apply performance overrides
                if 'performance' in override_config:
                    perf_overrides = override_config['performance']
                    if 'memory_limit' in perf_overrides:
                        self.config.performance.memory_limit = perf_overrides['memory_limit']
                    if 'generation_timeout' in perf_overrides:
                        self.config.performance.generation_timeout = perf_overrides['generation_timeout']
                    if 'program_timeout' in perf_overrides:
                        self.config.performance.program_timeout = perf_overrides['program_timeout']

                # Apply convergence overrides
                if 'convergence' in override_config:
                    conv_overrides = override_config['convergence']
                    if 'max_generations' in conv_overrides:
                        self.config.convergence.max_generations = conv_overrides['max_generations']
                    if 'stagnation_patience' in conv_overrides:
                        self.config.convergence.stagnation_patience = conv_overrides['stagnation_patience']

                # Log platform-specific configuration
                print(f"Applied {platform_key} platform overrides:")
                print(f"  Workers: {self.config.parallelization.workers}")
                print(f"  Batch size: {self.config.parallelization.batch_size}")
                print(f"  Memory limit: {self.config.performance.memory_limit}MB")
                print(f"  GPU acceleration: {self.config.parallelization.gpu_acceleration}")

        # Store platform info for later use
        self.platform_info = platform_info
        self.current_platform = current_platform

        # Initialize resource monitoring (Task 8.5)
        self._init_resource_monitoring()

    def _init_resource_monitoring(self) -> None:
        """Initialize platform-specific resource monitoring (Task 8.5)."""
        # Resource monitoring configuration
        self.resource_monitor_config = {
            'enabled': True,
            'check_interval': 60,  # seconds
            'memory_threshold': 0.9,
            'cpu_threshold': 0.95,
            'gpu_memory_threshold': 0.95,
            'throttle_enabled': True
        }

        # Platform-specific monitoring settings
        if hasattr(self, 'current_platform'):
            from src.infrastructure.components.platform_detector import Platform

            if self.current_platform == Platform.KAGGLE:
                self.resource_monitor_config.update({
                    'session_time_limit': 12 * 3600,  # 12 hours
                    'gpu_quota_hours': 30,  # Weekly quota
                    'check_interval': 30,  # More frequent checks
                    'memory_threshold': 0.85  # More conservative
                })
            elif self.current_platform == Platform.COLAB:
                self.resource_monitor_config.update({
                    'session_time_limit': 12 * 3600,
                    'gpu_quota_hours': 12,  # Daily quota
                    'check_interval': 30,
                    'memory_threshold': 0.9
                })
            elif self.current_platform == Platform.PAPERSPACE:
                self.resource_monitor_config.update({
                    'session_time_limit': 6 * 3600,
                    'gpu_quota_hours': 6,
                    'check_interval': 20,
                    'memory_threshold': 0.8  # Very conservative
                })

        # Initialize monitoring state
        self.resource_monitor_state = {
            'start_time': time.time(),
            'last_check': time.time(),
            'throttle_active': False,
            'throttle_level': 0,  # 0-3 (none, light, medium, heavy)
            'warnings_issued': []
        }

    async def _check_and_throttle_resources(self) -> None:
        """Check resource usage and apply throttling if needed (Task 8.5)."""
        if not self.resource_monitor_config['enabled']:
            return

        current_time = time.time()
        if current_time - self.resource_monitor_state['last_check'] < self.resource_monitor_config['check_interval']:
            return

        self.resource_monitor_state['last_check'] = current_time

        # Check memory usage
        memory_usage_ratio = self._get_memory_usage_mb() / self.config.performance.memory_limit

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Check GPU if available
        gpu_memory_ratio = 0.0
        if self.config.parallelization.gpu_acceleration:
            gpu_memory_ratio = await self._get_gpu_memory_usage_ratio()

        # Check session time
        session_time = current_time - self.resource_monitor_state['start_time']
        time_limit = self.resource_monitor_config.get('session_time_limit', float('inf'))
        time_ratio = session_time / time_limit if time_limit > 0 else 0

        # Determine throttle level
        old_throttle_level = self.resource_monitor_state['throttle_level']
        new_throttle_level = self._calculate_throttle_level(
            memory_usage_ratio, cpu_percent / 100, gpu_memory_ratio, time_ratio
        )

        # Apply throttling if changed
        if new_throttle_level != old_throttle_level:
            await self._apply_throttling(new_throttle_level)
            self.resource_monitor_state['throttle_level'] = new_throttle_level

        # Issue warnings if needed
        if memory_usage_ratio > self.resource_monitor_config['memory_threshold']:
            warning = f"High memory usage: {memory_usage_ratio:.1%}"
            if warning not in self.resource_monitor_state['warnings_issued']:
                print(f"WARNING: {warning}")
                self.resource_monitor_state['warnings_issued'].append(warning)

        if time_ratio > 0.9:
            warning = f"Session time limit approaching: {time_ratio:.1%}"
            if warning not in self.resource_monitor_state['warnings_issued']:
                print(f"WARNING: {warning}")
                self.resource_monitor_state['warnings_issued'].append(warning)

    def _calculate_throttle_level(
        self,
        memory_ratio: float,
        cpu_ratio: float,
        gpu_ratio: float,
        time_ratio: float
    ) -> int:
        """Calculate appropriate throttle level based on resource usage."""
        # Heavy throttling conditions
        if memory_ratio > 0.95 or time_ratio > 0.95:
            return 3

        # Medium throttling
        if memory_ratio > 0.9 or gpu_ratio > 0.9 or time_ratio > 0.9:
            return 2

        # Light throttling
        if memory_ratio > 0.85 or cpu_ratio > 0.9 or gpu_ratio > 0.85 or time_ratio > 0.8:
            return 1

        # No throttling needed
        return 0

    async def _apply_throttling(self, throttle_level: int) -> None:
        """Apply resource throttling based on level."""
        if throttle_level == 0:
            # Remove throttling
            self.resource_monitor_state['throttle_active'] = False
            print("Resource throttling removed")
            return

        self.resource_monitor_state['throttle_active'] = True
        print(f"Applying throttle level {throttle_level}")

        if throttle_level == 1:
            # Light throttling
            self.config.parallelization.batch_size = max(50, self.config.parallelization.batch_size // 2)
            self.config.performance.generation_timeout = 40

        elif throttle_level == 2:
            # Medium throttling
            self.config.parallelization.batch_size = max(25, self.config.parallelization.batch_size // 4)
            self.config.parallelization.workers = max(1, self.config.parallelization.workers - 1)
            self.config.performance.generation_timeout = 50
            self.config.population.size = max(100, self.config.population.size // 2)

        elif throttle_level == 3:
            # Heavy throttling
            self.config.parallelization.batch_size = 10
            self.config.parallelization.workers = 1
            self.config.performance.generation_timeout = 60
            self.config.population.size = 100
            self.config.convergence.early_stop = True
            # Force garbage collection
            gc.collect()

    async def _get_gpu_memory_usage_ratio(self) -> float:
        """Get GPU memory usage ratio."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                max_memory = torch.cuda.max_memory_allocated()
                if max_memory > 0:
                    return allocated / max_memory
        except ImportError:
            pass
        return 0.0

    async def _deserialize_operations(self, program_dict: list[dict[str, Any]]) -> list[Operation]:
        """Deserialize operation dictionaries into Operation objects."""
        # This is a placeholder - in real implementation would use DSL registry
        operations = []

        # For now, create mock operations
        from src.domain.dsl.base import Operation

        class MockOperation(Operation):
            def __init__(self, name: str, **params):
                self._name = name
                self.parameters = params  # Store params directly without validation
                # Don't call super().__init__ to avoid parameter validation

            def execute(self, grid, context=None):
                return {"success": True, "grid": grid}

            def get_name(self):
                return self._name

            def get_description(self):
                return f"Mock operation: {self._name}"

            def get_parameter_schema(self):
                # Accept any parameters for mock operations
                return {
                    'angle': {'type': 'int', 'required': False},
                    'axis': {'type': 'str', 'required': False},
                    'direction': {'type': 'str', 'required': False},
                    'factor': {'type': 'int', 'required': False},
                    'size': {'type': 'int', 'required': False},
                    'color': {'type': 'int', 'required': False},
                    'pattern': {'type': 'str', 'required': False},
                    'value': {'type': 'int', 'required': False},
                    'source_color': {'type': 'int', 'required': False},
                    'target_color': {'type': 'int', 'required': False},
                    'offset': {'type': 'int', 'required': False},
                    'threshold': {'type': 'float', 'required': False},
                    'min_size': {'type': 'int', 'required': False},
                    'max_size': {'type': 'int', 'required': False}
                }

        for op_dict in program_dict:
            op = MockOperation(
                name=op_dict.get("name", "unknown"),
                **op_dict.get("parameters", {})
            )
            operations.append(op)

        return operations

    async def evolve(
        self,
        task: ARCTask,
        callbacks: list[Callable] | None = None,
        experiment_name: str | None = None
    ) -> tuple[Individual, dict[str, Any]]:
        """
        Run genetic algorithm evolution for the given task.

        Args:
            task: ARC task to solve
            callbacks: Optional callbacks for generation updates
            experiment_name: Optional name for experiment tracking

        Returns:
            Tuple of (best individual, evolution statistics)
        """
        # Check if island model is enabled (Task 7.1)
        if self.config.island_model and self.config.island_model.enabled:
            return await self._evolve_with_islands(task, callbacks, experiment_name)

        # Otherwise use standard evolution
        # Store current task for evaluation
        self.current_task = task
        evolution_start_time = time.time()

        # Create experiment if orchestrator available
        if self.experiment_orchestrator and experiment_name:
            experiment_config = ExperimentConfig(
                name=experiment_name,
                description=f"Evolution search for task {task.task_id}",
                strategy_type=StrategyType.EVOLUTION,
                dataset_tasks=[task.task_id],
                parameters={
                    "population_size": self.config.population.size,
                    "max_generations": self.config.convergence.max_generations,
                    "crossover_rate": self.config.genetic_operators.crossover.rate,
                    "mutation_rate": self.config.genetic_operators.mutation.base_rate
                },
                tags=["evolution", "program_synthesis", task.task_id]
            )

            experiment_resources = ExperimentResources(
                memory_gb=self.config.performance.memory_limit / 1024,
                cpu_cores=self.config.parallelization.workers,
                max_runtime_hours=0.083,  # 5 minutes = 0.083 hours
                gpu_required=self.config.parallelization.gpu_acceleration,
                platform_preference=None
            )

            experiment = self.experiment_orchestrator.create_experiment(
                name=experiment_name,
                description=experiment_config.description,
                strategy_type=experiment_config.strategy_type,
                dataset_tasks=experiment_config.dataset_tasks,
                priority=ExperimentPriority.NORMAL,
                resources=experiment_resources,
                parameters=experiment_config.parameters,
                tags=experiment_config.tags
            )

            self.experiment_id = experiment.id
            await self.experiment_orchestrator.add_experiment(experiment)

        # Initialize fitness evaluator for this task
        self.fitness_evaluator = FitnessEvaluator(
            task=task,
            dsl_engine=self.dsl_engine,
            weights=self.config.fitness.metrics,
            cache_enabled=self.config.fitness.cache_enabled,
            early_termination_threshold=self.config.fitness.early_termination.get('threshold', 0.95)
        )

        # Initialize population
        await self._initialize_population(task)

        # Initialize co-evolution if enabled (Task 7.4)
        if self.config.coevolution and self.config.coevolution.get('enabled', False):
            await self._initialize_coevolution(task)

        # Evolution loop
        while not self.convergence_tracker.has_converged(self.population, self.total_programs_generated):
            generation_start_time = time.time()

            # Check and apply resource throttling (Task 8.5)
            await self._check_and_throttle_resources()

            # Evaluate fitness
            await self._evaluate_population()

            # Update statistics
            self.population.update_generation_stats()

            # Co-evolution update (Task 7.4)
            if self.coevolution_engine:
                self.population, best_fitness_func = self.coevolution_engine.coevolve_generation(
                    self.population, task
                )
                # Store best fitness function for export
                self.population.metadata['best_fitness_function'] = best_fitness_func

            # Update adaptive mutation controller (Task 7.2)
            if self.adaptive_controller:
                self.adaptive_controller.update_population_stats(self.population)

                # Update mutation success tracking for individuals
                if self.enhanced_adaptive_mutation:
                    for individual in self.population.individuals:
                        self.enhanced_adaptive_mutation.update_success(individual)

            # Track all individuals for export
            self.all_individuals_history.extend(self.population.individuals)

            # Save checkpoint if enabled
            if self._checkpoint_manager and self.population.generation % 10 == 0:
                await self._save_checkpoint()

            # Record visualization data
            self.visualizer.record_generation(self.population)
            self.visualizer.record_genealogy(self.population)

            # Checkpoint genealogy data periodically for large runs
            if self.population.generation % 10 == 0:  # Every 10 generations
                self._checkpoint_genealogy_data()

            # Check if we've generated enough programs (500+ target)
            if self.total_programs_generated >= 500 and self.population.best_individual:
                # If we have a good solution, consider stopping
                if self.population.best_individual.fitness >= 0.45:  # 45% accuracy target
                    break

            # Track generation time
            generation_time = time.time() - generation_start_time
            self.generation_times.append(generation_time)

            # Update experiment progress if tracking
            if self.experiment_orchestrator and self.experiment_id:
                progress = ExperimentProgress(
                    current_epoch=self.population.generation,
                    total_epochs=self.config.convergence.max_generations,
                    accuracy=self.population.best_individual.fitness if self.population.best_individual else 0.0,
                    elapsed_time_seconds=time.time() - evolution_start_time,
                    custom_metrics={
                        "unique_programs": self.population.diversity_metrics.get('unique_programs', 0),
                        "average_fitness": self.population.average_fitness(),
                        "programs_generated": self.total_programs_generated
                    }
                )
                await self.experiment_orchestrator.update_experiment_progress(
                    self.experiment_id,
                    progress
                )

            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self.population)

            # Check for early stopping
            if self._should_stop_early():
                break

            # Create next generation
            await self._create_next_generation()

            # Manage memory
            await self._manage_memory()

            # Increment generation
            self.population.increment_generation()

        # Export visualization data
        try:
            generation_data_path = self.visualizer.export_generation_data()
            fitness_history_path = self.visualizer.export_fitness_history()
            genealogy_data_path = self.visualizer.export_genealogy_data()
            evolution_report_path = self.visualizer.generate_html_report()
            genealogy_viz_path = self.visualizer.generate_genealogy_html()

            print("Visualization data exported:")
            print(f"  - Generation data: {generation_data_path}")
            print(f"  - Fitness history: {fitness_history_path}")
            print(f"  - Genealogy data: {genealogy_data_path}")
            print(f"  - Evolution report: {evolution_report_path}")
            print(f"  - Genealogy visualization: {genealogy_viz_path}")

            # Perform and export lineage analysis
            lineage_analysis = self.analyze_lineage_paths()
            lineage_path = self.visualizer.output_dir / "lineage_analysis.json"
            with open(lineage_path, 'w') as f:
                json.dump(lineage_analysis, f, indent=2, default=str)
            print(f"  - Lineage analysis: {lineage_path}")

            # Export top programs
            export_path = self.export_top_programs(top_n=10)
            print(f"  - Top programs exported to: {export_path}")
        except Exception as e:
            print(f"Warning: Failed to export visualization data: {e}")

        # Final cleanup and return best individual and statistics
        self.cleanup()

        # Complete experiment if tracking
        if self.experiment_orchestrator and self.experiment_id:
            from src.domain.services.experiment_orchestrator import ExperimentResults
            results = ExperimentResults(
                final_accuracy=self.population.best_individual.fitness if self.population.best_individual else 0.0,
                best_accuracy=max(self.population.best_fitness_history) if self.population.best_fitness_history else 0.0,
                evaluation_metrics={
                    "total_programs_generated": self.total_programs_generated,
                    "generations": self.population.generation,
                    "unique_programs_ratio": self.population.diversity_metrics.get('unique_programs', 0)
                }
            )
            await self.experiment_orchestrator.complete_experiment(self.experiment_id, results)

        return self.population.best_individual, self._get_evolution_stats()

    def _init_generation_strategies(self) -> None:
        """Initialize generation strategies for bandit-based selection (Task 8)."""
        self.generation_strategies = {}

        try:
            from src.adapters.strategies.crossover_focused_generation import (
                CrossoverFocusedGenerationStrategy,
            )

            self.generation_strategies["crossover_focused"] = CrossoverFocusedGenerationStrategy(
                crossover_rate=0.95, mutation_rate=0.05
            )

            self.generation_strategies["dsl_mutation"] = CrossoverFocusedGenerationStrategy(
                crossover_rate=0.10, mutation_rate=0.90
            )
        except ImportError as e:
            logger.warning(f"Could not load crossover strategies: {e}")

        # Only initialize LLM strategies if smart_model_router is available
        if self.smart_model_router:
            try:
                from src.adapters.strategies.hybrid_initialization import HybridLLMProgramGenerator
                from src.adapters.strategies.pure_llm_generation import PureLLMGenerationStrategy

                self.generation_strategies["hybrid_init"] = HybridLLMProgramGenerator(
                    self.smart_model_router
                )
                self.generation_strategies["pure_llm"] = PureLLMGenerationStrategy(
                    self.smart_model_router
                )
            except ImportError as e:
                logger.warning(f"Could not load LLM strategies: {e}")

        # Add adaptive mutation if available
        if hasattr(self, 'adaptive_mutation_controller'):
            self.generation_strategies["adaptive_mutation"] = self.adaptive_mutation_controller

    async def _apply_generation_strategy(
        self, strategy_id: str, parent1: Individual, parent2: Individual, task: ARCTask
    ) -> list[Individual]:
        """
        Apply selected generation strategy to create offspring (Task 8).

        Args:
            strategy_id: Strategy selected by bandit controller
            parent1: First parent individual
            parent2: Second parent individual
            task: Current ARC task

        Returns:
            List of offspring individuals
        """
        if strategy_id not in self.generation_strategies:
            logger.warning(f"Unknown strategy {strategy_id}, using default crossover")
            return await self._apply_crossover(parent1, parent2)

        strategy = self.generation_strategies[strategy_id]

        try:
            if strategy_id in ["hybrid_init", "pure_llm"]:
                offspring = await strategy.generate_offspring(parent1, parent2, task)
            elif strategy_id in ["crossover_focused", "dsl_mutation"]:
                offspring = await strategy.generate_offspring(parent1, parent2, task)
            elif strategy_id == "adaptive_mutation":
                child1 = await self._apply_mutation(parent1)
                child2 = await self._apply_mutation(parent2)
                offspring = [child1, child2]
            else:
                offspring = await self._apply_crossover(parent1, parent2)

            for ind in offspring:
                ind.metadata["generation_strategy"] = strategy_id

            return offspring

        except Exception as e:
            logger.error(f"Strategy {strategy_id} failed: {e}, using default crossover")
            return await self._apply_crossover(parent1, parent2)

    async def _initialize_population(self, task: ARCTask) -> None:
        """Initialize population with diverse individuals."""
        # Task 7.3: Use hybrid initialization with LLM-generated programs
        init_method = self.config.population.initialization.get("method", "hybrid")

        if init_method == "hybrid" and self.config.population.initialization.get("use_llm", True):
            await self._initialize_population_hybrid(task)
        else:
            # Fallback to traditional initialization
            await self._initialize_population_traditional(task)

    async def _initialize_population_hybrid(self, task: ARCTask) -> None:
        """Initialize population using hybrid LLM approach (Task 7.3)."""
        from src.adapters.strategies.hybrid_initialization import (
            HybridLLMProgramGenerator,
            HybridPopulationInitializer,
        )

        # Check if we have smart model router for LLM generation
        llm_generator = None
        if hasattr(self, 'smart_model_router') and self.smart_model_router:
            llm_generator = HybridLLMProgramGenerator(
                model_router=self.smart_model_router,
                available_operations=self._get_available_operations()
            )

        # Create hybrid initializer
        initializer = HybridPopulationInitializer(
            llm_generator=llm_generator
        )

        # Initialize population
        individuals = await initializer.initialize_population(
            task=task,
            population_size=self.config.population.size,
            config=self.config.population.initialization
        )

        # Add individuals to population
        for individual in individuals:
            self.population.add_individual(individual)
            self.total_programs_generated += 1

    async def _initialize_population_traditional(self, task: ARCTask) -> None:
        """Traditional initialization without LLM (fallback)."""
        from src.adapters.strategies.operation_templates import (
            OperationTemplateGenerator,
            create_seed_programs,
        )

        generator = OperationTemplateGenerator()

        # Get initialization parameters
        init_config = self.config.population.initialization
        pop_size = self.config.population.size
        llm_ratio = init_config.get("llm_seed_ratio", 0.2)
        template_ratio = init_config.get("template_ratio", 0.5)

        # Calculate population segments
        num_llm = int(pop_size * llm_ratio)
        num_seeds = min(len(create_seed_programs()), int(pop_size * 0.1))
        num_remaining = pop_size - num_llm - num_seeds

        # 1. Add seed programs (known good patterns)
        seed_programs = create_seed_programs()
        for i in range(num_seeds):
            program_dict = seed_programs[i % len(seed_programs)]
            operations = await self._deserialize_operations(program_dict)
            individual = Individual(operations=operations)
            individual.metadata.update({
                'generation': 0,
                'creation_method': 'seed',
                'source': 'predefined_patterns',
                'lineage': [],
                'mutation_history': []
            })
            self.population.add_individual(individual)
            self.total_programs_generated += 1

        # 2. Generate template and random programs
        generated_programs = generator.generate_diverse_population(
            size=num_remaining,
            random_ratio=1.0 - template_ratio,
            template_ratio=template_ratio
        )

        for program_dict in generated_programs:
            operations = await self._deserialize_operations(program_dict)
            individual = Individual(operations=operations)
            individual.metadata.update({
                'generation': 0,
                'creation_method': 'generated',
                'source': 'template_or_random',
                'lineage': [],
                'mutation_history': []
            })
            self.population.add_individual(individual)
            self.total_programs_generated += 1

        # 3. LLM-generated programs (placeholder for now)
        # In real implementation, this would call SmartModelRouter
        for _ in range(num_llm):
            # For now, use hybrid programs as placeholder
            program_dict = generator.generate_hybrid_program()
            operations = await self._deserialize_operations(program_dict)
            individual = Individual(operations=operations)
            individual.metadata.update({
                'generation': 0,
                'creation_method': 'llm_generated',
                'source': 'llm_placeholder',
                'lineage': [],
                'mutation_history': []
            })
            self.population.add_individual(individual)
            self.total_programs_generated += 1

    async def _evaluate_population(self) -> None:
        """Evaluate fitness of all individuals in population."""
        eval_start_time = time.time()

        # Get unevaluated individuals
        unevaluated = [ind for ind in self.population.individuals if ind.fitness == 0.0]

        if not unevaluated:
            return

        total_evaluated = 0
        successful_evals = 0

        # Batch evaluation based on configuration
        batch_size = self.config.parallelization.batch_size

        # Store execution results for novelty search
        execution_results = {}

        # Check if pruning is enabled
        use_pruning = getattr(self.config, 'enable_pruning', False)
        if use_pruning:
            # Use pruning-enabled evaluation
            await self._evaluate_population_with_pruning(unevaluated)
            return

        for i in range(0, len(unevaluated), batch_size):
            batch = unevaluated[i:i + batch_size]
            batch_start_time = time.time()

            # Evaluate batch in parallel
            if self.config.parallelization.backend == "multiprocessing":
                await self._evaluate_batch_multiprocessing(batch)
            else:
                # Fallback to sequential for now
                await self._evaluate_batch_sequential(batch)

            # Track successful evaluations and collect execution results
            for ind in batch:
                total_evaluated += 1
                if ind.fitness > 0:
                    successful_evals += 1

                # Collect execution results for novelty search
                if hasattr(ind, 'cached_execution') and ind.cached_execution:
                    execution_results[ind.id] = ind.cached_execution
                elif ind.metadata.get('execution_result'):
                    execution_results[ind.id] = ind.metadata['execution_result']

            # Track batch time
            batch_time = time.time() - batch_start_time
            self.performance_metrics["evaluation_times"].append(batch_time)

        # Apply novelty search if enabled (Task 7.5)
        if self.novelty_search_enabled and self.novelty_search_engine and self.current_task:
            # Store objective fitness before modification
            objective_fitness = {ind.id: ind.fitness for ind in self.population.individuals}

            # Calculate novelty scores
            novelty_scores = self.novelty_search_engine.evaluate_novelty(
                self.population, self.current_task, execution_results
            )

            # Update fitness with combined novelty-fitness scores
            self.novelty_search_engine.update_population_fitness(
                self.population, objective_fitness, novelty_scores
            )

            # Track novelty metrics
            self.population.diversity_metrics['novelty_archive_size'] = len(self.novelty_search_engine.archive.behaviors)
            self.population.diversity_metrics['average_novelty'] = np.mean(list(novelty_scores.values())) if novelty_scores else 0.0

        # Update performance metrics
        eval_time = time.time() - eval_start_time
        if total_evaluated > 0:
            self.performance_metrics["evaluation_success_rate"] = successful_evals / total_evaluated
            self.performance_metrics["programs_per_second"].append(total_evaluated / eval_time)

        # Update cache hit rate if available
        if self.fitness_evaluator and hasattr(self.fitness_evaluator, '_cache'):
            cache_size = len(self.fitness_evaluator._cache)
            if cache_size > 0:
                self.performance_metrics["cache_hit_rate"] = cache_size / self.total_programs_generated

        # Task 8.4: Update bandit rewards after evaluation
        await self._update_bandit_rewards()

    async def _update_bandit_rewards(self) -> None:
        """
        Update bandit controller rewards based on offspring fitness (Task 8.4).

        Tracks fitness improvement and API costs for each generation strategy.
        """
        if not self.bandit_controller or not self.task_feature_extractor:
            return

        # Group individuals by generation strategy
        strategy_rewards = {}

        for individual in self.population.individuals:
            if "bandit_strategy" not in individual.metadata:
                continue

            strategy_id = individual.metadata["bandit_strategy"]

            if strategy_id not in strategy_rewards:
                strategy_rewards[strategy_id] = {"rewards": [], "costs": []}

            # Reward is the fitness score (0.0-1.0)
            reward = individual.fitness

            # Estimate cost based on strategy type
            cost = self._estimate_strategy_cost(strategy_id)

            strategy_rewards[strategy_id]["rewards"].append(reward)
            strategy_rewards[strategy_id]["costs"].append(cost)

        # Update bandit controller with average rewards per strategy
        for strategy_id, data in strategy_rewards.items():
            avg_reward = sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0.0
            avg_cost = sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0.0

            # Check for failures (very low fitness)
            is_failure = avg_reward < 0.1

            self.bandit_controller.update_reward(strategy_id, avg_reward, avg_cost, is_failure)

    def _estimate_strategy_cost(self, strategy_id: str) -> float:
        """
        Estimate API cost for a generation strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Estimated cost (0.0-1.0 normalized)
        """
        cost_map = {
            "pure_llm": 1.0,
            "hybrid_init": 0.7,
            "adaptive_mutation": 0.15,
            "dsl_mutation": 0.1,
            "crossover_focused": 0.1,
        }
        return cost_map.get(strategy_id, 0.2)

    async def _evaluate_batch_sequential(self, batch: list[Individual]) -> None:
        """Evaluate a batch of individuals sequentially."""
        for individual in batch:
            fitness = self.fitness_evaluator.evaluate(individual)
            individual.fitness = fitness

            # Store execution result if available (for novelty search)
            if hasattr(self.fitness_evaluator, 'last_execution_result'):
                individual.metadata['execution_result'] = self.fitness_evaluator.last_execution_result

    async def _evaluate_population_with_pruning(self, unevaluated: list[Individual]) -> None:
        """Evaluate population with intelligent pruning to save computation time."""
        from src.domain.dsl.base import DSLProgram
        from src.domain.models import PruningStrategy
        from src.domain.services.evaluation_service import EvaluationService

        # Initialize evaluation service with pruning if not already done
        if not hasattr(self, '_evaluation_service'):
            # Get pruning strategy from config
            pruning_strategy = PruningStrategy(
                strategy_id="evolution-balanced",
                name="Evolution Balanced Strategy",
                aggressiveness=0.5,
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=True,
                confidence_threshold=0.6,
                max_partial_ops=3,
                timeout_ms=100,
            )

            self._evaluation_service = EvaluationService(
                enable_gpu_evaluation=self.config.parallelization.gpu_acceleration,
                enable_pruning=True,
                default_pruning_strategy=pruning_strategy,
            )

        # Convert individuals to DSL programs
        programs = []
        for ind in unevaluated:
            # Create DSLProgram from individual
            program = DSLProgram(
                operations=[
                    {"name": op.get_name(), "params": op.parameters}
                    for op in ind.operations
                ],
                metadata={"individual_id": ind.id},
            )
            programs.append(program)

        # Get test inputs from current task
        test_inputs = []
        if self.current_task and hasattr(self.current_task, 'train_inputs'):
            test_inputs = self.current_task.train_inputs[:2]  # Use first 2 training examples

        # Evaluate with pruning
        eval_results, pruning_metrics = await self._evaluation_service.evaluate_with_pruning(
            programs=programs,
            test_inputs=test_inputs,
        )

        # Update individual fitness scores
        for _i, (ind, result) in enumerate(zip(unevaluated, eval_results, strict=False)):
            if result.metadata.get("pruned", False):
                # Program was pruned - assign low fitness
                ind.fitness = 0.01  # Small non-zero value
                ind.metadata["pruned"] = True
                ind.metadata["pruning_reason"] = result.metadata.get("pruning_reason", "Unknown")
            else:
                # Program was evaluated - get actual fitness
                if result.best_attempt:
                    ind.fitness = result.best_attempt.pixel_accuracy.accuracy
                else:
                    ind.fitness = result.final_accuracy
                ind.metadata["pruned"] = False

        # Log pruning statistics
        if pruning_metrics.programs_pruned > 0:
            print(f"Pruning saved {pruning_metrics.time_saved_ms:.0f}ms by rejecting {pruning_metrics.programs_pruned}/{pruning_metrics.total_programs} programs")
            self.performance_metrics["pruning_rate"] = pruning_metrics.pruning_rate
            self.performance_metrics["time_saved_by_pruning_ms"] = pruning_metrics.time_saved_ms

    async def _evaluate_batch_multiprocessing(self, batch: list[Individual]) -> None:
        """Evaluate a batch of individuals using multiprocessing."""
        from src.adapters.strategies.parallel_evaluation import (
            GPUAcceleratedEvaluator,
            ParallelEvaluator,
        )

        # Choose evaluator based on GPU availability
        evaluator_class = GPUAcceleratedEvaluator if self.config.parallelization.gpu_acceleration else ParallelEvaluator

        # Use parallel evaluator
        async with evaluator_class(
            num_workers=self.config.parallelization.workers,
            batch_size=min(len(batch), self.config.parallelization.batch_size),
            timeout_per_individual=self.config.performance.program_timeout,
            memory_limit_mb=self.config.performance.memory_limit,
            gpu_batch_size=self.config.parallelization.gpu_batch_size
        ) as evaluator:
            if self.current_task:
                # Progress callback for experiment tracking
                def progress_callback(completed: int, total: int):
                    if self.experiment_orchestrator and self.experiment_id and completed % 50 == 0:
                        # Update progress every 50 evaluations
                        asyncio.create_task(self._update_experiment_progress())

                results = await evaluator.evaluate_population(
                    individuals=batch,
                    task=self.current_task,
                    progress_callback=progress_callback
                )

                # Update individual fitness scores
                successful_evals = 0
                for individual in batch:
                    if individual.id in results:
                        result = results[individual.id]
                        individual.fitness = result.fitness
                        if result.cached_outputs:
                            individual.cached_execution = result.cached_outputs
                        if not result.error:
                            successful_evals += 1

                # Log batch statistics
                if successful_evals < len(batch):
                    failed_evals = len(batch) - successful_evals
                    print(f"Warning: {failed_evals}/{len(batch)} evaluations failed in batch")
            else:
                # Fall back to sequential
                await self._evaluate_batch_sequential(batch)

    async def _create_next_generation(self) -> None:
        """Create next generation through selection and breeding."""
        new_population = []

        # Preserve elite
        elite_size = self.config.population.elite_size
        elite = self.population.get_elite(elite_size)
        new_population.extend(elite)

        # Generate offspring to fill population
        target_size = self.config.population.size

        # Initialize counters for deterministic mode
        breeding_counter = 0

        while len(new_population) < target_size:
            # Select parents
            parent1 = self._select_parent()
            parent2 = self._select_parent()

            # Task 8: Use bandit controller for strategy selection if available
            if self.bandit_controller and self.task_feature_extractor and hasattr(self, 'current_task'):
                # Extract task features for contextual selection
                task_features = self.task_feature_extractor.extract_features(self.current_task)

                # Select generation strategy via bandit
                strategy_id = self.bandit_controller.select_strategy(task_features)

                # Apply selected strategy
                offspring = await self._apply_generation_strategy(
                    strategy_id, parent1, parent2, self.current_task
                )

                # Track strategy usage
                for child in offspring:
                    child.metadata["bandit_strategy"] = strategy_id
            else:
                # Fallback to traditional approach
                if self._should_apply_crossover(breeding_counter):
                    offspring = await self._apply_crossover(parent1, parent2)
                else:
                    offspring = [self._clone_individual(parent1), self._clone_individual(parent2)]

                # Apply mutation
                for i, child in enumerate(offspring):
                    if self._should_apply_mutation(breeding_counter, i):
                        child = await self._apply_mutation(child)

            # Add offspring to new population
            for child in offspring:
                new_population.append(child)
                self.total_programs_generated += 1

                if len(new_population) >= target_size:
                    break

            breeding_counter += 1

        # Trim to exact size if needed
        new_population = new_population[:target_size]

        # Replace population
        self.population.individuals = new_population

    def _should_apply_crossover(self, breeding_counter: int) -> bool:
        """Determine whether to apply crossover (deterministic in deterministic mode)."""
        if self.config.reproducibility.deterministic:
            # Apply crossover deterministically based on counter
            crossover_rate = self.config.genetic_operators.crossover.rate
            # Convert rate to a pattern (e.g., 0.7 -> apply 7 out of 10 times)
            return (breeding_counter % 10) < int(crossover_rate * 10)
        else:
            return random.random() < self.config.genetic_operators.crossover.rate

    def _should_apply_mutation(self, breeding_counter: int, child_index: int) -> bool:
        """Determine whether to apply mutation (deterministic in deterministic mode)."""
        if self.config.reproducibility.deterministic:
            # Apply mutation deterministically based on counter and child index
            mutation_rate = self.config.genetic_operators.mutation.base_rate
            # Create a unique identifier for this decision
            decision_id = breeding_counter * 2 + child_index
            # Convert rate to a pattern
            return (decision_id % 10) < int(mutation_rate * 10)
        else:
            return random.random() < self.config.genetic_operators.mutation.base_rate

    def _select_parent(self) -> Individual:
        """Select a parent using tournament selection."""
        # Tournament selection
        tournament_size = 3

        if self.config.reproducibility.deterministic:
            # Deterministic parent selection based on fitness ranking
            sorted_individuals = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)
            # Select from top portion deterministically
            top_portion = len(sorted_individuals) // 2
            # Cycle through top individuals
            idx = self.total_programs_generated % top_portion
            return sorted_individuals[idx]
        else:
            tournament = random.sample(self.population.individuals, tournament_size)
            return max(tournament, key=lambda ind: ind.fitness)

    def _clone_individual(self, individual: Individual) -> Individual:
        """Create a clone of an individual."""
        from copy import deepcopy
        return Individual(
            operations=deepcopy(individual.operations),
            parent_ids={individual.id}
        )

    def _select_crossover_operator(self) -> GeneticOperator:
        """Select a crossover operator based on configured probabilities."""
        if self.config.reproducibility.deterministic:
            # Use deterministic selection based on generation counter
            operators = list(self.crossover_operators.keys())
            idx = self.population.generation % len(operators)
            return self.crossover_operators[operators[idx]]

        # Use weighted random selection
        methods = self.config.genetic_operators.crossover.methods
        operators = list(methods.keys())
        weights = list(methods.values())

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Select operator based on weights
        selected = np.random.choice(operators, p=weights)
        return self.crossover_operators[selected]

    def _select_mutation_operator(self) -> GeneticOperator:
        """Select a mutation operator based on configured probabilities."""
        if self.config.reproducibility.deterministic:
            # Use deterministic selection based on mutation counter
            operators = list(self.mutation_operators.keys())
            idx = self.mutation_total_count % len(operators)
            return self.mutation_operators[operators[idx]]

        # Use weighted random selection
        methods = self.config.genetic_operators.mutation.methods
        operators = list(methods.keys())
        weights = list(methods.values())

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Select operator based on weights
        selected = np.random.choice(operators, p=weights)
        return self.mutation_operators[selected]

    async def _apply_crossover(self, parent1: Individual, parent2: Individual) -> list[Individual]:
        """Apply crossover operator to create offspring with genealogy tracking."""
        crossover = self._select_crossover_operator()
        offspring = crossover.apply(parent1, parent2)

        # Track genealogy for each offspring
        for i, child in enumerate(offspring):
            # Set parent IDs
            child.parent_ids = {parent1.id, parent2.id}

            # Update metadata with genealogy information
            child.metadata.update({
                'generation': self.population.generation + 1,
                'creation_method': 'crossover',
                'operator_type': crossover.get_name(),
                'crossover_point': crossover.last_crossover_point if hasattr(crossover, 'last_crossover_point') else None,
                'parent1_id': parent1.id,
                'parent2_id': parent2.id,
                'parent1_fitness': parent1.fitness,
                'parent2_fitness': parent2.fitness,
                'child_index': i,
                'timestamp': datetime.now().isoformat()
            })

            # Inherit lineage information
            parent1_lineage = parent1.metadata.get('lineage', [])
            parent2_lineage = parent2.metadata.get('lineage', [])
            child.metadata['lineage'] = parent1_lineage + parent2_lineage + [
                {'generation': self.population.generation, 'parents': [parent1.id, parent2.id], 'method': 'crossover'}
            ]

        return offspring

    async def _apply_mutation(self, individual: Individual) -> Individual:
        """Apply mutation operator to individual with genealogy tracking."""
        mutation = self._select_mutation_operator()

        # Store original operations for comparison
        original_ops = [op.get_name() for op in individual.operations]

        mutated = mutation.apply(individual)
        result = mutated[0] if mutated else individual

        # Track mutation statistics
        self.mutation_total_count += 1

        # Track mutation genealogy
        if result != individual:  # Mutation occurred
            # Identify mutation details
            mutated_ops = [op.get_name() for op in result.operations]
            mutation_positions = []
            for i, (orig, mut) in enumerate(zip(original_ops, mutated_ops, strict=False)):
                if orig != mut:
                    mutation_positions.append(i)

            # Update metadata
            result.metadata.update({
                'mutation_generation': self.population.generation + 1,
                'mutation_type': mutation.get_name(),
                'mutation_positions': mutation_positions,
                'mutations_count': len(mutation_positions),
                'pre_mutation_fitness': individual.fitness,
                'mutation_timestamp': datetime.now().isoformat()
            })

            # Track mutation history
            mutation_history = result.metadata.get('mutation_history', [])
            mutation_history.append({
                'generation': self.population.generation,
                'type': mutation.get_name(),
                'positions': mutation_positions,
                'parent_fitness': individual.fitness,
                'timestamp': datetime.now().isoformat()
            })
            result.metadata['mutation_history'] = mutation_history

            # Update lineage
            lineage = result.metadata.get('lineage', [])
            lineage.append({
                'generation': self.population.generation,
                'parents': [individual.id],
                'method': 'mutation',
                'mutation_type': 'operation_replacement'
            })
            result.metadata['lineage'] = lineage

        return result

    def analyze_lineage_paths(self) -> dict[str, Any]:
        """
        Analyze lineage paths to identify beneficial genetic patterns.

        Returns:
            Dictionary containing:
            - successful_lineages: List of lineages leading to high-fitness individuals
            - mutation_impact: Analysis of mutation types and their success rates
            - crossover_impact: Analysis of parent fitness combinations
            - generation_improvements: Fitness improvements by generation
        """
        analysis = {
            'successful_lineages': [],
            'mutation_impact': {},
            'crossover_impact': {},
            'generation_improvements': {},
            'beneficial_patterns': []
        }

        # Find successful individuals (fitness > 0.5)
        successful_individuals = [
            ind for ind in self.population.individuals
            if ind.fitness > 0.5
        ]

        # Analyze lineage paths for successful individuals
        for individual in successful_individuals:
            lineage = individual.metadata.get('lineage', [])
            if lineage:
                lineage_info = {
                    'individual_id': individual.id,
                    'fitness': individual.fitness,
                    'lineage_depth': len(lineage),
                    'path': lineage,
                    'creation_methods': [step['method'] for step in lineage],
                    'generations': [step['generation'] for step in lineage]
                }
                analysis['successful_lineages'].append(lineage_info)

                # Track beneficial patterns
                pattern = tuple(lineage_info['creation_methods'])
                if pattern not in analysis['beneficial_patterns']:
                    analysis['beneficial_patterns'].append({
                        'pattern': pattern,
                        'avg_fitness': individual.fitness,
                        'count': 1
                    })
                else:
                    # Update existing pattern
                    for p in analysis['beneficial_patterns']:
                        if p['pattern'] == pattern:
                            p['avg_fitness'] = (p['avg_fitness'] * p['count'] + individual.fitness) / (p['count'] + 1)
                            p['count'] += 1
                            break

        # Analyze mutation impact
        for individual in self.population.individuals:
            mutation_history = individual.metadata.get('mutation_history', [])
            for mutation in mutation_history:
                mutation_type = mutation.get('type', 'unknown')
                parent_fitness = mutation.get('parent_fitness', 0)
                fitness_gain = individual.fitness - parent_fitness

                if mutation_type not in analysis['mutation_impact']:
                    analysis['mutation_impact'][mutation_type] = {
                        'total_applications': 0,
                        'successful_applications': 0,
                        'avg_fitness_gain': 0,
                        'max_fitness_gain': float('-inf'),
                        'min_fitness_gain': float('inf')
                    }

                impact = analysis['mutation_impact'][mutation_type]
                impact['total_applications'] += 1
                if fitness_gain > 0:
                    impact['successful_applications'] += 1
                impact['avg_fitness_gain'] = (
                    (impact['avg_fitness_gain'] * (impact['total_applications'] - 1) + fitness_gain) /
                    impact['total_applications']
                )
                impact['max_fitness_gain'] = max(impact['max_fitness_gain'], fitness_gain)
                impact['min_fitness_gain'] = min(impact['min_fitness_gain'], fitness_gain)

        # Analyze crossover impact
        for individual in self.population.individuals:
            if 'parent1_fitness' in individual.metadata and 'parent2_fitness' in individual.metadata:
                parent1_fitness = individual.metadata['parent1_fitness']
                parent2_fitness = individual.metadata['parent2_fitness']
                avg_parent_fitness = (parent1_fitness + parent2_fitness) / 2
                fitness_gain = individual.fitness - avg_parent_fitness

                # Categorize parent fitness combinations
                parent_category = (
                    'high' if parent1_fitness > 0.5 else 'low',
                    'high' if parent2_fitness > 0.5 else 'low'
                )

                if parent_category not in analysis['crossover_impact']:
                    analysis['crossover_impact'][parent_category] = {
                        'count': 0,
                        'avg_fitness_gain': 0,
                        'successful': 0
                    }

                impact = analysis['crossover_impact'][parent_category]
                impact['count'] += 1
                if fitness_gain > 0:
                    impact['successful'] += 1
                impact['avg_fitness_gain'] = (
                    (impact['avg_fitness_gain'] * (impact['count'] - 1) + fitness_gain) /
                    impact['count']
                )

        # Analyze generation improvements
        for generation in range(self.population.generation + 1):
            gen_individuals = [
                ind for ind in self.population.individuals
                if ind.metadata.get('generation', 0) == generation
            ]
            if gen_individuals:
                analysis['generation_improvements'][generation] = {
                    'avg_fitness': sum(ind.fitness for ind in gen_individuals) / len(gen_individuals),
                    'max_fitness': max(ind.fitness for ind in gen_individuals),
                    'count': len(gen_individuals)
                }

        return analysis

    def _checkpoint_genealogy_data(self) -> None:
        """
        Checkpoint genealogy data to disk with compression for efficiency.

        This helps manage memory for long-running evolutions and provides
        recovery points if the process is interrupted.
        """
        try:
            import gzip
            checkpoint_dir = Path("evolution_checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            # Create checkpoint filename with generation number
            checkpoint_file = checkpoint_dir / f"genealogy_checkpoint_gen{self.population.generation}.json.gz"

            # Prepare checkpoint data
            checkpoint_data = {
                'generation': self.population.generation,
                'total_programs_generated': self.total_programs_generated,
                'genealogy_data': self.visualizer.genealogy_data,
                'mutation_patterns': self.visualizer.mutation_success_patterns,
                'crossover_patterns': {
                    f"{k[0]},{k[1]}": v
                    for k, v in self.visualizer.crossover_success_patterns.items()
                },
                'population_metadata': {
                    ind.id: {
                        'fitness': ind.fitness,
                        'metadata': ind.metadata,
                        'parent_ids': list(ind.parent_ids),
                        'program_length': ind.program_length()
                    }
                    for ind in self.population.individuals
                }
            }

            # Write compressed checkpoint
            with gzip.open(checkpoint_file, 'wt', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            # Clean up old checkpoints to save space (keep only last 3)
            checkpoints = sorted(checkpoint_dir.glob("genealogy_checkpoint_gen*.json.gz"))
            if len(checkpoints) > 3:
                for old_checkpoint in checkpoints[:-3]:
                    old_checkpoint.unlink()

        except Exception as e:
            print(f"Warning: Failed to checkpoint genealogy data: {e}")

    def select_best_programs(self, top_n: int = 10, unique_only: bool = True) -> list[Individual]:
        """
        Select the best programs across all generations.

        Args:
            top_n: Number of top programs to select
            unique_only: If True, deduplicate programs by their operations

        Returns:
            List of best individuals sorted by fitness
        """
        # Combine current population with historical individuals
        all_individuals = list(self.all_individuals_history) + list(self.population.individuals)

        if unique_only:
            # Deduplicate by program hash
            seen_programs = {}
            for ind in all_individuals:
                program_hash = ind.id  # ID is already a hash of operations
                if program_hash not in seen_programs or ind.fitness > seen_programs[program_hash].fitness:
                    seen_programs[program_hash] = ind
            all_individuals = list(seen_programs.values())

        # Sort by fitness and return top N
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)
        return all_individuals[:top_n]

    def serialize_program_for_export(self, individual: Individual) -> dict[str, Any]:
        """
        Serialize an individual program for export with full metadata.

        Args:
            individual: Individual to serialize

        Returns:
            Dictionary with program and metadata for export
        """
        # Serialize operations
        operations_data = []
        for op in individual.operations:
            op_data = {
                'name': op.get_name(),
                'type': op.__class__.__name__,
                'parameters': op.parameters if hasattr(op, 'parameters') else {}
            }
            operations_data.append(op_data)

        # Prepare export data
        export_data = {
            'id': individual.id,
            'fitness': individual.fitness,
            'program': {
                'operations': operations_data,
                'length': individual.program_length()
            },
            'metadata': {
                'generation_created': individual.metadata.get('generation', 0),
                'creation_method': individual.metadata.get('creation_method', 'unknown'),
                'age': individual.age,
                'parent_ids': list(individual.parent_ids),
                'timestamp': individual.created_at.isoformat() if individual.created_at else None
            },
            'genealogy': {
                'lineage': individual.metadata.get('lineage', []),
                'mutation_history': individual.metadata.get('mutation_history', []),
                'mutations_count': len(individual.metadata.get('mutation_history', [])),
                'lineage_depth': len(individual.metadata.get('lineage', []))
            },
            'performance': {
                'cached_execution': individual.cached_execution is not None,
                'evaluation_error': individual.metadata.get('evaluation_error'),
                'exceeded_limits': individual.metadata.get('exceeded_limits', False)
            }
        }

        # Add complexity metrics
        complexity_metrics = self._calculate_program_complexity(individual)
        export_data['complexity'] = complexity_metrics

        return export_data

    def _calculate_program_complexity(self, individual: Individual) -> dict[str, Any]:
        """Calculate complexity metrics for a program."""
        operations = individual.operations

        # Count operation types
        op_type_counts = {}
        for op in operations:
            op_name = op.get_name()
            op_type_counts[op_name] = op_type_counts.get(op_name, 0) + 1

        # Calculate metrics
        return {
            'total_operations': len(operations),
            'unique_operations': len(op_type_counts),
            'operation_diversity': len(op_type_counts) / len(operations) if operations else 0,
            'operation_counts': op_type_counts,
            'has_loops': any(op.get_name() in ['repeat', 'iterate'] for op in operations),
            'has_conditionals': any(op.get_name() in ['conditional', 'if'] for op in operations),
            'max_nesting_depth': self._estimate_nesting_depth(operations)
        }

    def _estimate_nesting_depth(self, operations: list[Operation]) -> int:
        """Estimate maximum nesting depth of operations."""
        # Simple heuristic: count sequential control flow operations
        max_depth = 0
        current_depth = 0

        for op in operations:
            if op.get_name() in ['repeat', 'iterate', 'conditional', 'map', 'filter']:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif op.get_name() in ['end', 'break']:  # Assuming these exist
                current_depth = max(0, current_depth - 1)

        return max_depth

    def export_top_programs(self, output_dir: str | Path = "evolution_exports", top_n: int = 10) -> Path:
        """
        Export top programs in both DSL and Python formats.

        Args:
            output_dir: Directory to export programs to
            top_n: Number of top programs to export

        Returns:
            Path to export directory
        """
        from src.adapters.strategies.python_transpiler import PythonTranspiler

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get top programs
        top_programs = self.select_best_programs(top_n=top_n, unique_only=True)

        # Initialize transpiler
        transpiler = PythonTranspiler()

        # Export each program
        exported_programs = []
        for i, individual in enumerate(top_programs):
            program_dir = output_path / f"program_{i+1}_fitness_{individual.fitness:.3f}"
            program_dir.mkdir(exist_ok=True)

            # Export DSL format
            dsl_data = self.serialize_program_for_export(individual)
            dsl_path = program_dir / "program.dsl.json"
            with open(dsl_path, 'w') as f:
                json.dump(dsl_data, f, indent=2, default=str)

            # Export Python format
            try:
                # Create DSL program for transpilation
                dsl_program = {
                    'operations': [
                        {
                            'type': op.get_name(),
                            'params': op.parameters if hasattr(op, 'parameters') else {}
                        }
                        for op in individual.operations
                    ]
                }

                python_code = transpiler.transpile(dsl_program)
                python_path = program_dir / "program.py"
                with open(python_path, 'w') as f:
                    f.write(python_code)

                python_exported = True
                python_error = None
            except Exception as e:
                python_exported = False
                python_error = str(e)
                # Write error log
                error_path = program_dir / "python_transpilation_error.txt"
                with open(error_path, 'w') as f:
                    f.write(f"Failed to transpile to Python:\n{e}")

            # Create README for the program
            readme_content = f"""# Program {i+1}

## Performance
- Fitness Score: {individual.fitness:.3f}
- Generation Created: {individual.metadata.get('generation', 'unknown')}
- Program Length: {individual.program_length()} operations

## Metadata
- ID: {individual.id}
- Creation Method: {individual.metadata.get('creation_method', 'unknown')}
- Age: {individual.age} generations
- Parents: {', '.join(individual.parent_ids) if individual.parent_ids else 'None'}

## Files
- `program.dsl.json`: Full program data in DSL format with metadata
- `program.py`: Python implementation {'(transpiled)' if python_exported else '(failed - see error log)'}

## Complexity Metrics
{json.dumps(dsl_data.get('complexity', {}), indent=2)}

## Genealogy
- Lineage Depth: {dsl_data['genealogy']['lineage_depth']}
- Total Mutations: {dsl_data['genealogy']['mutations_count']}
"""
            readme_path = program_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)

            exported_programs.append({
                'rank': i + 1,
                'id': individual.id,
                'fitness': individual.fitness,
                'dsl_path': str(dsl_path),
                'python_path': str(python_path) if python_exported else None,
                'python_error': python_error
            })

        # Create export summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_programs_evaluated': self.total_programs_generated,
            'total_generations': self.population.generation,
            'programs_exported': len(exported_programs),
            'export_details': exported_programs,
            'evolution_stats': self._get_evolution_stats()
        }

        summary_path = output_path / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate analysis report
        self._generate_program_analysis_report(output_path, exported_programs)

        print(f"Exported {len(exported_programs)} programs to {output_path}")
        return output_path

    def _generate_program_analysis_report(self, output_dir: Path, exported_programs: list[dict]) -> None:
        """
        Generate comprehensive analysis report for exported programs.

        Args:
            output_dir: Directory containing exported programs
            exported_programs: List of exported program information
        """
        # Analyze all exported programs
        analysis_data = {
            'summary': {
                'total_programs_exported': len(exported_programs),
                'average_fitness': sum(p['fitness'] for p in exported_programs) / len(exported_programs) if exported_programs else 0,
                'best_fitness': max((p['fitness'] for p in exported_programs), default=0),
                'python_transpilation_success_rate': sum(1 for p in exported_programs if p['python_path']) / len(exported_programs) if exported_programs else 0
            },
            'operation_analysis': {},
            'complexity_distribution': {
                'length': [],
                'diversity': [],
                'nesting_depth': []
            },
            'genealogy_patterns': {},
            'performance_characteristics': {}
        }

        # Analyze each program
        for prog_info in exported_programs:
            prog_path = Path(prog_info['dsl_path'])
            with open(prog_path) as f:
                prog_data = json.load(f)

            # Operation frequency analysis
            for op_name, count in prog_data['complexity']['operation_counts'].items():
                if op_name not in analysis_data['operation_analysis']:
                    analysis_data['operation_analysis'][op_name] = {
                        'usage_count': 0,
                        'programs_using': 0,
                        'avg_fitness_when_used': 0
                    }
                op_stats = analysis_data['operation_analysis'][op_name]
                op_stats['usage_count'] += count
                op_stats['programs_using'] += 1
                op_stats['avg_fitness_when_used'] = (
                    (op_stats['avg_fitness_when_used'] * (op_stats['programs_using'] - 1) + prog_data['fitness']) /
                    op_stats['programs_using']
                )

            # Complexity distribution
            analysis_data['complexity_distribution']['length'].append(prog_data['program']['length'])
            analysis_data['complexity_distribution']['diversity'].append(prog_data['complexity']['operation_diversity'])
            analysis_data['complexity_distribution']['nesting_depth'].append(prog_data['complexity']['max_nesting_depth'])

            # Genealogy pattern analysis
            creation_method = prog_data['metadata']['creation_method']
            if creation_method not in analysis_data['genealogy_patterns']:
                analysis_data['genealogy_patterns'][creation_method] = {
                    'count': 0,
                    'avg_fitness': 0
                }
            gp = analysis_data['genealogy_patterns'][creation_method]
            gp['count'] += 1
            gp['avg_fitness'] = ((gp['avg_fitness'] * (gp['count'] - 1) + prog_data['fitness']) / gp['count'])

            # Performance characteristics
            if prog_data['performance']['exceeded_limits']:
                perf_key = 'exceeded_limits'
            elif prog_data['performance']['evaluation_error']:
                perf_key = 'had_errors'
            else:
                perf_key = 'successful'

            if perf_key not in analysis_data['performance_characteristics']:
                analysis_data['performance_characteristics'][perf_key] = 0
            analysis_data['performance_characteristics'][perf_key] += 1

        # Generate report
        report_content = f"""# Evolution Pipeline Analysis Report

Generated: {datetime.now().isoformat()}

## Executive Summary

- **Programs Exported**: {analysis_data['summary']['total_programs_exported']}
- **Average Fitness**: {analysis_data['summary']['average_fitness']:.3f}
- **Best Fitness**: {analysis_data['summary']['best_fitness']:.3f}
- **Python Transpilation Success**: {analysis_data['summary']['python_transpilation_success_rate']:.1%}

## Operation Usage Analysis

Most frequently used operations:
"""

        # Sort operations by usage
        sorted_ops = sorted(analysis_data['operation_analysis'].items(),
                          key=lambda x: x[1]['usage_count'], reverse=True)

        for op_name, stats in sorted_ops[:10]:
            report_content += f"\n- **{op_name}**:"
            report_content += f"\n  - Total usage: {stats['usage_count']}"
            report_content += f"\n  - Programs using: {stats['programs_using']}"
            report_content += f"\n  - Avg fitness when used: {stats['avg_fitness_when_used']:.3f}"

        report_content += f"""

## Complexity Analysis

### Program Length Distribution
- Min: {min(analysis_data['complexity_distribution']['length'], default=0)}
- Max: {max(analysis_data['complexity_distribution']['length'], default=0)}
- Average: {sum(analysis_data['complexity_distribution']['length']) / len(analysis_data['complexity_distribution']['length']) if analysis_data['complexity_distribution']['length'] else 0:.1f}

### Operation Diversity
- Min: {min(analysis_data['complexity_distribution']['diversity'], default=0):.3f}
- Max: {max(analysis_data['complexity_distribution']['diversity'], default=0):.3f}
- Average: {sum(analysis_data['complexity_distribution']['diversity']) / len(analysis_data['complexity_distribution']['diversity']) if analysis_data['complexity_distribution']['diversity'] else 0:.3f}

### Nesting Depth
- Max depth found: {max(analysis_data['complexity_distribution']['nesting_depth'], default=0)}

## Genealogy Patterns

Creation methods and their effectiveness:
"""

        for method, stats in analysis_data['genealogy_patterns'].items():
            report_content += f"\n- **{method}**: {stats['count']} programs, avg fitness {stats['avg_fitness']:.3f}"

        report_content += """

## Performance Characteristics

"""

        total_progs = sum(analysis_data['performance_characteristics'].values())
        for characteristic, count in analysis_data['performance_characteristics'].items():
            percentage = (count / total_progs * 100) if total_progs > 0 else 0
            report_content += f"- **{characteristic}**: {count} ({percentage:.1f}%)\n"

        report_content += f"""

## Recommendations

1. **High-performing operations**: Focus on {sorted_ops[0][0] if sorted_ops else 'N/A'} and {sorted_ops[1][0] if len(sorted_ops) > 1 else 'N/A'} which show strong correlation with fitness.

2. **Complexity sweet spot**: Programs with length around {sum(analysis_data['complexity_distribution']['length']) / len(analysis_data['complexity_distribution']['length']) if analysis_data['complexity_distribution']['length'] else 0:.0f} operations tend to perform well.

3. **Genealogy insights**: {max(analysis_data['genealogy_patterns'].items(), key=lambda x: x[1]['avg_fitness'], default=('N/A', {'avg_fitness': 0}))[0]} creation method produces highest fitness programs on average.

## Evolution Statistics

{json.dumps(self._get_evolution_stats(), indent=2)}
"""

        # Write report
        report_path = output_dir / "program_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Also save raw analysis data
        analysis_path = output_dir / "program_analysis_data.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.population.best_individual:
            return self.population.best_individual.fitness >= self.config.fitness.early_termination.get('threshold', 0.95)
        return False

    def _get_evolution_stats(self) -> dict[str, Any]:
        """Get evolution statistics."""
        # Calculate throughput metrics
        total_time = time.time() - self.convergence_tracker.start_time if hasattr(self, 'convergence_tracker') else 0
        avg_programs_per_sec = sum(self.performance_metrics["programs_per_second"]) / len(self.performance_metrics["programs_per_second"]) if self.performance_metrics["programs_per_second"] else 0

        return {
            "total_programs_generated": self.total_programs_generated,
            "generations": self.population.generation,
            "best_fitness": self.population.best_individual.fitness if self.population.best_individual else 0.0,
            "average_fitness": self.population.average_fitness(),
            "fitness_variance": self.population.fitness_variance(),
            "unique_programs": self.population.diversity_metrics.get('unique_programs', 0),
            "best_program_length": self.population.best_individual.program_length() if self.population.best_individual else 0,
            "average_generation_time": sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0.0,
            "fitness_history": self.population.best_fitness_history,
            "diversity_metrics": self.population.diversity_metrics,
            "population_size": self.population.size(),

            # Co-evolution metrics (Task 7.4)
            "coevolution_enabled": self.coevolution_enabled,
            "coevolution_stats": self.coevolution_engine.get_coevolution_stats() if self.coevolution_engine else None,

            # Novelty search metrics (Task 7.5)
            "novelty_search_enabled": self.novelty_search_enabled,
            "novelty_search_stats": self.novelty_search_engine.get_archive_diversity() if self.novelty_search_engine else None,
            "novelty_weight": self.novelty_search_engine.novelty_weight if self.novelty_search_engine else 0.0,

            # Performance metrics
            "total_evolution_time": total_time,
            "average_programs_per_second": avg_programs_per_sec,
            "evaluation_success_rate": self.performance_metrics["evaluation_success_rate"],
            "cache_hit_rate": self.performance_metrics["cache_hit_rate"],
            "peak_memory_usage_mb": max(self.performance_metrics["memory_usage"]) if self.performance_metrics["memory_usage"] else self._get_memory_usage_mb(),

            # Efficiency metrics
            "programs_per_generation": self.total_programs_generated / self.population.generation if self.population.generation > 0 else 0,
            "time_per_generation": total_time / self.population.generation if self.population.generation > 0 else 0,
            # Operator success rates (Task 5.3)
            "mutation_success_rate": self.mutation_success_count / self.mutation_total_count if self.mutation_total_count > 0 else 0.0,
            "crossover_success_rate": self.crossover_success_count / self.crossover_total_count if self.crossover_total_count > 0 else 0.0,
            "mutation_attempts": self.mutation_total_count,
            "successful_mutations": self._get_successful_mutations(),
            "operator_success_by_type": self.operator_success_by_type,
            # Genealogy metrics
            "max_genealogy_depth": self._calculate_max_genealogy_depth(),
            "average_lineage_length": self._calculate_avg_lineage_length()
        }

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _get_successful_mutations(self) -> list[dict[str, Any]]:
        """Get list of successful mutations that improved fitness."""
        successful_mutations = []
        for ind in self.all_individuals_history + self.population.individuals:
            mutation_history = ind.metadata.get('mutation_history', [])
            for mutation in mutation_history:
                if mutation.get('fitness_improvement', 0) > 0:
                    successful_mutations.append({
                        'generation': mutation.get('generation'),
                        'type': mutation.get('type'),
                        'improvement': mutation.get('fitness_improvement'),
                        'individual_id': ind.id
                    })
        return successful_mutations

    def _calculate_max_genealogy_depth(self) -> int:
        """Calculate maximum genealogy depth across all individuals."""
        max_depth = 0
        for ind in self.population.individuals:
            lineage = ind.metadata.get('lineage', [])
            max_depth = max(max_depth, len(lineage))
        return max_depth

    def _calculate_avg_lineage_length(self) -> float:
        """Calculate average lineage length."""
        total_length = 0
        count = 0
        for ind in self.population.individuals:
            lineage = ind.metadata.get('lineage', [])
            total_length += len(lineage)
            count += 1
        return total_length / count if count > 0 else 0.0

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        current_memory = self._get_memory_usage_mb()
        return current_memory < self.memory_limit_mb

    def _cleanup_population_memory(self) -> None:
        """Clean up memory by removing cached data from old individuals."""
        # Sort by fitness to keep best individuals' cache
        sorted_individuals = sorted(
            self.population.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )

        # Keep cache only for top 20% of population
        cache_keep_size = int(self.population.size() * 0.2)

        for i, individual in enumerate(sorted_individuals):
            if i >= cache_keep_size:
                # Clear cached execution results for lower fitness individuals
                individual.cached_execution = None

        # Force garbage collection
        gc.collect()

    async def _manage_memory(self) -> None:
        """Manage memory usage during evolution."""
        # Check every N generations or if time elapsed
        if (self.population.generation % self.memory_cleanup_frequency == 0 or
            time.time() - self.last_memory_check > 60):  # Also check every minute

            current_memory = self._get_memory_usage_mb()
            self.performance_metrics["memory_usage"].append(current_memory)

            # Log memory usage
            if self.experiment_orchestrator and self.experiment_id:
                await self.experiment_orchestrator.update_experiment_progress(
                    self.experiment_id,
                    ExperimentProgress(
                        current_epoch=self.population.generation,
                        total_epochs=self.config.convergence.max_generations,
                        memory_usage_mb=current_memory
                    )
                )

            # If approaching memory limit, cleanup
            if current_memory > self.memory_limit_mb * 0.8:  # 80% threshold
                print(f"Memory usage high ({current_memory:.1f}MB), cleaning up...")
                self._cleanup_population_memory()

                # Clear fitness evaluator cache
                if self.fitness_evaluator:
                    self.fitness_evaluator.clear_cache()

                # Check memory again
                new_memory = self._get_memory_usage_mb()
                print(f"Memory after cleanup: {new_memory:.1f}MB")

            self.last_memory_check = time.time()

    async def _update_experiment_progress(self) -> None:
        """Helper method to update experiment progress."""
        if self.experiment_orchestrator and self.experiment_id:
            progress = ExperimentProgress(
                current_epoch=self.population.generation,
                total_epochs=self.config.convergence.max_generations,
                accuracy=self.population.best_individual.fitness if self.population.best_individual else 0.0,
                elapsed_time_seconds=time.time() - self.convergence_tracker.start_time,
                memory_usage_mb=self._get_memory_usage_mb(),
                custom_metrics={
                    "unique_programs": self.population.diversity_metrics.get('unique_programs', 0),
                    "average_fitness": self.population.average_fitness(),
                    "programs_generated": self.total_programs_generated
                }
            )
            await self.experiment_orchestrator.update_experiment_progress(
                self.experiment_id,
                progress
            )

    async def _save_checkpoint(self) -> None:
        """Save current evolution state to checkpoint."""
        if not self._checkpoint_manager:
            return

        checkpoint_data = {
            'generation': self.population.generation,
            'random_seed': self.random_seed,
            'total_programs_generated': self.total_programs_generated,
            'experiment_id': self.experiment_id,
            'population': {
                'individuals': [self._serialize_individual(ind) for ind in self.population.individuals],
                'generation': self.population.generation,
                'diversity_metrics': self.population.diversity_metrics,
                'best_fitness': self.population.best_individual.fitness if self.population.best_individual else 0.0
            },
            'convergence_state': {
                'generations_since_improvement': self.convergence_tracker.generations_since_improvement,
                'last_best_fitness': self.convergence_tracker.last_best_fitness,
                'start_time': self.convergence_tracker.start_time
            },
            'operator_stats': {
                'mutation_success_count': self.mutation_success_count,
                'mutation_total_count': self.mutation_total_count,
                'crossover_success_count': self.crossover_success_count,
                'crossover_total_count': self.crossover_total_count,
                'operator_success_by_type': self.operator_success_by_type
            },
            'config_version': self.config.reproducibility.config_version,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_name = f"evolution_gen_{self.population.generation}_{self.experiment_id or 'default'}"
        checkpoint_dir = Path(self.config.reproducibility.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.json"

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            print(f"Saved checkpoint: {checkpoint_name}")

            # Save to checkpoint manager if available
            if self._checkpoint_manager:
                self._checkpoint_manager.save_checkpoint(
                    str(checkpoint_path),
                    checkpoint_name,
                    {'generation': self.population.generation}
                )
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    async def restore_from_checkpoint(self, checkpoint_path: str) -> None:
        """Restore evolution state from checkpoint."""
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Verify config version compatibility
            if checkpoint_data.get('config_version') != self.config.reproducibility.config_version:
                print(f"Config version mismatch: checkpoint={checkpoint_data.get('config_version')}, current={self.config.reproducibility.config_version}")

            # Restore random state
            self.random_seed = checkpoint_data['random_seed']
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

            # Restore counters
            self.total_programs_generated = checkpoint_data['total_programs_generated']
            self.experiment_id = checkpoint_data.get('experiment_id')

            # Restore population
            self.population = Population(generation=checkpoint_data['population']['generation'])
            self.population.diversity_metrics = checkpoint_data['population']['diversity_metrics']

            for ind_data in checkpoint_data['population']['individuals']:
                individual = await self._deserialize_individual(ind_data)
                self.population.add_individual(individual)

            # Restore convergence state
            conv_state = checkpoint_data['convergence_state']
            self.convergence_tracker.generations_since_improvement = conv_state['generations_since_improvement']
            self.convergence_tracker.last_best_fitness = conv_state['last_best_fitness']
            self.convergence_tracker.start_time = conv_state['start_time']

            # Restore operator stats
            op_stats = checkpoint_data['operator_stats']
            self.mutation_success_count = op_stats['mutation_success_count']
            self.mutation_total_count = op_stats['mutation_total_count']
            self.crossover_success_count = op_stats['crossover_success_count']
            self.crossover_total_count = op_stats['crossover_total_count']
            self.operator_success_by_type = op_stats['operator_success_by_type']

            print(f"Restored from checkpoint: generation {self.population.generation}")

        except Exception as e:
            print(f"Failed to restore from checkpoint: {e}")
            raise

    def _serialize_individual(self, individual: Individual) -> dict[str, Any]:
        """Serialize an individual to JSON-compatible format."""
        return {
            'operations': [{'name': op.get_name(), 'params': op.parameters} for op in individual.operations],
            'fitness': individual.fitness,
            'age': individual.age,
            'parent_ids': list(individual.parent_ids),
            'created_at': individual.created_at.isoformat(),
            'metadata': individual.metadata,
            'id': individual.id,
            'species_id': individual.species_id,
            'novelty_score': individual.novelty_score
        }

    async def _deserialize_individual(self, data: dict[str, Any]) -> Individual:
        """Deserialize an individual from JSON data."""
        # Reconstruct operations
        operations = []
        # This would need proper operation reconstruction from registry
        # For now, creating placeholder operations

        individual = Individual(operations=operations)
        individual.fitness = data['fitness']
        individual.age = data['age']
        individual.parent_ids = set(data['parent_ids'])
        individual.created_at = datetime.fromisoformat(data['created_at'])
        individual.metadata = data['metadata']
        individual.id = data['id']
        individual.species_id = data.get('species_id')
        individual.novelty_score = data.get('novelty_score')

        return individual

    async def _evolve_with_islands(
        self,
        task: ARCTask,
        callbacks: list[Callable] | None = None,
        experiment_name: str | None = None
    ) -> tuple[Individual, dict[str, Any]]:
        """
        Run evolution using island model for parallel sub-populations.

        This implements Task 7.1: Island model for parallel population evolution.

        Args:
            task: ARC task to solve
            callbacks: Optional callbacks for generation updates
            experiment_name: Optional name for experiment tracking

        Returns:
            Tuple of (best individual, evolution statistics)
        """
        from src.adapters.strategies.island_evolution import IslandEvolutionEngine, MigrationPolicy

        # Create migration policy from config
        migration_config = self.config.island_model.migration
        migration_policy = MigrationPolicy(
            frequency=migration_config.frequency,
            migration_rate=migration_config.migration_rate,
            selection_method=migration_config.selection_method,
            topology=migration_config.topology,
            adaptive=migration_config.adaptive
        )

        # Create island evolution engine
        island_engine = IslandEvolutionEngine(
            num_islands=self.config.island_model.num_islands,
            config=self.config,
            dsl_engine=self.dsl_engine,
            migration_policy=migration_policy
        )

        # Run island-based evolution
        best_individual, stats = await island_engine.evolve(
            task=task,
            max_generations=self.config.convergence.max_generations,
            callbacks=callbacks
        )

        # Add island-specific stats
        stats['evolution_type'] = 'island_model'
        stats['total_programs_generated'] = (
            self.config.island_model.num_islands *
            self.config.population.size *
            stats.get('total_generations', 0)
        )

        # Update engine state for compatibility
        self.population.best_individual = best_individual
        self.total_programs_generated = stats['total_programs_generated']

        return best_individual, stats

    async def _initialize_coevolution(self, task: ARCTask) -> None:
        """Initialize co-evolution engine (Task 7.4)."""
        from src.adapters.strategies.coevolution import CoevolutionEngine

        # Get co-evolution configuration
        coevol_config = self.config.coevolution

        self.coevolution_engine = CoevolutionEngine(
            program_pop_size=self.config.population.size,
            fitness_pop_size=coevol_config.get('fitness_pop_size', 20),
            elite_ratio=coevol_config.get('elite_ratio', 0.1)
        )

        # Initialize fitness function population
        self.coevolution_engine.initialize_populations()

        self.coevolution_enabled = True
        print("Co-evolution enabled: evolving both programs and fitness functions")

    def _get_available_operations(self) -> list[str]:
        """Get list of available DSL operations for LLM generation (Task 7.3)."""
        # This would normally query the DSL registry
        # For now, return common operations
        return [
            "rotate", "flip", "translate", "scale",
            "replace_color", "fill_background", "extract_objects",
            "find_pattern", "apply_pattern", "crop", "pad",
            "mirror", "tile", "overlay", "mask",
            "group_by_color", "connect_components", "fill_shape",
            "draw_line", "draw_rectangle", "flood_fill",
            "filter_by_size", "sort_objects", "align_objects",
            "repeat_pattern", "symmetrize", "denoise"
        ]

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        # Final memory cleanup
        self._cleanup_population_memory()
        gc.collect()
