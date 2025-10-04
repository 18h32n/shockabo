"""
Evolution configuration module.

This module provides aliases and imports for evolution configuration classes
to maintain compatibility with existing test files and provide a clean API.
"""

# Import all config classes from the main config module
from src.infrastructure.config import (
    ConvergenceConfig,
    DiversityConfig,
    FitnessConfig,
    GeneticAlgorithmConfig,
    GeneticOperatorsConfig,
    ParallelizationConfig,
    PerformanceConfig,
    PopulationConfig,
    ReproducibilityConfig,
)

# Create an alias for the main config class for backward compatibility
EvolutionConfig = GeneticAlgorithmConfig

# Also import the individual operator configs for completeness
from src.infrastructure.config import (
    CrossoverConfig,
    MutationConfig,
)

# Create an alias for genetic operators config
GeneticOperatorConfig = GeneticOperatorsConfig

__all__ = [
    "EvolutionConfig",
    "ConvergenceConfig",
    "DiversityConfig",
    "FitnessConfig",
    "GeneticOperatorConfig",
    "GeneticOperatorsConfig",
    "ParallelizationConfig",
    "PerformanceConfig",
    "PopulationConfig",
    "ReproducibilityConfig",
    "CrossoverConfig",
    "MutationConfig",
]
