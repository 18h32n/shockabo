"""Smart Model Router for intelligent LLM selection based on task complexity."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.domain.models import ARCTask
from src.infrastructure.components.budget_controller import BudgetController
from src.infrastructure.components.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.infrastructure.config import Config

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    BREAKTHROUGH = "breakthrough"


@dataclass
class ModelTier:
    """Configuration for a model tier."""
    name: str
    complexity_range: tuple[float, float]
    model_id: str
    max_tokens: int
    temperature: float
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float
    timeout_seconds: int = 30
    retry_attempts: int = 3

    def matches_complexity(self, score: float) -> bool:
        """Check if complexity score falls within this tier's range."""
        return self.complexity_range[0] <= score < self.complexity_range[1]


@dataclass
class ComplexityFeatures:
    """Features extracted from task for complexity analysis."""
    grid_size_score: float
    pattern_complexity: float
    color_diversity: float
    transformation_hints: float
    example_consistency: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for caching."""
        return {
            "grid_size_score": self.grid_size_score,
            "pattern_complexity": self.pattern_complexity,
            "color_diversity": self.color_diversity,
            "transformation_hints": self.transformation_hints,
            "example_consistency": self.example_consistency
        }


@dataclass
class RoutingDecision:
    """Routing decision with confidence and reasoning."""
    model_tier: ModelTier
    complexity_score: float
    complexity_level: ComplexityLevel
    confidence: float
    features: ComplexityFeatures
    reasoning: str


class LLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> tuple[str, int, int]:
        """Generate response. Returns (response, input_tokens, output_tokens)."""
        ...

    def get_name(self) -> str:
        """Get provider name."""
        ...


class SmartModelRouter:
    """Routes tasks to appropriate LLM models based on complexity analysis."""

    def __init__(
        self,
        config: Config,
        budget_controller: BudgetController,
        cache_dir: Path | None = None
    ):
        self.config = config
        self.budget_controller = budget_controller
        self.cache_dir = cache_dir or Path("data/llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model tiers configuration
        self.model_tiers = self._initialize_model_tiers()

        # Complexity detection weights
        self.complexity_weights = {
            "grid_size_score": 0.25,
            "pattern_complexity": 0.30,
            "color_diversity": 0.20,
            "transformation_hints": 0.15,
            "example_consistency": 0.10
        }

        # Complexity thresholds
        self.complexity_thresholds = {
            ComplexityLevel.SIMPLE: 0.3,
            ComplexityLevel.MEDIUM: 0.6,
            ComplexityLevel.COMPLEX: 0.85,
            ComplexityLevel.BREAKTHROUGH: 0.95
        }

        # Provider registry
        self.providers: dict[str, LLMProvider] = {}

        # Circuit breakers for each provider
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Cache for complexity analysis
        self._complexity_cache: dict[str, ComplexityFeatures] = {}

        # Performance tracking
        self.performance_stats: dict[str, dict[str, Any]] = {}

    def _initialize_model_tiers(self) -> list[ModelTier]:
        """Initialize model tier configurations."""
        return [
            ModelTier(
                name="Qwen2.5-Coder",
                complexity_range=(0.0, 0.3),
                model_id="qwen2.5-coder-32b",
                max_tokens=4096,
                temperature=0.7,
                cost_per_million_input_tokens=0.15,
                cost_per_million_output_tokens=0.15,
                timeout_seconds=20
            ),
            ModelTier(
                name="Gemini 2.5 Flash",
                complexity_range=(0.3, 0.6),
                model_id="gemini-2.5-flash",
                max_tokens=8192,
                temperature=0.8,
                cost_per_million_input_tokens=0.31,
                cost_per_million_output_tokens=2.62,
                timeout_seconds=30
            ),
            ModelTier(
                name="GLM-4.5",
                complexity_range=(0.6, 0.85),
                model_id="glm-4.5",
                max_tokens=16384,
                temperature=0.9,
                cost_per_million_input_tokens=0.59,
                cost_per_million_output_tokens=2.19,
                timeout_seconds=45
            ),
            ModelTier(
                name="GPT-5",
                complexity_range=(0.85, 1.0),
                model_id="gpt-5",
                max_tokens=32768,
                temperature=0.95,
                cost_per_million_input_tokens=1.25,
                cost_per_million_output_tokens=10.00,
                timeout_seconds=60
            ),
            ModelTier(
                name="Falcon Mamba 7B",
                complexity_range=(0.0, 1.0),  # Fallback for any complexity
                model_id="falcon-mamba-7b-local",
                max_tokens=2048,
                temperature=0.6,
                cost_per_million_input_tokens=0.0,
                cost_per_million_output_tokens=0.0,
                timeout_seconds=120  # Local model might be slower
            )
        ]

    def register_provider(self, model_id: str, provider: LLMProvider):
        """Register an LLM provider."""
        self.providers[model_id] = provider

        # Create circuit breaker for this provider
        self.circuit_breakers[model_id] = CircuitBreaker(
            name=f"llm_provider_{model_id}",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2
            )
        )

        logger.info(f"Registered LLM provider: {model_id}")

    def analyze_complexity(self, task: ARCTask) -> ComplexityFeatures:
        """Analyze task complexity and extract features."""
        # Check cache first
        task_hash = self._hash_task(task)
        if task_hash in self._complexity_cache:
            return self._complexity_cache[task_hash]

        # Calculate grid size score
        dimensions = task.get_grid_dimensions()
        max_size = max(
            max(d[0] * d[1] for d in dims) if dims else 0
            for dims in dimensions.values()
        )
        grid_size_score = min(1.0, max_size / 900.0)  # Normalize to 30x30 max

        # Calculate pattern complexity
        pattern_complexity = self._calculate_pattern_complexity(task)

        # Calculate color diversity
        color_diversity = self._calculate_color_diversity(task)

        # Calculate transformation hints
        transformation_hints = self._calculate_transformation_hints(task)

        # Calculate example consistency
        example_consistency = self._calculate_example_consistency(task)

        features = ComplexityFeatures(
            grid_size_score=grid_size_score,
            pattern_complexity=pattern_complexity,
            color_diversity=color_diversity,
            transformation_hints=transformation_hints,
            example_consistency=example_consistency
        )

        # Cache the result
        self._complexity_cache[task_hash] = features

        return features

    def _calculate_pattern_complexity(self, task: ARCTask) -> float:
        """Calculate pattern complexity based on grid structure."""
        complexity_scores = []

        for example in task.train_examples:
            input_grid = np.array(example["input"])
            output_grid = np.array(example.get("output", []))

            # Check for regular patterns
            input_entropy = self._calculate_grid_entropy(input_grid)
            output_entropy = self._calculate_grid_entropy(output_grid) if output_grid.size > 0 else 0

            # Check transformation complexity
            if output_grid.size > 0:
                size_change = abs(output_grid.size - input_grid.size) / max(input_grid.size, 1)
                complexity_scores.append((input_entropy + output_entropy) / 2 + size_change * 0.2)
            else:
                complexity_scores.append(input_entropy)

        return np.mean(complexity_scores) if complexity_scores else 0.5

    def _calculate_grid_entropy(self, grid: np.ndarray) -> float:
        """Calculate entropy of a grid."""
        if grid.size == 0:
            return 0.0

        # Flatten and calculate value frequencies
        values, counts = np.unique(grid.flatten(), return_counts=True)
        probabilities = counts / counts.sum()

        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(values))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_color_diversity(self, task: ARCTask) -> float:
        """Calculate color diversity across all grids."""
        all_colors = set()

        for example in task.train_examples:
            input_grid = np.array(example["input"])
            all_colors.update(np.unique(input_grid).tolist())

            if "output" in example:
                output_grid = np.array(example["output"])
                all_colors.update(np.unique(output_grid).tolist())

        # Test input
        if task.test_input:
            test_grid = np.array(task.test_input)
            all_colors.update(np.unique(test_grid).tolist())

        # Normalize to [0, 1] assuming max 10 colors
        return min(1.0, len(all_colors) / 10.0)

    def _calculate_transformation_hints(self, task: ARCTask) -> float:
        """Calculate hints about transformation complexity."""
        if not task.train_examples:
            return 0.5

        hints = []

        for example in task.train_examples:
            if "output" not in example:
                continue

            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])

            # Size change hint
            size_changed = input_grid.shape != output_grid.shape

            # Color mapping hint
            input_colors = set(np.unique(input_grid))
            output_colors = set(np.unique(output_grid))
            new_colors = output_colors - input_colors

            # Structure preservation hint
            if input_grid.shape == output_grid.shape:
                similarity = np.mean(input_grid == output_grid)
            else:
                similarity = 0.0

            hint_score = (
                (1.0 if size_changed else 0.0) * 0.3 +
                (len(new_colors) / max(len(output_colors), 1)) * 0.3 +
                (1.0 - similarity) * 0.4
            )
            hints.append(hint_score)

        return np.mean(hints) if hints else 0.5

    def _calculate_example_consistency(self, task: ARCTask) -> float:
        """Calculate consistency across training examples."""
        if len(task.train_examples) < 2:
            return 1.0  # Single example is perfectly consistent

        # Compare output shapes
        output_shapes = []
        for example in task.train_examples:
            if "output" in example:
                output_shapes.append(np.array(example["output"]).shape)

        if len(output_shapes) < 2:
            return 1.0

        # Check if all shapes are identical
        all_same_shape = all(shape == output_shapes[0] for shape in output_shapes)

        # Check color consistency
        color_sets = []
        for example in task.train_examples:
            if "output" in example:
                colors = set(np.unique(np.array(example["output"])))
                color_sets.append(colors)

        if color_sets:
            common_colors = set.intersection(*color_sets) if color_sets else set()
            avg_colors = np.mean([len(cs) for cs in color_sets])
            color_consistency = len(common_colors) / max(avg_colors, 1)
        else:
            color_consistency = 1.0

        # Combine metrics
        consistency = (
            (1.0 if all_same_shape else 0.5) * 0.5 +
            color_consistency * 0.5
        )

        return consistency

    def calculate_complexity_score(self, features: ComplexityFeatures) -> float:
        """Calculate weighted complexity score from features."""
        score = 0.0
        feature_dict = features.to_dict()

        for feature, weight in self.complexity_weights.items():
            score += feature_dict.get(feature, 0.0) * weight

        return np.clip(score, 0.0, 1.0)

    def determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score."""
        if score >= self.complexity_thresholds[ComplexityLevel.BREAKTHROUGH]:
            return ComplexityLevel.BREAKTHROUGH
        elif score >= self.complexity_thresholds[ComplexityLevel.COMPLEX]:
            return ComplexityLevel.COMPLEX
        elif score >= self.complexity_thresholds[ComplexityLevel.MEDIUM]:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE

    def calculate_routing_confidence(
        self,
        complexity_score: float,
        model_tier: ModelTier
    ) -> float:
        """Calculate confidence in routing decision."""
        # Distance from tier boundaries
        lower_bound, upper_bound = model_tier.complexity_range

        if complexity_score < lower_bound or complexity_score >= upper_bound:
            return 0.0  # Outside range

        # Calculate distance from boundaries
        range_size = upper_bound - lower_bound
        distance_from_lower = complexity_score - lower_bound
        distance_from_upper = upper_bound - complexity_score

        # Confidence is higher when score is in the middle of the range
        min_distance = min(distance_from_lower, distance_from_upper)
        confidence = 2 * min_distance / range_size

        return np.clip(confidence, 0.0, 1.0)

    def route(self, task: ARCTask, override_tier: str | None = None) -> RoutingDecision:
        """Route task to appropriate model tier."""
        # Analyze complexity
        features = self.analyze_complexity(task)
        complexity_score = self.calculate_complexity_score(features)
        complexity_level = self.determine_complexity_level(complexity_score)

        # Find matching tier
        if override_tier:
            # Find tier by name
            model_tier = next(
                (tier for tier in self.model_tiers if tier.name == override_tier),
                None
            )
            if not model_tier:
                raise ValueError(f"Unknown model tier: {override_tier}")
            confidence = 1.0  # Full confidence in override
            reasoning = f"Manually overridden to use {override_tier}"
        else:
            # Find tier by complexity score
            model_tier = None
            for tier in self.model_tiers[:-1]:  # Exclude fallback
                if tier.matches_complexity(complexity_score):
                    model_tier = tier
                    break

            if not model_tier:
                # Should not happen, but use fallback
                model_tier = self.model_tiers[-1]
                confidence = 0.5
                reasoning = "No matching tier found, using fallback"
            else:
                confidence = self.calculate_routing_confidence(complexity_score, model_tier)
                reasoning = self._generate_routing_reasoning(
                    features, complexity_score, complexity_level, model_tier
                )

        decision = RoutingDecision(
            model_tier=model_tier,
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            confidence=confidence,
            features=features,
            reasoning=reasoning
        )

        # Log routing decision
        logger.info(
            f"Routing decision for task {task.task_id}: "
            f"{model_tier.name} (complexity: {complexity_score:.2f}, "
            f"level: {complexity_level.value}, confidence: {confidence:.2f})"
        )

        return decision

    def _generate_routing_reasoning(
        self,
        features: ComplexityFeatures,
        score: float,
        level: ComplexityLevel,
        tier: ModelTier
    ) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasons = []

        # Analyze dominant features
        feature_dict = features.to_dict()
        sorted_features = sorted(
            feature_dict.items(),
            key=lambda x: x[1] * self.complexity_weights.get(x[0], 0),
            reverse=True
        )

        # Top contributing factors
        for feature_name, feature_value in sorted_features[:2]:
            weight = self.complexity_weights.get(feature_name, 0)
            contribution = feature_value * weight

            if contribution > 0.1:
                feature_desc = {
                    "grid_size_score": "large grid size",
                    "pattern_complexity": "complex patterns",
                    "color_diversity": "high color diversity",
                    "transformation_hints": "complex transformations",
                    "example_consistency": "inconsistent examples"
                }.get(feature_name, feature_name)

                reasons.append(f"{feature_desc} ({feature_value:.2f})")

        reasoning = (
            f"Task complexity {level.value} (score: {score:.2f}) "
            f"due to {', '.join(reasons)}. "
            f"Selected {tier.name} for optimal cost/performance balance."
        )

        return reasoning

    async def generate_with_routing(
        self,
        task: ARCTask,
        prompt: str,
        override_tier: str | None = None,
        use_cache: bool = True
    ) -> tuple[str, RoutingDecision, dict[str, Any]]:
        """Generate response with automatic routing."""
        # Route to appropriate model
        routing_decision = self.route(task, override_tier)
        model_tier = routing_decision.model_tier

        # Check cache if enabled
        if use_cache:
            cache_key = self._generate_cache_key(prompt, model_tier)
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached response for {model_tier.name}")
                return cached_response, routing_decision, {"cache_hit": True}

        # Check budget
        estimated_tokens = len(prompt.split()) * 2  # Rough estimate
        if not self.budget_controller.can_afford_request(model_tier.name, estimated_tokens):
            # Try fallback model
            logger.warning(f"Budget exceeded for {model_tier.name}, trying fallback")
            model_tier = self.model_tiers[-1]  # Local fallback
            routing_decision.model_tier = model_tier
            routing_decision.reasoning += " (Budget limit reached, using fallback)"

        # Get provider
        provider = self.providers.get(model_tier.model_id)
        if not provider:
            raise ValueError(f"No provider registered for {model_tier.model_id}")

        # Get circuit breaker
        circuit_breaker = self.circuit_breakers[model_tier.model_id]

        try:
            # Generate with circuit breaker protection
            response, input_tokens, output_tokens = await circuit_breaker.call_async(
                provider.generate,
                prompt,
                model_tier.max_tokens,
                model_tier.temperature
            )

            # Track usage
            await self.budget_controller.track_usage(
                model_name=model_tier.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                task_id=task.task_id
            )

            # Cache response
            if use_cache:
                await self._cache_response(
                    cache_key, response, routing_decision.features
                )

            # Track performance
            self._track_performance(
                model_tier.name,
                routing_decision.complexity_level,
                input_tokens + output_tokens
            )

            metadata = {
                "cache_hit": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": self.budget_controller.get_usage_summary()["total_cost"]
            }

            return response, routing_decision, metadata

        except Exception as e:
            logger.error(f"Error generating with {model_tier.name}: {e}")

            # Try fallback if not already using it
            if model_tier.model_id != self.model_tiers[-1].model_id:
                logger.info("Attempting fallback model")
                routing_decision.model_tier = self.model_tiers[-1]
                routing_decision.reasoning += f" (Fallback due to error: {str(e)})"
                return await self.generate_with_routing(
                    task, prompt, override_tier=self.model_tiers[-1].name, use_cache=use_cache
                )
            else:
                raise

    def _hash_task(self, task: ARCTask) -> str:
        """Generate hash for task."""
        task_data = {
            "train_examples": task.train_examples,
            "test_input": task.test_input
        }
        task_json = json.dumps(task_data, sort_keys=True)
        return hashlib.md5(task_json.encode()).hexdigest()

    def _generate_cache_key(self, prompt: str, model_tier: ModelTier) -> str:
        """Generate cache key for prompt and model."""
        key_data = f"{prompt}_{model_tier.model_id}_{model_tier.temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str) -> str | None:
        """Get cached response if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    return data["response"]
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        return None

    async def _cache_response(
        self,
        cache_key: str,
        response: str,
        features: ComplexityFeatures
    ):
        """Cache response with features."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "response": response,
                    "features": features.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

    def _track_performance(
        self,
        model_name: str,
        complexity_level: ComplexityLevel,
        tokens_used: int
    ):
        """Track model performance statistics."""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                "total_requests": 0,
                "complexity_distribution": {level.value: 0 for level in ComplexityLevel},
                "total_tokens": 0
            }

        stats = self.performance_stats[model_name]
        stats["total_requests"] += 1
        stats["complexity_distribution"][complexity_level.value] += 1
        stats["total_tokens"] += tokens_used

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all models."""
        return {
            "model_performance": self.performance_stats,
            "budget_status": self.budget_controller.get_usage_summary(),
            "circuit_breaker_status": {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            }
        }
