"""Unit tests for Smart Model Router."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.adapters.external.smart_model_router import (
    ComplexityFeatures,
    ComplexityLevel,
    ModelTier,
    RoutingDecision,
    SmartModelRouter,
)
from src.domain.models import ARCTask
from src.infrastructure.components.budget_controller import BudgetController
from src.infrastructure.config import Config


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, name: str):
        self.name = name
        self.generate = AsyncMock(return_value=("test response", 100, 50))

    def get_name(self) -> str:
        return self.name


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.platform = "local"
    return config


@pytest.fixture
def mock_budget_controller():
    """Create mock budget controller."""
    controller = Mock(spec=BudgetController)
    controller.can_afford_request.return_value = True
    controller.track_usage = AsyncMock()
    controller.get_usage_summary.return_value = {
        "total_cost": 10.0,
        "budget_limit": 100.0,
        "remaining_budget": 90.0,
        "usage_percent": 10.0,
        "status": "healthy"
    }
    return controller


@pytest.fixture
def router(mock_config, mock_budget_controller, tmp_path):
    """Create router instance for testing."""
    cache_dir = tmp_path / "test_cache"
    return SmartModelRouter(
        config=mock_config,
        budget_controller=mock_budget_controller,
        cache_dir=cache_dir
    )


@pytest.fixture
def sample_task():
    """Create sample ARC task for testing."""
    return ARCTask(
        task_id="test_task_001",
        task_source="test",
        train_examples=[
            {
                "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
            },
            {
                "input": [[2, 2, 2], [2, 3, 2], [2, 2, 2]],
                "output": [[3, 3, 3], [3, 2, 3], [3, 3, 3]]
            }
        ],
        test_input=[[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    )


@pytest.fixture
def complex_task():
    """Create complex ARC task for testing."""
    # Large grid with many colors
    large_input = [[i % 10 for i in range(30)] for _ in range(30)]
    large_output = [[9 - (i % 10) for i in range(30)] for _ in range(30)]

    return ARCTask(
        task_id="complex_task_001",
        task_source="test",
        train_examples=[
            {
                "input": large_input,
                "output": large_output
            }
        ],
        test_input=large_input
    )


class TestComplexityAnalysis:
    """Test complexity analysis functionality."""

    def test_analyze_complexity_simple_task(self, router, sample_task):
        """Test complexity analysis on simple task."""
        features = router.analyze_complexity(sample_task)

        assert isinstance(features, ComplexityFeatures)
        assert 0 <= features.grid_size_score <= 1
        assert 0 <= features.pattern_complexity <= 1
        assert 0 <= features.color_diversity <= 1
        assert 0 <= features.transformation_hints <= 1
        assert 0 <= features.example_consistency <= 1

    def test_analyze_complexity_complex_task(self, router, complex_task):
        """Test complexity analysis on complex task."""
        features = router.analyze_complexity(complex_task)

        # Complex task should have higher scores
        assert features.grid_size_score > 0.5  # Large grid
        assert features.color_diversity > 0.7   # Many colors (0-9)

    def test_complexity_caching(self, router, sample_task):
        """Test that complexity analysis is cached."""
        # First call
        features1 = router.analyze_complexity(sample_task)

        # Second call should use cache
        features2 = router.analyze_complexity(sample_task)

        assert features1 == features2
        assert len(router._complexity_cache) == 1

    def test_calculate_complexity_score(self, router):
        """Test complexity score calculation."""
        features = ComplexityFeatures(
            grid_size_score=0.5,
            pattern_complexity=0.6,
            color_diversity=0.4,
            transformation_hints=0.7,
            example_consistency=0.3
        )

        score = router.calculate_complexity_score(features)

        # Check weighted sum
        expected = (
            0.5 * 0.25 +  # grid_size
            0.6 * 0.30 +  # pattern
            0.4 * 0.20 +  # color
            0.7 * 0.15 +  # transformation
            0.3 * 0.10    # consistency
        )
        assert abs(score - expected) < 0.001

    def test_determine_complexity_level(self, router):
        """Test complexity level determination."""
        assert router.determine_complexity_level(0.2) == ComplexityLevel.SIMPLE
        assert router.determine_complexity_level(0.5) == ComplexityLevel.MEDIUM
        assert router.determine_complexity_level(0.7) == ComplexityLevel.COMPLEX
        assert router.determine_complexity_level(0.9) == ComplexityLevel.COMPLEX
        assert router.determine_complexity_level(0.96) == ComplexityLevel.BREAKTHROUGH


class TestModelRouting:
    """Test model routing functionality."""

    def test_route_simple_task(self, router, sample_task):
        """Test routing for simple task."""
        decision = router.route(sample_task)

        assert isinstance(decision, RoutingDecision)
        assert decision.model_tier.name == "Qwen2.5-Coder"  # Tier 1 for simple
        assert decision.complexity_level == ComplexityLevel.SIMPLE
        assert 0 <= decision.confidence <= 1
        assert decision.reasoning != ""

    def test_route_complex_task(self, router, complex_task):
        """Test routing for complex task."""
        decision = router.route(complex_task)

        # Should route to higher tier
        assert decision.model_tier.name in ["Gemini 2.5 Flash", "GLM-4.5", "GPT-5"]
        assert decision.complexity_level in [ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]

    def test_route_with_override(self, router, sample_task):
        """Test routing with tier override."""
        decision = router.route(sample_task, override_tier="GPT-5")

        assert decision.model_tier.name == "GPT-5"
        assert decision.confidence == 1.0  # Full confidence on override
        assert "override" in decision.reasoning.lower()

    def test_route_invalid_override(self, router, sample_task):
        """Test routing with invalid tier override."""
        with pytest.raises(ValueError, match="Unknown model tier"):
            router.route(sample_task, override_tier="InvalidModel")

    def test_routing_confidence_calculation(self, router):
        """Test routing confidence calculation."""
        tier = ModelTier(
            name="Test",
            complexity_range=(0.3, 0.7),
            model_id="test",
            max_tokens=1000,
            temperature=0.5,
            cost_per_million_input_tokens=0.1,
            cost_per_million_output_tokens=0.1
        )

        # Score at boundaries should have low confidence
        assert router.calculate_routing_confidence(0.3, tier) < 0.1
        assert router.calculate_routing_confidence(0.69, tier) < 0.1

        # Score in middle should have high confidence
        assert router.calculate_routing_confidence(0.5, tier) > 0.9

        # Score outside range should have zero confidence
        assert router.calculate_routing_confidence(0.2, tier) == 0.0
        assert router.calculate_routing_confidence(0.8, tier) == 0.0


class TestLLMGeneration:
    """Test LLM generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_with_routing(self, router, sample_task):
        """Test generation with automatic routing."""
        # Register mock provider
        provider = MockLLMProvider("qwen")
        router.register_provider("qwen2.5-coder-32b", provider)

        response, decision, metadata = await router.generate_with_routing(
            sample_task,
            "Generate a program to solve this task"
        )

        assert response == "test response"
        assert decision.model_tier.name == "Qwen2.5-Coder"
        assert metadata["cache_hit"] == False
        assert metadata["input_tokens"] == 100
        assert metadata["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_with_cache_hit(self, router, sample_task):
        """Test generation with cache hit."""
        provider = MockLLMProvider("qwen")
        router.register_provider("qwen2.5-coder-32b", provider)

        # First call
        response1, _, metadata1 = await router.generate_with_routing(
            sample_task,
            "test prompt",
            use_cache=True
        )

        # Second call should hit cache
        response2, _, metadata2 = await router.generate_with_routing(
            sample_task,
            "test prompt",
            use_cache=True
        )

        assert response2 == response1
        assert metadata2["cache_hit"] == True
        # Provider should only be called once
        provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_budget_exceeded(self, router, sample_task, mock_budget_controller):
        """Test generation when budget exceeded."""
        # Set up budget controller to reject request
        mock_budget_controller.can_afford_request.return_value = False

        # Register providers
        expensive_provider = MockLLMProvider("gpt5")
        fallback_provider = MockLLMProvider("falcon")
        router.register_provider("gpt-5", expensive_provider)
        router.register_provider("falcon-mamba-7b-local", fallback_provider)

        # Should fall back to local model
        response, decision, metadata = await router.generate_with_routing(
            sample_task,
            "test prompt",
            override_tier="GPT-5"  # Try to use expensive model
        )

        assert decision.model_tier.name == "Falcon Mamba 7B"  # Fallback
        assert "budget" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_generate_with_circuit_breaker_open(self, router, sample_task):
        """Test generation when circuit breaker is open."""
        # Register failing provider
        failing_provider = MockLLMProvider("qwen")
        failing_provider.generate.side_effect = Exception("API Error")

        fallback_provider = MockLLMProvider("falcon")

        router.register_provider("qwen2.5-coder-32b", failing_provider)
        router.register_provider("falcon-mamba-7b-local", fallback_provider)

        # Make requests to trip circuit breaker
        for _ in range(5):
            try:
                await router.generate_with_routing(sample_task, "test")
            except:
                pass

        # Circuit should be open, should use fallback
        response, decision, metadata = await router.generate_with_routing(
            sample_task,
            "test prompt"
        )

        assert decision.model_tier.name == "Falcon Mamba 7B"
        assert "fallback" in decision.reasoning.lower()


class TestProviderManagement:
    """Test provider registration and management."""

    def test_register_provider(self, router):
        """Test provider registration."""
        provider = MockLLMProvider("test")
        router.register_provider("test-model", provider)

        assert "test-model" in router.providers
        assert "test-model" in router.circuit_breakers
        assert router.providers["test-model"] == provider

    def test_get_performance_summary(self, router):
        """Test performance summary generation."""
        # Track some performance
        router._track_performance("GPT-5", ComplexityLevel.COMPLEX, 1000)
        router._track_performance("Qwen2.5-Coder", ComplexityLevel.SIMPLE, 500)
        router._track_performance("Qwen2.5-Coder", ComplexityLevel.SIMPLE, 600)

        summary = router.get_performance_summary()

        assert "model_performance" in summary
        assert "GPT-5" in summary["model_performance"]
        assert summary["model_performance"]["GPT-5"]["total_requests"] == 1
        assert summary["model_performance"]["Qwen2.5-Coder"]["total_requests"] == 2
        assert summary["model_performance"]["Qwen2.5-Coder"]["total_tokens"] == 1100


class TestHelperMethods:
    """Test helper methods."""

    def test_hash_task(self, router, sample_task):
        """Test task hashing."""
        hash1 = router._hash_task(sample_task)
        hash2 = router._hash_task(sample_task)

        # Same task should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest length

    def test_generate_cache_key(self, router):
        """Test cache key generation."""
        tier = router.model_tiers[0]
        key = router._generate_cache_key("test prompt", tier)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

    def test_generate_routing_reasoning(self, router):
        """Test routing reasoning generation."""
        features = ComplexityFeatures(
            grid_size_score=0.8,
            pattern_complexity=0.6,
            color_diversity=0.4,
            transformation_hints=0.3,
            example_consistency=0.2
        )

        reasoning = router._generate_routing_reasoning(
            features,
            0.65,
            ComplexityLevel.COMPLEX,
            router.model_tiers[2]  # GLM-4.5
        )

        assert "complex" in reasoning.lower()
        assert "GLM-4.5" in reasoning
        assert "large grid size" in reasoning  # Dominant feature
