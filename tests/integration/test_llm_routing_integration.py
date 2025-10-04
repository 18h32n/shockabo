"""Integration tests for LLM routing system."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.adapters.external.base_llm_client import BaseLLMClient
from src.adapters.external.llm_cache_manager import LLMCacheManager
from src.adapters.external.smart_model_router import SmartModelRouter
from src.adapters.strategies.program_synthesis_enhanced import (
    EnhancedProgramSynthesisAdapter,
    EnhancedProgramSynthesisConfig,
)
from src.domain.models import ARCTask, StrategyType
from src.infrastructure.components.budget_controller import create_default_budget_controller
from src.infrastructure.config import Config


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for integration testing."""

    def __init__(self, name: str, response: str = "def solve(grid): return grid",
                 cost_per_million_input: float = 0.15, cost_per_million_output: float = 0.15):
        self.name = name
        self.mock_response = response
        self.cost_per_million_input = cost_per_million_input
        self.cost_per_million_output = cost_per_million_output
        super().__init__(api_key="test_key", timeout=5)

    def get_name(self) -> str:
        return self.name

    def _get_api_key_from_credentials(self) -> str:
        return "test_key"

    async def _prepare_request(self, prompt, max_tokens, temperature, **kwargs):
        return {"prompt": prompt, "max_tokens": max_tokens}

    async def _parse_response(self, response_data):
        # Simulate token counting
        tokens = len(self.mock_response.split())
        return self.mock_response, tokens * 2, tokens

    def _estimate_request_cost(self, prompt_length: int, max_tokens: int) -> float:
        """Estimate the cost of a request before making it."""
        estimated_input_tokens = prompt_length / 4
        input_cost = (estimated_input_tokens / 1_000_000) * self.cost_per_million_input
        output_cost = (max_tokens / 1_000_000) * self.cost_per_million_output
        return input_cost + output_cost

    def _calculate_actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the actual cost based on token usage."""
        input_cost = (input_tokens / 1_000_000) * self.cost_per_million_input
        output_cost = (output_tokens / 1_000_000) * self.cost_per_million_output
        return input_cost + output_cost

    async def generate(self, prompt, max_tokens, temperature, **kwargs):
        # Simulate some processing time
        await asyncio.sleep(0.1)
        return await self._parse_response({})


@pytest.fixture
def test_cache_dir(tmp_path):
    """Create test cache directory."""
    return tmp_path / "llm_cache"


@pytest.fixture
def budget_controller():
    """Create budget controller for testing."""
    controller = create_default_budget_controller(budget_limit=10.0)  # Small budget for testing
    return controller


@pytest.fixture
def cache_manager(test_cache_dir):
    """Create cache manager for testing."""
    return LLMCacheManager(
        cache_dir=test_cache_dir,
        max_cache_size_gb=0.1,  # Small size for testing
        similarity_threshold=0.85
    )


@pytest.fixture
def sample_arc_task():
    """Create sample ARC task."""
    return ARCTask(
        task_id="integration_test_001",
        task_source="test",
        train_examples=[
            {
                "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                "output": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            }
        ],
        test_input=[[2, 0, 2], [0, 2, 0], [2, 0, 2]]
    )


class TestLLMRoutingIntegration:
    """Test integrated LLM routing functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_routing_and_generation(
        self,
        budget_controller,
        cache_manager,
        sample_arc_task,
        test_cache_dir
    ):
        """Test complete routing and generation flow."""
        # Create router
        config = Config()
        router = SmartModelRouter(
            config=config,
            budget_controller=budget_controller,
            cache_dir=test_cache_dir
        )

        # Register mock clients
        clients = {
            "qwen2.5-coder-32b": MockLLMClient("Qwen", "def solve(g): return [[0]*3 for _ in range(3)]"),
            "gemini-2.5-flash": MockLLMClient("Gemini", "def solve(g): return g[::-1]"),
            "glm-4.5": MockLLMClient("GLM", "def solve(g): return [[r[i] for r in g] for i in range(3)]"),
            "gpt-5": MockLLMClient("GPT5", "def solve(g): return [[1-c for c in r] for r in g]"),
            "falcon-mamba-7b-local": MockLLMClient("Falcon", "def solve(g): return g")
        }

        for model_id, client in clients.items():
            router.register_provider(model_id, client)

        # Test routing for different tasks
        response, decision, metadata = await router.generate_with_routing(
            sample_arc_task,
            "Generate a Python function to solve this ARC task"
        )

        assert response in [client.mock_response for client in clients.values()]
        assert decision.model_tier is not None
        assert not metadata["cache_hit"]
        assert "total_cost" in metadata

        # Second call should use cache
        response2, decision2, metadata2 = await router.generate_with_routing(
            sample_arc_task,
            "Generate a Python function to solve this ARC task"
        )

        assert response2 == response
        assert metadata2["cache_hit"]

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, budget_controller, sample_arc_task, test_cache_dir):
        """Test that budget limits are enforced."""
        # Create router with very limited budget
        budget_controller = create_default_budget_controller(budget_limit=0.01)  # $0.01 budget

        config = Config()
        router = SmartModelRouter(
            config=config,
            budget_controller=budget_controller,
            cache_dir=test_cache_dir
        )

        # Register expensive and free clients
        expensive_client = MockLLMClient("GPT5")
        free_client = MockLLMClient("Falcon")

        router.register_provider("gpt-5", expensive_client)
        router.register_provider("falcon-mamba-7b-local", free_client)

        # Make requests until budget exhausted
        responses = []
        for i in range(10):
            try:
                # Try to use expensive model
                response, decision, _ = await router.generate_with_routing(
                    sample_arc_task,
                    f"Test prompt {i}",
                    override_tier="GPT-5"
                )
                responses.append((response, decision.model_tier.name))
            except Exception:
                break

        # Should have fallen back to free model after budget exhausted
        model_names = [name for _, name in responses]
        assert "Falcon Mamba 7B" in model_names

    @pytest.mark.asyncio
    async def test_cache_similarity_matching(self, cache_manager, sample_arc_task):
        """Test cache similarity matching functionality."""
        # Store a cached response
        await cache_manager.put(
            prompt="Generate solution for grid transformation",
            model_name="test-model",
            temperature=0.7,
            response="cached_solution",
            token_count=100,
            task_features={
                "grid_size_score": 0.3,
                "pattern_complexity": 0.4,
                "color_diversity": 0.2,
                "transformation_hints": 0.5,
                "example_consistency": 0.8
            }
        )

        # Try to retrieve with similar features
        similar_features = {
            "grid_size_score": 0.32,  # Slightly different
            "pattern_complexity": 0.38,
            "color_diversity": 0.21,
            "transformation_hints": 0.48,
            "example_consistency": 0.79
        }

        cached = await cache_manager.get(
            prompt="Different prompt",  # Different prompt
            model_name="test-model",
            temperature=0.7,
            task_features=similar_features
        )

        # Should find similar cache entry
        assert cached == "cached_solution"

    @pytest.mark.asyncio
    async def test_performance_tracking(self, budget_controller, sample_arc_task, test_cache_dir):
        """Test performance tracking and reporting."""
        from src.adapters.external.llm_monitoring_dashboard import LLMMonitoringDashboard

        # Create router and dashboard
        config = Config()
        router = SmartModelRouter(
            config=config,
            budget_controller=budget_controller,
            cache_dir=test_cache_dir
        )

        dashboard = LLMMonitoringDashboard(
            budget_controller=budget_controller,
            model_router=router,
            output_dir=test_cache_dir / "reports"
        )

        # Register mock client
        client = MockLLMClient("TestModel")
        router.register_provider("test-model", client)
        router.model_tiers[0].model_id = "test-model"  # Map to first tier

        # Make some requests and track
        for i in range(5):
            response, decision, metadata = await router.generate_with_routing(
                sample_arc_task,
                f"Test prompt {i}"
            )

            # Log to dashboard
            dashboard.log_routing_decision(
                task_id=sample_arc_task.task_id,
                model_name=decision.model_tier.name,
                complexity_score=decision.complexity_score,
                complexity_level=decision.complexity_level.value,
                confidence=decision.confidence,
                reasoning=decision.reasoning
            )

            dashboard.log_request_performance(
                task_id=sample_arc_task.task_id,
                model_name=decision.model_tier.name,
                success=True,
                latency_seconds=0.1,
                input_tokens=metadata.get("input_tokens", 10),
                output_tokens=metadata.get("output_tokens", 5),
                cost=0.001
            )

        # Generate reports
        cost_report = dashboard.generate_cost_report()
        performance_report = dashboard.generate_performance_report()

        assert cost_report["summary"]["total_spent"] > 0
        assert len(performance_report["model_performance"]) > 0
        assert performance_report["routing_analysis"]["total_routing_decisions"] == 5


class TestProgramSynthesisIntegration:
    """Test enhanced program synthesis with LLM routing."""

    @pytest.mark.asyncio
    async def test_enhanced_program_synthesis(self, sample_arc_task, test_cache_dir):
        """Test program synthesis with LLM generation."""
        # Create enhanced synthesis adapter
        config = EnhancedProgramSynthesisConfig(
            use_llm_generation=True,
            budget_limit=1.0,
            llm_generation_ratio=0.5,  # 50% LLM programs
            max_generation_attempts=10,
            cache_dir=test_cache_dir
        )

        adapter = EnhancedProgramSynthesisAdapter(config)

        # Mock LLM clients
        with patch.object(adapter, '_register_llm_providers'):
            # Manually register a mock client
            if adapter.model_router:
                mock_client = MockLLMClient(
                    "TestLLM",
                    """
program1 = [
    rotate(90),
    color_replace(1, 0)
]
                    """
                )
                adapter.model_router.register_provider("qwen2.5-coder-32b", mock_client)

            # Solve task
            solution = await adapter._solve_async(sample_arc_task)

            assert solution.task_id == sample_arc_task.task_id
            assert solution.strategy_used == StrategyType.PROGRAM_SYNTHESIS
            assert "llm_usage" in solution.metadata

    @pytest.mark.asyncio
    async def test_llm_program_parsing(self, test_cache_dir):
        """Test parsing of LLM-generated programs."""
        config = EnhancedProgramSynthesisConfig(cache_dir=test_cache_dir)
        adapter = EnhancedProgramSynthesisAdapter(config)

        # Test various LLM response formats
        test_responses = [
            # Python code block
            """```python
program = [
    rotate(90),
    flip("horizontal"),
    color_replace(1, 2)
]
```""",
            # Multiple programs
            """```
program1 = [
    rotate(180),
    color_filter(1)
]

program2 = [
    flip("vertical"),
    translate(1, 0)
]
```""",
            # Inline operations
            "Use rotate(90) followed by color_replace(0, 1) and flood_fill(0, 0, 2)"
        ]

        for response in test_responses:
            programs = adapter._parse_llm_programs(response)
            assert len(programs) > 0
            assert all(hasattr(p, 'operations') for p in programs)

    @pytest.mark.asyncio
    async def test_hybrid_generation_strategy(self, sample_arc_task, test_cache_dir):
        """Test hybrid program generation with templates and LLM."""
        config = EnhancedProgramSynthesisConfig(
            use_llm_generation=True,
            generation_strategy="HYBRID",
            llm_generation_ratio=0.3,
            max_generation_attempts=20,
            cache_dir=test_cache_dir
        )

        adapter = EnhancedProgramSynthesisAdapter(config)

        # Mock the router
        with patch.object(adapter, 'model_router') as mock_router:
            mock_router.generate_with_routing = AsyncMock(
                return_value=(
                    "program = [rotate(90)]",
                    Mock(model_tier=Mock(name="TestModel")),
                    {"cache_hit": False}
                )
            )

            # Generate programs
            programs = await adapter._generate_enhanced_programs(
                sample_arc_task,
                {"pattern_complexity": "medium"}
            )

            assert len(programs) > 0
            # Should have both template-based and LLM programs
            assert len(programs) >= 10  # At least some programs generated


class TestCostOptimization:
    """Test cost optimization features."""

    @pytest.mark.asyncio
    async def test_cost_simulation(self, test_cache_dir):
        """Test cost simulation for different routing strategies."""
        # Create two routers with different strategies
        budget1 = create_default_budget_controller(100.0)
        budget2 = create_default_budget_controller(100.0)

        config = Config()

        # Conservative router (prefer cheaper models)
        conservative_router = SmartModelRouter(config, budget1, test_cache_dir / "conservative")
        conservative_router.complexity_thresholds = {
            "simple": 0.5,  # More tasks classified as simple
            "medium": 0.8,
            "complex": 0.95,
            "breakthrough": 0.99
        }

        # Aggressive router (prefer better models)
        aggressive_router = SmartModelRouter(config, budget2, test_cache_dir / "aggressive")
        aggressive_router.complexity_thresholds = {
            "simple": 0.2,  # Fewer tasks classified as simple
            "medium": 0.5,
            "complex": 0.7,
            "breakthrough": 0.9
        }

        # Simulate routing decisions for various tasks
        tasks = []
        for i in range(50):
            # Create tasks with varying complexity
            grid_size = 3 + i % 27  # 3 to 30
            num_colors = 2 + i % 8   # 2 to 9

            task = ARCTask(
                task_id=f"sim_task_{i}",
                task_source="simulation",
                train_examples=[{
                    "input": [[j % num_colors for j in range(grid_size)] for _ in range(grid_size)],
                    "output": [[(j+1) % num_colors for j in range(grid_size)] for _ in range(grid_size)]
                }],
                test_input=[[0] * grid_size for _ in range(grid_size)]
            )
            tasks.append(task)

        # Route all tasks with both strategies
        conservative_costs = []
        aggressive_costs = []

        for task in tasks:
            cons_decision = conservative_router.route(task)
            aggr_decision = aggressive_router.route(task)

            # Estimate costs (simplified)
            cons_cost = cons_decision.model_tier.cost_per_million_input_tokens * 0.001
            aggr_cost = aggr_decision.model_tier.cost_per_million_input_tokens * 0.001

            conservative_costs.append(cons_cost)
            aggressive_costs.append(aggr_cost)

        # Conservative should be cheaper on average
        assert sum(conservative_costs) < sum(aggressive_costs)

        # But aggressive should use higher-tier models more often
        cons_model_usage = {}
        aggr_model_usage = {}

        for task in tasks[:10]:  # Sample subset
            cons_dec = conservative_router.route(task)
            aggr_dec = aggressive_router.route(task)

            cons_model_usage[cons_dec.model_tier.name] = cons_model_usage.get(cons_dec.model_tier.name, 0) + 1
            aggr_model_usage[aggr_dec.model_tier.name] = aggr_model_usage.get(aggr_dec.model_tier.name, 0) + 1

        # Verify routing patterns match strategy
        assert cons_model_usage.get("Qwen2.5-Coder", 0) > aggr_model_usage.get("Qwen2.5-Coder", 0)


class TestBudgetEnforcement:
    """Comprehensive tests for $100 budget hard limit enforcement."""

    @pytest.mark.asyncio
    async def test_budget_hard_limit_enforcement(self, sample_arc_task):
        """Test that requests stop completely at $100 budget limit."""
        from src.infrastructure.components.budget_controller import BudgetExceededException

        # Create budget controller with $100 limit
        budget_controller = create_default_budget_controller(budget_limit=100.0)

        # Create expensive mock clients
        expensive_clients = {
            "qwen2.5-coder-32b": MockLLMClient("Qwen", cost_per_million_input=0.15, cost_per_million_output=0.15),
            "gemini-2.5-flash": MockLLMClient("Gemini", cost_per_million_input=0.31, cost_per_million_output=2.62),
            "glm-4.5": MockLLMClient("GLM", cost_per_million_input=0.59, cost_per_million_output=2.19),
            "gpt-5": MockLLMClient("GPT5", cost_per_million_input=1.25, cost_per_million_output=10.00),
            "falcon-mamba-7b-local": MockLLMClient("Falcon", cost_per_million_input=0.0, cost_per_million_output=0.0)
        }

        # Track costs
        total_cost = 0.0
        successful_requests = 0

        # Simulate heavy usage with GPT-5
        for i in range(1000):  # Many requests to ensure we hit the limit
            client = expensive_clients["gpt-5"]
            client.budget_controller = budget_controller

            try:
                # Large request (8K tokens output)
                response, input_tokens, output_tokens = await client.generate(
                    prompt="Generate a complex solution" * 100,  # ~400 tokens input
                    max_tokens=8192,
                    temperature=0.9
                )

                # Calculate actual cost
                actual_cost = client._calculate_actual_cost(400, 8192)
                total_cost += actual_cost
                successful_requests += 1

            except BudgetExceededException:
                # This is expected - budget limit reached
                break

        # Verify we didn't exceed $100
        assert total_cost < 100.0
        assert budget_controller.get_remaining_budget() >= 0

        # Verify we can't make any more expensive requests
        with pytest.raises(BudgetExceededException):
            await client.generate("test", max_tokens=1000, temperature=0.5)

        # But we should still be able to use free local model
        local_client = expensive_clients["falcon-mamba-7b-local"]
        local_client.budget_controller = budget_controller

        response, _, _ = await local_client.generate("test", max_tokens=100, temperature=0.5)
        assert response  # Should work since it's free

    @pytest.mark.asyncio
    async def test_budget_tracking_across_all_clients(self):
        """Test that budget is properly tracked across all 5 LLM clients."""
        from src.infrastructure.components.budget_controller import BudgetController

        # Create shared budget controller
        budget_controller = BudgetController(budget_limit=10.0, persistence_path=None)

        # Import all real clients
        from src.adapters.external.gemini_client import GeminiClient
        from src.adapters.external.glm_client import GLMClient
        from src.adapters.external.gpt5_client import GPT5Client
        from src.adapters.external.local_model_client import LocalModelClient
        from src.adapters.external.qwen_client import QwenClient

        # Create all clients with shared budget controller
        clients = [
            QwenClient(budget_controller=budget_controller),
            GeminiClient(budget_controller=budget_controller),
            GLMClient(budget_controller=budget_controller),
            GPT5Client(budget_controller=budget_controller),
            LocalModelClient(budget_controller=budget_controller)
        ]

        # Verify all clients have cost estimation methods
        for client in clients:
            assert hasattr(client, '_estimate_request_cost')
            assert hasattr(client, '_calculate_actual_cost')
            assert client.budget_controller is budget_controller

            # Test cost estimation
            est_cost = client._estimate_request_cost(1000, 500)  # 1000 chars, 500 max tokens
            assert isinstance(est_cost, float)
            assert est_cost >= 0

            # Test actual cost calculation
            actual_cost = client._calculate_actual_cost(250, 500)  # 250 input, 500 output tokens
            assert isinstance(actual_cost, float)
            assert actual_cost >= 0

    @pytest.mark.asyncio
    async def test_budget_persistence_and_recovery(self, tmp_path):
        """Test budget state persistence across restarts."""
        persistence_file = tmp_path / "budget_state.json"

        # Phase 1: Use some budget
        budget_controller1 = BudgetController(
            budget_limit=50.0,
            persistence_path=persistence_file
        )

        # Track usage
        budget_controller1.track_usage("GPT-5", 10.0, 1000, 2000)
        budget_controller1.track_usage("GLM-4.5", 5.0, 500, 1000)

        assert budget_controller1.get_total_spent() == 15.0
        assert budget_controller1.get_remaining_budget() == 35.0

        # Phase 2: Create new controller with same persistence file
        budget_controller2 = BudgetController(
            budget_limit=50.0,
            persistence_path=persistence_file
        )

        # Should restore previous state
        assert budget_controller2.get_total_spent() == 15.0
        assert budget_controller2.get_remaining_budget() == 35.0

        # Continue using budget
        budget_controller2.track_usage("Gemini", 20.0, 2000, 4000)

        # Phase 3: Verify final state
        budget_controller3 = BudgetController(
            budget_limit=50.0,
            persistence_path=persistence_file
        )

        assert budget_controller3.get_total_spent() == 35.0
        assert budget_controller3.get_remaining_budget() == 15.0

    @pytest.mark.asyncio
    async def test_budget_enforcement_in_synthesis_adapter(self, sample_arc_task, tmp_path):
        """Test budget enforcement in EnhancedProgramSynthesisAdapter."""

        # Create adapter with very limited budget
        config = EnhancedProgramSynthesisConfig(
            use_llm_generation=True,
            budget_limit=0.001,  # $0.001 - extremely small
            llm_generation_ratio=1.0,  # Try to use LLM for everything
            max_generation_attempts=10,
            cache_dir=tmp_path
        )

        adapter = EnhancedProgramSynthesisAdapter(config)

        # Mock expensive LLM client
        expensive_client = MockLLMClient(
            "ExpensiveModel",
            response="def solve(g): return g",
            cost_per_million_input=1000.0,  # Very expensive
            cost_per_million_output=1000.0
        )
        expensive_client.budget_controller = adapter.budget_controller

        # Register the expensive client
        if adapter.model_router:
            adapter.model_router.register_provider("gpt-5", expensive_client)
            # Force router to use expensive model
            adapter.model_router.model_tiers[-1].model_id = "gpt-5"

        # Try to solve - should fail or fall back to templates
        solution = await adapter._solve_async(sample_arc_task)

        # Verify budget wasn't exceeded
        assert adapter.budget_controller.get_total_spent() <= 0.001

        # Solution should still be generated (via templates or fallback)
        assert solution.task_id == sample_arc_task.task_id

    @pytest.mark.asyncio
    async def test_budget_alert_system(self):
        """Test budget warning alerts at 80% threshold."""
        from src.infrastructure.components.budget_controller import BudgetController

        # Create controller with $100 budget
        budget_controller = BudgetController(budget_limit=100.0)

        # Track usage up to 79%
        budget_controller.track_usage("Model1", 79.0, 10000, 20000)
        alerts = budget_controller.check_budget_alerts()
        assert len(alerts) == 0  # No alert yet

        # Track usage to 80%
        budget_controller.track_usage("Model2", 1.0, 100, 200)
        alerts = budget_controller.check_budget_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "WARNING"
        assert "80%" in alerts[0]["message"]

        # Track usage to 95%
        budget_controller.track_usage("Model3", 15.0, 1500, 3000)
        alerts = budget_controller.check_budget_alerts()
        assert len(alerts) >= 1
        assert any(alert["type"] == "CRITICAL" for alert in alerts)


class TestRoutingPerformance:
    """Performance benchmarks for routing decisions."""

    @pytest.mark.asyncio
    async def test_routing_decision_latency(self, sample_arc_task, test_cache_dir):
        """Test that routing decisions complete within 10ms."""
        import time

        from src.adapters.external.smart_model_router import SmartModelRouter
        from src.infrastructure.config import Config

        # Create router
        config = Config()
        budget_controller = create_default_budget_controller(budget_limit=100.0)
        router = SmartModelRouter(
            config=config,
            budget_controller=budget_controller,
            cache_dir=test_cache_dir
        )

        # Warm up the router
        _ = router.route(sample_arc_task)

        # Measure routing latency over multiple iterations
        latencies = []
        iterations = 100

        for i in range(iterations):
            # Create varied tasks to test different complexity levels
            grid_size = 3 + (i % 28)  # 3 to 30
            num_colors = 2 + (i % 8)   # 2 to 9

            task = ARCTask(
                task_id=f"perf_test_{i}",
                task_source="performance_test",
                train_examples=[{
                    "input": [[j % num_colors for j in range(grid_size)] for _ in range(grid_size)],
                    "output": [[(j+1) % num_colors for j in range(grid_size)] for _ in range(grid_size)]
                }],
                test_input=[[0] * grid_size for _ in range(grid_size)]
            )

            # Measure routing decision time
            start_time = time.perf_counter()
            decision = router.route(task)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            assert decision is not None
            assert decision.model_tier is not None

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        # Performance assertions
        assert avg_latency < 10.0, f"Average routing latency {avg_latency:.2f}ms exceeds 10ms"
        assert p95_latency < 15.0, f"95th percentile latency {p95_latency:.2f}ms exceeds 15ms"
        assert max_latency < 20.0, f"Max routing latency {max_latency:.2f}ms exceeds 20ms"

    @pytest.mark.asyncio
    async def test_concurrent_routing_performance(self, test_cache_dir):
        """Test routing performance under concurrent load."""
        import asyncio
        import time

        from src.adapters.external.smart_model_router import SmartModelRouter
        from src.infrastructure.config import Config

        # Create router
        config = Config()
        budget_controller = create_default_budget_controller(budget_limit=1000.0)
        router = SmartModelRouter(
            config=config,
            budget_controller=budget_controller,
            cache_dir=test_cache_dir
        )

        # Register mock clients
        for tier in ["qwen2.5-coder-32b", "gemini-2.5-flash", "glm-4.5", "gpt-5", "falcon-mamba-7b-local"]:
            router.register_provider(tier, MockLLMClient(tier))

        # Create diverse tasks
        tasks = []
        for i in range(50):
            task = ARCTask(
                task_id=f"concurrent_test_{i}",
                task_source="concurrent_test",
                train_examples=[{
                    "input": [[j % 5 for j in range(10)] for _ in range(10)],
                    "output": [[j % 5 for j in range(10)] for _ in range(10)]
                }],
                test_input=[[0] * 10 for _ in range(10)]
            )
            tasks.append(task)

        # Measure concurrent routing
        start_time = time.perf_counter()

        # Create concurrent routing tasks
        async def route_task(task):
            return await asyncio.create_task(
                asyncio.to_thread(router.route, task)
            )

        # Route all tasks concurrently
        routing_tasks = [route_task(task) for task in tasks]
        decisions = await asyncio.gather(*routing_tasks)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Verify all decisions were made
        assert len(decisions) == 50
        assert all(d.model_tier is not None for d in decisions)

        # Performance check - should handle 50 concurrent requests quickly
        avg_time_per_request = total_time_ms / 50
        assert avg_time_per_request < 20.0, f"Average time per concurrent request {avg_time_per_request:.2f}ms exceeds 20ms"

    @pytest.mark.asyncio
    async def test_cache_lookup_performance(self, test_cache_dir):
        """Test cache lookup performance."""
        import time

        from src.adapters.external.llm_cache_manager import LLMCacheManager

        # Create cache manager
        cache_manager = LLMCacheManager(
            cache_dir=test_cache_dir,
            max_cache_size_gb=1.0,
            similarity_threshold=0.85
        )

        # Populate cache with many entries
        num_entries = 1000
        for i in range(num_entries):
            await cache_manager.put(
                prompt=f"Test prompt {i}",
                model_name="test-model",
                temperature=0.7,
                response=f"Response {i}",
                token_count=100,
                task_features={
                    "grid_size_score": i / num_entries,
                    "pattern_complexity": (i % 100) / 100,
                    "color_diversity": ((i + 50) % 100) / 100,
                    "transformation_hints": ((i + 25) % 100) / 100,
                    "example_consistency": 0.8
                }
            )

        # Measure lookup performance
        lookup_times = []
        hits = 0

        for i in range(100):
            features = {
                "grid_size_score": (i * 10 % num_entries) / num_entries,
                "pattern_complexity": (i % 100) / 100,
                "color_diversity": ((i + 50) % 100) / 100,
                "transformation_hints": ((i + 25) % 100) / 100,
                "example_consistency": 0.8
            }

            start_time = time.perf_counter()
            result = await cache_manager.get(
                prompt=f"Different prompt {i}",
                model_name="test-model",
                temperature=0.7,
                task_features=features
            )
            end_time = time.perf_counter()

            lookup_time_ms = (end_time - start_time) * 1000
            lookup_times.append(lookup_time_ms)

            if result is not None:
                hits += 1

        # Calculate statistics
        avg_lookup = sum(lookup_times) / len(lookup_times)
        max_lookup = max(lookup_times)

        # Performance assertions
        assert avg_lookup < 50.0, f"Average cache lookup {avg_lookup:.2f}ms exceeds 50ms"
        assert max_lookup < 100.0, f"Max cache lookup {max_lookup:.2f}ms exceeds 100ms"
        assert hits > 0, "Cache similarity matching should find some hits"
