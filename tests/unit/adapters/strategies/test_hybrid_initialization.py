"""
Unit tests for hybrid initialization with LLM-generated programs.

Tests Task 7.3: Create hybrid initialization using LLM-generated programs.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adapters.strategies.hybrid_initialization import (
    HybridLLMProgramGenerator,
    HybridPopulationInitializer,
    LLMProgramPrompt,
)
from src.domain.models import ARCTask, Grid


class TestLLMProgramPrompt:
    """Test LLM program prompt generation."""

    def test_prompt_creation(self):
        """Test creating LLM prompt."""
        prompt = LLMProgramPrompt(
            task_description="Transform 3x3 grid by rotating",
            input_examples=["1 2 3\n4 5 6\n7 8 9"],
            output_examples=["7 4 1\n8 5 2\n9 6 3"],
            constraints=["Use rotation operations", "Preserve all values"],
            available_operations=["rotate", "flip"],
            program_style="simple"
        )

        prompt_str = prompt.to_prompt()

        assert "Transform 3x3 grid by rotating" in prompt_str
        assert "rotate" in prompt_str
        assert "flip" in prompt_str
        assert "simple program" in prompt_str
        assert "Example 1:" in prompt_str

    def test_example_formatting(self):
        """Test formatting of input/output examples."""
        prompt = LLMProgramPrompt(
            task_description="Test",
            input_examples=["1 2\n3 4", "5 6\n7 8"],
            output_examples=["4 3\n2 1", "8 7\n6 5"],
            constraints=[],
            available_operations=["flip"],
            program_style="simple"
        )

        formatted = prompt._format_examples()

        assert "Example 1:" in formatted
        assert "Example 2:" in formatted
        assert "Input:\n1 2\n3 4" in formatted
        assert "Output:\n4 3\n2 1" in formatted


class TestHybridLLMProgramGenerator:
    """Test hybrid LLM program generator."""

    @pytest.fixture
    def mock_router(self):
        """Create mock smart model router."""
        router = MagicMock()
        router.route_task = AsyncMock()
        router.generate_with_routing = AsyncMock()
        return router

    @pytest.fixture
    def generator(self, mock_router):
        """Create generator with mock router."""
        return HybridLLMProgramGenerator(
            model_router=mock_router,
            available_operations=["rotate", "flip", "translate"]
        )

    @pytest.fixture
    def mock_task(self):
        """Create mock ARC task."""
        task = MagicMock(spec=ARCTask)
        task.task_id = "test_task"

        # Create mock training examples with proper structure
        # Each training example is a dict with "input" and "output" keys
        input_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        output_grid = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

        training_example = {
            "input": input_grid,
            "output": output_grid
        }

        # Create mock training pairs for backward compatibility with hybrid_initialization.py
        class MockTrainingPair:
            def __init__(self, input_data, output_data):
                self.input = MagicMock()
                self.input.shape = (len(input_data), len(input_data[0]))
                self.input.data = input_data
                self.output = MagicMock()
                self.output.shape = (len(output_data), len(output_data[0]))
                self.output.data = output_data

        mock_pair = MockTrainingPair(input_grid, output_grid)
        
        task.train_examples = [training_example]
        task.train_pairs = [mock_pair]  # For compatibility with source file
        return task

    @pytest.mark.asyncio
    async def test_generate_programs(self, generator, mock_router, mock_task):
        """Test generating programs."""
        # Mock routing decision
        routing_decision = MagicMock()
        routing_decision.complexity_level = MagicMock()
        routing_decision.complexity_level.value = "simple"
        mock_router.route_task.return_value = routing_decision

        # Mock LLM response
        llm_response = json.dumps([
            {"name": "rotate", "parameters": {"angle": 90}},
            {"name": "flip", "parameters": {"direction": "horizontal"}}
        ])
        mock_router.generate_with_routing.return_value = llm_response

        # Generate programs
        programs = await generator.generate_programs(
            task=mock_task,
            num_programs=2,
            diversity_level="high"
        )

        assert len(programs) >= 2
        assert mock_router.route_task.called
        assert mock_router.generate_with_routing.called

    def test_get_program_styles(self, generator):
        """Test getting program styles based on diversity."""
        from src.adapters.external.smart_model_router import ComplexityLevel

        # High diversity
        styles = generator._get_program_styles("high", ComplexityLevel.MEDIUM)
        assert len(styles) == 5
        assert "simple" in styles
        assert "complex" in styles

        # Low diversity
        styles = generator._get_program_styles("low", ComplexityLevel.SIMPLE)
        assert len(styles) == 1
        assert styles[0] == "simple"

    def test_parse_llm_response(self, generator):
        """Test parsing LLM response."""
        # Valid JSON response
        response = '''Here is the program:
        [
            {"name": "rotate", "parameters": {"angle": 90}},
            {"name": "flip", "parameters": {"direction": "vertical"}}
        ]'''

        program = generator._parse_llm_response(response)
        assert program is not None
        assert len(program) == 2
        assert program[0]["name"] == "rotate"

        # Invalid response
        invalid_response = "This is not JSON"
        program = generator._parse_llm_response(invalid_response)
        assert program is None

    def test_validate_program(self, generator):
        """Test program validation."""
        # Valid program
        valid_program = [
            {"name": "rotate", "parameters": {"angle": 90}},
            {"name": "flip", "parameters": {"direction": "horizontal"}}
        ]
        assert generator._validate_program(valid_program) is True

        # Invalid programs
        assert generator._validate_program([]) is False
        assert generator._validate_program("not a list") is False
        assert generator._validate_program([{"missing_name": "test"}]) is False

    def test_create_program_variation(self, generator):
        """Test creating program variations."""
        base_program = [
            {"name": "rotate", "parameters": {"angle": 90}},
            {"name": "flip", "parameters": {"direction": "horizontal"}}
        ]

        # Simple style variation (should remove operation)
        variation = generator._create_program_variation(base_program, "simple")
        assert len(variation) <= len(base_program)

        # Complex style variation (might add operation)
        variation = generator._create_program_variation(base_program, "complex")
        assert len(variation) >= 1

        # Creative style variation (might reorder)
        variation = generator._create_program_variation(base_program, "creative")
        assert len(variation) == len(base_program)

    def test_analyze_task_pattern(self, generator, mock_task):
        """Test task pattern analysis."""
        description = generator._analyze_task_pattern(mock_task)

        assert isinstance(description, str)
        assert len(description) > 0

    def test_cache_functionality(self, generator, mock_task):
        """Test program caching."""
        # First call should generate and cache
        cache_key = f"{mock_task.task_id}_5_high"
        assert cache_key not in generator.generation_cache

        # Mock some programs
        programs = [
            [{"name": "rotate", "parameters": {"angle": 90}}],
            [{"name": "flip", "parameters": {"direction": "horizontal"}}]
        ]
        generator.generation_cache[cache_key] = programs

        # Second call should return cached
        cached = generator.generation_cache.get(cache_key, [])
        assert len(cached) == 2


class TestHybridPopulationInitializer:
    """Test hybrid population initializer."""

    @pytest.fixture
    def mock_llm_generator(self):
        """Create mock LLM generator."""
        generator = MagicMock()
        generator.generate_programs = AsyncMock()
        return generator

    @pytest.fixture
    def initializer(self, mock_llm_generator):
        """Create initializer with mocks."""
        return HybridPopulationInitializer(
            llm_generator=mock_llm_generator
        )

    @pytest.fixture
    def init_config(self):
        """Create initialization config."""
        return {
            "llm_seed_ratio": 0.2,
            "template_ratio": 0.5,
            "use_seed_programs": True
        }

    @pytest.mark.asyncio
    async def test_initialize_population(self, initializer, mock_llm_generator, init_config):
        """Test population initialization."""
        # Mock LLM programs
        mock_llm_generator.generate_programs.return_value = [
            [{"name": "rotate", "parameters": {"angle": 90}}],
            [{"name": "flip", "parameters": {"direction": "horizontal"}}]
        ]

        # Mock task
        task = MagicMock()

        # Initialize population
        population = await initializer.initialize_population(
            task=task,
            population_size=10,
            config=init_config
        )

        assert len(population) == 10

        # Check initialization methods
        methods = [ind.metadata.get("initialization_method") for ind in population]
        assert "llm" in methods
        assert "template" in methods
        assert "random" in methods

    @pytest.mark.asyncio
    async def test_generate_llm_individuals(self, initializer, mock_llm_generator):
        """Test LLM individual generation."""
        # Mock LLM response
        mock_llm_generator.generate_programs.return_value = [
            [{"name": "rotate", "parameters": {"angle": 90}}],
            [{"name": "flip", "parameters": {"direction": "vertical"}}]
        ]

        task = MagicMock()
        individuals = await initializer._generate_llm_individuals(task, 2)

        assert len(individuals) == 2
        for ind in individuals:
            assert ind.metadata.get("llm_generated") is True
            assert ind.metadata.get("generation_style") == "llm_diverse"

    def test_create_seed_individuals(self, initializer):
        """Test seed individual creation."""
        with patch('src.adapters.strategies.hybrid_initialization.create_seed_programs') as mock_seeds:
            mock_seeds.return_value = [
                [{"name": "rotate", "parameters": {"angle": 180}}],
                [{"name": "scale", "parameters": {"factor": 2}}]
            ]

            individuals = initializer._create_seed_individuals(2)

            assert len(individuals) == 2
            for ind in individuals:
                assert ind.metadata.get("seed_program") is True

    def test_generate_template_individuals(self, initializer):
        """Test template individual generation."""
        individuals = initializer._generate_template_individuals(3)

        assert len(individuals) == 3
        for ind in individuals:
            assert ind.metadata.get("template_generated") is True

    def test_generate_random_individuals(self, initializer):
        """Test random individual generation."""
        individuals = initializer._generate_random_individuals(5)

        assert len(individuals) == 5
        for ind in individuals:
            assert ind.metadata.get("random_generated") is True

    def test_convert_to_operations(self, initializer):
        """Test converting program data to operations."""
        program_data = [
            {"name": "rotate", "parameters": {"angle": 90}},
            {"name": "flip", "parameters": {"direction": "horizontal"}}
        ]

        operations = initializer._convert_to_operations(program_data)

        assert operations is not None
        assert len(operations) == 2
        assert operations[0].get_name() == "rotate"
        assert operations[1].get_name() == "flip"

    def test_initialization_method_assignment(self, initializer):
        """Test correct assignment of initialization methods."""
        # Test boundary conditions
        assert initializer._get_init_method(0, 2, 2, 3) == "llm"
        assert initializer._get_init_method(1, 2, 2, 3) == "llm"
        assert initializer._get_init_method(2, 2, 2, 3) == "seed"
        assert initializer._get_init_method(3, 2, 2, 3) == "seed"
        assert initializer._get_init_method(4, 2, 2, 3) == "template"
        assert initializer._get_init_method(7, 2, 2, 3) == "random"

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, initializer):
        """Test initialization without LLM generator."""
        initializer.llm_generator = None

        population = await initializer.initialize_population(
            task=MagicMock(),
            population_size=10,
            config={"llm_seed_ratio": 0.2, "template_ratio": 0.5}
        )

        assert len(population) == 10
        # Should not have LLM individuals
        methods = [ind.metadata.get("initialization_method") for ind in population]
        assert "llm" not in methods
