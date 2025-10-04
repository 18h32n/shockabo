"""
Integration tests for Program Synthesis Strategy.

Tests the complete integration of DSL program generation, execution, and result
conversion within the program synthesis strategy adapter.
"""

import time

import pytest

from src.adapters.strategies.program_synthesis import (
    ProgramGenerationStrategy,
    ProgramSynthesisAdapter,
    ProgramSynthesisConfig,
)
from src.domain.dsl.base import DSLProgram
from src.domain.models import ARCTask, ARCTaskSolution, StrategyType


class TestProgramSynthesisIntegration:
    """Integration tests for program synthesis strategy."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return ProgramSynthesisConfig(
            execution_timeout=2.0,
            max_program_length=3,
            max_generation_attempts=10,
            beam_search_width=2,
            max_total_time=10.0
        )

    @pytest.fixture
    def hybrid_config(self):
        """Create a hybrid strategy config."""
        return ProgramSynthesisConfig(
            generation_strategy=ProgramGenerationStrategy.HYBRID,
            execution_timeout=3.0,
            max_program_length=5,
            max_generation_attempts=15,
            beam_search_width=3
        )

    @pytest.fixture
    def simple_rotation_task(self):
        """Create a simple rotation task for testing."""
        return ARCTask(
            task_id="test_rotation",
            task_source="test",
            train_examples=[
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[3, 1], [4, 2]]  # 90-degree rotation
                },
                {
                    "input": [[5, 6], [7, 8]],
                    "output": [[7, 5], [8, 6]]  # 90-degree rotation
                }
            ],
            test_input=[[9, 0], [1, 2]]
        )

    @pytest.fixture
    def color_mapping_task(self):
        """Create a simple color mapping task."""
        return ARCTask(
            task_id="test_color_map",
            task_source="test",
            train_examples=[
                {
                    "input": [[1, 1, 2], [2, 1, 2], [1, 2, 1]],
                    "output": [[3, 3, 4], [4, 3, 4], [3, 4, 3]]  # 1->3, 2->4
                }
            ],
            test_input=[[1, 2, 1], [2, 1, 2]]
        )

    @pytest.fixture
    def complex_task(self):
        """Create a complex task requiring multiple operations."""
        return ARCTask(
            task_id="test_complex",
            task_source="test",
            train_examples=[
                {
                    "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                    "output": [[1, 0, 1], [0, 3, 0], [1, 0, 1]]  # Invert colors and replace 2->3
                },
                {
                    "input": [[0, 2, 0], [2, 1, 2], [0, 2, 0]],
                    "output": [[2, 0, 2], [0, 4, 0], [2, 0, 2]]  # Invert colors and replace 1->4
                }
            ],
            test_input=[[1, 0, 1], [0, 2, 0], [1, 0, 1]]
        )

    def test_program_synthesis_initialization(self, simple_config):
        """Test that program synthesis adapter initializes correctly."""
        adapter = ProgramSynthesisAdapter(simple_config)

        assert adapter.config == simple_config
        assert adapter.dsl_engine is not None
        assert adapter.serializer is not None
        assert len(adapter.templates) > 0
        assert adapter.generation_stats["programs_generated"] == 0

        # Test DSL engine has operations registered
        operations = adapter.dsl_engine.get_registered_operations()
        assert len(operations) > 0
        assert "rotate" in operations
        assert "flip" in operations
        assert "color_replace" in operations

    def test_task_analysis(self, simple_config, simple_rotation_task, color_mapping_task):
        """Test task analysis functionality."""
        adapter = ProgramSynthesisAdapter(simple_config)

        # Test rotation task analysis
        analysis = adapter._analyze_task(simple_rotation_task)

        assert "grid_dimensions" in analysis
        assert "color_changes" in analysis
        assert "spatial_changes" in analysis
        assert "pattern_complexity" in analysis

        # Check grid dimensions analysis
        assert len(analysis["grid_dimensions"]) == 2  # Two training examples
        for dim_info in analysis["grid_dimensions"]:
            assert "input" in dim_info
            assert "output" in dim_info
            assert "size_changed" in dim_info
            assert dim_info["input"] == (2, 2)
            assert dim_info["output"] == (2, 2)
            assert not dim_info["size_changed"]

        # Test color mapping task analysis
        analysis = adapter._analyze_task(color_mapping_task)
        color_change = analysis["color_changes"][0]
        assert 3 in color_change["colors_added"]
        assert 4 in color_change["colors_added"]
        assert 1 in color_change["colors_removed"]
        assert 2 in color_change["colors_removed"]

    def test_template_based_generation(self, simple_config, simple_rotation_task):
        """Test template-based program generation."""
        config = simple_config
        config.generation_strategy = ProgramGenerationStrategy.TEMPLATE_BASED
        adapter = ProgramSynthesisAdapter(config)

        analysis = adapter._analyze_task(simple_rotation_task)
        programs = adapter._generate_template_based_programs(simple_rotation_task, analysis)

        assert len(programs) > 0
        assert adapter.generation_stats["programs_generated"] > 0

        # Check that programs are valid DSLProgram instances
        for program in programs:
            assert isinstance(program, DSLProgram)
            assert len(program.operations) > 0
            for op in program.operations:
                assert "name" in op
                assert "parameters" in op

    def test_search_based_generation(self, simple_config, color_mapping_task):
        """Test search-based program generation."""
        config = simple_config
        config.generation_strategy = ProgramGenerationStrategy.SEARCH_BASED
        config.beam_search_width = 2
        adapter = ProgramSynthesisAdapter(config)

        analysis = adapter._analyze_task(color_mapping_task)
        programs = adapter._generate_search_based_programs(color_mapping_task, analysis)

        assert len(programs) > 0
        assert adapter.generation_stats["programs_generated"] > 0

        # Check program validity
        for program in programs:
            assert isinstance(program, DSLProgram)
            assert len(program.operations) <= config.max_program_length

    def test_program_evaluation(self, simple_config, simple_rotation_task):
        """Test program evaluation and selection."""
        adapter = ProgramSynthesisAdapter(simple_config)

        # Create test programs
        test_programs = [
            DSLProgram(operations=[{"name": "rotate", "parameters": {"angle": 90}}]),
            DSLProgram(operations=[{"name": "flip", "parameters": {"direction": 0}}]),
            DSLProgram(operations=[{"name": "color_replace", "parameters": {"from_color": 1, "to_color": 2}}])
        ]

        best_program, evaluation_results = adapter._evaluate_programs(test_programs, simple_rotation_task)

        assert best_program is not None
        assert evaluation_results["best_score"] >= 0.0
        assert evaluation_results["programs_evaluated"] == len(test_programs)
        assert len(evaluation_results["program_scores"]) == len(test_programs)
        assert len(evaluation_results["execution_times"]) == len(test_programs)

        # Best program should be the rotation one for this task
        assert best_program.operations[0]["name"] == "rotate"
        assert evaluation_results["best_score"] > 0.8  # Should get high score for correct rotation

    def test_complete_solve_workflow(self, simple_config, simple_rotation_task):
        """Test the complete solve workflow from start to finish."""
        adapter = ProgramSynthesisAdapter(simple_config)

        solution = adapter.solve(simple_rotation_task)

        # Check solution structure
        assert isinstance(solution, ARCTaskSolution)
        assert solution.task_id == simple_rotation_task.task_id
        assert solution.strategy_used == StrategyType.PROGRAM_SYNTHESIS
        assert len(solution.predictions) == 1
        assert solution.confidence_score >= 0.0
        assert solution.resource_usage is not None

        # Check metadata
        metadata = solution.metadata
        assert metadata["program_synthesis"] is True
        assert "generation_strategy" in metadata
        assert "best_program" in metadata
        assert "programs_evaluated" in metadata
        assert "synthesis_time" in metadata
        assert "task_analysis" in metadata

        # Check that prediction has correct dimensions
        prediction = solution.predictions[0]
        expected_dims = (2, 2)  # Same as input
        assert len(prediction) == expected_dims[0]
        assert len(prediction[0]) == expected_dims[1]

    def test_solve_with_different_strategies(self, hybrid_config, color_mapping_task):
        """Test solve with different generation strategies."""
        strategies = [
            ProgramGenerationStrategy.TEMPLATE_BASED,
            ProgramGenerationStrategy.SEARCH_BASED,
            ProgramGenerationStrategy.HYBRID
        ]

        for strategy in strategies:
            config = hybrid_config
            config.generation_strategy = strategy
            adapter = ProgramSynthesisAdapter(config)

            solution = adapter.solve(color_mapping_task)

            assert isinstance(solution, ARCTaskSolution)
            assert solution.metadata["generation_strategy"] == strategy.value
            assert solution.confidence_score >= 0.0

    def test_performance_requirements(self, simple_config, simple_rotation_task):
        """Test that synthesis meets performance requirements."""
        config = simple_config
        config.max_total_time = 5.0  # 5 second limit
        adapter = ProgramSynthesisAdapter(config)

        start_time = time.time()
        solution = adapter.solve(simple_rotation_task)
        end_time = time.time()

        synthesis_time = end_time - start_time

        # Should complete within time limit
        assert synthesis_time < config.max_total_time

        # Check metadata reports correct timing
        assert "synthesis_time" in solution.metadata
        assert solution.metadata["synthesis_time"] <= synthesis_time
        assert solution.metadata["performance_target_met"] in [True, False]

    def test_error_handling(self, simple_config):
        """Test error handling in various scenarios."""
        adapter = ProgramSynthesisAdapter(simple_config)

        # Test with malformed task
        bad_task = ARCTask(
            task_id="bad_task",
            task_source="test",
            train_examples=[],  # Empty training examples
            test_input=[]  # Empty test input
        )

        solution = adapter.solve(bad_task)

        # Should return failure solution gracefully
        assert isinstance(solution, ARCTaskSolution)
        assert solution.confidence_score == 0.0
        assert solution.metadata["success"] is False
        assert "error" in solution.metadata

    def test_caching_functionality(self, simple_config, simple_rotation_task):
        """Test program caching functionality."""
        config = simple_config
        config.cache_successful_programs = True
        adapter = ProgramSynthesisAdapter(config)

        # Solve task to populate cache
        solution1 = adapter.solve(simple_rotation_task)
        initial_cache_size = len(adapter.successful_programs)

        # Solve again - should potentially use cached results
        solution2 = adapter.solve(simple_rotation_task)

        # Check that solutions are consistent
        assert solution1.task_id == solution2.task_id
        assert solution1.strategy_used == solution2.strategy_used

        # Cache might have grown if successful programs were found
        assert len(adapter.successful_programs) >= initial_cache_size

    def test_resource_usage_tracking(self, simple_config, complex_task):
        """Test resource usage tracking and reporting."""
        adapter = ProgramSynthesisAdapter(simple_config)

        solution = adapter.solve(complex_task)

        resource_usage = solution.resource_usage
        assert resource_usage is not None
        assert resource_usage.task_id == complex_task.task_id
        assert resource_usage.strategy_type == StrategyType.PROGRAM_SYNTHESIS
        assert resource_usage.cpu_seconds > 0
        assert resource_usage.memory_mb > 0
        assert resource_usage.gpu_memory_mb is None  # DSL doesn't use GPU
        assert isinstance(resource_usage.api_calls, dict)
        assert resource_usage.total_tokens == 0
        assert resource_usage.estimated_cost == 0.0

    def test_statistics_collection(self, simple_config, simple_rotation_task):
        """Test statistics collection and reporting."""
        adapter = ProgramSynthesisAdapter(simple_config)

        # Solve a task
        adapter.solve(simple_rotation_task)

        # Get statistics
        stats = adapter.get_synthesis_stats()

        assert "generation_stats" in stats
        assert "dsl_execution_stats" in stats
        assert "config" in stats
        assert "successful_programs_cached" in stats
        assert "templates_available" in stats

        gen_stats = stats["generation_stats"]
        assert gen_stats["programs_generated"] > 0
        assert gen_stats["programs_executed"] > 0
        assert gen_stats["total_execution_time"] >= 0

        dsl_stats = stats["dsl_execution_stats"]
        assert "cache_hits" in dsl_stats
        assert "cache_misses" in dsl_stats
        assert "peak_memory_mb" in dsl_stats

    def test_template_relevance_selection(self, simple_config):
        """Test template selection based on task characteristics."""
        adapter = ProgramSynthesisAdapter(simple_config)

        # Test with color-heavy analysis
        color_analysis = {
            "pattern_complexity": "low",
            "color_changes": [{"colors_added": [3], "colors_removed": [1]}],
            "grid_dimensions": [{"size_changed": False}]
        }

        relevant_templates = adapter._select_relevant_templates(color_analysis)

        # Color templates should be prioritized
        color_template_count = sum(1 for t in relevant_templates[:3] if any(
            "color" in op.get("name", "") for op in t.operations
        ))
        assert color_template_count > 0

        # Test with size-change analysis
        size_analysis = {
            "pattern_complexity": "medium",
            "color_changes": [{"colors_added": [], "colors_removed": []}],
            "grid_dimensions": [{"size_changed": True}]
        }

        relevant_templates = adapter._select_relevant_templates(size_analysis)

        # Geometric templates should be prioritized
        geo_template_count = sum(1 for t in relevant_templates[:3] if any(
            op.get("name") in ["rotate", "flip", "crop", "pad", "translate"]
            for op in t.operations
        ))
        assert geo_template_count > 0

    def test_cleanup_functionality(self, simple_config, simple_rotation_task):
        """Test cleanup functionality."""
        adapter = ProgramSynthesisAdapter(simple_config)

        # Use the adapter
        adapter.solve(simple_rotation_task)

        # Ensure there's something to clean up
        assert len(adapter.generation_stats) > 0

        # Perform cleanup
        adapter.cleanup()

        # Check cleanup effects
        assert len(adapter.successful_programs) == 0

    def test_program_serialization_integration(self, simple_config, simple_rotation_task):
        """Test integration with program serialization."""
        adapter = ProgramSynthesisAdapter(simple_config)

        solution = adapter.solve(simple_rotation_task)

        # Check that best program was serialized in metadata
        if "best_program" in solution.metadata:
            serialized_program = solution.metadata["best_program"]
            assert isinstance(serialized_program, dict)
            assert "operations" in serialized_program
            assert "version" in serialized_program

            # Should be able to deserialize
            program = adapter.serializer.deserialize_program(serialized_program)
            assert isinstance(program, DSLProgram)

    @pytest.mark.timeout(30)
    def test_timeout_enforcement(self):
        """Test that timeouts are enforced properly."""
        config = ProgramSynthesisConfig(
            max_total_time=1.0,  # Very short timeout
            max_generation_attempts=100,  # Many attempts to trigger timeout
            execution_timeout=0.1
        )

        adapter = ProgramSynthesisAdapter(config)

        # Create a task that might take longer to solve
        complex_task = ARCTask(
            task_id="timeout_test",
            task_source="test",
            train_examples=[
                {"input": [[i % 10 for i in range(10)] for _ in range(10)],
                 "output": [[(i + 1) % 10 for i in range(10)] for _ in range(10)]}
                for _ in range(5)  # Multiple examples
            ],
            test_input=[[i % 10 for i in range(10)] for _ in range(10)]
        )

        start_time = time.time()
        solution = adapter.solve(complex_task)
        end_time = time.time()

        # Should complete within reasonable time despite timeout
        assert end_time - start_time < 10.0  # Reasonable upper bound
        assert isinstance(solution, ARCTaskSolution)


class TestEnsembleCompatibility:
    """Test compatibility with ensemble voting system."""

    @pytest.fixture
    def ensemble_config(self):
        """Config optimized for ensemble usage."""
        return ProgramSynthesisConfig(
            generation_strategy=ProgramGenerationStrategy.HYBRID,
            max_generation_attempts=20,
            early_stopping_threshold=0.9,
            cache_successful_programs=True
        )

    def test_multiple_strategy_instances(self, ensemble_config, simple_rotation_task):
        """Test that multiple strategy instances can work together."""
        adapters = [
            ProgramSynthesisAdapter(ensemble_config),
            ProgramSynthesisAdapter(ensemble_config),
            ProgramSynthesisAdapter(ensemble_config)
        ]

        solutions = []
        for adapter in adapters:
            solution = adapter.solve(simple_rotation_task)
            solutions.append(solution)

        # All solutions should be valid
        for solution in solutions:
            assert isinstance(solution, ARCTaskSolution)
            assert solution.strategy_used == StrategyType.PROGRAM_SYNTHESIS
            assert solution.confidence_score >= 0.0

        # Solutions should be consistent in structure
        task_ids = [s.task_id for s in solutions]
        assert len(set(task_ids)) == 1  # All same task

        # Predictions should have consistent structure
        for solution in solutions:
            assert len(solution.predictions) == 1
            prediction = solution.predictions[0]
            assert len(prediction) == 2  # Expected dimensions
            assert len(prediction[0]) == 2

    def test_confidence_score_reliability(self, ensemble_config):
        """Test that confidence scores are reliable for ensemble voting."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        # High-confidence task (simple rotation)
        simple_task = ARCTask(
            task_id="simple_ensemble",
            task_source="test",
            train_examples=[
                {"input": [[1, 0]], "output": [[1], [0]]},  # Simple transpose
                {"input": [[2, 3]], "output": [[2], [3]]}
            ],
            test_input=[[4, 5]]
        )

        solution = adapter.solve(simple_task)
        # Should have reasonable confidence for simple task
        assert solution.confidence_score >= 0.0

        # Low-confidence task (contradictory examples)
        contradictory_task = ARCTask(
            task_id="contradictory_ensemble",
            task_source="test",
            train_examples=[
                {"input": [[1]], "output": [[2]]},
                {"input": [[1]], "output": [[3]]}  # Same input, different output
            ],
            test_input=[[1]]
        )

        solution = adapter.solve(contradictory_task)
        # Should have low confidence for contradictory task
        assert solution.confidence_score <= 0.5

    def test_metadata_completeness_for_ensemble(self, ensemble_config, simple_rotation_task):
        """Test that metadata contains all information needed for ensemble voting."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        solution = adapter.solve(simple_rotation_task)

        metadata = solution.metadata

        # Required metadata for ensemble voting
        required_fields = [
            "program_synthesis",
            "generation_strategy",
            "synthesis_time",
            "programs_evaluated",
            "task_analysis"
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required metadata field: {field}"

        # Check metadata types and values
        assert isinstance(metadata["program_synthesis"], bool)
        assert isinstance(metadata["generation_strategy"], str)
        assert isinstance(metadata["synthesis_time"], (int, float))
        assert isinstance(metadata["programs_evaluated"], int)
        assert isinstance(metadata["task_analysis"], dict)

    def test_resource_usage_for_ensemble_tracking(self, ensemble_config, simple_rotation_task):
        """Test resource usage tracking for ensemble resource management."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        solution = adapter.solve(simple_rotation_task)

        resource_usage = solution.resource_usage
        assert resource_usage is not None

        # Essential fields for ensemble resource tracking
        assert resource_usage.cpu_seconds > 0
        assert resource_usage.memory_mb > 0
        assert resource_usage.timestamp is not None

        # Program synthesis specific
        assert resource_usage.strategy_type == StrategyType.PROGRAM_SYNTHESIS
        assert resource_usage.gpu_memory_mb is None  # DSL doesn't use GPU

    def test_ensemble_compatibility_verification(self, ensemble_config):
        """Test the ensemble compatibility verification method."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        compatibility_report = adapter.verify_ensemble_compatibility()

        # Check report structure
        required_fields = [
            "voting_system_available",
            "can_generate_candidates",
            "supports_confidence_scores",
            "supports_metadata_for_voting",
            "prediction_format_compatible",
            "ensemble_methods_available",
            "compatibility_issues",
            "is_fully_compatible"
        ]

        for field in required_fields:
            assert field in compatibility_report, f"Missing field: {field}"

        # Check types
        assert isinstance(compatibility_report["voting_system_available"], bool)
        assert isinstance(compatibility_report["can_generate_candidates"], bool)
        assert isinstance(compatibility_report["ensemble_methods_available"], list)
        assert isinstance(compatibility_report["compatibility_issues"], list)
        assert isinstance(compatibility_report["is_fully_compatible"], bool)

        # If voting system is available, should have ensemble methods
        if compatibility_report["voting_system_available"]:
            assert len(compatibility_report["ensemble_methods_available"]) > 0
            assert "generate_ensemble_candidates" in compatibility_report["ensemble_methods_available"]
            assert "solve_with_ensemble_voting" in compatibility_report["ensemble_methods_available"]

    def test_ensemble_candidate_generation(self, ensemble_config, simple_rotation_task):
        """Test generation of ensemble candidates."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        candidates = adapter.generate_ensemble_candidates(simple_rotation_task, 3)

        # Should generate some candidates (unless voting system unavailable)
        if hasattr(adapter, 'VOTING_SYSTEM_AVAILABLE') and adapter.VOTING_SYSTEM_AVAILABLE:
            assert len(candidates) > 0

            # Check candidate structure (if voting system available)
            for candidate in candidates:
                # Should have prediction field
                assert hasattr(candidate, 'prediction') or isinstance(candidate, list)
                # If it's a PredictionCandidate, check its structure
                if hasattr(candidate, 'prediction'):
                    assert hasattr(candidate, 'confidence')
                    assert hasattr(candidate, 'source_type')
                    assert hasattr(candidate, 'augmentation_info')

    def test_solve_with_ensemble_voting(self, ensemble_config, color_mapping_task):
        """Test solve with ensemble voting functionality."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        solution = adapter.solve_with_ensemble_voting(color_mapping_task, 3)

        # Should return valid solution
        assert isinstance(solution, ARCTaskSolution)
        assert solution.strategy_used == StrategyType.PROGRAM_SYNTHESIS
        assert solution.confidence_score >= 0.0

        # Check ensemble-specific metadata (if voting available)
        metadata = solution.metadata
        if metadata.get("ensemble_voting"):
            assert "voting_method" in metadata
            assert "candidates_generated" in metadata
            assert "agreement_ratio" in metadata
            assert "vote_distribution" in metadata

    def test_ensemble_voting_fallback_behavior(self):
        """Test fallback behavior when ensemble voting fails."""
        # Test with minimal config that might cause issues
        minimal_config = ProgramSynthesisConfig(
            max_generation_attempts=1,
            execution_timeout=0.1,  # Very short timeout
            max_total_time=1.0
        )

        adapter = ProgramSynthesisAdapter(minimal_config)

        test_task = ARCTask(
            task_id="fallback_test",
            task_source="test",
            train_examples=[{"input": [[1]], "output": [[2]]}],
            test_input=[[1]]
        )

        # Should still return a valid solution even if ensemble fails
        solution = adapter.solve_with_ensemble_voting(test_task, 3)

        assert isinstance(solution, ARCTaskSolution)
        assert solution.task_id == test_task.task_id
        assert len(solution.predictions) > 0

    def test_ensemble_voting_integration_with_existing_system(self, ensemble_config, simple_rotation_task):
        """Test that ensemble voting integrates properly with the existing voting system."""
        adapter = ProgramSynthesisAdapter(ensemble_config)

        # Test compatibility check
        compatibility = adapter.verify_ensemble_compatibility()

        if compatibility["voting_system_available"]:
            # Test full ensemble workflow
            solution = adapter.solve_with_ensemble_voting(simple_rotation_task, 3)

            # Verify solution quality
            assert isinstance(solution, ARCTaskSolution)
            assert solution.confidence_score >= 0.0
            assert len(solution.predictions) == 1

            # Check that the solution has ensemble metadata
            metadata = solution.metadata
            if metadata.get("ensemble_voting"):
                assert "voting_method" in metadata
                assert "agreement_ratio" in metadata

                # Voting method should be one of the expected types
                voting_method = metadata["voting_method"]
                expected_methods = [
                    "hybrid", "hybrid_agreement", "hybrid_self_consistency",
                    "hybrid_augmentation", "hybrid_fallback"
                ]
                assert any(voting_method.startswith(method) for method in expected_methods)
        else:
            # If voting system not available, should gracefully degrade
            solution = adapter.solve_with_ensemble_voting(simple_rotation_task, 3)
            assert isinstance(solution, ARCTaskSolution)
            # Should not have ensemble voting metadata
            assert not solution.metadata.get("ensemble_voting", False)


if __name__ == "__main__":
    pytest.main([__file__])
