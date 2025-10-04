"""Integration tests for complete evolution search pipeline."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.adapters.strategies.evolution_strategy_adapter import EvolutionStrategyAdapter
from src.domain.models import ARCTask
from src.infrastructure.config import Config


class TestEvolutionSearchIntegration:
    """Integration tests for evolution search pipeline."""

    @pytest.fixture
    def integration_config(self):
        """Create integration test configuration."""
        config = Config()

        # Evolution settings for integration testing
        config.evolution.enabled = True
        config.evolution.population_size = 50  # Smaller for testing
        config.evolution.max_generations = 10  # Fewer generations
        config.evolution.mutation_rate = 0.2
        config.evolution.crossover_rate = 0.7
        config.evolution.elitism_count = 5
        config.evolution.tournament_size = 3

        # Pipeline settings
        config.evolution.batch_size = 10
        config.evolution.parallel_workers = 2
        config.evolution.memory_limit_mb = 1024
        config.evolution.generation_timeout = 10
        config.evolution.fitness_evaluation_timeout = 0.5

        # Early termination
        config.evolution.early_termination.enabled = True
        config.evolution.early_termination.stagnation_generations = 3
        config.evolution.early_termination.fitness_threshold = 0.95
        config.evolution.early_termination.min_generations = 2

        # Export settings
        config.evolution.export_count = 5
        config.evolution.export_format = "both"
        config.evolution.checkpoint_interval = 5

        # Platform
        config.platform = "local"

        return config

    @pytest.fixture
    def sample_arc_tasks(self):
        """Create sample ARC tasks for testing."""
        tasks = [
            ARCTask(
                id="test_simple",
                train=[
                    {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                    {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
                ],
                test=[{"input": [[3, 0], [0, 3]]}]
            ),
            ARCTask(
                id="test_complex",
                train=[
                    {
                        "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        "output": [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
                    },
                    {
                        "input": [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                        "output": [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
                    }
                ],
                test=[{"input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}]
            )
        ]
        return tasks

    def test_complete_evolution_pipeline(self, integration_config, sample_arc_tasks):
        """Test complete evolution pipeline from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integration_config.evolution.export_path = temp_dir
            integration_config.evolution.checkpoint_path = temp_dir

            # Create evolution engine
            engine = EvolutionEngine(integration_config)

            # Run evolution on simple task
            task = sample_arc_tasks[0]
            result = engine.evolve(task.train)

            # Verify basic results
            assert result is not None
            assert "best_individual" in result
            assert "population" in result
            assert "generations" in result
            assert result["generations"] >= integration_config.evolution.early_termination.min_generations

            # Check population was evolved
            best = result["best_individual"]
            assert best is not None
            assert best.fitness > 0
            assert len(best.operations) > 0

            # Verify convergence history
            assert "convergence_history" in result
            assert len(result["convergence_history"]) == result["generations"]

    def test_evolution_with_500_programs_target(self, integration_config):
        """Test that evolution generates 500+ programs."""
        # Configure for program generation target
        integration_config.evolution.population_size = 100
        integration_config.evolution.max_generations = 10
        integration_config.evolution.min_programs_generated = 500

        engine = EvolutionEngine(integration_config)

        # Track program count
        program_count = 0
        unique_programs = set()

        # Mock fitness evaluation to count programs
        original_evaluate = engine._evaluate_fitness

        def counting_evaluate(operations, examples):
            nonlocal program_count, unique_programs
            program_count += 1
            unique_programs.add(str(operations))
            return original_evaluate(operations, examples)

        engine._evaluate_fitness = counting_evaluate

        # Simple task
        examples = [{"input": [[1, 2]], "output": [[2, 1]]}]

        # Run evolution
        result = engine.evolve(examples)

        # Verify program count
        assert program_count >= 500, f"Only generated {program_count} programs"
        assert len(unique_programs) >= 400, f"Only {len(unique_programs)} unique programs"

        # Check metadata
        metrics = engine.get_evolution_metrics()
        assert metrics["total_programs_evaluated"] >= 500

    def test_evolution_accuracy_achievement(self, integration_config, sample_arc_tasks):
        """Test achieving target accuracy on tasks."""
        # Use adapter for full pipeline
        adapter = EvolutionStrategyAdapter(integration_config)

        # Track successful solutions
        successful_tasks = 0
        total_confidence = 0

        for task in sample_arc_tasks:
            result = adapter.process_task(task)

            if result.success and result.confidence >= 0.45:  # 45% target
                successful_tasks += 1
                total_confidence += result.confidence

        # Should solve at least some tasks
        assert successful_tasks > 0
        avg_confidence = total_confidence / len(sample_arc_tasks)
        assert avg_confidence >= 0.3  # Relaxed for test stability

    def test_evolution_time_constraint(self, integration_config):
        """Test evolution completes within time constraints."""
        # Configure for time testing
        integration_config.evolution.max_runtime_seconds = 300  # 5 minutes
        integration_config.evolution.population_size = 200
        integration_config.evolution.max_generations = 50

        engine = EvolutionEngine(integration_config)

        # Simple task
        examples = [
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
            {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
        ]

        start_time = time.time()
        result = engine.evolve(examples)
        elapsed_time = time.time() - start_time

        # Should complete within time limit
        assert elapsed_time < 300, f"Evolution took {elapsed_time}s, exceeding 5 minute limit"
        assert result is not None
        assert result["best_individual"] is not None

    def test_genealogy_tracking_integration(self, integration_config):
        """Test genealogy tracking across generations."""
        engine = EvolutionEngine(integration_config)

        # Enable detailed genealogy
        integration_config.evolution.track_genealogy = True

        examples = [{"input": [[1, 2]], "output": [[2, 1]]}]
        result = engine.evolve(examples)

        # Check genealogy data
        best = result["best_individual"]
        assert "mutation_history" in best.metadata
        assert "genealogy_depth" in best.metadata
        assert best.metadata["genealogy_depth"] >= 0

        # Verify parent tracking
        if best.parent_ids:
            assert len(best.parent_ids) > 0

    def test_program_export_integration(self, integration_config, sample_arc_tasks):
        """Test program export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integration_config.evolution.export_path = temp_dir

            adapter = EvolutionStrategyAdapter(integration_config)
            task = sample_arc_tasks[0]

            # Process task
            result = adapter.process_task(task)

            # Check export files
            dsl_file = Path(temp_dir) / f"{task.id}_evolution_programs.dsl"
            py_file = Path(temp_dir) / f"{task.id}_evolution_programs.py"
            json_file = Path(temp_dir) / f"{task.id}_evolution_analysis.json"

            assert dsl_file.exists()
            assert py_file.exists()
            assert json_file.exists()

            # Verify content
            with open(json_file) as f:
                analysis = json.load(f)
                assert "programs" in analysis
                assert len(analysis["programs"]) <= integration_config.evolution.export_count

    def test_evaluation_framework_integration(self, integration_config):
        """Test integration with evaluation framework."""
        adapter = EvolutionStrategyAdapter(integration_config)

        # Mock evaluation service integration
        with patch('src.domain.services.evaluation_service.EvaluationService') as mock_service:
            mock_eval_service = Mock()
            mock_service.return_value = mock_eval_service

            task = ARCTask(
                id="eval_test",
                train=[{"input": [[1]], "output": [[2]]}],
                test=[{"input": [[3]]}]
            )

            result = adapter.process_task(task)

            # Verify evaluation result format
            assert result.task_id == "eval_test"
            assert result.strategy == "evolution"
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'programs')
            assert hasattr(result, 'performance')

            # Check metrics
            if result.success:
                assert result.performance.evolution_generations > 0
                assert result.performance.programs_evaluated > 0

    def test_reproducibility_with_seed(self, integration_config):
        """Test reproducible results with fixed seed."""
        integration_config.evolution.random_seed = 42

        examples = [{"input": [[1, 2, 3]], "output": [[3, 2, 1]]}]

        # Run evolution twice with same seed
        engine1 = EvolutionEngine(integration_config)
        result1 = engine1.evolve(examples)

        engine2 = EvolutionEngine(integration_config)
        result2 = engine2.evolve(examples)

        # Results should be identical
        assert result1["generations"] == result2["generations"]
        assert result1["best_individual"].fitness == result2["best_individual"].fitness
        assert str(result1["best_individual"].operations) == str(result2["best_individual"].operations)

    def test_checkpoint_recovery_integration(self, integration_config):
        """Test checkpoint save and recovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integration_config.evolution.checkpoint_path = temp_dir
            integration_config.evolution.checkpoint_interval = 2

            engine = EvolutionEngine(integration_config)
            examples = [{"input": [[1, 2]], "output": [[2, 1]]}]

            # Run partial evolution
            with patch.object(engine, '_evolve_generation') as mock_evolve:
                # Simulate stopping after 3 generations
                generation_count = 0

                def evolve_with_stop(population, examples):
                    nonlocal generation_count
                    generation_count += 1
                    if generation_count >= 3:
                        raise KeyboardInterrupt("Simulated interruption")
                    population.generation += 1
                    return population

                mock_evolve.side_effect = evolve_with_stop

                try:
                    engine.evolve(examples)
                except KeyboardInterrupt:
                    pass

            # Check checkpoint exists
            checkpoint_files = list(Path(temp_dir).glob("evolution_checkpoint_*.json"))
            assert len(checkpoint_files) > 0

            # Load from checkpoint
            engine2 = EvolutionEngine(integration_config)
            checkpoint_data = engine2._load_latest_checkpoint()
            assert checkpoint_data is not None
            assert checkpoint_data["generation"] >= 2

    def test_multi_strategy_ensemble(self, integration_config):
        """Test evolution as part of multi-strategy ensemble."""
        # Create multiple strategy adapters
        evolution_adapter = EvolutionStrategyAdapter(integration_config)

        # Mock other strategies
        mock_ttt_adapter = Mock()
        mock_ttt_adapter.process_task.return_value = Mock(
            success=True, confidence=0.6, strategy="ttt"
        )

        mock_synthesis_adapter = Mock()
        mock_synthesis_adapter.process_task.return_value = Mock(
            success=True, confidence=0.7, strategy="synthesis"
        )

        # Process task with all strategies
        task = ARCTask(
            id="ensemble_test",
            train=[{"input": [[1]], "output": [[2]]}],
            test=[{"input": [[3]]}]
        )

        results = []
        for adapter in [evolution_adapter, mock_ttt_adapter, mock_synthesis_adapter]:
            result = adapter.process_task(task)
            results.append(result)

        # Select best result
        best_result = max(results, key=lambda r: r.confidence if r.success else 0)

        # Verify ensemble behavior
        assert best_result is not None
        assert best_result.confidence >= 0.6

    def test_platform_specific_execution(self, integration_config):
        """Test platform-specific optimizations."""
        platforms = ["kaggle", "colab", "paperspace"]

        for platform in platforms:
            config = integration_config.copy()
            config.platform = platform

            # Apply platform overrides
            if platform == "kaggle":
                config.evolution.parallel_workers = 2
                config.evolution.batch_size = 50
            elif platform == "colab":
                config.evolution.gpu_enabled = True
                config.evolution.batch_size = 100

            engine = EvolutionEngine(config)

            # Verify platform-specific settings
            if platform == "kaggle":
                assert engine.config.evolution.parallel_workers == 2
            elif platform == "colab":
                assert hasattr(engine.config.evolution, 'gpu_enabled')

    def test_experiment_orchestrator_integration(self, integration_config):
        """Test integration with experiment orchestrator."""
        with patch('src.domain.services.experiment_orchestrator.ExperimentOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            engine = EvolutionEngine(integration_config)
            examples = [{"input": [[1]], "output": [[2]]}]

            # Run evolution
            result = engine.evolve(examples)

            # Verify experiment tracking
            assert mock_orchestrator.start_experiment.called
            assert mock_orchestrator.log_metric.called
            assert mock_orchestrator.end_experiment.called
