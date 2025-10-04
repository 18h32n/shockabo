"""Integration tests for ensemble voting system with mock strategies."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.adapters.repositories.ensemble_interface import (
    EnsembleInterface,
    EnsembleResult,
)
from src.adapters.repositories.program_cache import ProgramCache, ProgramCacheEntry
from src.adapters.repositories.program_cache_config import ProgramCacheConfig
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache_config(cache_dir):
    """Create cache configuration for integration tests."""
    config = ProgramCacheConfig.from_yaml(
        str(Path(__file__).parent.parent.parent / "configs" / "strategies" / "program_cache.yaml")
    )
    # Update cache directory
    config.storage.cache_dir = cache_dir
    return config


@pytest.fixture
def program_cache(cache_config):
    """Create program cache instance."""
    cache = ProgramCache(config=cache_config)
    yield cache
    cache.close()


@pytest.fixture
def ensemble_interface(program_cache):
    """Create ensemble interface with real cache."""
    config = {
        'voting_method': 'weighted_majority',
        'min_programs_for_vote': 2,
        'confidence_threshold': 0.7
    }
    return EnsembleInterface(program_cache=program_cache, config=config)


@pytest.fixture
def mock_strategy():
    """Create a mock evolution strategy."""
    strategy = Mock()
    strategy.name = "mock_evolution_strategy"
    return strategy


@pytest.fixture
def sample_programs():
    """Create sample programs with different characteristics."""
    programs = []

    # High-performing program
    prog1 = DSLProgram(operations=[
        {"type": "rotate", "params": {"angle": 90}},
        {"type": "flip", "params": {"axis": "horizontal"}}
    ])

    # Medium-performing program
    prog2 = DSLProgram(operations=[
        {"type": "fill", "params": {"color": 1}},
        {"type": "mask", "params": {"pattern": [[1, 0], [0, 1]]}}
    ])

    # Low-performing program
    prog3 = DSLProgram(operations=[
        {"type": "crop", "params": {"x": 0, "y": 0, "width": 5, "height": 5}}
    ])

    # Complex program
    prog4 = DSLProgram(operations=[
        {"type": "rotate", "params": {"angle": 180}},
        {"type": "flip", "params": {"axis": "vertical"}},
        {"type": "fill", "params": {"color": 2}},
        {"type": "resize", "params": {"scale": 2.0}}
    ])

    return [prog1, prog2, prog3, prog4]


class TestEnsembleIntegration:
    """Integration tests for ensemble voting with mock strategies."""

    def test_ensemble_with_mock_strategy_execution(self, program_cache, ensemble_interface, sample_programs, mock_strategy):
        """Test ensemble integration with mock strategy execution."""
        # Save sample programs to cache
        saved_ids = []
        for i, program in enumerate(sample_programs):
            program_id = program_cache.save_program(
                program=program,
                task_id=f"ensemble_test_{i}",
                task_source="training",
                success=i < 3,  # First 3 are successful
                accuracy_score=0.9 - i * 0.1,  # Decreasing accuracy
                execution_time_ms=100 + i * 20,
                generation=i,
                parents=[f"parent_{i}"] if i > 0 else [],
                mutation_type="crossover" if i > 0 else None,
                fitness_score=0.85 - i * 0.05
            )
            saved_ids.append(program_id)

        # Create mock execution function
        def mock_execution(program, input_grid):
            """Mock execution that returns different outputs based on program."""
            if "rotate" in str(program.operations):
                # Rotation programs produce inverted output
                return 1 - input_grid
            elif "fill" in str(program.operations):
                # Fill programs produce filled output
                return np.ones_like(input_grid)
            else:
                # Others return input unchanged
                return input_grid.copy()

        # Get candidate programs from cache
        candidates = program_cache.get_ensemble_programs(
            task_id="ensemble_test",
            min_accuracy=0.5,
            limit=10
        )

        assert len(candidates) > 0

        # Run ensemble voting
        input_grid = np.array([[0, 1], [1, 0]])
        result = ensemble_interface.ensemble_vote(
            input_grid=input_grid,
            candidate_programs=candidates,
            execution_func=mock_execution
        )

        assert isinstance(result, EnsembleResult)
        assert result.final_output is not None
        assert result.voting_method == "weighted_majority"
        assert len(result.votes) == len(candidates)
        assert result.confidence > 0

        print("\nEnsemble result:")
        print(f"  Method: {result.voting_method}")
        print(f"  Votes: {len(result.votes)}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Consensus: {result.consensus_level:.3f}")

    def test_mock_strategy_program_generation(self, program_cache, ensemble_interface, mock_strategy):
        """Test integration with mock strategy generating new programs."""
        # Mock strategy generates programs
        def generate_mock_programs(n=5):
            """Generate mock programs from strategy."""
            programs = []
            for i in range(n):
                program = DSLProgram(operations=[
                    {"type": "rotate", "params": {"angle": 90 * (i % 4)}},
                    {"type": "flip", "params": {"axis": "horizontal" if i % 2 == 0 else "vertical"}}
                ])
                programs.append(program)
            return programs

        mock_strategy.generate_programs = Mock(return_value=generate_mock_programs())

        # Generate programs from strategy
        new_programs = mock_strategy.generate_programs()

        # Save to cache with mock performance metrics
        saved_entries = []
        for i, program in enumerate(new_programs):
            program_id = program_cache.save_program(
                program=program,
                task_id="mock_strategy_test",
                task_source="evolution",
                success=True,
                accuracy_score=0.7 + i * 0.05,
                execution_time_ms=80 + i * 10,
                generation=1,
                parents=[],
                mutation_type="random",
                fitness_score=0.75 + i * 0.03
            )

            entry = program_cache.get_program(program_id)
            if entry:
                saved_entries.append(entry)

        # Use ensemble to evaluate generated programs
        candidates = saved_entries

        # Mock evaluation function
        def mock_evaluate(program, input_grid):
            # Programs with higher index perform better
            prog_index = next((i for i, p in enumerate(new_programs) if p == program), 0)
            noise = np.random.rand(*input_grid.shape) * 0.1
            return input_grid + prog_index * 0.1 + noise

        # Run ensemble on mock-generated programs
        test_input = np.array([[1, 2], [3, 4]], dtype=float)
        # Create new interface with confidence_weighted method
        confidence_interface = EnsembleInterface(
            program_cache=program_cache,
            config={'voting_method': 'confidence_weighted'}
        )
        result = confidence_interface.ensemble_vote(
            input_grid=test_input,
            candidate_programs=candidates,
            execution_func=lambda p, g: mock_evaluate(p, g)
        )

        assert result.voting_method == "confidence_weighted"
        assert mock_strategy.generate_programs.called

        # Check that ensemble voting worked
        assert len(result.votes) > 0
        assert result.confidence > 0

    def test_multi_strategy_ensemble_integration(self, program_cache, ensemble_interface):
        """Test ensemble with programs from multiple mock strategies."""
        # Create multiple mock strategies
        strategies = {
            "genetic": Mock(name="genetic_strategy"),
            "random": Mock(name="random_strategy"),
            "hill_climbing": Mock(name="hill_climbing_strategy")
        }

        # Each strategy produces different types of programs
        strategy_programs = {
            "genetic": [
                DSLProgram(operations=[
                    {"type": "rotate", "params": {"angle": 90}},
                    {"type": "flip", "params": {"axis": "horizontal"}}
                ]),
                DSLProgram(operations=[
                    {"type": "rotate", "params": {"angle": 180}},
                    {"type": "flip", "params": {"axis": "vertical"}}
                ])
            ],
            "random": [
                DSLProgram(operations=[
                    {"type": "fill", "params": {"color": 1}}
                ]),
                DSLProgram(operations=[
                    {"type": "mask", "params": {"pattern": [[1, 0], [0, 1]]}}
                ])
            ],
            "hill_climbing": [
                DSLProgram(operations=[
                    {"type": "crop", "params": {"x": 0, "y": 0, "width": 5, "height": 5}},
                    {"type": "resize", "params": {"scale": 2.0}}
                ])
            ]
        }

        # Save programs from each strategy
        all_entries = []
        for strategy_name, programs in strategy_programs.items():
            for i, program in enumerate(programs):
                # Simulate different performance characteristics per strategy
                if strategy_name == "genetic":
                    accuracy = 0.85 + i * 0.05
                    success = True
                elif strategy_name == "random":
                    accuracy = 0.65 + i * 0.1
                    success = i == 0  # Only first is successful
                else:  # hill_climbing
                    accuracy = 0.75
                    success = True

                program_id = program_cache.save_program(
                    program=program,
                    task_id="multi_strategy_test",
                    task_source=strategy_name,
                    success=success,
                    accuracy_score=accuracy,
                    execution_time_ms=100 + i * 10,
                    generation=i + 1,
                    parents=[],
                    mutation_type=strategy_name,
                    fitness_score=accuracy * 0.9
                )

                entry = program_cache.get_program(program_id)
                if entry:
                    all_entries.append(entry)

        # Get ensemble candidates (should include best from each strategy)
        candidates = program_cache.get_ensemble_programs(
            task_id="multi_strategy_test",
            min_accuracy=0.6,
            limit=10
        )

        print(f"\nCandidates from {len(set(e.task_source for e in candidates))} different strategies")

        # Verify we have programs from multiple strategies
        strategy_sources = set(entry.task_source for entry in candidates)
        assert len(strategy_sources) >= 2, "Ensemble should include programs from multiple strategies"

        # Mock execution that favors genetic strategy programs
        def strategy_aware_execution(program, input_grid):
            # Need to find which strategy this program came from
            # For simplicity, use program structure to determine behavior
            if any("rotate" in str(op) for op in program.operations):
                return np.rot90(input_grid)  # Genetic programs rotate
            elif any("fill" in str(op) for op in program.operations):
                return np.ones_like(input_grid)  # Random programs fill
            else:
                return input_grid * 2  # Hill climbing doubles values

        # Run ensemble voting
        test_input = np.array([[1, 0], [0, 1]])
        # Create interface with consensus voting
        consensus_interface = EnsembleInterface(
            program_cache=program_cache,
            config={'voting_method': 'consensus'}
        )
        result = consensus_interface.ensemble_vote(
            input_grid=test_input,
            candidate_programs=candidates,
            execution_func=strategy_aware_execution
        )

        # Consensus might fail if programs produce different outputs
        assert result.voting_method in ["consensus", "consensus_failed"]

        # Analyze voting distribution by strategy
        strategy_votes = {}
        for vote in result.votes:
            program_entry = next(e for e in candidates if e.program_id == vote.program_id)
            strategy = program_entry.task_source
            if strategy not in strategy_votes:
                strategy_votes[strategy] = []
            strategy_votes[strategy].append(vote.confidence)

        print("\nVoting distribution by strategy:")
        for strategy, confidences in strategy_votes.items():
            print(f"  {strategy}: {len(confidences)} votes, avg confidence {np.mean(confidences):.3f}")

    def test_ensemble_feedback_to_strategy(self, program_cache, ensemble_interface, mock_strategy):
        """Test ensemble results feeding back to strategy."""
        # Initial programs
        initial_programs = [
            DSLProgram(operations=[
                {"type": "rotate", "params": {"angle": 90}}
            ]),
            DSLProgram(operations=[
                {"type": "flip", "params": {"axis": "horizontal"}}
            ])
        ]

        # Save initial programs
        entries = []
        for i, program in enumerate(initial_programs):
            program_id = program_cache.save_program(
                program=program,
                task_id="feedback_test",
                task_source="evolution",
                success=True,
                accuracy_score=0.7,
                execution_time_ms=100,
                generation=0
            )
            entry = program_cache.get_program(program_id)
            if entry:
                entries.append(entry)

        # Run ensemble
        input_grid = np.array([[1, 0], [0, 1]])
        # Create interface with simple_majority voting
        simple_interface = EnsembleInterface(
            program_cache=program_cache,
            config={'voting_method': 'simple_majority'}
        )
        result = simple_interface.ensemble_vote(
            input_grid=input_grid,
            candidate_programs=entries,
            execution_func=lambda p, g: np.rot90(g)  # Simple rotation
        )

        # Mock strategy uses ensemble results to generate improved programs
        def generate_improved_programs(ensemble_result):
            """Generate new programs based on ensemble feedback."""
            # Find winning program characteristics
            best_vote = max(ensemble_result.votes, key=lambda v: v.confidence)

            # Generate variations of successful program
            new_programs = []
            for i in range(3):
                # Add operations to the successful base
                new_ops = [
                    {"type": "rotate", "params": {"angle": 90}},
                    {"type": "flip", "params": {"axis": "horizontal"}},
                    {"type": "fill", "params": {"color": i}}
                ]
                new_programs.append(DSLProgram(operations=new_ops[:i+1]))

            return new_programs

        mock_strategy.evolve_from_ensemble = Mock(side_effect=generate_improved_programs)

        # Strategy generates new programs based on ensemble
        improved_programs = mock_strategy.evolve_from_ensemble(result)

        # Save improved programs
        for i, program in enumerate(improved_programs):
            program_cache.save_program(
                program=program,
                task_id="feedback_test",
                task_source="evolution",
                success=True,
                accuracy_score=0.8 + i * 0.05,  # Better than initial
                execution_time_ms=90,
                generation=1,
                parents=[result.winning_program_id] if hasattr(result, 'winning_program_id') else []
            )

        # Verify strategy was called with ensemble results
        assert mock_strategy.evolve_from_ensemble.called

        # Check that new generation has better performance
        gen0_programs = program_cache.get_programs_by_task("feedback_test")
        gen0_programs = [p for p in gen0_programs if p.generation == 0]
        gen1_programs = [p for p in program_cache.get_programs_by_task("feedback_test") if p.generation == 1]

        if gen0_programs and gen1_programs:
            avg_accuracy_gen0 = np.mean([p.accuracy_score for p in gen0_programs])
            avg_accuracy_gen1 = np.mean([p.accuracy_score for p in gen1_programs])

            print(f"\nGeneration 0 avg accuracy: {avg_accuracy_gen0:.3f}")
            print(f"Generation 1 avg accuracy: {avg_accuracy_gen1:.3f}")

            assert avg_accuracy_gen1 >= avg_accuracy_gen0, "New generation should perform at least as well"

    def test_ensemble_error_handling_with_mock_strategies(self, ensemble_interface):
        """Test ensemble handles errors from mock strategies gracefully."""
        # Create programs where some will fail execution
        failing_program = ProgramCacheEntry(
            program_id="failing_prog",
            program_hash="hash_fail",
            program=DSLProgram(operations=[
                {"type": "invalid_op", "params": {}}
            ]),
            task_id="error_test",
            task_source="evolution",
            success=True,
            accuracy_score=0.8,
            execution_time_ms=100
        )

        working_program = ProgramCacheEntry(
            program_id="working_prog",
            program_hash="hash_work",
            program=DSLProgram(operations=[
                {"type": "rotate", "params": {"angle": 90}}
            ]),
            task_id="error_test",
            task_source="evolution",
            success=True,
            accuracy_score=0.85,
            execution_time_ms=100
        )

        # Mock execution that fails for invalid operations
        def mock_execution_with_errors(program, input_grid):
            if "invalid_op" in str(program.operations):
                raise ValueError("Invalid operation")
            return np.rot90(input_grid)

        # Run ensemble with mixed programs
        candidates = [failing_program, working_program]
        input_grid = np.array([[1, 0], [0, 1]])

        result = ensemble_interface.ensemble_vote(
            input_grid=input_grid,
            candidate_programs=candidates,
            execution_func=mock_execution_with_errors
        )

        # Ensemble should handle the error and still produce result
        assert result is not None
        assert result.final_output is not None

        # Only working program should have voted
        valid_votes = [v for v in result.votes if v.output_grid is not None]
        assert len(valid_votes) == 1
        assert valid_votes[0].program_id == "working_prog"

        print(f"\nHandled {len(candidates) - len(valid_votes)} failed executions")
        print(f"Final result confidence: {result.confidence:.3f}")
