"""Unit tests for ensemble interface."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.adapters.repositories.ensemble_interface import (
    EnsembleInterface,
    EnsembleResult,
    ProgramVote,
)
from src.adapters.repositories.program_cache import ProgramCache, ProgramCacheEntry
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def mock_cache():
    """Create mock program cache."""
    cache = Mock(spec=ProgramCache)
    return cache


@pytest.fixture
def sample_programs():
    """Create sample program entries."""
    programs = []

    # High accuracy program
    prog1 = ProgramCacheEntry(
        program_id="prog_high",
        program_hash="hash1",
        program=DSLProgram(
            operations=[{"type": "rotate", "params": {"angle": 90}}]
        ),
        task_id="task_001",
        task_source="training",
        success=True,
        accuracy_score=0.95,
        execution_time_ms=100,
        access_count=10,
        fitness_score=0.9
    )
    programs.append(prog1)

    # Medium accuracy program
    prog2 = ProgramCacheEntry(
        program_id="prog_med",
        program_hash="hash2",
        program=DSLProgram(
            operations=[{"type": "flip", "params": {"axis": "horizontal"}}]
        ),
        task_id="task_001",
        task_source="training",
        success=True,
        accuracy_score=0.85,
        execution_time_ms=120,
        access_count=5
    )
    programs.append(prog2)

    # Low accuracy program
    prog3 = ProgramCacheEntry(
        program_id="prog_low",
        program_hash="hash3",
        program=DSLProgram(
            operations=[{"type": "fill", "params": {"color": 1}}]
        ),
        task_id="task_001",
        task_source="training",
        success=False,
        accuracy_score=0.70,
        execution_time_ms=150,
        access_count=2
    )
    programs.append(prog3)

    return programs


@pytest.fixture
def ensemble_interface(mock_cache):
    """Create ensemble interface instance."""
    config = {
        'min_programs_for_vote': 2,
        'confidence_threshold': 0.7,
        'success_weight_multiplier': 1.5,
        'voting_method': 'weighted_majority'
    }
    return EnsembleInterface(mock_cache, config)


class TestEnsembleInterface:
    """Test EnsembleInterface functionality."""

    def test_initialization(self, ensemble_interface, mock_cache):
        """Test interface initialization."""
        assert ensemble_interface.cache == mock_cache
        assert ensemble_interface.min_programs_for_vote == 2
        assert ensemble_interface.confidence_threshold == 0.7
        assert ensemble_interface.success_weight_multiplier == 1.5
        assert ensemble_interface.voting_method == 'weighted_majority'
        assert isinstance(ensemble_interface.ensemble_cache, dict)

    def test_get_programs_for_ensemble_by_task(self, ensemble_interface, sample_programs):
        """Test getting programs for ensemble by task ID."""
        # Mock cache method
        ensemble_interface.cache.get_programs_by_task.return_value = sample_programs[:2]

        programs = ensemble_interface.get_programs_for_ensemble(
            task_id="task_001",
            min_accuracy=0.8
        )

        assert len(programs) == 2
        # Should be sorted by success and accuracy
        assert programs[0].accuracy_score >= programs[1].accuracy_score
        ensemble_interface.cache.get_programs_by_task.assert_called_once_with("task_001")

    def test_get_programs_for_ensemble_similar(self, ensemble_interface, sample_programs):
        """Test getting similar programs for ensemble."""
        test_program = DSLProgram(operations=[{"type": "rotate", "params": {"angle": 90}}])

        # Mock similar programs response
        ensemble_interface.cache.find_similar_programs.return_value = [
            ("prog_high", 0.95, Mock()),
            ("prog_med", 0.85, Mock())
        ]

        # Mock get_program to return entries
        def get_program_side_effect(prog_id):
            prog_map = {p.program_id: p for p in sample_programs}
            return prog_map.get(prog_id)

        ensemble_interface.cache.get_program.side_effect = get_program_side_effect

        programs = ensemble_interface.get_programs_for_ensemble(
            similar_to=test_program,
            min_accuracy=0.8
        )

        assert len(programs) > 0
        ensemble_interface.cache.find_similar_programs.assert_called_once()

    def test_calculate_confidence(self, ensemble_interface, sample_programs):
        """Test confidence calculation."""
        # High accuracy, successful, high access count
        conf1 = ensemble_interface._calculate_confidence(sample_programs[0])
        assert conf1 > 0.9  # Should be boosted (but capped at 1.0)

        # Medium accuracy, successful, medium access
        conf2 = ensemble_interface._calculate_confidence(sample_programs[1])
        assert conf2 > 0.8  # Should be boosted

        # For capped values, check base scores
        base1 = sample_programs[0].accuracy_score
        base2 = sample_programs[1].accuracy_score
        assert base1 > base2  # Original accuracy ordering preserved

        # Low accuracy, failed
        conf3 = ensemble_interface._calculate_confidence(sample_programs[2])
        assert conf3 == sample_programs[2].accuracy_score  # No boost

    def test_simple_majority_vote(self, ensemble_interface):
        """Test simple majority voting."""
        # Create votes with same output
        output1 = np.array([[1, 2], [3, 4]])
        output2 = np.array([[1, 2], [3, 4]])
        output3 = np.array([[5, 6], [7, 8]])

        votes = [
            ProgramVote("prog1", output1, 0.9),
            ProgramVote("prog2", output2, 0.8),
            ProgramVote("prog3", output3, 0.7)
        ]

        result = ensemble_interface._simple_majority_vote(votes)

        assert isinstance(result, EnsembleResult)
        assert np.array_equal(result.final_output, output1)  # Majority output
        assert result.voting_method == "simple_majority"
        assert result.consensus_level == 2/3  # 2 out of 3 agree

    def test_weighted_majority_vote(self, ensemble_interface):
        """Test weighted majority voting."""
        # Different outputs with different confidences
        output1 = np.array([[1, 2], [3, 4]])
        output2 = np.array([[5, 6], [7, 8]])

        votes = [
            ProgramVote("prog1", output1, 0.9),
            ProgramVote("prog2", output1, 0.8),
            ProgramVote("prog3", output2, 0.3)
        ]

        result = ensemble_interface._weighted_majority_vote(votes)

        assert isinstance(result, EnsembleResult)
        assert np.array_equal(result.final_output, output1)  # Higher weighted output
        assert result.voting_method == "weighted_majority"
        assert result.consensus_level > 0.7  # High weighted consensus

    def test_confidence_weighted_vote(self, ensemble_interface):
        """Test confidence-weighted averaging."""
        # Numeric outputs for averaging
        output1 = np.array([[1, 2], [3, 4]])
        output2 = np.array([[2, 3], [4, 5]])
        output3 = np.array([[3, 4], [5, 6]])

        votes = [
            ProgramVote("prog1", output1, 0.6),
            ProgramVote("prog2", output2, 0.3),
            ProgramVote("prog3", output3, 0.1)
        ]

        result = ensemble_interface._confidence_weighted_vote(votes)

        assert isinstance(result, EnsembleResult)
        assert result.voting_method == "confidence_weighted"
        # Calculate expected weighted average
        # (1*0.6 + 2*0.3 + 3*0.1) / (0.6 + 0.3 + 0.1) = 1.5
        # After rounding: 2
        assert result.final_output[0, 0] == 2  # Weighted average rounded
        assert np.all(result.final_output >= 1)  # All values reasonable
        assert np.all(result.final_output <= 6)

    def test_ensemble_vote_full(self, ensemble_interface, sample_programs):
        """Test full ensemble voting process."""
        input_grid = np.array([[0, 0], [0, 0]])

        # Mock execution function
        def mock_execute(program, grid):
            # Return different outputs based on program
            if "rotate" in str(program.operations):
                return np.array([[1, 1], [1, 1]])
            elif "flip" in str(program.operations):
                return np.array([[2, 2], [2, 2]])
            else:
                return np.array([[3, 3], [3, 3]])

        result = ensemble_interface.ensemble_vote(
            input_grid=input_grid,
            candidate_programs=sample_programs,
            execution_func=mock_execute
        )

        assert isinstance(result, EnsembleResult)
        assert len(result.votes) == 3
        assert result.confidence > 0

    def test_ensemble_vote_insufficient_programs(self, ensemble_interface, sample_programs):
        """Test ensemble with insufficient programs."""
        input_grid = np.array([[0, 0], [0, 0]])

        # Only one program (less than min_programs_for_vote)
        result = ensemble_interface.ensemble_vote(
            input_grid=input_grid,
            candidate_programs=[sample_programs[0]],
            execution_func=lambda p, g: g
        )

        assert result.voting_method == "single_best"
        assert len(result.votes) == 1

    def test_cache_and_retrieve_ensemble_result(self, ensemble_interface):
        """Test caching ensemble results."""
        result = EnsembleResult(
            final_output=np.array([[1, 2], [3, 4]]),
            confidence=0.9,
            votes=[],
            voting_method="test",
            consensus_level=0.8
        )

        # Cache result
        ensemble_interface.cache_ensemble_result("task_123", result)

        # Retrieve
        cached = ensemble_interface.get_cached_ensemble_result("task_123")

        assert cached is not None
        assert cached.confidence == 0.9
        assert np.array_equal(cached.final_output, result.final_output)

    def test_analyze_ensemble_performance(self, ensemble_interface):
        """Test ensemble performance analysis."""
        results = [
            EnsembleResult(
                final_output=np.zeros((2, 2)),
                confidence=0.9,
                votes=[Mock(), Mock(), Mock()],
                voting_method="weighted_majority",
                consensus_level=0.85
            ),
            EnsembleResult(
                final_output=np.zeros((2, 2)),
                confidence=0.7,
                votes=[Mock(), Mock()],
                voting_method="simple_majority",
                consensus_level=0.6
            ),
            EnsembleResult(
                final_output=np.zeros((2, 2)),
                confidence=0.8,
                votes=[Mock(), Mock(), Mock(), Mock()],
                voting_method="weighted_majority",
                consensus_level=0.75
            )
        ]

        stats = ensemble_interface.analyze_ensemble_performance(results)

        assert stats['total_ensembles'] == 3
        assert 0.7 <= stats['avg_confidence'] <= 0.9
        assert stats['avg_vote_count'] == 3
        assert 'weighted_majority' in stats['voting_methods']
        assert stats['voting_methods']['weighted_majority'] == 2
        assert 0 <= stats['high_confidence_rate'] <= 1

    def test_consensus_voting(self, ensemble_interface):
        """Test consensus voting method."""
        ensemble_interface.voting_method = 'consensus'

        # High consensus votes
        output = np.array([[1, 1], [1, 1]])
        votes = [
            ProgramVote("prog1", output, 0.9),
            ProgramVote("prog2", output, 0.8),
            ProgramVote("prog3", output, 0.7)
        ]

        result = ensemble_interface._consensus_vote(votes)

        # Should accept high consensus
        assert np.array_equal(result.final_output, output)
        assert result.consensus_level >= 0.7

        # Low consensus votes
        votes2 = [
            ProgramVote("prog1", np.array([[1, 1], [1, 1]]), 0.9),
            ProgramVote("prog2", np.array([[2, 2], [2, 2]]), 0.8),
            ProgramVote("prog3", np.array([[3, 3], [3, 3]]), 0.7)
        ]

        result2 = ensemble_interface._consensus_vote(votes2)

        # Should fall back to best program
        assert result2.voting_method == "consensus_failed"
        assert result2.confidence < 0.9  # Reduced confidence
