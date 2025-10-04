"""Unit tests for pattern analyzer."""


import pytest

from src.adapters.repositories.pattern_analyzer import (
    ProgramPatternAnalyzer,
)
from src.adapters.repositories.program_cache import ProgramCacheEntry
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def pattern_analyzer():
    """Create pattern analyzer instance."""
    return ProgramPatternAnalyzer({
        'min_frequency': 2,
        'min_success_rate': 0.5,
        'max_patterns': 10,
        'pattern_types': ['sequence', 'structure', 'parameter']
    })


@pytest.fixture
def sample_programs():
    """Create sample program cache entries."""
    programs = []

    # Pattern 1: rotate -> flip (appears 3 times, high success)
    for i in range(3):
        prog = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}}
            ]
        )
        entry = ProgramCacheEntry(
            program_id=f"prog_rotate_flip_{i}",
            program_hash=f"hash_{i}",
            program=prog,
            task_id=f"task_{i}",
            task_source="training",
            success=True,
            accuracy_score=0.9 + i * 0.01,
            execution_time_ms=100 + i * 10
        )
        programs.append(entry)

    # Pattern 2: fill -> mask (appears 2 times, mixed success)
    for i in range(2):
        prog = DSLProgram(
            operations=[
                {"type": "fill", "params": {"color": i}},
                {"type": "mask", "params": {"pattern": [[1, 0], [0, 1]]}}
            ]
        )
        entry = ProgramCacheEntry(
            program_id=f"prog_fill_mask_{i}",
            program_hash=f"hash_fm_{i}",
            program=prog,
            task_id=f"task_fm_{i}",
            task_source="training",
            success=i == 0,  # Only first one succeeds
            accuracy_score=0.8 if i == 0 else 0.3,
            execution_time_ms=150 + i * 20
        )
        programs.append(entry)

    # Different program (no pattern)
    prog = DSLProgram(
        operations=[
            {"type": "crop", "params": {"x": 0, "y": 0}},
            {"type": "resize", "params": {"scale": 2}}
        ]
    )
    entry = ProgramCacheEntry(
        program_id="prog_different",
        program_hash="hash_diff",
        program=prog,
        task_id="task_diff",
        task_source="evaluation",
        success=True,
        accuracy_score=0.85,
        execution_time_ms=200
    )
    programs.append(entry)

    return programs


class TestPatternAnalyzer:
    """Test ProgramPatternAnalyzer functionality."""

    def test_initialization(self, pattern_analyzer):
        """Test analyzer initialization."""
        assert pattern_analyzer.min_frequency == 2
        assert pattern_analyzer.min_success_rate == 0.5
        assert pattern_analyzer.max_patterns == 10
        assert 'sequence' in pattern_analyzer.pattern_types

    def test_find_sequence_patterns(self, pattern_analyzer, sample_programs):
        """Test finding sequence patterns."""
        patterns = pattern_analyzer.analyze_programs(sample_programs)

        # Should find rotate->flip pattern (appears 3 times, 100% success)
        rotate_flip_found = False
        for pattern in patterns.values():
            if (pattern.pattern_type == 'sequence' and
                pattern.operation_sequence == ['rotate', 'flip']):
                rotate_flip_found = True
                assert pattern.frequency == 3
                assert pattern.success_rate == 1.0
                assert len(pattern.program_ids) == 3

        assert rotate_flip_found, "rotate->flip pattern not found"

        # fill->mask should be found but with lower success rate
        fill_mask_found = False
        for pattern in patterns.values():
            if (pattern.pattern_type == 'sequence' and
                pattern.operation_sequence == ['fill', 'mask']):
                fill_mask_found = True
                assert pattern.frequency == 2
                assert pattern.success_rate == 0.5
                assert len(pattern.program_ids) == 2

        assert fill_mask_found, "fill->mask pattern not found"

    def test_find_structure_patterns(self, pattern_analyzer, sample_programs):
        """Test finding structural patterns."""
        patterns = pattern_analyzer.analyze_programs(sample_programs)

        # Should find structure patterns
        structure_patterns = [p for p in patterns.values()
                            if p.pattern_type == 'structure']
        assert len(structure_patterns) > 0

    def test_find_parameter_patterns(self, pattern_analyzer, sample_programs):
        """Test finding parameter patterns."""
        patterns = pattern_analyzer.analyze_programs(sample_programs)

        # Should find parameter patterns
        param_patterns = [p for p in patterns.values()
                         if p.pattern_type == 'parameter']
        assert len(param_patterns) > 0

    def test_pattern_filtering(self, pattern_analyzer):
        """Test pattern filtering by frequency and success rate."""
        # Create programs with low frequency pattern
        programs = []
        prog = DSLProgram(
            operations=[
                {"type": "invert", "params": {}},
                {"type": "border", "params": {"width": 1}}
            ]
        )
        entry = ProgramCacheEntry(
            program_id="prog_rare",
            program_hash="hash_rare",
            program=prog,
            task_id="task_rare",
            task_source="training",
            success=True,
            accuracy_score=0.95,
            execution_time_ms=50
        )
        programs.append(entry)

        patterns = pattern_analyzer.analyze_programs(programs)

        # Should not find pattern (frequency = 1 < min_frequency = 2)
        assert len(patterns) == 0

    def test_pattern_ranking(self, pattern_analyzer, sample_programs):
        """Test pattern ranking by importance."""
        patterns = pattern_analyzer.analyze_programs(sample_programs)

        if len(patterns) >= 2:
            pattern_list = list(patterns.values())
            # First pattern should have higher or equal combined score
            # (success_rate * 0.6 + frequency_score * 0.4)
            first_score = (pattern_list[0].success_rate * 0.6 +
                          min(pattern_list[0].frequency / 100, 1.0) * 0.4)
            second_score = (pattern_list[1].success_rate * 0.6 +
                           min(pattern_list[1].frequency / 100, 1.0) * 0.4)
            assert first_score >= second_score

    def test_extract_structure_signature(self, pattern_analyzer):
        """Test structure signature extraction."""
        prog = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}},
                {"type": "rotate", "params": {"angle": 180}}
            ]
        )

        signature = pattern_analyzer._extract_structure_signature(prog)

        # Should be sorted tuple of (op_type, count)
        assert isinstance(signature, tuple)
        assert ('flip', 1) in signature
        assert ('rotate', 2) in signature

    def test_bucket_numeric_values(self, pattern_analyzer):
        """Test numeric value bucketing."""
        assert pattern_analyzer._bucket_numeric_value(0) == "zero"
        assert pattern_analyzer._bucket_numeric_value(1) == "one"
        assert pattern_analyzer._bucket_numeric_value(-5) == "negative"
        assert pattern_analyzer._bucket_numeric_value(5) == "small"
        assert pattern_analyzer._bucket_numeric_value(50) == "medium"
        assert pattern_analyzer._bucket_numeric_value(500) == "large"

    def test_find_patterns_in_program(self, pattern_analyzer, sample_programs):
        """Test finding patterns in a specific program."""
        # First analyze to discover patterns
        patterns = pattern_analyzer.analyze_programs(sample_programs)

        # Create a new program with known pattern
        new_prog = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}},
                {"type": "fill", "params": {"color": 2}}
            ]
        )

        found_patterns = pattern_analyzer.find_patterns_in_program(new_prog, patterns)

        # Should find the rotate->flip pattern
        assert len(found_patterns) > 0

    def test_suggest_patterns_for_task(self, pattern_analyzer, sample_programs):
        """Test pattern suggestion for tasks."""
        patterns = pattern_analyzer.analyze_programs(sample_programs)

        # Mock task features
        task_features = {
            "grid_size": (10, 10),
            "colors_used": [0, 1, 2],
            "symmetry": "horizontal"
        }

        suggestions = pattern_analyzer.suggest_patterns_for_task(
            task_features, patterns, top_k=3
        )

        assert len(suggestions) <= 3
        # Suggestions should be sorted by score (descending)
        if len(suggestions) >= 2:
            assert suggestions[0][1] >= suggestions[1][1]
