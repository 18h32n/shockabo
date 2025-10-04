"""Unit tests for program similarity detector."""

import pytest

from src.adapters.repositories.similarity_detector import (
    ProgramSimilarityDetector,
    SimilarityResult,
)
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def similarity_detector():
    """Create similarity detector instance."""
    return ProgramSimilarityDetector({
        'semantic_threshold': 0.95,
        'fuzzy_threshold': 0.85,
        'weights': {
            'semantic': 0.4,
            'structural': 0.3,
            'parameter': 0.3
        }
    })


@pytest.fixture
def program_rotate_flip():
    """Create program with rotate and flip operations."""
    return DSLProgram(
        operations=[
            {"type": "rotate", "params": {"angle": 90}},
            {"type": "flip", "params": {"axis": "horizontal"}}
        ],
        version="1.0"
    )


@pytest.fixture
def program_rotate_flip_fill():
    """Create program with rotate, flip, and fill operations."""
    return DSLProgram(
        operations=[
            {"type": "rotate", "params": {"angle": 90}},
            {"type": "flip", "params": {"axis": "horizontal"}},
            {"type": "fill", "params": {"color": 1}}
        ],
        version="1.0"
    )


@pytest.fixture
def program_different():
    """Create program with completely different operations."""
    return DSLProgram(
        operations=[
            {"type": "crop", "params": {"x": 0, "y": 0, "width": 10, "height": 10}},
            {"type": "resize", "params": {"scale": 2.0}},
            {"type": "invert", "params": {}}
        ],
        version="1.0"
    )


class TestSimilarityDetector:
    """Test ProgramSimilarityDetector functionality."""

    def test_compute_hash(self, similarity_detector, program_rotate_flip):
        """Test hash computation for programs."""
        # Same program should have same hash
        hash1 = similarity_detector.compute_hash(program_rotate_flip)
        hash2 = similarity_detector.compute_hash(program_rotate_flip)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

        # Different program should have different hash
        different_prog = DSLProgram(
            operations=[{"type": "rotate", "params": {"angle": 180}}],
            version="1.0"
        )
        hash3 = similarity_detector.compute_hash(different_prog)
        assert hash1 != hash3

    def test_exact_duplicate_detection(self, similarity_detector, program_rotate_flip):
        """Test exact duplicate detection."""
        # Same program is exact duplicate
        assert similarity_detector.detect_exact_duplicate(
            program_rotate_flip, program_rotate_flip
        )

        # Program with different parameters is not exact duplicate
        modified_prog = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 180}},  # Different angle
                {"type": "flip", "params": {"axis": "horizontal"}}
            ],
            version="1.0"
        )
        assert not similarity_detector.detect_exact_duplicate(
            program_rotate_flip, modified_prog
        )

    def test_semantic_similarity(self, similarity_detector, program_rotate_flip,
                                 program_rotate_flip_fill, program_different):
        """Test semantic similarity computation."""
        # Same program should have similarity 1.0
        sim_same = similarity_detector.compute_semantic_similarity(
            program_rotate_flip, program_rotate_flip
        )
        assert sim_same == 1.0

        # Similar programs should have high similarity
        sim_similar = similarity_detector.compute_semantic_similarity(
            program_rotate_flip, program_rotate_flip_fill
        )
        assert 0.6 < sim_similar < 1.0

        # Different programs should have low similarity
        sim_diff = similarity_detector.compute_semantic_similarity(
            program_rotate_flip, program_different
        )
        assert sim_diff < 0.3

    def test_structural_similarity(self, similarity_detector, program_rotate_flip,
                                  program_rotate_flip_fill, program_different):
        """Test structural similarity computation."""
        # Same program should have similarity 1.0
        sim_same = similarity_detector.compute_structural_similarity(
            program_rotate_flip, program_rotate_flip
        )
        assert sim_same == 1.0

        # Similar structure should have high similarity
        sim_similar = similarity_detector.compute_structural_similarity(
            program_rotate_flip, program_rotate_flip_fill
        )
        assert sim_similar > 0.5

        # Different structure should have lower similarity
        sim_diff = similarity_detector.compute_structural_similarity(
            program_rotate_flip, program_different
        )
        assert sim_diff < sim_similar

    def test_parameter_similarity(self, similarity_detector):
        """Test parameter similarity computation."""
        # Programs with same parameters
        prog1 = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}}
            ]
        )
        prog2 = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}}
            ]
        )

        sim_same = similarity_detector.compute_parameter_similarity(prog1, prog2)
        assert sim_same == 1.0

        # Programs with different parameter values
        prog3 = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 180}},
                {"type": "flip", "params": {"axis": "vertical"}}
            ]
        )

        sim_diff = similarity_detector.compute_parameter_similarity(prog1, prog3)
        assert 0.0 <= sim_diff < 1.0

    def test_fuzzy_similarity(self, similarity_detector, program_rotate_flip,
                             program_rotate_flip_fill):
        """Test comprehensive fuzzy similarity."""
        result = similarity_detector.compute_fuzzy_similarity(
            program_rotate_flip, program_rotate_flip_fill
        )

        assert isinstance(result, SimilarityResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert not result.exact_match
        assert 0.0 <= result.semantic_score <= 1.0
        assert 0.0 <= result.structural_score <= 1.0
        assert 0.0 <= result.parameter_score <= 1.0
        assert isinstance(result.operation_overlap, list)
        assert "rotate" in result.operation_overlap
        assert "flip" in result.operation_overlap

    def test_cluster_programs(self, similarity_detector):
        """Test program clustering."""
        # Create test programs
        programs = [
            ("prog1", DSLProgram(operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}}
            ])),
            ("prog2", DSLProgram(operations=[
                {"type": "rotate", "params": {"angle": 90}},
                {"type": "flip", "params": {"axis": "horizontal"}}
            ])),
            ("prog3", DSLProgram(operations=[
                {"type": "crop", "params": {"x": 0, "y": 0}},
                {"type": "resize", "params": {"scale": 2}}
            ])),
            ("prog4", DSLProgram(operations=[
                {"type": "crop", "params": {"x": 0, "y": 0}},
                {"type": "resize", "params": {"scale": 2}}
            ]))
        ]

        # Cluster with high threshold
        clusters = similarity_detector.cluster_similar_programs(programs, threshold=0.9)

        # Should have two clusters (rotate+flip and crop+resize)
        assert len(clusters) >= 2

        # Each similar pair should be in same cluster
        prog1_cluster = None
        prog3_cluster = None

        for cluster in clusters:
            if "prog1" in cluster:
                prog1_cluster = cluster
            if "prog3" in cluster:
                prog3_cluster = cluster

        assert prog1_cluster is not None
        assert prog3_cluster is not None
        assert "prog2" in prog1_cluster
        assert "prog4" in prog3_cluster
        assert prog1_cluster != prog3_cluster

    def test_lcs_computation(self, similarity_detector):
        """Test longest common subsequence computation."""
        seq1 = ["rotate", "flip", "fill", "mask"]
        seq2 = ["rotate", "crop", "flip", "mask"]

        lcs_length = similarity_detector._longest_common_subsequence(seq1, seq2)
        assert lcs_length == 3  # ["rotate", "flip", "mask"]

    def test_edit_distance(self, similarity_detector):
        """Test edit distance computation."""
        seq1 = ["rotate", "flip"]
        seq2 = ["rotate", "flip", "fill"]

        edit_dist = similarity_detector._edit_distance(seq1, seq2)
        assert edit_dist == 1  # One insertion

        seq3 = ["crop", "resize"]
        edit_dist2 = similarity_detector._edit_distance(seq1, seq3)
        assert edit_dist2 == 2  # Two substitutions

    def test_ngram_similarity(self, similarity_detector):
        """Test n-gram similarity computation."""
        seq1 = ["rotate", "flip", "fill"]
        seq2 = ["rotate", "flip", "mask"]

        # Bigram similarity
        sim_2gram = similarity_detector._ngram_similarity(seq1, seq2, n=2)
        assert 0.0 < sim_2gram < 1.0  # ["rotate", "flip"] is common

        # Unigram similarity
        sim_1gram = similarity_detector._ngram_similarity(seq1, seq2, n=1)
        assert sim_1gram == 2/4  # 2 common out of 4 total unique
