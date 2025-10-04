"""Unit tests for population merger with hash-based deduplication."""

import pytest

from src.adapters.strategies.population_merger import PopulationMerger


@pytest.fixture
def merger():
    """Create population merger instance."""
    return PopulationMerger()


@pytest.fixture
def population1():
    """Create first sample population."""
    return [
        {"program": "rotate_90 | flip_horizontal", "fitness": 0.85, "hash": "hash1"},
        {"program": "flip_vertical | crop", "fitness": 0.72, "hash": "hash2"},
        {"program": "translate_x(2) | translate_y(1)", "fitness": 0.68, "hash": "hash3"},
    ]


@pytest.fixture
def population2():
    """Create second sample population."""
    return [
        {"program": "rotate_90 | flip_horizontal", "fitness": 0.80, "hash": "hash1"},
        {"program": "scale(2) | filter_color(1)", "fitness": 0.91, "hash": "hash4"},
        {"program": "pad(1) | crop", "fitness": 0.55, "hash": "hash5"},
    ]


@pytest.fixture
def population3():
    """Create third sample population."""
    return [
        {"program": "flip_vertical | crop", "fitness": 0.75, "hash": "hash2"},
        {"program": "new_program", "fitness": 0.60, "hash": "hash6"},
    ]


def test_merge_no_duplicates(merger, population1, population2):
    """Test merging populations with no duplicates."""
    pop2_unique = [
        {"program": "scale(2) | filter_color(1)", "fitness": 0.91, "hash": "hash4"},
        {"program": "pad(1) | crop", "fitness": 0.55, "hash": "hash5"},
    ]

    result = merger.merge_populations({"platform1": population1, "platform2": pop2_unique})
    merged = result.merged_population

    assert len(merged) == 5

    hashes = [ind["hash"] for ind in merged]
    assert len(hashes) == len(set(hashes))


def test_merge_with_duplicates(merger, population1, population2):
    """Test merging populations with duplicates - keeps higher fitness."""
    result = merger.merge_populations({"platform1": population1, "platform2": population2})
    merged = result.merged_population

    hashes = [ind["hash"] for ind in merged]
    assert len(hashes) == len(set(hashes))

    hash1_individuals = [ind for ind in merged if ind["hash"] == "hash1"]
    assert len(hash1_individuals) == 1
    assert hash1_individuals[0]["fitness"] == 0.85


def test_merge_multiple_populations(merger, population1, population2, population3):
    """Test merging three populations."""
    result = merger.merge_populations({"platform1": population1, "platform2": population2, "platform3": population3})
    merged = result.merged_population

    unique_hashes = set(ind["hash"] for ind in merged)
    assert len(merged) == len(unique_hashes)

    all_input_hashes = set()
    for pop in [population1, population2, population3]:
        all_input_hashes.update(ind["hash"] for ind in pop)

    assert unique_hashes == all_input_hashes


def test_fitness_based_conflict_resolution(merger):
    """Test that higher fitness individual is kept on conflict."""
    pop1 = [
        {"program": "program_a", "fitness": 0.50, "hash": "same_hash"}
    ]
    pop2 = [
        {"program": "program_a", "fitness": 0.90, "hash": "same_hash"}
    ]
    pop3 = [
        {"program": "program_a", "fitness": 0.70, "hash": "same_hash"}
    ]

    result = merger.merge_populations({"platform1": pop1, "platform2": pop2, "platform3": pop3})
    merged = result.merged_population

    assert len(merged) == 1
    assert merged[0]["fitness"] == 0.90


def test_merge_empty_populations(merger):
    """Test merging with empty populations."""
    pop1 = [{"program": "program_1", "fitness": 0.5, "hash": "hash1"}]
    pop2 = []
    pop3 = [{"program": "program_2", "fitness": 0.6, "hash": "hash2"}]

    result = merger.merge_populations({"platform1": pop1, "platform2": pop2, "platform3": pop3})
    merged = result.merged_population

    assert len(merged) == 2


def test_merge_all_empty(merger):
    """Test merging all empty populations."""
    result = merger.merge_populations({"platform1": [], "platform2": [], "platform3": []})
    merged = result.merged_population

    assert len(merged) == 0


def test_merge_validation(merger, population1, population2):
    """Test merge validation checks."""
    result = merger.merge_populations({"platform1": population1, "platform2": population2})

    is_valid = merger.validate_merge(
        populations={"platform1": population1, "platform2": population2},
        merge_result=result
    )

    assert is_valid is True
    assert result.duplicates_removed > 0
    assert result.platforms_merged == 2


def test_hash_generation_consistency(merger):
    """Test that hash generation is consistent."""
    program = "rotate_90 | flip_horizontal"

    hash1 = merger._calculate_hash(program)
    hash2 = merger._calculate_hash(program)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_hash_uniqueness(merger):
    """Test that different programs generate different hashes."""
    program1 = "rotate_90 | flip_horizontal"
    program2 = "flip_vertical | crop"

    hash1 = merger._calculate_hash(program1)
    hash2 = merger._calculate_hash(program2)

    assert hash1 != hash2


def test_merge_preserves_program_data(merger):
    """Test that merge preserves all program data."""
    pop = [
        {
            "program": "test_program",
            "fitness": 0.75,
            "hash": "test_hash",
            "metadata": {"extra": "data"}
        }
    ]

    result = merger.merge_populations({"platform1": pop})
    merged = result.merged_population

    assert merged[0]["program"] == "test_program"
    assert merged[0]["fitness"] == 0.75
    assert merged[0]["hash"] == "test_hash"
    assert merged[0].get("metadata") == {"extra": "data"}


def test_deduplication_rate_calculation(merger):
    """Test deduplication rate calculation."""
    pop1 = [
        {"program": "prog1", "fitness": 0.5, "hash": "hash1"},
        {"program": "prog2", "fitness": 0.6, "hash": "hash2"},
    ]
    pop2 = [
        {"program": "prog1", "fitness": 0.7, "hash": "hash1"},
        {"program": "prog3", "fitness": 0.8, "hash": "hash3"},
    ]

    result = merger.merge_populations({"platform1": pop1, "platform2": pop2})

    total_input = len(pop1) + len(pop2)

    assert result.total_individuals == total_input
    assert result.duplicates_removed == 1
    assert len(result.merged_population) == 3


def test_merge_large_populations(merger):
    """Test merging large populations."""
    pop1 = [
        {"program": f"program_{i}", "fitness": 0.5 + (i * 0.001), "hash": f"hash_{i}"}
        for i in range(1000)
    ]
    pop2 = [
        {"program": f"program_{i}", "fitness": 0.6 + (i * 0.001), "hash": f"hash_{i}"}
        for i in range(500, 1500)
    ]

    result = merger.merge_populations({"platform1": pop1, "platform2": pop2})
    merged = result.merged_population

    unique_hashes = set(ind["hash"] for ind in merged)
    assert len(merged) == len(unique_hashes)
    assert len(merged) == 1500


def test_no_data_loss_guarantee(merger, population1, population2):
    """Test that no unique individuals are lost during merge."""
    result = merger.merge_populations({"platform1": population1, "platform2": population2})
    merged = result.merged_population

    all_unique_hashes = set()
    for pop in [population1, population2]:
        all_unique_hashes.update(ind["hash"] for ind in pop)

    merged_hashes = set(ind["hash"] for ind in merged)

    assert merged_hashes == all_unique_hashes


def test_merge_with_missing_hashes(merger):
    """Test merging when some individuals lack hash - they get auto-generated."""
    pop1 = [
        {"program": "prog1", "fitness": 0.5},
        {"program": "prog3", "fitness": 0.7},
    ]
    pop2 = [
        {"program": "prog2", "fitness": 0.6},
    ]

    result = merger.merge_populations({"platform1": pop1, "platform2": pop2})
    merged = result.merged_population

    for ind in merged:
        assert "hash" in ind
        assert len(ind["hash"]) == 64


def test_fitness_ordering_preserved(merger):
    """Test that merged population can be sorted by fitness."""
    pop1 = [
        {"program": "prog1", "fitness": 0.8, "hash": "hash1"},
        {"program": "prog2", "fitness": 0.5, "hash": "hash2"},
    ]
    pop2 = [
        {"program": "prog3", "fitness": 0.9, "hash": "hash3"},
        {"program": "prog4", "fitness": 0.6, "hash": "hash4"},
    ]

    result = merger.merge_populations({"platform1": pop1, "platform2": pop2})
    merged = result.merged_population
    sorted_merged = sorted(merged, key=lambda x: x["fitness"], reverse=True)

    assert sorted_merged[0]["fitness"] == 0.9
    assert sorted_merged[-1]["fitness"] == 0.5
