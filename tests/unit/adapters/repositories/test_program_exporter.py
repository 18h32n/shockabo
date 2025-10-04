"""Unit tests for program exporter."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.adapters.repositories.program_cache import ProgramCacheEntry
from src.adapters.repositories.program_exporter import ProgramExporter
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def sample_program_entry():
    """Create sample program cache entry."""
    program = DSLProgram(
        operations=[
            {"type": "rotate", "params": {"angle": 90}},
            {"type": "flip", "params": {"axis": "horizontal"}},
            {"type": "fill", "params": {"color": 1}}
        ],
        version="1.0",
        metadata={"author": "test"}
    )

    return ProgramCacheEntry(
        program_id="test_prog_123",
        program_hash="abcd1234",
        program=program,
        task_id="task_001",
        task_source="training",
        success=True,
        accuracy_score=0.95,
        execution_time_ms=150.5,
        generation=3,
        parents=["parent1", "parent2"],
        mutation_type="crossover",
        fitness_score=0.88,
        created_at=datetime(2025, 1, 1, 10, 0, 0),
        last_accessed=datetime(2025, 1, 2, 11, 0, 0),
        access_count=5
    )


@pytest.fixture
def program_exporter():
    """Create program exporter instance."""
    return ProgramExporter(include_metadata=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestProgramExporter:
    """Test ProgramExporter functionality."""

    def test_initialization(self, program_exporter):
        """Test exporter initialization."""
        assert program_exporter.include_metadata is True
        assert 'dsl' in program_exporter.SUPPORTED_FORMATS
        assert 'python' in program_exporter.SUPPORTED_FORMATS
        assert 'json' in program_exporter.SUPPORTED_FORMATS
        assert 'markdown' in program_exporter.SUPPORTED_FORMATS

    def test_export_as_dsl(self, program_exporter, sample_program_entry):
        """Test DSL format export."""
        content = program_exporter.export_program(sample_program_entry, format='dsl')

        assert "# Program: test_prog_123" in content
        assert "# Task: task_001 (training)" in content
        assert "# Success: True (accuracy: 0.95)" in content
        assert "PROGRAM {" in content
        assert "rotate(angle=90)" in content
        assert "flip(axis=\"horizontal\")" in content
        assert "fill(color=1)" in content
        assert "->" in content
        assert "}" in content

    def test_export_as_python(self, program_exporter, sample_program_entry):
        """Test Python code export."""
        content = program_exporter.export_program(sample_program_entry, format='python')

        assert "import numpy as np" in content
        assert "def transform_grid(input_grid: np.ndarray) -> np.ndarray:" in content
        assert "grid = input_grid.copy()" in content
        assert "# Step 1: rotate" in content
        assert "grid = np.rot90(grid, k=1)" in content
        assert "# Step 2: flip" in content
        assert "grid = np.flipud(grid)" in content
        assert "# Step 3: fill" in content
        assert "return grid" in content

        # Check metadata comments
        assert "Success rate: 95.0%" in content
        assert "Fitness score: 0.880" in content

    def test_export_as_json(self, program_exporter, sample_program_entry):
        """Test JSON format export."""
        content = program_exporter.export_program(sample_program_entry, format='json')

        # Parse JSON
        data = json.loads(content)

        assert data['program_id'] == "test_prog_123"
        assert data['task_id'] == "task_001"
        assert data['success'] is True
        assert data['accuracy_score'] == 0.95
        assert data['generation'] == 3
        assert data['parents'] == ["parent1", "parent2"]

        # Check program structure
        assert 'program' in data
        assert len(data['program']['operations']) == 3
        assert data['program']['operations'][0]['type'] == 'rotate'

    def test_export_as_markdown(self, program_exporter, sample_program_entry):
        """Test Markdown documentation export."""
        content = program_exporter.export_program(sample_program_entry, format='markdown')

        assert "# Program: test_prog_123" in content
        assert "## Metadata" in content
        assert "- **Task ID**: task_001" in content
        assert "- **Success**: ✓" in content
        assert "- **Accuracy Score**: 95.00%" in content
        assert "## Operations Pipeline" in content
        assert "### Step 1: Rotate" in content
        assert "- `angle`: `90`" in content
        assert "## Genealogy" in content
        assert "- parent1" in content
        assert "## Code Examples" in content
        assert "```dsl" in content

    def test_export_to_file(self, program_exporter, sample_program_entry, temp_output_dir):
        """Test export to file."""
        output_file = Path(temp_output_dir) / "test_export.dsl"

        result = program_exporter.export_program(
            sample_program_entry,
            format='dsl',
            output_file=str(output_file)
        )

        assert result == str(output_file)
        assert output_file.exists()

        # Check file content
        content = output_file.read_text()
        assert "PROGRAM {" in content

    def test_export_batch_separate_files(self, program_exporter, sample_program_entry, temp_output_dir):
        """Test batch export to separate files."""
        # Create multiple entries
        entries = []
        for i in range(3):
            entry = ProgramCacheEntry(
                program_id=f"prog_{i}",
                program_hash=f"hash_{i}",
                program=sample_program_entry.program,
                task_id=f"task_{i}",
                task_source="training",
                success=True,
                accuracy_score=0.9,
                execution_time_ms=100
            )
            entries.append(entry)

        # Export batch
        exported_files = program_exporter.export_programs_batch(
            entries,
            format='json',
            output_dir=temp_output_dir,
            single_file=False
        )

        assert len(exported_files) == 3

        # Check files exist
        for i, file_path in enumerate(exported_files):
            assert Path(file_path).exists()
            assert f"prog_{i}.json" in file_path

    def test_export_batch_single_file(self, program_exporter, sample_program_entry, temp_output_dir):
        """Test batch export to single file."""
        entries = [sample_program_entry]

        exported_files = program_exporter.export_programs_batch(
            entries,
            format='json',
            output_dir=temp_output_dir,
            single_file=True
        )

        assert len(exported_files) == 1

        # Check single file with all programs
        file_path = Path(exported_files[0])
        assert file_path.exists()
        assert "programs_export_" in file_path.name

        # Check content
        data = json.loads(file_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['program_id'] == 'test_prog_123'

    def test_validate_export(self, program_exporter, sample_program_entry):
        """Test export validation."""
        # Valid JSON
        json_content = program_exporter.export_program(sample_program_entry, format='json')
        assert program_exporter.validate_export(json_content, 'json') is True

        # Valid Python
        python_content = program_exporter.export_program(sample_program_entry, format='python')
        assert program_exporter.validate_export(python_content, 'python') is True

        # Valid DSL
        dsl_content = program_exporter.export_program(sample_program_entry, format='dsl')
        assert program_exporter.validate_export(dsl_content, 'dsl') is True

        # Invalid JSON
        assert program_exporter.validate_export("{invalid json", 'json') is False

        # Invalid Python
        assert program_exporter.validate_export("def broken(", 'python') is False

    def test_get_extension(self, program_exporter):
        """Test file extension mapping."""
        assert program_exporter._get_extension('dsl') == 'dsl'
        assert program_exporter._get_extension('python') == 'py'
        assert program_exporter._get_extension('json') == 'json'
        assert program_exporter._get_extension('markdown') == 'md'
        assert program_exporter._get_extension('unknown') == 'txt'

    def test_export_without_metadata(self):
        """Test export without metadata."""
        exporter = ProgramExporter(include_metadata=False)
        entry = ProgramCacheEntry(
            program_id="simple",
            program_hash="hash",
            program=DSLProgram(
                operations=[{"type": "rotate", "params": {"angle": 90}}]
            ),
            task_id="task",
            task_source="test",
            success=True,
            accuracy_score=1.0,
            execution_time_ms=10
        )

        # DSL should not have metadata comments
        dsl_content = exporter.export_program(entry, format='dsl')
        assert "# Program:" not in dsl_content
        assert "PROGRAM {" in dsl_content

        # JSON should not have optional fields
        json_data = json.loads(exporter.export_program(entry, format='json'))
        assert 'generation' not in json_data
        assert 'parents' not in json_data

    def test_unsupported_format(self, program_exporter, sample_program_entry):
        """Test error handling for unsupported format."""
        with pytest.raises(ValueError) as exc_info:
            program_exporter.export_program(sample_program_entry, format='invalid')

        assert "Unsupported format" in str(exc_info.value)
