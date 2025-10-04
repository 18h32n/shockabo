"""Program export functionality for various formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ProgramExporter:
    """Export DSL programs to various readable formats."""

    SUPPORTED_FORMATS = ['dsl', 'python', 'json', 'markdown']

    def __init__(self, include_metadata: bool = True):
        """Initialize exporter."""
        self.include_metadata = include_metadata

    def export_program(
        self,
        program_entry: Any,  # ProgramCacheEntry
        format: str = 'dsl',
        output_file: str | None = None
    ) -> str:
        """Export a single program to specified format."""
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")

        if format == 'dsl':
            content = self._export_as_dsl(program_entry)
        elif format == 'python':
            content = self._export_as_python(program_entry)
        elif format == 'json':
            content = self._export_as_json(program_entry)
        elif format == 'markdown':
            content = self._export_as_markdown(program_entry)

        if output_file:
            Path(output_file).write_text(content, encoding='utf-8')
            return output_file

        return content

    def export_programs_batch(
        self,
        program_entries: list[Any],  # List[ProgramCacheEntry]
        format: str = 'dsl',
        output_dir: str = None,
        single_file: bool = False
    ) -> list[str]:
        """Export multiple programs."""
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        if single_file and output_dir:
            # Export all to single file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f"programs_export_{timestamp}.{self._get_extension(format)}"

            if format == 'json':
                # Special handling for JSON batch export
                all_programs = []
                for entry in program_entries:
                    all_programs.append(self._program_entry_to_dict(entry))
                content = json.dumps(all_programs, indent=2)
            else:
                # Concatenate other formats
                contents = []
                for entry in program_entries:
                    contents.append(self.export_program(entry, format))
                content = "\n\n" + ("=" * 80) + "\n\n".join(contents)

            output_file.write_text(content, encoding='utf-8')
            return [str(output_file)]
        else:
            # Export each to separate file
            exported_files = []
            for _i, entry in enumerate(program_entries):
                if output_dir:
                    filename = f"{entry.program_id}.{self._get_extension(format)}"
                    output_file = str(output_path / filename)
                    self.export_program(entry, format, output_file)
                    exported_files.append(output_file)
                else:
                    content = self.export_program(entry, format)
                    exported_files.append(content)

            return exported_files

    def _export_as_dsl(self, entry: Any) -> str:  # ProgramCacheEntry
        """Export as pretty-printed DSL format."""
        lines = []

        # Header with metadata
        if self.include_metadata:
            lines.append(f"# Program: {entry.program_id}")
            lines.append(f"# Task: {entry.task_id} ({entry.task_source})")
            lines.append(f"# Success: {entry.success} (accuracy: {entry.accuracy_score:.2f})")
            lines.append(f"# Execution time: {entry.execution_time_ms:.1f}ms")
            if entry.generation is not None:
                lines.append(f"# Generation: {entry.generation}")
            if entry.parents:
                lines.append(f"# Parents: {', '.join(entry.parents)}")
            lines.append(f"# Created: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")

        # DSL operations
        lines.append("PROGRAM {")
        for i, op in enumerate(entry.program.operations):
            op_type = op.get('type', 'unknown')
            params = op.get('params', {})

            # Format operation
            if params:
                param_strs = []
                for key, value in params.items():
                    if isinstance(value, str):
                        param_strs.append(f'{key}="{value}"')
                    elif isinstance(value, list):
                        param_strs.append(f'{key}={value}')
                    else:
                        param_strs.append(f'{key}={value}')

                lines.append(f"  {op_type}({', '.join(param_strs)})")
            else:
                lines.append(f"  {op_type}()")

            # Add arrow between operations except last
            if i < len(entry.program.operations) - 1:
                lines.append("    ->")

        lines.append("}")

        return "\n".join(lines)

    def _export_as_python(self, entry: Any) -> str:  # ProgramCacheEntry
        """Export as Python code with comments."""
        lines = []

        # Header
        lines.append('"""')
        if self.include_metadata:
            lines.append(f"Program: {entry.program_id}")
            lines.append(f"Task: {entry.task_id}")
            lines.append(f"Success rate: {entry.accuracy_score:.2%}")
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append('"""')
        lines.append("")
        lines.append("import numpy as np")
        lines.append("from typing import List")
        lines.append("")
        lines.append("")

        # Main function
        lines.append("def transform_grid(input_grid: np.ndarray) -> np.ndarray:")
        lines.append('    """Apply transformation pipeline to input grid."""')
        lines.append("    grid = input_grid.copy()")
        lines.append("")

        # Operations
        for i, op in enumerate(entry.program.operations):
            op_type = op.get('type', 'unknown')
            params = op.get('params', {})

            # Add operation comment
            lines.append(f"    # Step {i+1}: {op_type}")

            # Generate Python code for operation
            if op_type == 'rotate':
                angle = params.get('angle', 0)
                k = angle // 90
                lines.append(f"    grid = np.rot90(grid, k={k})")

            elif op_type == 'flip':
                axis = params.get('axis', 'horizontal')
                if axis == 'horizontal':
                    lines.append("    grid = np.flipud(grid)")
                else:
                    lines.append("    grid = np.fliplr(grid)")

            elif op_type == 'fill':
                color = params.get('color', 0)
                if 'position' in params:
                    pos = params['position']
                    lines.append(f"    # Fill at position {pos}")
                    lines.append("    from scipy.ndimage import label")
                    lines.append("    # Implementation depends on fill algorithm")
                else:
                    lines.append(f"    grid.fill({color})")

            elif op_type == 'mask':
                pattern = params.get('pattern', [])
                lines.append(f"    # Apply mask with pattern: {pattern}")
                lines.append(f"    # mask = np.array({pattern})")
                lines.append("    # grid = apply_mask(grid, mask)")

            elif op_type == 'crop':
                lines.append(f"    # Crop with params: {params}")
                if all(k in params for k in ['x', 'y', 'width', 'height']):
                    x, y = params['x'], params['y']
                    w, h = params['width'], params['height']
                    lines.append(f"    grid = grid[{y}:{y+h}, {x}:{x+w}]")

            elif op_type == 'resize':
                scale = params.get('scale', 1)
                lines.append(f"    # Resize by factor {scale}")
                lines.append("    from scipy.ndimage import zoom")
                lines.append(f"    grid = zoom(grid, {scale}, order=0)")

            else:
                # Generic operation
                lines.append(f"    # TODO: Implement {op_type} with params: {params}")
                lines.append(f"    grid = apply_{op_type}(grid, **{params})")

            lines.append("")

        lines.append("    return grid")
        lines.append("")

        # Add helper functions if needed
        if self.include_metadata and entry.fitness_score is not None:
            lines.append("")
            lines.append("# Performance metrics:")
            lines.append(f"# - Fitness score: {entry.fitness_score:.3f}")
            lines.append(f"# - Execution time: {entry.execution_time_ms:.1f}ms")

        return "\n".join(lines)

    def _export_as_json(self, entry: Any) -> str:  # ProgramCacheEntry
        """Export as JSON format."""
        data = self._program_entry_to_dict(entry)
        return json.dumps(data, indent=2, default=str)

    def _export_as_markdown(self, entry: Any) -> str:  # ProgramCacheEntry
        """Export as Markdown documentation."""
        lines = []

        # Title
        lines.append(f"# Program: {entry.program_id}")
        lines.append("")

        # Metadata section
        if self.include_metadata:
            lines.append("## Metadata")
            lines.append("")
            lines.append(f"- **Task ID**: {entry.task_id}")
            lines.append(f"- **Task Source**: {entry.task_source}")
            lines.append(f"- **Success**: {'✓' if entry.success else '✗'}")
            lines.append(f"- **Accuracy Score**: {entry.accuracy_score:.2%}")
            lines.append(f"- **Execution Time**: {entry.execution_time_ms:.1f}ms")

            if entry.generation is not None:
                lines.append(f"- **Generation**: {entry.generation}")
            if entry.mutation_type:
                lines.append(f"- **Mutation Type**: {entry.mutation_type}")
            if entry.fitness_score is not None:
                lines.append(f"- **Fitness Score**: {entry.fitness_score:.3f}")

            lines.append(f"- **Created**: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Last Accessed**: {entry.last_accessed.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Access Count**: {entry.access_count}")
            lines.append("")

        # Operations section
        lines.append("## Operations Pipeline")
        lines.append("")

        for i, op in enumerate(entry.program.operations):
            op_type = op.get('type', 'unknown')
            params = op.get('params', {})

            lines.append(f"### Step {i+1}: {op_type.title()}")
            lines.append("")

            if params:
                lines.append("**Parameters:**")
                for key, value in params.items():
                    lines.append(f"- `{key}`: `{value}`")
                lines.append("")
            else:
                lines.append("*No parameters*")
                lines.append("")

        # Genealogy section (for evolution programs)
        if entry.parents:
            lines.append("## Genealogy")
            lines.append("")
            lines.append("**Parent Programs:**")
            for parent in entry.parents:
                lines.append(f"- {parent}")
            lines.append("")

        # Code examples section
        lines.append("## Code Examples")
        lines.append("")

        lines.append("### DSL Format")
        lines.append("```dsl")
        dsl_code = self._export_as_dsl(entry)
        # Remove metadata comments for code block
        dsl_lines = dsl_code.split('\n')
        code_start = next(i for i, line in enumerate(dsl_lines) if line.startswith('PROGRAM'))
        lines.extend(dsl_lines[code_start:])
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def _program_entry_to_dict(self, entry: Any) -> dict[str, Any]:  # ProgramCacheEntry
        """Convert program entry to dictionary."""
        data = {
            'program_id': entry.program_id,
            'program_hash': entry.program_hash,
            'task_id': entry.task_id,
            'task_source': entry.task_source,
            'success': entry.success,
            'accuracy_score': entry.accuracy_score,
            'execution_time_ms': entry.execution_time_ms,
            'created_at': entry.created_at.isoformat(),
            'last_accessed': entry.last_accessed.isoformat(),
            'access_count': entry.access_count,
            'program': {
                'operations': entry.program.operations,
                'version': entry.program.version,
                'metadata': entry.program.metadata
            }
        }

        if self.include_metadata:
            if entry.generation is not None:
                data['generation'] = entry.generation
            if entry.parents:
                data['parents'] = entry.parents
            if entry.mutation_type:
                data['mutation_type'] = entry.mutation_type
            if entry.fitness_score is not None:
                data['fitness_score'] = entry.fitness_score
            if entry.metadata:
                data['metadata'] = entry.metadata

        return data

    def _get_extension(self, format: str) -> str:
        """Get file extension for format."""
        extensions = {
            'dsl': 'dsl',
            'python': 'py',
            'json': 'json',
            'markdown': 'md'
        }
        return extensions.get(format, 'txt')

    def validate_export(self, content: str, format: str) -> bool:
        """Validate exported content."""
        try:
            if format == 'json':
                json.loads(content)
                return True
            elif format == 'python':
                # Basic syntax check
                compile(content, '<export>', 'exec')
                return True
            else:
                # For DSL and markdown, just check non-empty
                return len(content.strip()) > 0
        except Exception:
            return False
