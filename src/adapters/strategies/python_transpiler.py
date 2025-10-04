"""
Python transpiler for converting DSL programs to executable Python functions.

This module provides transpilation from DSL operations to optimized Python code,
with runtime safety checks, performance monitoring, and sandboxed execution.
"""

import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from src.adapters.strategies.ast_generator import ASTGenerator
from src.adapters.strategies.transpiler_templates import get_all_templates

# Import operation registry to access DSL operations
from src.infrastructure.config import TranspilerSandboxConfig


class TranspilationError(Exception):
    """Raised when transpilation fails."""
    pass


class ExecutionError(Exception):
    """Raised when generated code execution fails."""
    def __init__(self, message: str, operation_context: str | None = None):
        super().__init__(message)
        self.operation_context = operation_context


@dataclass
class TranspilationResult:
    """Result of transpiling a DSL program."""
    source_code: str
    function_name: str
    imports: list[str]
    helpers: dict[str, str]  # helper function name -> source
    program_hash: str
    estimated_memory_mb: float
    source_map: dict[int, str]  # line number -> DSL operation context


@dataclass
class ProfileData:
    """CPU profiling data."""
    enabled: bool = False
    total_calls: int = 0
    primitive_calls: int = 0
    total_time: float = 0.0
    cumulative_time: float = 0.0
    top_functions: list[tuple[str, int, float, float]] = None  # (function, calls, tottime, cumtime)
    raw_stats: Any | None = None  # Raw pstats.Stats object

    def __post_init__(self):
        if self.top_functions is None:
            self.top_functions = []


@dataclass
class MemoryAllocationData:
    """Memory allocation tracking data."""
    enabled: bool = False
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    allocation_count: int = 0
    deallocation_count: int = 0
    net_allocations: int = 0
    traced_allocations: list[tuple[str, int, float]] = None  # (filename:lineno, size_mb, timestamp)
    top_allocators: list[tuple[str, float]] = None  # (location, size_mb)

    def __post_init__(self):
        if self.traced_allocations is None:
            self.traced_allocations = []
        if self.top_allocators is None:
            self.top_allocators = []


@dataclass
class ExecutionMetrics:
    """Metrics collected during function execution."""
    execution_time_ms: float
    memory_used_mb: float
    operation_timings: dict[str, float]  # operation_name -> time_ms
    slow_operations: list[tuple[str, float]]  # operations taking >threshold ms

    # Profiling data (optional)
    profile_data: ProfileData | None = None
    memory_allocation_data: MemoryAllocationData | None = None


class PythonTranspiler:
    """Transpiles DSL programs to executable Python functions."""

    def __init__(self, enable_cache: bool = True, cache_size: int = 1000,
                 config: TranspilerSandboxConfig | None = None):
        self.config = config or TranspilerSandboxConfig()
        self.operation_templates = get_all_templates()
        self.helper_functions = self._initialize_helper_functions()
        self.ast_generator = ASTGenerator()
        self._operation_registry = self._build_operation_registry()
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache: OrderedDict[str, TranspilationResult] = OrderedDict()  # program_hash -> result (LRU)
        self._cache_hits = 0
        self._cache_misses = 0

        # Initialize profiling support
        self._profiling_enabled = (
            self.config.cpu_profiling_enabled or
            self.config.memory_tracking_enabled or
            self.config.resource_monitoring_enabled
        )

    def transpile(self, program: dict[str, Any], function_name: str = "solve_task") -> TranspilationResult:
        """
        Transpile a DSL program to Python code.

        Args:
            program: The DSL program dictionary to transpile
            function_name: Name for the generated function

        Returns:
            TranspilationResult with generated code and metadata
        """
        # Check cache if enabled
        if self.enable_cache:
            program_hash = self._compute_program_hash(program)
            if program_hash in self._cache:
                self._cache_hits += 1
                # Move to end (most recently used) for LRU ordering
                cached = self._cache[program_hash]
                self._cache.move_to_end(program_hash)

                if cached.function_name != function_name:
                    # Update function name in cached code
                    updated_code = cached.source_code.replace(
                        f"def {cached.function_name}",
                        f"def {function_name}"
                    )
                    return TranspilationResult(
                        source_code=updated_code,
                        function_name=function_name,
                        imports=cached.imports,
                        helpers=cached.helpers,
                        program_hash=program_hash,
                        estimated_memory_mb=cached.estimated_memory_mb,
                        source_map=cached.source_map
                    )
                return cached
            else:
                self._cache_misses += 1

        try:
            # Parse the program
            operations = self._parse_program(program)

            # Generate code for each operation
            code_blocks = []
            imports = {"import numpy as np", "import time", "from typing import List, Dict, Any"}
            helpers = {}
            source_map = {}
            current_line = 1  # Track line numbers

            # Add profiling imports if enabled
            if self._profiling_enabled:
                if self.config.cpu_profiling_enabled:
                    imports.add("import cProfile")
                    imports.add("import pstats")
                    imports.add("import io")
                if self.config.memory_tracking_enabled:
                    imports.add("import tracemalloc")
                if self.config.resource_monitoring_enabled:
                    imports.add("import sys")
                    if sys.platform != "win32":
                        imports.add("import resource")

            # Handle psutil import with try/except
            psutil_code = []
            if self._profiling_enabled and self.config.resource_monitoring_enabled:
                psutil_code.extend([
                    "try:",
                    "    import psutil",
                    "    HAS_PSUTIL = True",
                    "except ImportError:",
                    "    HAS_PSUTIL = False"
                ])

            # Add input validation
            validation_code = self._generate_input_validation()
            code_blocks.append(validation_code)
            current_line += validation_code.count('\n') + 1

            # Convert input to numpy array
            code_blocks.append("    # Convert to numpy array for efficient operations")
            code_blocks.append("    grid = np.array(grid, dtype=np.int32)")
            code_blocks.append("")
            current_line += 3

            # Add profiling initialization if enabled
            if self._profiling_enabled:
                profiling_init = self._generate_profiling_initialization()
                code_blocks.extend(profiling_init)
                current_line += len(profiling_init)

            # Generate code for each operation
            for i, (op_name, op_params) in enumerate(operations):
                # Track source location for this operation
                operation_context = f"{op_name} (operation #{i})"

                op_code, op_imports, op_helpers = self._transpile_operation(op_name, op_params, i)

                # Record source map entries for this operation's code
                for line in op_code:
                    if line.strip() and not line.strip().startswith('#'):
                        source_map[current_line] = operation_context
                    current_line += 1

                code_blocks.extend(op_code)
                imports.update(op_imports)
                helpers.update(op_helpers)

            # Add profiling finalization if enabled
            if self._profiling_enabled:
                profiling_final = self._generate_profiling_finalization()
                code_blocks.extend(profiling_final)

            # Convert result back to list and return
            code_blocks.append("")
            code_blocks.append("    # Convert back to list")
            code_blocks.append("    return grid.tolist()")

            # Build the complete function
            function_code = self._build_function(function_name, code_blocks)

            # Add imports and helpers
            source_code = self._build_complete_source(function_code, sorted(imports), helpers, psutil_code)

            # Compute program hash
            program_hash = self._compute_program_hash(program)

            # Estimate memory usage
            estimated_memory = self._estimate_memory_usage(program)

            result = TranspilationResult(
                source_code=source_code,
                function_name=function_name,
                imports=sorted(imports),
                helpers=helpers,
                program_hash=program_hash,
                estimated_memory_mb=estimated_memory,
                source_map=source_map
            )

            # Store in cache if enabled
            if self.enable_cache:
                self._add_to_cache(program_hash, result)

            return result

        except Exception as e:
            raise TranspilationError(f"Failed to transpile program: {str(e)}") from e

    def _parse_program(self, program: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Parse a DSL program into a list of (operation_name, parameters) tuples."""
        operations = []

        if "operations" in program:
            # Handle standard DSLProgram format
            for op in program["operations"]:
                if isinstance(op, dict):
                    op_name = op.get("type") or op.get("name") or op.get("operation")
                    params = {k: v for k, v in op.items() if k not in ["type", "name", "operation"]}
                    operations.append((op_name, params))
                else:
                    raise TranspilationError(f"Invalid operation format: {op}")
        else:
            raise TranspilationError("Program missing 'operations' field")

        return operations

    def _transpile_operation(self, op_name: str, params: dict[str, Any],
                           op_index: int) -> tuple[list[str], set, dict[str, str]]:
        """
        Transpile a single operation to Python code.

        Returns:
            (code_lines, imports, helpers)
        """
        code_lines = []
        imports = set()
        helpers = {}

        # Add operation timing
        code_lines.append(f"    # Operation {op_index}: {op_name}")
        code_lines.append("    _op_start = time.time()")

        # Add error context wrapper
        code_lines.append("    try:")

        # Find the appropriate template
        template = self._find_operation_template(op_name)

        if template:
            # Add default values for optional parameters
            template_params = params.copy()
            if op_name == "pattern_match" and "mask" not in template_params:
                template_params["mask"] = None

            # Generate code from template
            op_code = self._apply_template(template, template_params)

            # Check if the operation code needs to be assigned to grid
            needs_assignment = self._needs_grid_assignment(op_code, op_name)
            if needs_assignment:
                op_code = f"grid = {op_code}"

            # Indent the operation code
            indented_code = "\n".join(f"        {line}" if line.strip() else line
                                     for line in op_code.split("\n"))
            code_lines.append(indented_code)
        else:
            # Fallback: generate generic operation call
            code_lines.append(f"        # WARNING: No template found for {op_name}")
            code_lines.append("        # Using fallback implementation")
            code_lines.append("        grid = grid  # No-op")

        # Add error context
        code_lines.append("    except Exception as e:")
        code_lines.append(f"        e.operation_context = '{op_name} (operation #{op_index})'")
        code_lines.append("        raise")

        # Record timing
        code_lines.append(f"    _operation_timings['{op_name}_{op_index}'] = (time.time() - _op_start) * 1000")

        # Check if operation produced metadata and store it globally
        code_lines.append("    # Note: _operation_metadata is already in local scope and will be accessible")
        code_lines.append("")

        return code_lines, imports, helpers

    def _find_operation_template(self, op_name: str) -> Any | None:
        """Find the template for an operation."""
        # Search through all template categories
        for _category, templates in self.operation_templates.items():
            if op_name in templates:
                return templates[op_name]
            # Also check with underscores replaced
            normalized_name = op_name.replace("_", "").lower()
            for template_name, template in templates.items():
                if normalized_name == template_name.replace("_", "").lower():
                    return template
        return None

    def _needs_grid_assignment(self, op_code: str, op_name: str) -> bool:
        """Check if operation code needs to be assigned to grid."""
        # Check if the code already has a variable assignment
        # Look for patterns like "grid = " or "result = " at the start of lines
        lines = op_code.strip().split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped and ('=' in stripped):
                # Check if it's an actual assignment (not comparison)
                if ('==' not in stripped and '>=' not in stripped and
                    '<=' not in stripped and '!=' not in stripped):
                    # Check if it's a real assignment (has variable = expression)
                    parts = stripped.split('=', 1)
                    if len(parts) == 2 and parts[0].strip().isidentifier():
                        return False

        # If the code is multi-line with intermediate operations, check more carefully
        if len(lines) > 1:
            # Multi-line operations might handle assignment internally
            # Look for patterns that suggest the operation manages its own result
            for line in lines:
                if 'result' in line.lower() and '=' in line:
                    return False
            # If no internal result handling found, still needs assignment
            return True

        # Single expression operations need assignment
        return True

    def _apply_template(self, template: Any, params: dict[str, Any]) -> str:
        """Apply parameters to a template."""
        if isinstance(template, str):
            # Simple string template - handle missing params gracefully
            try:
                return template.format(input="grid", **params)
            except KeyError:
                # Return template with available params
                format_dict = {"input": "grid"}
                format_dict.update(params)
                # Use safe substitution
                import string
                temp = string.Template(template.replace("{", "${").replace("}", "}"))
                try:
                    return temp.safe_substitute(**format_dict)
                except Exception:
                    return template.format(input="grid", **{k: v for k, v in params.items() if "{" + k + "}" in template})
        elif isinstance(template, dict):
            # Template with parameter-based selection
            # Check params for matching keys
            for param_key, param_value in params.items():
                if param_value in template:
                    return template[param_value].format(input="grid", **params)
                elif param_key in template:
                    return template[param_key].format(input="grid", **params)

            # Check numeric angle values for rotate
            if "angle" in params and params["angle"] in template:
                return template[params["angle"]].format(input="grid", **params)

            # Default to first template
            return list(template.values())[0].format(input="grid", **params)
        else:
            return "grid"  # No-op

    def _initialize_operation_templates(self) -> dict[str, str]:
        """Initialize code templates for each operation type."""
        return get_all_templates()

    def _initialize_helper_functions(self) -> dict[str, str]:
        """Initialize reusable helper functions."""
        helpers = {
            "validate_grid": f"""
def validate_grid(grid):
    '''Validate grid dimensions and values.'''
    if not grid or not grid[0]:
        raise ValueError("Empty grid")
    h, w = len(grid), len(grid[0])
    if h > {self.config.max_grid_height} or w > {self.config.max_grid_width}:
        raise ValueError(f"Grid too large: {{h}}x{{w}} (max: {self.config.max_grid_height}x{self.config.max_grid_width})")
    for row in grid:
        if len(row) != w:
            raise ValueError("Inconsistent row widths")
        for val in row:
            if not isinstance(val, (int, np.integer)) or val < 0 or val > 9:
                raise ValueError(f"Invalid grid value: {{val}}")
""",
            "safe_bounds": """
def safe_bounds(y, x, h, w):
    '''Check if coordinates are within bounds.'''
    return 0 <= y < h and 0 <= x < w
"""
        }
        return helpers

    def _generate_function_signature(self, function_name: str) -> str:
        """Generate the function signature with type hints."""
        return f"def {function_name}(grid: List[List[int]]) -> List[List[int]]:"

    def _generate_input_validation(self) -> str:
        """Generate input validation code."""
        return f"""    # Validate input
    if not grid or not grid[0]:
        raise ValueError("Empty grid provided")
    if len(grid) > {self.config.max_grid_height} or len(grid[0]) > {self.config.max_grid_width}:
        raise ValueError(f"Grid dimensions exceed {self.config.max_grid_height}x{self.config.max_grid_width}: {{len(grid)}}x{{len(grid[0])}}")"""

    def _generate_bounds_check(self, var_name: str) -> str:
        """Generate bounds checking code for a grid variable."""
        return f"""
    if {var_name}.shape[0] > {self.config.max_grid_height} or {var_name}.shape[1] > {self.config.max_grid_width}:
        raise ValueError("Operation resulted in grid larger than {self.config.max_grid_height}x{self.config.max_grid_width}")
"""

    def _build_function(self, function_name: str, code_blocks: list[str]) -> str:
        """Build the complete function from code blocks."""
        signature = self._generate_function_signature(function_name)
        body = "\n".join(code_blocks)
        return f"{signature}\n{body}"

    def _build_complete_source(self, function_code: str, imports: list[str],
                              helpers: dict[str, str], psutil_code: list[str] = None) -> str:
        """Build the complete source code with imports and helpers."""
        parts = []

        # Add imports
        parts.extend(imports)
        parts.append("")

        # Add psutil try/except block if needed
        if psutil_code:
            parts.extend(psutil_code)
            parts.append("")

        # Add ExecutionError class
        parts.append("class ExecutionError(Exception):")
        parts.append("    def __init__(self, message, operation_context=None):")
        parts.append("        super().__init__(message)")
        parts.append("        self.operation_context = operation_context")
        parts.append("")

        # Add global metadata variables
        parts.append("# Global variables for operation data")
        parts.append("_operation_timings = {}")
        parts.append("_operation_metadata = {}")
        parts.append("")

        # Add helper functions
        for helper_code in helpers.values():
            parts.append(helper_code)
            parts.append("")

        # Add main function
        parts.append(function_code)

        return "\n".join(parts)

    def _estimate_memory_usage(self, program: dict[str, Any], max_grid_size: int | None = None) -> float:
        """Estimate memory usage for the program in MB."""
        if max_grid_size is None:
            max_grid_size = max(self.config.max_grid_height, self.config.max_grid_width)

        # Base memory for a max_grid_size x max_grid_size int32 grid
        base_memory = (max_grid_size * max_grid_size * 4) / (1024 * 1024)  # 4 bytes per int32

        # Add overhead for intermediate operations
        num_operations = len(program.get("operations", []))
        operation_overhead = num_operations * base_memory * self.config.max_operation_memory_overhead_factor

        # Add Python overhead
        python_overhead = 5.0  # MB

        return base_memory + operation_overhead + python_overhead

    def _compute_program_hash(self, program: dict[str, Any]) -> str:
        """Compute a hash for the program for caching."""
        # Serialize program to JSON for consistent hashing
        program_str = json.dumps(program, sort_keys=True)
        return hashlib.sha256(program_str.encode()).hexdigest()[:16]

    def _build_operation_registry(self) -> dict[str, type]:
        """Build a registry of available operations."""
        # Import all DSL operations
        from src.domain.dsl.color import (
            ColorFilterOperation,
            ColorInvertOperation,
            ColorMapOperation,
            ColorReplaceOperation,
            ColorThresholdOperation,
        )
        from src.domain.dsl.composition import CropOperation as CompositionCropOperation
        from src.domain.dsl.composition import OverlayOperation
        from src.domain.dsl.composition import PadOperation as CompositionPadOperation
        from src.domain.dsl.connectivity import (
            ConnectedComponentsOperation,
            FilterComponentsOperation,
        )
        from src.domain.dsl.edges import (
            BoundaryTracingOperation,
            ContourExtractionOperation,
            EdgeDetectionOperation,
        )
        from src.domain.dsl.geometric import CropOperation as GeometricCropOperation
        from src.domain.dsl.geometric import FlipOperation, RotateOperation, TranslateOperation
        from src.domain.dsl.geometric import PadOperation as GeometricPadOperation
        from src.domain.dsl.pattern import (
            FloodFillOperation,
            PatternFillOperation,
            PatternMatchOperation,
            PatternReplaceOperation,
        )
        from src.domain.dsl.symmetry import CreateSymmetryOperation

        # Build registry mapping operation names to their classes
        registry = {
            # Geometric operations
            "rotate": RotateOperation,
            "flip": FlipOperation,
            "translate": TranslateOperation,
            "geometric_crop": GeometricCropOperation,
            "geometric_pad": GeometricPadOperation,

            # Color operations
            "color_map": ColorMapOperation,
            "color_filter": ColorFilterOperation,
            "color_replace": ColorReplaceOperation,
            "color_invert": ColorInvertOperation,
            "color_threshold": ColorThresholdOperation,

            # Pattern operations
            "pattern_fill": PatternFillOperation,
            "pattern_match": PatternMatchOperation,
            "pattern_replace": PatternReplaceOperation,
            "flood_fill": FloodFillOperation,

            # Composition operations
            "crop": CompositionCropOperation,
            "pad": CompositionPadOperation,
            "overlay": OverlayOperation,

            # Connectivity operations
            "connected_components": ConnectedComponentsOperation,
            "filter_components": FilterComponentsOperation,

            # Edge operations
            "edge_detection": EdgeDetectionOperation,
            "boundary_tracing": BoundaryTracingOperation,
            "contour_extraction": ContourExtractionOperation,

            # Symmetry operations
            "create_symmetry": CreateSymmetryOperation,
        }

        return registry

    def _add_to_cache(self, program_hash: str, result: TranspilationResult):
        """Add a transpilation result to the cache, with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # LRU eviction - remove least recently used entry (first item in OrderedDict)
            self._cache.popitem(last=False)

        # Add new entry (will be placed at the end, marking it as most recently used)
        self._cache[program_hash] = result

    def clear_cache(self):
        """Clear the transpilation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "enabled": self.enable_cache
        }

    def _generate_profiling_initialization(self) -> list[str]:
        """Generate profiling initialization code."""
        code_lines = []

        if not self._profiling_enabled:
            return code_lines

        code_lines.append("    # Profiling initialization")

        if self.config.resource_monitoring_enabled:
            code_lines.append("    _start_time = time.time()")
            code_lines.append("    _initial_memory = 0.0")
            code_lines.append("    try:")
            code_lines.append("        if 'psutil' in globals() and HAS_PSUTIL:")
            code_lines.append("            import psutil")
            code_lines.append("            _initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)")
            code_lines.append("        else:")
            code_lines.append("            import sys")
            code_lines.append("            if sys.platform != 'win32':")
            code_lines.append("                import resource")
            code_lines.append("                _initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024")
            code_lines.append("            else:")
            code_lines.append("                _initial_memory = 0.0")
            code_lines.append("    except Exception as e:")
            code_lines.append("        _initial_memory = 0.0")

        if self.config.cpu_profiling_enabled:
            code_lines.append("    _profiler = cProfile.Profile()")
            code_lines.append("    _profiler.enable()")

        if self.config.memory_tracking_enabled:
            code_lines.append("    _memory_start_snapshot = None")
            code_lines.append("    try:")
            code_lines.append("        if not tracemalloc.is_tracing():")
            code_lines.append("            tracemalloc.start()")
            code_lines.append("        _memory_start_snapshot = tracemalloc.take_snapshot()")
            code_lines.append("    except Exception as e:")
            code_lines.append("        pass")

        code_lines.append("")
        return code_lines

    def _generate_profiling_finalization(self) -> list[str]:
        """Generate profiling finalization code."""
        code_lines = []

        if not self._profiling_enabled:
            return code_lines

        code_lines.append("")
        code_lines.append("    # Profiling finalization")

        if self.config.cpu_profiling_enabled:
            code_lines.append("    try:")
            code_lines.append("        _profiler.disable()")
            code_lines.append("        # Store profiler in global for retrieval")
            code_lines.append("        globals()['_last_profiler'] = _profiler")
            code_lines.append("    except Exception as e:")
            code_lines.append("        pass")

        if self.config.memory_tracking_enabled:
            code_lines.append("    try:")
            code_lines.append("        if _memory_start_snapshot and tracemalloc.is_tracing():")
            code_lines.append("            _memory_current_snapshot = tracemalloc.take_snapshot()")
            code_lines.append("            globals()['_memory_snapshots'] = (_memory_start_snapshot, _memory_current_snapshot)")
            code_lines.append("    except Exception as e:")
            code_lines.append("        pass")

        if self.config.resource_monitoring_enabled:
            code_lines.append("    try:")
            code_lines.append("        _end_time = time.time()")
            code_lines.append("        _execution_duration = _end_time - _start_time")
            code_lines.append("        _final_memory = 0.0")
            code_lines.append("        if 'psutil' in globals() and HAS_PSUTIL:")
            code_lines.append("            _final_memory = psutil.Process().memory_info().rss / (1024 * 1024)")
            code_lines.append("        else:")
            code_lines.append("            import sys")
            code_lines.append("            if sys.platform != 'win32':")
            code_lines.append("                import resource")
            code_lines.append("                _final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024")
            code_lines.append("        globals()['_profiling_metrics'] = {")
            code_lines.append("            'execution_duration': _execution_duration,")
            code_lines.append("            'initial_memory': _initial_memory,")
            code_lines.append("            'final_memory': _final_memory,")
            code_lines.append("            'memory_delta': _final_memory - _initial_memory")
            code_lines.append("        }")
            code_lines.append("    except Exception as e:")
            code_lines.append("        pass")

        return code_lines
