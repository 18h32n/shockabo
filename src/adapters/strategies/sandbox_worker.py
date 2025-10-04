"""
Standalone worker module for sandbox execution.

This module contains the worker process functions that can be properly pickled
for multiprocessing on Windows.
"""

import sys
import time
from typing import Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Platform-specific imports
if sys.platform != "win32":
    import resource

from multiprocessing import Queue

from src.adapters.strategies.execution_profiler import create_profiling_manager
from src.infrastructure.config import TranspilerSandboxConfig


def execute_code_in_sandbox(code: str, function_name: str, grid: list[list[int]],
                          result_queue: Queue, metrics_queue: Queue,
                          memory_limit_mb: int, allowed_imports: list[str],
                          timeout_seconds: float,
                          transpiler_config: TranspilerSandboxConfig | None = None):
    """Execute code in a sandboxed subprocess with optional profiling."""
    try:
        # Initialize config if not provided
        if transpiler_config is None:
            transpiler_config = TranspilerSandboxConfig()

        # Set resource limits
        _set_resource_limits(memory_limit_mb, timeout_seconds)

        # Create profiling manager
        profiler = create_profiling_manager(
            cpu_profiling=transpiler_config.cpu_profiling_enabled,
            memory_tracking=transpiler_config.memory_tracking_enabled,
            resource_monitoring=transpiler_config.resource_monitoring_enabled,
            transpiler_config=transpiler_config
        )

        # Create restricted execution environment
        restricted_globals = _create_restricted_globals(allowed_imports)

        # Measure initial memory
        initial_memory = _measure_memory_usage()

        # Execute the code to define the function
        exec(code, restricted_globals, restricted_globals)

        # Check if function was defined
        if function_name not in restricted_globals:
            result_queue.put((False, None, f"Function '{function_name}' not found in code", None))
            return

        # Get the function
        func = restricted_globals[function_name]

        # Execute the function with profiling
        try:
            with profiler.profile_execution():
                result = func(grid)

            # Measure final memory
            final_memory = _measure_memory_usage()
            memory_used = final_memory - initial_memory

            # Check memory limit
            memory_exceeded = memory_used > memory_limit_mb

            # Get operation timings if available
            operation_timings = restricted_globals.get('_operation_timings', {})

            # Get profiling data
            profile_data = None
            memory_allocation_data = None

            if transpiler_config.cpu_profiling_enabled or transpiler_config.memory_tracking_enabled:
                try:
                    profile_data = profiler.get_profile_data()
                    memory_allocation_data = profiler.get_memory_allocation_data()
                except Exception:
                    # Profiling failed, but don't let it break execution
                    pass

            # Send metrics with profiling data
            metrics_queue.put({
                'memory_used_mb': memory_used,
                'memory_exceeded': memory_exceeded,
                'operation_timings': operation_timings,
                'profile_data': profile_data,
                'memory_allocation_data': memory_allocation_data
            })

            if memory_exceeded:
                result_queue.put((
                    False,
                    None,
                    f"Memory limit exceeded: {memory_used:.1f}MB > {memory_limit_mb}MB",
                    function_name
                ))
            else:
                result_queue.put((True, result, None, None))

        except Exception as e:
            # Execution error
            error_msg = str(e)
            error_context = getattr(e, 'operation_context', function_name)
            result_queue.put((False, None, error_msg, error_context))

            # Send partial metrics with any available profiling data
            try:
                profile_data = profiler.get_profile_data() if transpiler_config.cpu_profiling_enabled else None
                memory_allocation_data = profiler.get_memory_allocation_data() if transpiler_config.memory_tracking_enabled else None
            except Exception as e:
                profile_data = None
                memory_allocation_data = None

            metrics_queue.put({
                'memory_used_mb': _measure_memory_usage() - initial_memory,
                'operation_timings': restricted_globals.get('_operation_timings', {}),
                'profile_data': profile_data,
                'memory_allocation_data': memory_allocation_data
            })

    except Exception as e:
        # Setup or compilation error
        result_queue.put((False, None, f"Setup error: {str(e)}", None))
        metrics_queue.put({})


def _set_resource_limits(memory_limit_mb: int, timeout_seconds: float):
    """Set resource limits for the worker process."""
    if sys.platform == "win32":
        # Windows: Basic limits
        try:
            if HAS_PSUTIL:
                process = psutil.Process()
                process.memory_info()  # Prime the cache
        except Exception:
            pass
    else:
        # Unix/Linux: Use resource module
        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # Set CPU time limit (as backup to timeout)
        cpu_seconds = int(timeout_seconds * 2)  # Allow some overhead
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def _create_restricted_globals(allowed_imports: list[str]) -> dict[str, Any]:
    """Create restricted global namespace for execution."""
    # Pre-import commonly used modules that transpiled code might try to import
    restricted_modules = {}

    try:
        import psutil
        restricted_modules['psutil'] = psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False

    import sys
    restricted_modules['sys'] = sys

    if sys.platform != "win32":
        try:
            import resource
            restricted_modules['resource'] = resource
        except ImportError:
            pass

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Restricted import that only allows pre-imported modules."""
        if name in restricted_modules:
            return restricted_modules[name]
        elif name in ['typing', 'collections', 'math', 'itertools', 'numpy', 'time', 'scipy']:
            # Allow these safe modules
            import importlib
            module = importlib.import_module(name)
            restricted_modules[name] = module
            return module
        else:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")

    # Import allowed modules
    allowed_globals = {
        '__builtins__': {
            # Allow only safe built-ins
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'min': min,
            'max': max,
            'abs': abs,
            'sum': sum,
            'all': all,
            'any': any,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'isinstance': isinstance,
            'type': type,
            'print': print,  # For debugging
            'bytearray': bytearray,  # For memory tests
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'IndexError': IndexError,
            '__import__': restricted_import,  # Allow restricted imports
            '__build_class__': __builtins__['__build_class__'],  # Allow class creation
            '__name__': '__main__',  # Provide module name
        }
    }

    # Add pre-imported modules to globals
    allowed_globals.update(restricted_modules)
    allowed_globals['HAS_PSUTIL'] = HAS_PSUTIL

    # Add allowed imports
    for module_name in allowed_imports:
        if module_name == 'numpy':
            import numpy as np
            allowed_globals['np'] = np
            allowed_globals['numpy'] = np
        elif module_name == 'math':
            import math
            allowed_globals['math'] = math
        elif module_name == 'itertools':
            import itertools
            allowed_globals['itertools'] = itertools
        elif module_name == 'collections':
            import collections
            allowed_globals['collections'] = collections
        elif module_name == 'scipy.ndimage':
            import scipy.ndimage
            allowed_globals['scipy'] = {'ndimage': scipy.ndimage}

    # Add time for performance tracking
    allowed_globals['time'] = time

    # Add typing imports
    from typing import Any
    allowed_globals.update({
        'List': list,
        'Dict': dict,
        'Any': Any,
        'Tuple': tuple
    })

    return allowed_globals


def _measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    if sys.platform == "win32":
        # Windows: use psutil if available
        try:
            if HAS_PSUTIL:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            else:
                # Fallback: rough estimate
                import gc
                gc.collect()
                return 10.0  # Default estimate
        except Exception:
            # Fallback: rough estimate
            import gc
            gc.collect()
            return 10.0  # Default estimate
    else:
        # Unix: use resource module
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB
