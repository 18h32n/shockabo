"""
Sandboxed execution environment with process pooling for transpiled Python functions.

Provides process-based isolation, timeout enforcement, memory limits,
safe execution of generated code, and process pooling for improved performance.
"""

import atexit
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import uuid
from concurrent.futures import Future
from enum import Enum
from queue import Empty as QueueEmpty
from queue import Queue as ThreadQueue

# Platform-specific imports
if sys.platform == "win32":
    try:
        import win32api
        import win32con
        import win32job
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        # Fallback to ctypes for basic Windows operations
else:
    import resource

from src.adapters.strategies.python_transpiler import ExecutionMetrics
from src.infrastructure.config import TranspilerSandboxConfig


def worker_main_loop(worker_id: str, request_queue: Queue, response_queue: Queue):
    """Main loop for worker processes (standalone function)."""
    try:
        # Set basic resource limits
        set_worker_resource_limits()

        while True:
            try:
                # Wait for request
                request = request_queue.get(timeout=5.0)

                if request is None:  # Shutdown signal
                    break

                # Execute the request
                response = execute_request_in_worker(request)

                # Clear any state between executions
                clear_worker_state()

                # Send response
                response_queue.put(response, timeout=1.0)

            except Exception:
                # Timeout or error, continue waiting
                pass

    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    finally:
        # Cleanup
        try:
            response_queue.put(None)  # Signal death
        except Exception:
            pass


def set_worker_resource_limits():
    """Set resource limits for worker processes."""
    # Default limits for worker processes
    memory_limit_mb = 100
    timeout_seconds = 2.0

    if sys.platform == "win32":
        # Windows: Basic limits
        try:
            if HAS_PSUTIL:
                import psutil
                process = psutil.Process()
                process.memory_info()  # Prime the cache
        except Exception:
            pass
    else:
        # Unix/Linux: Use resource module
        import resource
        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # Set CPU time limit (as backup to timeout)
        cpu_seconds = int(timeout_seconds * 2)  # Allow some overhead
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def execute_request_in_worker(request: 'ExecutionRequest') -> dict[str, Any]:
    """Execute a request within a worker process."""
    try:
        # Create restricted execution environment
        restricted_globals = create_restricted_globals_for_request(request)

        # Measure initial memory
        initial_memory = measure_memory_usage()
        start_time = time.time()

        # Simple timeout implementation for all platforms
        # Note: For process pooling, external timeout handling is more reliable

        # Execute the code to define the function
        exec(request.code, restricted_globals, restricted_globals)

        # Check if function was defined
        if request.function_name not in restricted_globals:
            return {
                'success': False,
                'result': None,
                'error': f"Function '{request.function_name}' not found in code",
                'error_context': None,
                'metrics': create_error_metrics(time.time() - start_time),
                'timed_out': False,
                'memory_exceeded': False
            }

        # Get the function
        func = restricted_globals[request.function_name]

        # Check for timeout before execution
        if time.time() - start_time > request.timeout:
            return {
                'success': False,
                'result': None,
                'error': f"Execution timed out after {request.timeout}s",
                'error_context': request.function_name,
                'metrics': create_error_metrics(time.time() - start_time),
                'timed_out': True,
                'memory_exceeded': False,
                'metadata': {}
            }

        # Execute the function
        result = func(request.grid)

        # Measure final memory and time
        execution_time = time.time() - start_time
        final_memory = measure_memory_usage()
        memory_used = final_memory - initial_memory

        # Check memory limit
        memory_exceeded = memory_used > request.memory_limit

        # Get operation timings if available
        operation_timings = restricted_globals.get('_operation_timings', {})

        # Get operation metadata if available
        operation_metadata = restricted_globals.get('_operation_metadata', {})

        # Build metrics
        metrics = ExecutionMetrics(
            execution_time_ms=execution_time * 1000,
            memory_used_mb=memory_used,
            operation_timings=operation_timings,
            slow_operations=[(op, time_ms) for op, time_ms in operation_timings.items()
                            if time_ms > request.transpiler_config.slow_operation_threshold_ms],
            profile_data=None,  # Not available in worker process mode
            memory_allocation_data=None  # Not available in worker process mode
        )

        if memory_exceeded:
            return {
                'success': False,
                'result': None,
                'error': f"Memory limit exceeded: {memory_used:.1f}MB > {request.memory_limit}MB",
                'error_context': request.function_name,
                'metrics': metrics,
                'timed_out': False,
                'memory_exceeded': True,
                'metadata': operation_metadata
            }
        else:
            return {
                'success': True,
                'result': result,
                'error': None,
                'error_context': None,
                'metrics': metrics,
                'timed_out': False,
                'memory_exceeded': False,
                'metadata': operation_metadata
            }

    except TimeoutError as e:
        # Timeout occurred
        execution_time = time.time() - start_time if 'start_time' in locals() else request.timeout
        return {
            'success': False,
            'result': None,
            'error': str(e),
            'error_context': request.function_name,
            'metrics': create_error_metrics(execution_time),
            'timed_out': True,
            'memory_exceeded': False,
            'metadata': {}
        }
    except Exception as e:
        # Execution error
        error_msg = str(e)
        error_context = getattr(e, 'operation_context', request.function_name)

        return {
            'success': False,
            'result': None,
            'error': error_msg,
            'error_context': error_context,
            'metrics': create_error_metrics(time.time() - start_time if 'start_time' in locals() else 0),
            'timed_out': False,
            'memory_exceeded': False,
            'metadata': None
        }


def clear_worker_state():
    """Clear any residual state between executions to ensure isolation."""
    import gc

    # Force garbage collection
    gc.collect()


def create_restricted_globals_for_request(request: 'ExecutionRequest') -> dict[str, Any]:
    """Create restricted global namespace for execution based on request."""
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
            'NameError': NameError,
            '__import__': restricted_import,  # Allow restricted imports
            '__build_class__': __builtins__['__build_class__'],  # Allow class creation
            '__name__': '__main__',  # Provide module name
        }
    }

    # Add pre-imported modules to globals
    allowed_globals.update(restricted_modules)
    allowed_globals['HAS_PSUTIL'] = HAS_PSUTIL

    # Add allowed imports
    for module_name in request.allowed_imports:
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
    import time
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


def create_error_metrics(execution_time: float) -> 'ExecutionMetrics':
    """Create error metrics for failed executions."""
    return ExecutionMetrics(
        execution_time_ms=execution_time * 1000,
        memory_used_mb=0.0,
        operation_timings={},
        slow_operations=[],
        profile_data=None,
        memory_allocation_data=None
    )


def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    import sys

    if sys.platform == "win32":
        # Windows: use psutil if available
        try:
            if HAS_PSUTIL:
                import psutil
                process = psutil.Process()
                # Use working set size which includes all memory allocations
                memory_info = process.memory_info()
                return memory_info.rss / (1024 * 1024)  # RSS in MB
            else:
                # Fallback: try to estimate using gc and sys
                import gc
                import sys
                gc.collect()
                # Try to get some reasonable estimate
                try:
                    # Count objects in memory as rough estimate
                    total_objects = len(gc.get_objects())
                    return max(10.0, total_objects * 0.001)  # Rough heuristic
                except Exception:
                    return 15.0  # Conservative estimate
        except Exception:
            # Fallback: conservative estimate
            return 15.0
    else:
        # Unix: use resource module
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux, maxrss is in KB, on macOS it might be in bytes
        maxrss = usage.ru_maxrss
        if sys.platform == 'darwin':
            return maxrss / (1024 * 1024)  # macOS: bytes to MB
        else:
            return maxrss / 1024  # Linux: KB to MB


def standalone_worker_process(code: str, function_name: str, grid: list[list[int]],
                             result_queue: Queue, metrics_queue: Queue,
                             memory_limit_mb: int, allowed_imports: list[str]):
    """Standalone worker process that executes the code."""
    try:
        # Set resource limits
        set_worker_resource_limits()

        # Create restricted execution environment
        restricted_globals = create_restricted_globals_for_allowed_imports(allowed_imports)

        # Measure initial memory
        initial_memory = measure_memory_usage()

        # Execute the code to define the function
        exec(code, restricted_globals, restricted_globals)

        # Check if function was defined
        if function_name not in restricted_globals:
            result_queue.put((False, None, f"Function '{function_name}' not found in code", None))
            return

        # Get the function
        func = restricted_globals[function_name]

        # Execute the function
        try:
            result = func(grid)

            # Measure final memory
            final_memory = measure_memory_usage()
            memory_used = final_memory - initial_memory

            # Check memory limit
            memory_exceeded = memory_used > memory_limit_mb

            # Get operation timings if available
            operation_timings = restricted_globals.get('_operation_timings', {})

            # Send metrics
            metrics_queue.put({
                'memory_used_mb': memory_used,
                'memory_exceeded': memory_exceeded,
                'operation_timings': operation_timings
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

            # Send partial metrics
            metrics_queue.put({
                'memory_used_mb': measure_memory_usage() - initial_memory,
                'operation_timings': restricted_globals.get('_operation_timings', {})
            })

    except Exception as e:
        # Setup or compilation error
        result_queue.put((False, None, f"Setup error: {str(e)}", None))
        metrics_queue.put({'metadata': {}})


def create_restricted_globals_for_allowed_imports(allowed_imports: list[str]) -> dict[str, Any]:
    """Create restricted global namespace for execution based on allowed imports."""
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
            'NameError': NameError,
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
    import time
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


class WorkerState(Enum):
    """Worker process states."""
    IDLE = "idle"
    BUSY = "busy"
    DEAD = "dead"
    STARTING = "starting"


@dataclass
class ProcessPoolConfig:
    """Configuration for the process pool."""
    pool_size: int = 4  # Number of worker processes
    enabled: bool = True  # Enable/disable pooling
    max_executions_per_worker: int = 100  # Replace worker after N executions
    worker_timeout_seconds: float = 5.0  # Time to wait for worker to start
    idle_worker_timeout: float = 300.0  # Kill idle workers after 5 minutes


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    timeout_seconds: float = 1.0
    memory_limit_mb: int = 100
    allowed_imports: list[str] = None
    process_pool: ProcessPoolConfig = None
    transpiler_config: TranspilerSandboxConfig = None

    def __post_init__(self):
        if self.allowed_imports is None:
            # Include scipy.ndimage for connectivity operations
            self.allowed_imports = ['numpy', 'math', 'itertools', 'collections', 'scipy.ndimage']
        if self.process_pool is None:
            self.process_pool = ProcessPoolConfig()
        if self.transpiler_config is None:
            self.transpiler_config = TranspilerSandboxConfig()

        # Override defaults with transpiler config values if not explicitly set
        if self.timeout_seconds == 1.0:  # If still default value
            self.timeout_seconds = self.transpiler_config.timeout_seconds
        if self.memory_limit_mb == 100:  # If still default value
            self.memory_limit_mb = self.transpiler_config.memory_limit_mb


@dataclass
class WorkerInfo:
    """Information about a worker process."""
    process: Process
    request_queue: Queue
    response_queue: Queue
    state: WorkerState
    execution_count: int
    last_used_time: float
    worker_id: str
    pid: int | None = None


@dataclass
class ExecutionRequest:
    """Request to execute code in a worker process."""
    request_id: str
    code: str
    function_name: str
    grid: list[list[int]]
    timeout: float
    memory_limit: int
    allowed_imports: list[str]
    transpiler_config: TranspilerSandboxConfig = None

    def __post_init__(self):
        if self.transpiler_config is None:
            self.transpiler_config = TranspilerSandboxConfig()


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    success: bool
    result: list[list[int]] | None
    error: str | None
    error_context: str | None
    metrics: ExecutionMetrics
    timed_out: bool = False
    memory_exceeded: bool = False
    used_pooled_worker: bool = False  # Track if pooling was used
    metadata: dict[str, Any] | None = None  # Additional execution metadata


class SandboxExecutor:
    """Executes transpiled Python code in a sandboxed environment with process pooling."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._active_processes = set()
        self._job_handle = None

        # Process pool components (temporarily disabled due to Windows multiprocessing issues)
        self._workers: dict[str, WorkerInfo] = {}
        self._worker_lock = threading.RLock()
        self._pending_requests: ThreadQueue = ThreadQueue()
        self._pool_manager_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        # Enable process pooling with platform-specific handling
        self._pool_enabled = self.config.process_pool.enabled

        # Initialize Windows job object for process management
        # Temporarily disable job object due to compatibility issues
        # if sys.platform == "win32":
        #     self._setup_windows_job()

        # Initialize process pool if enabled
        if self._pool_enabled:
            try:
                self._initialize_pool()
            except Exception as e:
                print(f"Warning: Failed to initialize process pool: {e}. Falling back to single-process mode.")
                self._pool_enabled = False

        # Register cleanup on exit
        atexit.register(self._cleanup_all_processes)

    def _setup_windows_job(self):
        """Setup Windows job object for resource limiting."""
        if not HAS_WIN32:
            print("Warning: Win32 modules not available. Using basic process management.")
            self._job_handle = None
            return

        try:
            # Create a job object
            self._job_handle = win32job.CreateJobObject(None, "")

            # Configure job limits with basic settings
            # Using simplified configuration to avoid argument count mismatch
            basic_info = win32job.QueryInformationJobObject(
                self._job_handle,
                win32job.JobObjectBasicLimitInformation
            )

            # Enable kill on job close
            basic_info['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            win32job.SetInformationJobObject(
                self._job_handle,
                win32job.JobObjectBasicLimitInformation,
                basic_info
            )

        except Exception as e:
            print(f"Warning: Failed to create Windows job object: {e}")
            self._job_handle = None

    def _initialize_pool(self):
        """Initialize the process pool."""
        try:
            # Start the pool manager thread
            self._pool_manager_thread = threading.Thread(
                target=self._pool_manager,
                daemon=True,
                name="SandboxPoolManager"
            )
            self._pool_manager_thread.start()

            # Warm up initial workers
            for _ in range(min(2, self.config.process_pool.pool_size)):
                self._create_worker()

        except Exception as e:
            print(f"Warning: Failed to initialize process pool: {e}. Falling back to single-process mode.")
            self._pool_enabled = False

    def _pool_manager(self):
        """Main loop for managing the process pool."""
        while not self._shutdown_event.is_set():
            try:
                # Check for pending requests
                try:
                    request = self._pending_requests.get(timeout=0.1)
                    self._handle_execution_request(request)
                except QueueEmpty:
                    pass

                # Maintain pool health
                self._maintain_pool_health()

                # Brief sleep to avoid busy waiting
                time.sleep(0.01)

            except Exception as e:
                print(f"Error in pool manager: {e}")
                time.sleep(0.1)

    def _create_worker(self) -> str | None:
        """Create a new worker process."""
        with self._worker_lock:
            if len(self._workers) >= self.config.process_pool.pool_size:
                return None

            worker_id = str(uuid.uuid4())
            request_queue = Queue(maxsize=1)
            response_queue = Queue(maxsize=1)

            process = Process(
                target=worker_main_loop,
                args=(worker_id, request_queue, response_queue),
                daemon=True
            )

            worker = WorkerInfo(
                process=process,
                request_queue=request_queue,
                response_queue=response_queue,
                state=WorkerState.STARTING,
                execution_count=0,
                last_used_time=time.time(),
                worker_id=worker_id
            )

            self._workers[worker_id] = worker
            self._active_processes.add(process)

            try:
                process.start()
                worker.pid = process.pid

                # Assign to Windows job if available
                if sys.platform == "win32" and self._job_handle and HAS_WIN32:
                    try:
                        process_handle = win32api.OpenProcess(
                            win32con.PROCESS_ALL_ACCESS,
                            False,
                            process.pid
                        )
                        win32job.AssignProcessToJobObject(self._job_handle, process_handle)
                        win32api.CloseHandle(process_handle)
                    except Exception as e:
                        print(f"Warning: Failed to assign worker to job: {e}")

                # Wait for worker to be ready
                start_time = time.time()
                worker_ready = False
                while (not worker_ready and
                       time.time() - start_time < self.config.process_pool.worker_timeout_seconds):
                    time.sleep(0.1)
                    # Check if worker is responsive
                    if process.is_alive():
                        worker.state = WorkerState.IDLE
                        worker_ready = True

                if not worker_ready:
                    # Worker failed to start
                    self._remove_worker(worker_id)
                    return None

                return worker_id

            except Exception as e:
                self._remove_worker(worker_id)
                print(f"Failed to create worker: {e}")
                return None

    def _remove_worker(self, worker_id: str):
        """Remove a worker process."""
        with self._worker_lock:
            if worker_id not in self._workers:
                return

            worker = self._workers[worker_id]

            # Terminate process
            try:
                if worker.process.is_alive():
                    worker.process.terminate()
                    worker.process.join(timeout=1.0)
                    if worker.process.is_alive():
                        self._force_kill_process(worker.process)
            except Exception:
                pass
            finally:
                self._active_processes.discard(worker.process)
                del self._workers[worker_id]

    def _maintain_pool_health(self):
        """Maintain pool health by checking workers and replacing dead ones."""
        current_time = time.time()
        workers_to_remove = []

        with self._worker_lock:
            for worker_id, worker in self._workers.items():
                # Check if process is still alive
                if not worker.process.is_alive():
                    worker.state = WorkerState.DEAD
                    workers_to_remove.append(worker_id)
                    continue

                # Check if worker has been idle too long
                idle_time = current_time - worker.last_used_time
                if (worker.state == WorkerState.IDLE and
                    idle_time > self.config.process_pool.idle_worker_timeout):
                    workers_to_remove.append(worker_id)
                    continue

                # Check if worker has exceeded execution limit
                if worker.execution_count >= self.config.process_pool.max_executions_per_worker:
                    workers_to_remove.append(worker_id)
                    continue

        # Remove dead/old workers
        for worker_id in workers_to_remove:
            self._remove_worker(worker_id)

        # Ensure minimum number of workers
        current_workers = len(self._workers)
        min_workers = min(2, self.config.process_pool.pool_size)
        for _ in range(min_workers - current_workers):
            self._create_worker()

    def _get_available_worker(self) -> str | None:
        """Get an available worker or create one if needed."""
        with self._worker_lock:
            # First, look for idle workers
            for worker_id, worker in self._workers.items():
                if worker.state == WorkerState.IDLE:
                    worker.state = WorkerState.BUSY
                    worker.last_used_time = time.time()
                    return worker_id

            # If no idle workers, try to create a new one
            if len(self._workers) < self.config.process_pool.pool_size:
                worker_id = self._create_worker()
                if worker_id:
                    worker = self._workers[worker_id]
                    worker.state = WorkerState.BUSY
                    return worker_id

        return None

    def _handle_execution_request(self, request_data: tuple):
        """Handle an execution request using the pool."""
        request, future = request_data

        try:
            # Get an available worker
            worker_id = self._get_available_worker()
            if not worker_id:
                # No workers available, execute in new process
                result = self._execute_in_new_process(request)
                future.set_result(result)
                return

            with self._worker_lock:
                worker = self._workers[worker_id]

            # Send request to worker
            try:
                worker.request_queue.put(request, timeout=1.0)
            except Exception:
                # Queue full or worker dead, fall back to new process
                self._remove_worker(worker_id)
                result = self._execute_in_new_process(request)
                future.set_result(result)
                return

            # Wait for response with minimal timeout buffer
            try:
                response = worker.response_queue.get(timeout=request.timeout + 0.5)

                # Update worker stats
                with self._worker_lock:
                    if worker_id in self._workers:  # Worker might have been removed
                        worker.execution_count += 1
                        worker.state = WorkerState.IDLE
                        worker.last_used_time = time.time()

                # Convert response to ExecutionResult
                result = ExecutionResult(
                    success=response['success'],
                    result=response['result'],
                    error=response['error'],
                    error_context=response['error_context'],
                    metrics=response['metrics'],
                    timed_out=response['timed_out'],
                    memory_exceeded=response['memory_exceeded'],
                    used_pooled_worker=True,
                    metadata=response.get('metadata')
                )

                future.set_result(result)

            except Exception:
                # Worker didn't respond in time, remove it and return timeout
                self._remove_worker(worker_id)
                result = ExecutionResult(
                    success=False,
                    result=None,
                    error="Execution timed out",
                    error_context=f"Function '{request.function_name}' exceeded {request.timeout}s timeout",
                    metrics=ExecutionMetrics(
                        execution_time_ms=request.timeout * 1000,
                        memory_used_mb=0.0,
                        operation_timings={},
                        slow_operations=[],
                        profile_data=None,
                        memory_allocation_data=None
                    ),
                    timed_out=True,
                    memory_exceeded=False,
                    used_pooled_worker=True,
                    metadata=None
                )
                future.set_result(result)

        except Exception as e:
            # Something went wrong, return error
            result = ExecutionResult(
                success=False,
                result=None,
                error=f"Pool execution error: {str(e)}",
                error_context=None,
                metrics=ExecutionMetrics(
                    execution_time_ms=0.0,
                    memory_used_mb=0.0,
                    operation_timings={},
                    slow_operations=[],
                    profile_data=None,
                    memory_allocation_data=None
                ),
                timed_out=False,
                memory_exceeded=False,
                used_pooled_worker=False,
                metadata=None
            )
            future.set_result(result)



    def execute(self, code: str, function_name: str, grid: list[list[int]]) -> ExecutionResult:
        """
        Execute transpiled code in a sandboxed process.

        Args:
            code: The Python code to execute
            function_name: Name of the function to call
            grid: Input grid data

        Returns:
            ExecutionResult with output or error information
        """
        # Use pool if enabled and available
        if self._pool_enabled and hasattr(self, '_pool_manager_thread') and self._pool_manager_thread.is_alive():
            return self._execute_with_pool(code, function_name, grid)
        else:
            # Fallback to creating a new process
            request = ExecutionRequest(
                request_id=str(uuid.uuid4()),
                code=code,
                function_name=function_name,
                grid=grid,
                timeout=self.config.timeout_seconds,
                memory_limit=self.config.memory_limit_mb,
                allowed_imports=self.config.allowed_imports,
                transpiler_config=self.config.transpiler_config
            )
            return self._execute_in_new_process(request)

    def _execute_with_pool(self, code: str, function_name: str, grid: list[list[int]]) -> ExecutionResult:
        """
        Execute code using the process pool.
        """
        request = ExecutionRequest(
            request_id=str(uuid.uuid4()),
            code=code,
            function_name=function_name,
            grid=grid,
            timeout=self.config.timeout_seconds,
            memory_limit=self.config.memory_limit_mb,
            allowed_imports=self.config.allowed_imports,
            transpiler_config=self.config.transpiler_config
        )

        # Create a future for the result
        future = Future()

        # Add request to pending queue
        try:
            self._pending_requests.put((request, future), timeout=1.0)
        except Exception:
            # Queue full, fallback to new process
            return self._execute_in_new_process(request)

        # Wait for result with minimal timeout buffer
        try:
            return future.result(timeout=request.timeout + 1.0)
        except Exception:
            # Timeout or error, fallback to new process
            return self._execute_in_new_process(request)

    def _execute_in_new_process(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute request in a new process (fallback when pool is unavailable)."""
        # This is the original execution logic
        result_queue = Queue()
        metrics_queue = Queue()

        from src.adapters.strategies.sandbox_worker import execute_code_in_sandbox

        process = Process(
            target=execute_code_in_sandbox,
            args=(request.code, request.function_name, request.grid, result_queue, metrics_queue,
                  request.memory_limit, request.allowed_imports, request.timeout, request.transpiler_config)
        )

        self._active_processes.add(process)

        start_time = time.time()
        process.start()

        # Assign to job object on Windows
        if sys.platform == "win32" and self._job_handle and HAS_WIN32:
            try:
                process_handle = win32api.OpenProcess(
                    win32con.PROCESS_ALL_ACCESS,
                    False,
                    process.pid
                )
                win32job.AssignProcessToJobObject(self._job_handle, process_handle)
                win32api.CloseHandle(process_handle)
            except Exception as e:
                print(f"Warning: Failed to assign process to job: {e}")

        # Wait for process to complete or timeout
        process.join(timeout=request.timeout)

        execution_time = time.time() - start_time
        timed_out = process.is_alive()

        if timed_out:
            self._terminate_process(process)

            return ExecutionResult(
                success=False,
                result=None,
                error="Execution timed out",
                error_context=f"Function '{request.function_name}' exceeded {request.timeout}s timeout",
                metrics=ExecutionMetrics(
                    execution_time_ms=execution_time * 1000,
                    memory_used_mb=0.0,
                    operation_timings={},
                    slow_operations=[],
                    profile_data=None,
                    memory_allocation_data=None
                ),
                timed_out=True,
                used_pooled_worker=False,
                metadata=None
            )

        self._active_processes.discard(process)

        # Get results from queues
        try:
            success, result_data, error_msg, error_context = result_queue.get_nowait()
            metrics_data = metrics_queue.get_nowait() if not metrics_queue.empty() else {}
        except Exception:
            return ExecutionResult(
                success=False,
                result=None,
                error="Process crashed or failed to return results",
                error_context=None,
                metrics=ExecutionMetrics(
                    execution_time_ms=execution_time * 1000,
                    memory_used_mb=0.0,
                    operation_timings={},
                    slow_operations=[],
                    profile_data=None,
                    memory_allocation_data=None
                ),
                used_pooled_worker=False,
                metadata=None
            )

        # Build metrics
        operation_timings = metrics_data.get('operation_timings', {})
        slow_operations = [(op, time_ms) for op, time_ms in operation_timings.items()
                          if time_ms > (getattr(self, 'config', None) and
                                      getattr(self.config, 'transpiler_config', None) and
                                      self.config.transpiler_config.slow_operation_threshold_ms or 50)]

        metrics = ExecutionMetrics(
            execution_time_ms=execution_time * 1000,
            memory_used_mb=metrics_data.get('memory_used_mb', 0.0),
            operation_timings=operation_timings,
            slow_operations=slow_operations,
            profile_data=metrics_data.get('profile_data'),
            memory_allocation_data=metrics_data.get('memory_allocation_data')
        )

        return ExecutionResult(
            success=success,
            result=result_data if success else None,
            error=error_msg,
            error_context=error_context,
            metrics=metrics,
            memory_exceeded=metrics_data.get('memory_exceeded', False),
            used_pooled_worker=False,
            metadata=metrics_data.get('metadata')
        )


    def _terminate_process(self, process):
        """Safely terminate a process with escalation."""
        try:
            process.terminate()
            process.join(timeout=1.0)

            if process.is_alive():
                # Escalate to force kill
                self._force_kill_process(process)

                # Final wait
                process.join(timeout=0.5)
        except Exception:
            pass
        finally:
            self._active_processes.discard(process)

    def _force_kill_process(self, process: Process):
        """Force kill a process."""
        try:
            if sys.platform == "win32":
                if HAS_WIN32:
                    handle = win32api.OpenProcess(1, False, process.pid)
                    win32api.TerminateProcess(handle, 1)
                    win32api.CloseHandle(handle)
                else:
                    import subprocess
                    subprocess.run(["taskkill", "/F", "/PID", str(process.pid)],
                                 capture_output=True, encoding='utf-8', errors='replace')
            else:
                os.kill(process.pid, signal.SIGKILL)
        except Exception:
            pass

    def _set_resource_limits(self, memory_limit_mb: int = None, timeout_seconds: float = None):
        """Set resource limits for the worker process."""
        memory_limit = memory_limit_mb or self.config.memory_limit_mb
        timeout = timeout_seconds or self.config.timeout_seconds

        if sys.platform == "win32":
            # Windows: Resource limits are enforced via job objects at process level
            try:
                if HAS_PSUTIL:
                    process = psutil.Process()
                    # Set working set size (soft limit)
                    memory_bytes = memory_limit * 1024 * 1024
                    # This is a soft limit - job object provides hard enforcement
                    process.memory_info()  # Prime the cache
            except Exception:
                pass
        else:
            # Unix/Linux: Use resource module
            memory_bytes = memory_limit * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # Set CPU time limit (as backup to timeout)
            cpu_seconds = int(timeout * 2)  # Allow some overhead
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    def _set_worker_resource_limits(self):
        """Set resource limits for worker processes (without accessing self.config)."""
        # Default limits for worker processes
        memory_limit_mb = 100
        timeout_seconds = 2.0

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

    def _create_restricted_globals(self) -> dict[str, Any]:
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
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'IndexError': IndexError,
                '__import__': restricted_import,  # Allow restricted imports
            }
        }

        # Add pre-imported modules to globals
        allowed_globals.update(restricted_modules)
        allowed_globals['HAS_PSUTIL'] = HAS_PSUTIL

        # Add allowed imports
        for module_name in self.config.allowed_imports:
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
        import time
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

    def _measure_memory_usage(self) -> float:
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

    def get_pool_stats(self) -> dict[str, Any]:
        """Get statistics about the process pool."""
        if not self._pool_enabled:
            return {
                "pool_enabled": False,
                "total_workers": 0,
                "idle_workers": 0,
                "busy_workers": 0,
                "dead_workers": 0,
                "pending_requests": 0
            }

        with self._worker_lock:
            idle_count = sum(1 for w in self._workers.values() if w.state == WorkerState.IDLE)
            busy_count = sum(1 for w in self._workers.values() if w.state == WorkerState.BUSY)
            dead_count = sum(1 for w in self._workers.values() if w.state == WorkerState.DEAD)

            return {
                "pool_enabled": True,
                "pool_size_limit": self.config.process_pool.pool_size,
                "total_workers": len(self._workers),
                "idle_workers": idle_count,
                "busy_workers": busy_count,
                "dead_workers": dead_count,
                "pending_requests": self._pending_requests.qsize(),
                "worker_details": {
                    worker_id: {
                        "state": worker.state.value,
                        "execution_count": worker.execution_count,
                        "last_used": worker.last_used_time,
                        "pid": worker.pid
                    }
                    for worker_id, worker in self._workers.items()
                }
            }

    def _cleanup_all_processes(self):
        """Clean up all active processes on exit."""
        # Signal shutdown to pool
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()

        # Clean up pool workers first
        if hasattr(self, '_workers'):
            with self._worker_lock:
                for worker_id in list(self._workers.keys()):
                    try:
                        worker = self._workers[worker_id]
                        worker.request_queue.put(None, timeout=0.1)  # Shutdown signal
                    except Exception:
                        pass
                    self._remove_worker(worker_id)

        # Wait for pool manager thread to finish
        if hasattr(self, '_pool_manager_thread') and self._pool_manager_thread:
            self._pool_manager_thread.join(timeout=2.0)

        # Clean up remaining active processes
        for proc in list(self._active_processes):
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        self._force_kill_process(proc)
            except Exception:
                pass
            finally:
                self._active_processes.discard(proc)

        # Clean up job handle
        if sys.platform == "win32" and self._job_handle and HAS_WIN32:
            try:
                win32api.CloseHandle(self._job_handle)
            except Exception:
                pass
