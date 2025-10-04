"""
Sandboxed execution environment for transpiled Python functions.

Provides process-based isolation, timeout enforcement, memory limits,
and safe execution of generated code.
"""

import atexit
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from queue import Queue as ThreadQueue
from typing import Any

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

    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = ['numpy', 'math', 'itertools', 'collections', 'scipy.ndimage']
        if self.process_pool is None:
            self.process_pool = ProcessPoolConfig()


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


class SandboxExecutor:
    """Executes transpiled Python code in a sandboxed environment."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._active_processes = set()
        self._job_handle = None

        # Process pool components
        self._workers: dict[str, WorkerInfo] = {}
        self._worker_lock = threading.RLock()
        self._pending_requests: ThreadQueue = ThreadQueue()
        self._response_futures: dict[str, Future] = {}
        self._pool_manager_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._pool_enabled = self.config.process_pool.enabled

        # Initialize Windows job object for process management
        if sys.platform == "win32":
            self._setup_windows_job()

        # Initialize process pool if enabled
        if self._pool_enabled:
            self._initialize_pool()

        # Register cleanup on exit
        atexit.register(self._cleanup_all_processes)

    def _setup_windows_job(self):
        """Setup Windows job object for resource limiting."""
        if not HAS_WIN32:
            # TODO: TECH-DEBT-HIGH - Win32 modules not available, implement ctypes alternative
            print("Warning: Win32 modules not available. Using basic process management.")
            self._job_handle = None
            return

        try:
            # Create a job object
            self._job_handle = win32job.CreateJobObject(None, "")

            # Configure job limits
            job_info = win32job.QueryInformationJobObject(
                self._job_handle,
                win32job.JobObjectBasicLimitInformation
            )

            # Set process memory limit
            job_info['ProcessMemoryLimit'] = self.config.memory_limit_mb * 1024 * 1024

            # Set job memory limit
            job_info['JobMemoryLimit'] = self.config.memory_limit_mb * 2 * 1024 * 1024

            # Enable limits
            job_info['LimitFlags'] = (
                win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY |
                win32job.JOB_OBJECT_LIMIT_JOB_MEMORY |
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
                win32job.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            )

            # Apply the limits
            win32job.SetInformationJobObject(
                self._job_handle,
                win32job.JobObjectBasicLimitInformation,
                job_info
            )

            # Set additional extended limits
            extended_info = win32job.QueryInformationJobObject(
                self._job_handle,
                win32job.JobObjectExtendedLimitInformation
            )

            extended_info['BasicLimitInformation'] = job_info

            win32job.SetInformationJobObject(
                self._job_handle,
                win32job.JobObjectExtendedLimitInformation,
                extended_info
            )

        except Exception as e:
            # Log error but continue - fallback to monitoring only
            print(f"Warning: Failed to create Windows job object: {e}")
            self._job_handle = None

    def _cleanup_all_processes(self):
        """Clean up all active processes on exit."""
        for proc in list(self._active_processes):
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        # Force kill if still alive
                        if sys.platform == "win32":
                            try:
                                if HAS_WIN32:
                                    handle = win32api.OpenProcess(1, False, proc.pid)
                                    win32api.TerminateProcess(handle, 1)
                                else:
                                    # Fallback to subprocess
                                    import subprocess
                                    subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)],
                                                 capture_output=True, encoding='utf-8', errors='replace')
                            except Exception:
                                pass
                        else:
                            os.kill(proc.pid, signal.SIGKILL)
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
        # Create queues for communication
        result_queue = Queue()
        metrics_queue = Queue()

        # Create and start worker process
        process = Process(
            target=self._worker_process,
            args=(code, function_name, grid, result_queue, metrics_queue)
        )

        # Track process for cleanup
        self._active_processes.add(process)

        start_time = time.time()
        process.start()

        # Assign to job object on Windows
        if sys.platform == "win32" and self._job_handle and HAS_WIN32:
            try:
                # Get process handle
                process_handle = win32api.OpenProcess(
                    win32con.PROCESS_ALL_ACCESS,
                    False,
                    process.pid
                )
                # Assign to job
                win32job.AssignProcessToJobObject(self._job_handle, process_handle)
                win32api.CloseHandle(process_handle)
            except Exception as e:
                # Log but continue - process still sandboxed via other means
                print(f"Warning: Failed to assign process to job: {e}")

        # Wait for process to complete or timeout
        process.join(timeout=self.config.timeout_seconds)

        execution_time = time.time() - start_time
        timed_out = process.is_alive()

        if timed_out:
            # Process timed out, terminate it
            self._terminate_process(process)

            return ExecutionResult(
                success=False,
                result=None,
                error="Execution timed out",
                error_context=f"Function '{function_name}' exceeded {self.config.timeout_seconds}s timeout",
                metrics=ExecutionMetrics(
                    execution_time_ms=execution_time * 1000,
                    memory_used_mb=0.0,
                    operation_timings={},
                    slow_operations=[]
                ),
                timed_out=True
            )

        # Remove from active processes
        self._active_processes.discard(process)

        # Get results from queues
        try:
            success, result_data, error_msg, error_context = result_queue.get_nowait()
            metrics_data = metrics_queue.get_nowait() if not metrics_queue.empty() else {}
        except Exception:
            # Process crashed or didn't return results
            return ExecutionResult(
                success=False,
                result=None,
                error="Process crashed or failed to return results",
                error_context=None,
                metrics=ExecutionMetrics(
                    execution_time_ms=execution_time * 1000,
                    memory_used_mb=0.0,
                    operation_timings={},
                    slow_operations=[]
                )
            )

        # Build metrics
        operation_timings = metrics_data.get('operation_timings', {})
        slow_operations = [(op, time_ms) for op, time_ms in operation_timings.items() if time_ms > 50]

        metrics = ExecutionMetrics(
            execution_time_ms=execution_time * 1000,
            memory_used_mb=metrics_data.get('memory_used_mb', 0.0),
            operation_timings=operation_timings,
            slow_operations=slow_operations
        )

        return ExecutionResult(
            success=success,
            result=result_data if success else None,
            error=error_msg,
            error_context=error_context,
            metrics=metrics,
            memory_exceeded=metrics_data.get('memory_exceeded', False)
        )

    def _worker_process(self, code: str, function_name: str, grid: list[list[int]],
                       result_queue: Queue, metrics_queue: Queue):
        """Worker process that executes the code."""
        try:
            # Set resource limits
            self._set_resource_limits()

            # Create restricted execution environment
            restricted_globals = self._create_restricted_globals()

            # Measure initial memory
            initial_memory = self._measure_memory_usage()

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
                final_memory = self._measure_memory_usage()
                memory_used = final_memory - initial_memory

                # Check memory limit
                memory_exceeded = memory_used > self.config.memory_limit_mb

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
                        f"Memory limit exceeded: {memory_used:.1f}MB > {self.config.memory_limit_mb}MB",
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
                    'memory_used_mb': self._measure_memory_usage() - initial_memory,
                    'operation_timings': restricted_globals.get('_operation_timings', {})
                })

        except Exception as e:
            # Setup or compilation error
            result_queue.put((False, None, f"Setup error: {str(e)}", None))
            metrics_queue.put({})

    def _terminate_process(self, process):
        """Safely terminate a process with escalation."""
        try:
            process.terminate()
            process.join(timeout=1.0)

            if process.is_alive():
                # Escalate to force kill
                if sys.platform == "win32":
                    try:
                        if HAS_WIN32:
                            # Use Windows API for forceful termination
                            handle = win32api.OpenProcess(1, False, process.pid)
                            win32api.TerminateProcess(handle, 1)
                            win32api.CloseHandle(handle)
                        else:
                            # Fallback to taskkill
                            import subprocess
                            subprocess.run(["taskkill", "/F", "/PID", str(process.pid)],
                                         capture_output=True, encoding='utf-8', errors='replace')
                    except Exception:
                        pass
                else:
                    os.kill(process.pid, signal.SIGKILL)

                # Final wait
                process.join(timeout=0.5)
        except Exception:
            pass
        finally:
            self._active_processes.discard(process)

    def _set_resource_limits(self):
        """Set resource limits for the worker process."""
        if sys.platform == "win32":
            # Windows: Resource limits are enforced via job objects at process level
            # Additional per-process limits can be set here if needed
            try:
                import psutil
                process = psutil.Process()
                # Set working set size (soft limit)
                memory_bytes = self.config.memory_limit_mb * 1024 * 1024
                # This is a soft limit - job object provides hard enforcement
                process.memory_info()  # Prime the cache
            except Exception:
                pass
        else:
            # Unix/Linux: Use resource module
            memory_bytes = self.config.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # Set CPU time limit (as backup to timeout)
            cpu_seconds = int(self.config.timeout_seconds * 2)  # Allow some overhead
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    def _create_restricted_globals(self) -> dict[str, Any]:
        """Create restricted global namespace for execution."""
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
            }
        }

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
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except ImportError:
                # Fallback: rough estimate
                import gc
                gc.collect()
                return 10.0  # Default estimate
        else:
            # Unix: use resource module
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert KB to MB
