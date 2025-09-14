"""
Progressive inference timeout mechanisms and fallback strategies.

This module implements a sophisticated timeout system with multiple fallback levels
to ensure inference stays within the 7.2-minute limit while maximizing solution quality.
"""
import logging
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class FallbackLevel(Enum):
    """Progressive fallback levels for inference timeout."""
    NONE = "none"
    FAST_SAMPLING = "fast_sampling"
    DETERMINISTIC = "deterministic"
    SHORT_GENERATION = "short_generation"
    MINIMAL_OUTPUT = "minimal_output"
    INPUT_COPY = "input_copy"


@dataclass
class TimeoutConfig:
    """Configuration for progressive inference timeout."""
    # Total time limits
    total_limit_seconds: float = 432.0  # 7.2 minutes
    warning_threshold: float = 0.8  # 80% of total limit
    
    # Progressive timeout stages
    stage_1_limit: float = 0.4  # 40% of total for full quality
    stage_2_limit: float = 0.7  # 70% of total for fast sampling  
    stage_3_limit: float = 0.85  # 85% of total for deterministic
    stage_4_limit: float = 0.95  # 95% of total for short generation
    
    # Fallback parameters
    fast_sampling_temperature: float = 1.2
    deterministic_beams: int = 1
    short_max_tokens: int = 100
    minimal_max_tokens: int = 50
    
    # Monitoring
    check_interval: float = 1.0  # Check timeout every second
    enable_progressive_reduction: bool = True


@dataclass
class InferenceResult:
    """Result from progressive inference attempt."""
    output: Any
    success: bool
    fallback_level: FallbackLevel
    execution_time: float
    timeout_triggered: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProgressiveTimeoutManager:
    """Manages progressive timeout with multiple fallback strategies."""
    
    def __init__(self, config: TimeoutConfig):
        """Initialize timeout manager with configuration."""
        self.config = config
        self._start_time: Optional[float] = None
        self._timeout_triggered = False
        self._current_stage = 0
        self._interrupt_flag = threading.Event()
        
    def get_current_stage_limit(self) -> float:
        """Get the time limit for the current stage."""
        if self._start_time is None:
            return self.config.total_limit_seconds
            
        elapsed = time.time() - self._start_time
        remaining = self.config.total_limit_seconds - elapsed
        
        # Return remaining time, but respect stage limits
        stage_limits = [
            self.config.stage_1_limit * self.config.total_limit_seconds,
            self.config.stage_2_limit * self.config.total_limit_seconds,
            self.config.stage_3_limit * self.config.total_limit_seconds,
            self.config.stage_4_limit * self.config.total_limit_seconds,
        ]
        
        for i, limit in enumerate(stage_limits):
            if elapsed < limit:
                return min(remaining, limit - elapsed)
        
        # Final stage - use all remaining time
        return max(0, remaining)
    
    def get_current_fallback_level(self) -> FallbackLevel:
        """Determine current fallback level based on elapsed time."""
        if self._start_time is None:
            return FallbackLevel.NONE
            
        elapsed = time.time() - self._start_time
        total_limit = self.config.total_limit_seconds
        
        if elapsed > total_limit * self.config.stage_4_limit:
            return FallbackLevel.MINIMAL_OUTPUT
        elif elapsed > total_limit * self.config.stage_3_limit:
            return FallbackLevel.SHORT_GENERATION
        elif elapsed > total_limit * self.config.stage_2_limit:
            return FallbackLevel.DETERMINISTIC
        elif elapsed > total_limit * self.config.stage_1_limit:
            return FallbackLevel.FAST_SAMPLING
        else:
            return FallbackLevel.NONE
    
    def start_monitoring(self) -> None:
        """Start timeout monitoring."""
        self._start_time = time.time()
        self._timeout_triggered = False
        self._current_stage = 0
        self._interrupt_flag.clear()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_timeout,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop timeout monitoring."""
        self._interrupt_flag.set()
        
    def _monitor_timeout(self) -> None:
        """Monitor timeout and set interrupt flag when limit exceeded."""
        while not self._interrupt_flag.is_set():
            if self._start_time is None:
                break
                
            elapsed = time.time() - self._start_time
            
            # Check if total limit exceeded
            if elapsed >= self.config.total_limit_seconds:
                self._timeout_triggered = True
                logger.error(f"Total timeout limit ({self.config.total_limit_seconds}s) exceeded")
                break
            
            # Check if warning threshold reached
            if elapsed >= self.config.total_limit_seconds * self.config.warning_threshold:
                if not hasattr(self, '_warning_logged'):
                    logger.warning(f"Approaching timeout limit: {elapsed:.1f}s / {self.config.total_limit_seconds}s")
                    self._warning_logged = True
            
            time.sleep(self.config.check_interval)
    
    def is_timeout_triggered(self) -> bool:
        """Check if timeout has been triggered."""
        return self._timeout_triggered or (
            self._start_time is not None and 
            time.time() - self._start_time >= self.config.total_limit_seconds
        )
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since monitoring started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time before timeout."""
        if self._start_time is None:
            return self.config.total_limit_seconds
        elapsed = time.time() - self._start_time
        return max(0, self.config.total_limit_seconds - elapsed)


class ProgressiveInferenceEngine:
    """Engine for progressive inference with timeout handling and fallbacks."""
    
    def __init__(self, config: TimeoutConfig):
        """Initialize progressive inference engine."""
        self.config = config
        self.timeout_manager = ProgressiveTimeoutManager(config)
        
    def execute_with_fallbacks(
        self,
        primary_function: Callable[..., Any],
        fallback_functions: Dict[FallbackLevel, Callable[..., Any]],
        *args,
        **kwargs
    ) -> InferenceResult:
        """
        Execute function with progressive fallbacks.
        
        Args:
            primary_function: Primary inference function
            fallback_functions: Dictionary mapping fallback levels to functions
            *args: Arguments for functions
            **kwargs: Keyword arguments for functions
            
        Returns:
            InferenceResult with output and metadata
        """
        logger.info("Starting progressive inference with fallbacks")
        self.timeout_manager.start_monitoring()
        
        start_time = time.time()
        result = None
        
        try:
            # Try progressive fallback levels
            for attempt, (level, func) in enumerate([
                (FallbackLevel.NONE, primary_function),
                *fallback_functions.items()
            ]):
                if self.timeout_manager.is_timeout_triggered():
                    logger.warning("Timeout triggered, moving to emergency fallback")
                    break
                
                current_fallback = self.timeout_manager.get_current_fallback_level()
                
                # Skip levels that are too early
                if self._should_skip_level(level, current_fallback):
                    continue
                
                logger.info(f"Attempting inference with level: {level.value}")
                
                try:
                    # Execute with timeout monitoring
                    with self._execution_timeout_context():
                        if level == FallbackLevel.NONE:
                            output = func(*args, **kwargs)
                        else:
                            # Apply fallback-specific modifications to kwargs
                            modified_kwargs = self._modify_kwargs_for_fallback(level, kwargs.copy())
                            output = func(*args, **modified_kwargs)
                    
                    # Success - return result
                    execution_time = time.time() - start_time
                    result = InferenceResult(
                        output=output,
                        success=True,
                        fallback_level=level,
                        execution_time=execution_time,
                        timeout_triggered=False,
                        metadata={
                            "attempt_number": attempt + 1,
                            "elapsed_time": self.timeout_manager.get_elapsed_time(),
                            "remaining_time": self.timeout_manager.get_remaining_time(),
                        }
                    )
                    
                    logger.info(f"Inference successful with level: {level.value} ({execution_time:.2f}s)")
                    break
                    
                except TimeoutError:
                    logger.warning(f"Timeout at level: {level.value}, trying next fallback")
                    continue
                    
                except Exception as e:
                    logger.warning(f"Error at level {level.value}: {e}, trying next fallback")
                    continue
            
            # If no result yet, use emergency fallback
            if result is None:
                logger.error("All fallback levels exhausted, using emergency fallback")
                result = self._emergency_fallback(*args, **kwargs)
                
        finally:
            self.timeout_manager.stop_monitoring()
        
        if result is None:
            execution_time = time.time() - start_time
            result = InferenceResult(
                output=None,
                success=False,
                fallback_level=FallbackLevel.INPUT_COPY,
                execution_time=execution_time,
                timeout_triggered=True,
                error_message="All inference attempts failed",
                metadata={
                    "total_elapsed": execution_time,
                    "timeout_limit": self.config.total_limit_seconds,
                }
            )
        
        return result
    
    def _should_skip_level(self, level: FallbackLevel, current_fallback: FallbackLevel) -> bool:
        """Determine if a fallback level should be skipped."""
        level_order = [
            FallbackLevel.NONE,
            FallbackLevel.FAST_SAMPLING,
            FallbackLevel.DETERMINISTIC,
            FallbackLevel.SHORT_GENERATION,
            FallbackLevel.MINIMAL_OUTPUT,
            FallbackLevel.INPUT_COPY,
        ]
        
        try:
            level_idx = level_order.index(level)
            current_idx = level_order.index(current_fallback)
            return level_idx < current_idx
        except ValueError:
            return False
    
    def _modify_kwargs_for_fallback(self, level: FallbackLevel, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Modify generation kwargs based on fallback level."""
        if level == FallbackLevel.FAST_SAMPLING:
            kwargs.update({
                "do_sample": True,
                "temperature": self.config.fast_sampling_temperature,
                "top_p": 0.95,
                "top_k": 0,  # No top-k filtering for speed
            })
            
        elif level == FallbackLevel.DETERMINISTIC:
            kwargs.update({
                "do_sample": False,
                "num_beams": self.config.deterministic_beams,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            })
            
        elif level == FallbackLevel.SHORT_GENERATION:
            kwargs.update({
                "max_new_tokens": min(
                    kwargs.get("max_new_tokens", 512),
                    self.config.short_max_tokens
                ),
                "do_sample": False,
                "num_beams": 1,
            })
            
        elif level == FallbackLevel.MINIMAL_OUTPUT:
            kwargs.update({
                "max_new_tokens": min(
                    kwargs.get("max_new_tokens", 512),
                    self.config.minimal_max_tokens
                ),
                "do_sample": False,
                "num_beams": 1,
                "min_new_tokens": 1,
            })
        
        return kwargs
    
    @contextmanager
    def _execution_timeout_context(self):
        """Context manager for execution timeout."""
        # For now, just yield - timeout is managed by the monitoring thread
        # In production, could implement signal-based timeout for Unix systems
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutError("Execution interrupted by timeout")
    
    def _emergency_fallback(self, *args, **kwargs) -> InferenceResult:
        """Emergency fallback when all else fails."""
        logger.warning("Using emergency fallback - returning minimal result")
        
        execution_time = time.time() - (self.timeout_manager._start_time or time.time())
        
        # Try to extract input data for copying
        emergency_output = None
        if args and hasattr(args[0], 'test_input'):  # ARCTask
            emergency_output = args[0].test_input
        elif 'task' in kwargs and hasattr(kwargs['task'], 'test_input'):
            emergency_output = kwargs['task'].test_input
        else:
            emergency_output = "TIMEOUT_FALLBACK"
        
        return InferenceResult(
            output=emergency_output,
            success=False,
            fallback_level=FallbackLevel.INPUT_COPY,
            execution_time=execution_time,
            timeout_triggered=True,
            error_message="Emergency fallback activated",
            metadata={
                "emergency_fallback": True,
                "total_time": execution_time,
            }
        )


class OptimizedGenerationFallbacks:
    """Optimized generation functions for different fallback levels."""
    
    @staticmethod
    def fast_sampling_generate(model, tokenizer, inputs, **kwargs):
        """Fast sampling generation with high temperature."""
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
                max_new_tokens=kwargs.get("max_new_tokens", 200),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        return outputs
    
    @staticmethod
    def deterministic_generate(model, tokenizer, inputs, **kwargs):
        """Deterministic generation with greedy decoding."""
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=kwargs.get("max_new_tokens", 200),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        return outputs
    
    @staticmethod
    def short_generate(model, tokenizer, inputs, **kwargs):
        """Short generation with minimal tokens."""
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(kwargs.get("max_new_tokens", 200), 100),
                min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        return outputs
    
    @staticmethod
    def minimal_generate(model, tokenizer, inputs, **kwargs):
        """Minimal generation with very few tokens."""
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(kwargs.get("max_new_tokens", 200), 50),
                min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        return outputs


def create_default_fallback_functions(model, tokenizer) -> Dict[FallbackLevel, Callable]:
    """Create default fallback functions for a model and tokenizer."""
    fallbacks = OptimizedGenerationFallbacks()
    
    return {
        FallbackLevel.FAST_SAMPLING: lambda *args, **kwargs: fallbacks.fast_sampling_generate(
            model, tokenizer, args[0], **kwargs
        ),
        FallbackLevel.DETERMINISTIC: lambda *args, **kwargs: fallbacks.deterministic_generate(
            model, tokenizer, args[0], **kwargs
        ),
        FallbackLevel.SHORT_GENERATION: lambda *args, **kwargs: fallbacks.short_generate(
            model, tokenizer, args[0], **kwargs
        ),
        FallbackLevel.MINIMAL_OUTPUT: lambda *args, **kwargs: fallbacks.minimal_generate(
            model, tokenizer, args[0], **kwargs
        ),
    }


# Usage example and testing functions
def example_progressive_inference():
    """Example of how to use progressive inference."""
    config = TimeoutConfig(
        total_limit_seconds=60,  # 1 minute for testing
        stage_1_limit=0.4,
        stage_2_limit=0.7,
        stage_3_limit=0.85,
        stage_4_limit=0.95,
    )
    
    engine = ProgressiveInferenceEngine(config)
    
    def primary_function(input_data, **kwargs):
        """Simulate a slow primary function."""
        time.sleep(30)  # Simulate slow processing
        return "Primary result"
    
    def fast_fallback(input_data, **kwargs):
        """Simulate a faster fallback."""
        time.sleep(5)  # Simulate faster processing
        return "Fast fallback result"
    
    def minimal_fallback(input_data, **kwargs):
        """Simulate minimal fallback."""
        time.sleep(1)  # Very fast
        return "Minimal result"
    
    fallback_functions = {
        FallbackLevel.FAST_SAMPLING: fast_fallback,
        FallbackLevel.MINIMAL_OUTPUT: minimal_fallback,
    }
    
    result = engine.execute_with_fallbacks(
        primary_function,
        fallback_functions,
        "test_input"
    )
    
    print(f"Result: {result.output}")
    print(f"Success: {result.success}")
    print(f"Fallback Level: {result.fallback_level}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    return result


if __name__ == "__main__":
    # Run example
    logging.basicConfig(level=logging.INFO)
    example_progressive_inference()