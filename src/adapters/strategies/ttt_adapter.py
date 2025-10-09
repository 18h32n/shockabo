"""
MIT Test-Time Training (TTT) adapter for ARC tasks.

This module provides integration with the actual MIT TTT methodology,
implementing per-instance adaptation, self-consistency, and data augmentation
as described in the MIT research paper.
"""
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.domain.models import (
    ARCTask,
    ARCTaskSolution,
    ResourceUsage,
    StrategyOutput,
    StrategyType,
    TTTAdaptation,
)
from src.domain.ports.strategy import StrategyPort
from src.domain.ports.timing import TimingCoordinator
from src.infrastructure.monitoring import MetricsCollector, get_metrics_collector
from src.utils.advanced_memory_optimization import (
    MemoryOptimizationConfig,
    MemoryOptimizationLevel,
    apply_memory_optimizations,
)
from src.utils.comprehensive_error_handling import (
    CheckpointManager,
    ErrorCategory,
    ErrorContext,
    ErrorReporter,
    ErrorSeverity,
    ModelLoadingHandler,
    OutOfMemoryHandler,
    resilient_operation,
)
from src.utils.error_recovery import FallbackStrategy, RetryStrategy
from src.utils.progressive_inference import (
    FallbackLevel,
    ProgressiveInferenceEngine,
    TimeoutConfig,
)
from src.utils.ttt_batch_processor import MemoryConfig, MemoryEfficientBatchProcessor
from src.utils.ttt_data_conversion import AugmentationType
from src.utils.ttt_leave_one_out import LeaveOneOutConfig, LeaveOneOutGenerator
from src.utils.ttt_lora_optimizer import LoRAOptimizer, LoRAOptimizerConfig
from src.utils.ttt_methodology import MIT_TTTStrategy, TTTTrainingConfig
from src.utils.ttt_self_consistency import SelfConsistencyConfig, SelfConsistencyValidator

logger = logging.getLogger(__name__)


@dataclass
class TTTConfig:
    """Configuration for MIT Test-Time Training adapter."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3-8B"
    device: str = "auto"
    quantization: bool = True
    quantization_config: dict[str, Any] = None
    mixed_precision: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"

    # QLoRA specific parameters for 8B model
    lora_rank: int = 64  # Rank 64 for better capacity with 8B model
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = None

    # Training parameters
    learning_rate: float = 5e-5
    per_instance_lr: float = 1e-4
    num_epochs: int = 2
    per_instance_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 1

    # Self-consistency parameters
    permute_n: int = 1
    temperature: float = 0.0

    # Augmentation configuration
    use_basic_augmentation: bool = True
    use_size_augmentation: bool = False
    use_chain_augmentation: bool = False

    # Data processing configuration
    max_examples: int = 5  # Maximum training examples per task

    # Memory and performance (optimized for 8B model)
    max_length: int = 2048
    gradient_checkpointing: bool = True
    selective_checkpointing: bool = True
    checkpointing_layers: int = 3  # Checkpoint every 3 layers
    memory_limit_mb: float = 24576  # 24GB for 8B model
    max_training_time: float = 300.0
    use_flash_attention: bool = True
    memory_optimization_level: str = "balanced"  # conservative, balanced, aggressive

    # Inference optimization parameters
    max_inference_time: float = 432.0  # 7.2 minutes
    enable_torch_compile: bool = True
    enable_kv_cache_optimization: bool = True
    enable_static_cache: bool = True
    inference_batch_size: int = 1
    low_cpu_mem_usage: bool = True

    # Progressive inference parameters
    enable_progressive_inference: bool = True
    progressive_warning_threshold: float = 0.8  # 80% of time limit
    progressive_stage_1_limit: float = 0.4  # 40% for full quality
    progressive_stage_2_limit: float = 0.7  # 70% for fast sampling
    progressive_stage_3_limit: float = 0.85  # 85% for deterministic
    progressive_stage_4_limit: float = 0.95  # 95% for short generation

    # Paths
    checkpoint_dir: Path = Path("data/models/ttt")
    cache_dir: Path = Path("data/cache/ttt")
    config_path: Path | None = Path("configs/strategies/ttt.yaml")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TTTConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Map YAML structure to config fields
        kwargs = {}

        if 'model' in config_data:
            model_config = config_data['model']
            kwargs.update({
                'model_name': model_config.get('name', "meta-llama/Llama-3-8B"),
                'device': model_config.get('device', "auto"),
                'quantization': model_config.get('quantization', True),
                'load_in_4bit': model_config.get('load_in_4bit', True),
                'bnb_4bit_quant_type': model_config.get('bnb_4bit_quant_type', 'nf4'),
                'bnb_4bit_compute_dtype': model_config.get('bnb_4bit_compute_dtype', 'float16'),
                'use_flash_attention': model_config.get('use_flash_attention', True),
                'max_length': model_config.get('max_length', 2048)
            })

        if 'training' in config_data:
            training_config = config_data['training']
            kwargs.update({
                'learning_rate': training_config.get('learning_rate', 5e-5),
                'per_instance_lr': training_config.get('per_instance_lr', 1e-4),
                'num_epochs': training_config.get('num_epochs', 2),
                'per_instance_epochs': training_config.get('per_instance_epochs', 1),
                'batch_size': training_config.get('batch_size', 2),
                'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 1),
                'mixed_precision': training_config.get('mixed_precision', True),
                'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
                'selective_checkpointing': training_config.get('selective_checkpointing', True),
                'checkpointing_layers': training_config.get('checkpointing_layers', 3),
                'memory_limit_mb': training_config.get('memory_limit_mb', 24576),
                'max_training_time': training_config.get('max_training_time', 300.0),
                'max_examples': training_config.get('max_examples', 5),
                'memory_optimization_level': training_config.get('memory_optimization_level', 'balanced')
            })

        if 'lora' in config_data:
            lora_config = config_data['lora']
            kwargs.update({
                'lora_rank': lora_config.get('rank', 64),
                'lora_alpha': lora_config.get('alpha', 16),
                'lora_dropout': lora_config.get('dropout', 0.1),
                'lora_target_modules': lora_config.get('target_modules')
            })

        if 'inference' in config_data:
            inference_config = config_data['inference']
            kwargs.update({
                'temperature': inference_config.get('temperature', 0.0),
                'max_inference_time': inference_config.get('max_inference_time', 432.0),
                'enable_torch_compile': inference_config.get('enable_torch_compile', True),
                'enable_kv_cache_optimization': inference_config.get('enable_kv_cache_optimization', True),
                'enable_static_cache': inference_config.get('enable_static_cache', True),
                'inference_batch_size': inference_config.get('inference_batch_size', 1),
                'low_cpu_mem_usage': inference_config.get('low_cpu_mem_usage', True)
            })

        if 'adaptation' in config_data:
            adaptation_config = config_data['adaptation']
            kwargs.update({
                'use_basic_augmentation': adaptation_config.get('use_basic_augmentation', True),
                'use_size_augmentation': adaptation_config.get('use_size_augmentation', False),
                'use_chain_augmentation': adaptation_config.get('use_chain_augmentation', False),
                'permute_n': adaptation_config.get('permute_n', 1)
            })

        return cls(**kwargs)

    def __post_init__(self):
        """Initialize QLoRA quantization configuration."""
        if self.quantization_config is None:
            self.quantization_config = {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": True,  # Double quantization for better memory
            }

        if self.lora_target_modules is None:
            # Default target modules for Llama-3 8B model
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class TTTAdapter(StrategyPort):
    """MIT Test-Time Training adapter for ARC tasks."""

    def __init__(self, config: TTTConfig | None = None):
        """Initialize MIT TTT adapter with configuration."""
        self.config = config or TTTConfig()

        # Load configuration from YAML only if no config was explicitly provided
        if config is None and self.config.config_path and self.config.config_path.exists():
            self.config = TTTConfig.from_yaml(self.config.config_path)

        # Create TTT training configuration
        self.ttt_config = self._create_ttt_training_config()

        # Initialize MIT TTT strategy
        self.mit_ttt_strategy = MIT_TTTStrategy(self.ttt_config)

        # Track adaptations
        self.adaptations: dict[str, TTTAdaptation] = {}

        # Initialize model components (loaded on demand)
        self.model = None
        self.tokenizer = None
        self.optimized_model_wrapper = None

        # Setup device
        self.device = self._setup_device()

        # Initialize error handling components
        self.oom_handler = OutOfMemoryHandler(
            min_batch_size=1,
            memory_threshold_mb=22000
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.config.checkpoint_dir),
            save_interval_minutes=5  # More frequent saves for safety
        )
        self.model_loading_handler = ModelLoadingHandler()
        self.error_reporter = ErrorReporter(log_file=str(self.config.checkpoint_dir / "error_log.json"))

        # Initialize retry and fallback strategies
        self.retry_strategy = RetryStrategy(
            max_attempts=3,
            base_delay=2.0
        )

        self.training_fallback = FallbackStrategy("ttt_training")
        self.inference_fallback = FallbackStrategy("ttt_inference")

        # Setup error-specific fallbacks
        self._setup_error_fallbacks()

        # Circuit breakers for critical operations
        from src.utils.error_recovery import get_circuit_breaker
        self.model_loading_breaker = get_circuit_breaker("model_loading")
        self.training_breaker = get_circuit_breaker("training")
        self.inference_breaker = get_circuit_breaker("inference")

        # Memory optimization configuration
        self.memory_opt_config = self._create_memory_optimization_config()

        # Progressive inference configuration
        self.progressive_inference_engine = None
        if self.config.enable_progressive_inference:
            self.progressive_inference_engine = self._create_progressive_inference_engine()

        # Ensure directories exist
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize enhanced TTT components (Story 3.1)
        self.leave_one_out_generator = LeaveOneOutGenerator(
            LeaveOneOutConfig(min_examples=2, max_examples=self.config.max_examples)
        )

        self.self_consistency_validator = SelfConsistencyValidator(
            SelfConsistencyConfig(
                permute_n=self.config.permute_n,
                consensus_threshold=0.6,
                enable_geometric=True,
                enable_color_remap=False
            )
        )

        self.lora_optimizer = LoRAOptimizer(
            LoRAOptimizerConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                learning_rate=self.config.per_instance_lr,
                warmup_ratio=0.1,
                max_grad_norm=1.0,
                early_stopping_patience=3,
                target_epochs=self.config.per_instance_epochs
            )
        )

        self.batch_processor = MemoryEfficientBatchProcessor(
            MemoryConfig(
                memory_limit_mb=self.config.memory_limit_mb,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_checkpointing=self.config.gradient_checkpointing,
                checkpointing_layers=self.config.checkpointing_layers
            )
        )

        # Initialize metrics collector and timing coordinator (optional)
        self.metrics_collector: MetricsCollector = get_metrics_collector()
        self.timing_coordinator: TimingCoordinator | None = None  # Set externally if needed

        # Setup health monitoring
        from src.utils.error_recovery import get_health_monitor
        self.health_monitor = get_health_monitor("ttt_adapter")
        self._setup_health_checks()

        logger.info("Enhanced TTT components initialized: leave-one-out, self-consistency, LoRA optimizer, batch processor")

    def _create_ttt_training_config(self) -> TTTTrainingConfig:
        """Create TTT training configuration from adapter config."""
        # Determine augmentation types
        augmentation_types = []
        if self.config.use_basic_augmentation:
            augmentation_types.append(AugmentationType.BASIC)
        if self.config.use_size_augmentation:
            augmentation_types.append(AugmentationType.SIZE)
        if self.config.use_chain_augmentation:
            augmentation_types.append(AugmentationType.CHAIN)

        if not augmentation_types:
            augmentation_types = [AugmentationType.BASIC]

        return TTTTrainingConfig(
            model_name=self.config.model_name,
            device=self.config.device,
            quantization=self.config.quantization,
            mixed_precision=self.config.mixed_precision,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            per_instance_epochs=self.config.per_instance_epochs,
            per_instance_lr=self.config.per_instance_lr,
            max_training_time=self.config.max_training_time,
            permute_n=self.config.permute_n,
            temperature=self.config.temperature,
            max_sequence_length=self.config.max_length,
            gradient_checkpointing=self.config.gradient_checkpointing,
            memory_limit_mb=self.config.memory_limit_mb,
            augmentation_types=augmentation_types,
            cache_dir=self.config.cache_dir,
            checkpoint_dir=self.config.checkpoint_dir
        )

    def _setup_device(self) -> torch.device:
        """Set up computing device based on availability and configuration."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _create_memory_optimization_config(self) -> MemoryOptimizationConfig:
        """Create memory optimization configuration."""
        # Map string level to enum
        level_mapping = {
            "conservative": MemoryOptimizationLevel.CONSERVATIVE,
            "balanced": MemoryOptimizationLevel.BALANCED,
            "aggressive": MemoryOptimizationLevel.AGGRESSIVE
        }

        optimization_level = level_mapping.get(
            self.config.memory_optimization_level,
            MemoryOptimizationLevel.BALANCED
        )

        # Calculate gradient checkpointing ratio based on layers
        checkpointing_ratio = 1.0 / self.config.checkpointing_layers if self.config.selective_checkpointing else 0.3

        return MemoryOptimizationConfig(
            level=optimization_level,
            gradient_checkpointing_ratio=checkpointing_ratio,
            activation_checkpointing=True,
            mixed_precision=self.config.mixed_precision,
            cpu_offload=optimization_level == MemoryOptimizationLevel.AGGRESSIVE,
            zero_stage=2 if optimization_level != MemoryOptimizationLevel.CONSERVATIVE else 1,
            max_memory_utilization=0.85 if optimization_level == MemoryOptimizationLevel.AGGRESSIVE else 0.9,
            dynamic_batching=True,
            memory_defragmentation=True,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

    def _create_progressive_inference_engine(self) -> ProgressiveInferenceEngine:
        """Create progressive inference engine with timeout configuration."""
        timeout_config = TimeoutConfig(
            total_limit_seconds=self.config.max_inference_time,
            warning_threshold=self.config.progressive_warning_threshold,
            stage_1_limit=self.config.progressive_stage_1_limit,
            stage_2_limit=self.config.progressive_stage_2_limit,
            stage_3_limit=self.config.progressive_stage_3_limit,
            stage_4_limit=self.config.progressive_stage_4_limit,
            check_interval=1.0,
            enable_progressive_reduction=True,
        )

        return ProgressiveInferenceEngine(timeout_config)

    @resilient_operation(
        max_attempts=3,
        delay_seconds=2.0,
        handle_oom=True,
        handle_cuda_errors=True
    )
    def initialize_model(self) -> None:
        """Initialize MIT TTT strategy with comprehensive error handling and recovery."""
        try:
            # Use model loading circuit breaker
            async def load_model():
                return self.mit_ttt_strategy.initialize_model()

            # Execute with circuit breaker protection
            import asyncio
            loop = asyncio.new_event_loop() if not asyncio.get_event_loop().is_running() else asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, create a task
                loop.create_task(self.model_loading_breaker.call(load_model))
            else:
                # Run in new event loop
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.model_loading_breaker.call(load_model))

            # Apply memory optimizations to the model if available
            if hasattr(self.mit_ttt_strategy.trainer, 'model') and self.mit_ttt_strategy.trainer.model is not None:
                logger.info("Applying advanced memory optimizations to TTT model")

                try:
                    self.optimized_model_wrapper = apply_memory_optimizations(
                        self.mit_ttt_strategy.trainer.model,
                        self.memory_opt_config.level
                    )
                    logger.info(f"Memory optimizations applied with level: {self.memory_opt_config.level.value}")

                    # Apply inference optimizations with error handling
                    self._apply_inference_optimizations()

                    # Log optimization details
                    if self.config.selective_checkpointing:
                        logger.info(f"Selective gradient checkpointing enabled: every {self.config.checkpointing_layers} layers")
                    if self.config.mixed_precision:
                        logger.info("Mixed precision training enabled with automatic loss scaling")

                except Exception as opt_error:
                    logger.warning(f"Memory optimizations failed: {opt_error}")
                    # Continue without optimizations rather than fail completely
                    self.error_reporter.report_error(
                        opt_error,
                        ErrorContext(operation="memory_optimization"),
                        ErrorSeverity.MEDIUM,
                        ErrorCategory.MEMORY
                    )
            else:
                logger.warning("Model not available for memory optimization")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            # Report the error
            self.error_reporter.report_error(
                e,
                ErrorContext(operation="model_initialization"),
                ErrorSeverity.CRITICAL,
                ErrorCategory.MODEL_LOADING
            )
            raise

    def _apply_inference_optimizations(self) -> None:
        """Apply inference-specific optimizations to the model."""
        if not hasattr(self.mit_ttt_strategy.trainer, 'model') or self.mit_ttt_strategy.trainer.model is None:
            logger.warning("No model available for inference optimizations")
            return

        model = self.mit_ttt_strategy.trainer.model

        try:
            # Apply torch.compile for inference speedup
            if self.config.enable_torch_compile and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Applied torch.compile for inference optimization")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

            # Enable KV cache optimizations
            if self.config.enable_kv_cache_optimization:
                if hasattr(model.config, 'use_cache'):
                    model.config.use_cache = True
                    logger.info("KV cache enabled for inference")

                # Enable static cache if supported
                if self.config.enable_static_cache and hasattr(model, 'enable_static_cache'):
                    try:
                        model.enable_static_cache()
                        logger.info("Static KV cache enabled")
                    except Exception as e:
                        logger.warning(f"Static cache enabling failed: {e}")

            # Configure for inference mode
            model.eval()

            # Enable memory efficient attention if available
            if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                try:
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("Memory efficient attention enabled for inference")
                except Exception as e:
                    logger.warning(f"Memory efficient attention failed: {e}")

            logger.info("Inference optimizations applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply inference optimizations: {e}")

    def get_memory_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive memory optimization statistics."""
        if self.optimized_model_wrapper is not None:
            return self.optimized_model_wrapper.get_optimization_stats()
        return {
            "error": "No optimized model wrapper available",
            "basic_stats": {
                "memory_usage_mb": self._estimate_memory_usage(),
                "gpu_memory_mb": self._estimate_gpu_memory(),
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "selective_checkpointing": self.config.selective_checkpointing,
                "mixed_precision": self.config.mixed_precision
            }
        }

    @resilient_operation(
        max_attempts=3,
        delay_seconds=2.0,
        handle_oom=True,
        handle_cuda_errors=True
    )
    def adapt_to_task(self, task: ARCTask) -> TTTAdaptation:
        """
        Adapt model to specific task using MIT TTT methodology with comprehensive error recovery.

        Args:
            task: ARC task to adapt to

        Returns:
            TTTAdaptation containing adapted model information
        """
        adaptation_id = f"mit_ttt_{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        adaptation_start = time.time()

        error_context = ErrorContext(
            operation="ttt_adaptation",
            model_name=self.config.model_name,
            batch_size=self.config.batch_size
        )

        try:
            # Use training circuit breaker for protection
            async def perform_adaptation():
                return self.mit_ttt_strategy.solve_task(task, use_self_consistency=True)

            # Execute with fallback strategy
            async def adaptation_with_fallback():
                return await self.training_fallback.execute(
                    perform_adaptation,
                    fallback_args={
                        "reduce_batch_size": {"batch_size": max(1, self.config.batch_size // 2)},
                        "disable_augmentation": {"use_augmentation": False},
                        "simple_training": {"per_instance_epochs": 1, "batch_size": 1}
                    }
                )

            # Execute with circuit breaker and error handling
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task if loop is already running
                    task_result = loop.create_task(self.training_breaker.call(adaptation_with_fallback))
                    prediction, metadata = task_result.result() if hasattr(task_result, 'result') else (None, {})
                else:
                    prediction, metadata = loop.run_until_complete(self.training_breaker.call(adaptation_with_fallback))
            except Exception:
                # Fallback to synchronous execution
                prediction, metadata = self.mit_ttt_strategy.solve_task(task, use_self_consistency=True)

            adaptation_time = time.time() - adaptation_start

            # Extract adaptation metrics from metadata
            adaptation_result = metadata.get('adaptation_result')
            training_metrics = adaptation_result.training_metrics if adaptation_result else {}

            # Create TTT adaptation record with enhanced metrics
            adaptation = TTTAdaptation(
                adaptation_id=adaptation_id,
                task_id=task.task_id,
                base_model_checkpoint=str(self.config.checkpoint_dir / "base_model.pt"),
                adapted_weights_path=adaptation_result.adapter_path if adaptation_result else None,
                training_examples=self._prepare_training_examples(task),
                adaptation_metrics={
                    "mit_ttt_training_loss": training_metrics.get("avg_loss", 0.0),
                    "adaptation_time": adaptation_time,
                    "num_epochs": self.config.per_instance_epochs,
                    "num_steps": training_metrics.get("num_steps", 0),
                    "confidence": metadata.get("confidence", 0.0),
                    "permutations": metadata.get("permutations", 1),
                    "augmentations": metadata.get("augmentations", []),
                    "memory_usage": adaptation_result.memory_usage if adaptation_result else {},
                    "memory_optimization_stats": self.get_memory_optimization_stats(),
                    "selective_checkpointing": self.config.selective_checkpointing,
                    "checkpointing_layers": self.config.checkpointing_layers,
                    "mixed_precision_enabled": self.config.mixed_precision,
                    "mit_ttt_success": metadata.get("success", False),
                    "error_recovery_used": metadata.get("fallback_used", False),
                    "circuit_breaker_stats": self.training_breaker.get_stats(),
                    "oom_recoveries": self.oom_handler.batch_size_history,
                    "retry_attempts": metadata.get("retry_attempts", 0)
                },
                created_at=datetime.now(),
            )

            # Store adaptation
            self.adaptations[task.task_id] = adaptation

            # Cache successful batch size
            if metadata.get("success", False) and hasattr(self, 'oom_handler'):
                self.oom_handler.cache_successful_batch_size("ttt_adaptation", self.config.batch_size)

            return adaptation

        except Exception as e:
            adaptation_time = time.time() - adaptation_start

            # Report error with comprehensive context
            self.error_reporter.report_error(
                e,
                error_context,
                ErrorSeverity.HIGH,
                ErrorCategory.TRAINING
            )

            # Create failed adaptation record with detailed error information
            adaptation = TTTAdaptation(
                adaptation_id=adaptation_id,
                task_id=task.task_id,
                base_model_checkpoint=str(self.config.checkpoint_dir / "base_model.pt"),
                adapted_weights_path=None,
                training_examples=[],
                adaptation_metrics={
                    "adaptation_time": adaptation_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "memory_optimization_stats": self.get_memory_optimization_stats(),
                    "mit_ttt_success": False,
                    "circuit_breaker_stats": self.training_breaker.get_stats(),
                    "error_recovery_attempted": True,
                    "fallback_strategy_stats": self.training_fallback.get_stats(),
                    "retry_strategy_stats": self.retry_strategy.get_recovery_stats()
                },
                created_at=datetime.now(),
            )

            self.adaptations[task.task_id] = adaptation
            return adaptation

    @resilient_operation(
        max_attempts=2,  # Fewer retries for inference to stay within time limits
        delay_seconds=1.0,
        handle_oom=True,
        handle_cuda_errors=True
    )
    def solve(self, task: ARCTask, batch_size: int = None) -> ARCTaskSolution:
        """
        Solve ARC task using MIT TTT methodology with comprehensive error handling and recovery.

        Args:
            task: ARC task to solve
            batch_size: Optional batch size override for OOM recovery

        Returns:
            Solution with predictions and metadata
        """
        start_time = datetime.now()
        inference_start = time.time()

        # Use batch_size parameter if provided (for OOM recovery)
        if batch_size is not None:
            original_batch_size = self.config.batch_size
            self.config.batch_size = batch_size
            logger.info(f"Using adjusted batch size: {batch_size}")

        error_context = ErrorContext(
            operation="ttt_inference",
            model_name=self.config.model_name,
            batch_size=self.config.batch_size
        )

        try:
            # Use progressive inference if enabled
            if self.config.enable_progressive_inference and self.progressive_inference_engine:
                result = self._solve_with_progressive_inference(task)
                prediction = result.output
                predictions = [prediction] if prediction is not None else [task.test_input]
                success = result.success
                confidence = 0.8 if result.success else 0.0

                # Create metadata from progressive inference result
                metadata = result.metadata or {}
                metadata.update({
                    'success': result.success,
                    'confidence': confidence,
                    'fallback_level': result.fallback_level.value,
                    'inference_time_seconds': result.execution_time,
                    'within_time_limit': not result.timeout_triggered,
                    'progressive_inference_used': True,
                    'timeout_triggered': result.timeout_triggered,
                    'error_recovery_used': False
                })

                if result.error_message:
                    metadata['error'] = result.error_message

            else:
                # Use circuit breaker protection for inference
                async def perform_inference():
                    with self._inference_timeout_context():
                        return self.mit_ttt_strategy.solve_task(
                            task,
                            use_self_consistency=True
                        )

                # Execute with fallback strategy
                async def inference_with_fallback():
                    return await self.inference_fallback.execute(
                        perform_inference,
                        fallback_args={
                            "fast_inference": {"use_self_consistency": False, "temperature": 0.0},
                            "minimal_inference": {"max_tokens": 100, "temperature": 0.0},
                            "emergency_fallback": {"return_input_copy": True}
                        }
                    )

                # Execute with error handling
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        task_result = loop.create_task(self.inference_breaker.call(inference_with_fallback))
                        prediction, metadata = task_result.result() if hasattr(task_result, 'result') else (task.test_input, {})
                    else:
                        prediction, metadata = loop.run_until_complete(self.inference_breaker.call(inference_with_fallback))
                except Exception:
                    # Fallback to synchronous execution
                    with self._inference_timeout_context():
                        prediction, metadata = self.mit_ttt_strategy.solve_task(
                            task,
                            use_self_consistency=True
                        )

                predictions = [prediction]
                success = metadata.get('success', False)
                confidence = metadata.get('confidence', 0.0)

                # Calculate inference time
                inference_time = time.time() - inference_start
                metadata.update({
                    'inference_time_seconds': inference_time,
                    'within_time_limit': inference_time <= self.config.max_inference_time,
                    'progressive_inference_used': False,
                    'error_recovery_used': metadata.get('fallback_used', False),
                    'circuit_breaker_stats': self.inference_breaker.get_stats()
                })

                if not metadata['within_time_limit']:
                    logger.warning(f"Inference time ({inference_time:.1f}s) exceeded limit ({self.config.max_inference_time:.1f}s)")

            # Store adaptation if successful
            if success and 'adaptation_result' in metadata:
                adaptation_result = metadata['adaptation_result']
                adaptation = TTTAdaptation(
                    adaptation_id=adaptation_result.adaptation_id,
                    task_id=task.task_id,
                    base_model_checkpoint=str(self.config.checkpoint_dir / "base_model.pt"),
                    adapted_weights_path=adaptation_result.adapter_path,
                    training_examples=self._prepare_training_examples(task),
                    adaptation_metrics=adaptation_result.training_metrics,
                    created_at=datetime.now(),
                )
                self.adaptations[task.task_id] = adaptation

            # Cache successful configuration
            if success:
                self.oom_handler.cache_successful_batch_size("ttt_inference", self.config.batch_size)

        except TimeoutError as e:
            # Handle inference timeout with error reporting
            inference_time = time.time() - inference_start
            prediction = task.test_input
            predictions = [prediction]
            success = False
            confidence = 0.0

            self.error_reporter.report_error(
                e,
                error_context,
                ErrorSeverity.HIGH,
                ErrorCategory.INFERENCE
            )

            metadata = {
                'success': False,
                'error': f'Inference timeout after {inference_time:.1f}s',
                'inference_time_seconds': inference_time,
                'within_time_limit': False,
                'timeout_exceeded': True,
                'fallback_used': True,
                'error_recovery_used': True,
                'circuit_breaker_stats': self.inference_breaker.get_stats()
            }
            logger.error(f"Inference timeout after {inference_time:.1f}s for task {task.task_id}")

        except Exception as e:
            # Handle other errors with comprehensive reporting
            inference_time = time.time() - inference_start
            prediction = task.test_input
            predictions = [prediction]
            success = False
            confidence = 0.0

            # Report error with context
            self.error_reporter.report_error(
                e,
                error_context,
                ErrorSeverity.HIGH,
                ErrorCategory.INFERENCE
            )

            metadata = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'inference_time_seconds': inference_time,
                'within_time_limit': inference_time <= self.config.max_inference_time,
                'fallback_used': True,
                'error_recovery_used': True,
                'circuit_breaker_stats': self.inference_breaker.get_stats(),
                'fallback_strategy_stats': self.inference_fallback.get_stats()
            }

        finally:
            # Restore original batch size if it was modified
            if batch_size is not None:
                self.config.batch_size = original_batch_size

        # Calculate resource usage
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        resource_usage = ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            cpu_seconds=duration,
            memory_mb=self._estimate_memory_usage(),
            gpu_memory_mb=self._estimate_gpu_memory(),
            api_calls={},
            total_tokens=metadata.get('total_tokens', 0),
            estimated_cost=0.0,
            timestamp=datetime.now(),
        )

        return ARCTaskSolution(
            task_id=task.task_id,
            predictions=predictions,
            strategy_used=StrategyType.TEST_TIME_TRAINING,
            confidence_score=confidence,
            metadata={
                "mit_ttt_strategy": True,
                "success": success,
                "adaptation_id": metadata.get('adaptation_result', {}).get('adaptation_id') if 'adaptation_result' in metadata else None,
                "model_name": self.config.model_name,
                "permutations": metadata.get('permutations', 1),
                "augmentations": metadata.get('augmentations', []),
                "total_time": metadata.get('total_time', duration),
                "methodology": "MIT_TTT",
                **metadata
            },
            resource_usage=resource_usage,
        )

    def _prepare_training_examples(self, task: ARCTask) -> list[dict[str, Any]]:
        """Prepare training examples from task for TTT adaptation."""
        examples = []

        from src.utils.grid_ops import grid_to_string

        for i, train_example in enumerate(task.train_examples):
            # Convert grids to string representation for LLM
            input_str = grid_to_string(train_example["input"])
            output_str = grid_to_string(train_example["output"])

            # Create prompt for training
            prompt = f"Task: Transform the input grid to output grid.\n\nInput:\n{input_str}\n\nOutput:\n{output_str}"

            examples.append({
                "index": i,
                "prompt": prompt,
                "input_grid": train_example["input"],
                "output_grid": train_example["output"],
            })

        return examples

    def _solve_with_progressive_inference(self, task: ARCTask):
        """Solve task using progressive inference with fallbacks."""
        logger.info(f"Starting progressive inference for task: {task.task_id}")

        def primary_solve_function(arc_task, **kwargs):
            """Primary solve function with full quality."""
            logger.debug("Executing primary solve function")
            prediction, metadata = self.mit_ttt_strategy.solve_task(
                arc_task,
                use_self_consistency=True
            )
            return prediction, metadata

        def fast_solve_function(arc_task, **kwargs):
            """Fast solve with reduced quality."""
            logger.debug("Executing fast solve fallback")
            prediction, metadata = self.mit_ttt_strategy.solve_task(
                arc_task,
                use_self_consistency=False,  # Disable self-consistency for speed
                temperature=1.2,
                max_tokens=200,
            )
            return prediction, metadata

        def deterministic_solve_function(arc_task, **kwargs):
            """Deterministic solve with greedy decoding."""
            logger.debug("Executing deterministic solve fallback")
            prediction, metadata = self.mit_ttt_strategy.solve_task(
                arc_task,
                use_self_consistency=False,
                temperature=0.0,  # Deterministic
                max_tokens=150,
            )
            return prediction, metadata

        def minimal_solve_function(arc_task, **kwargs):
            """Minimal solve with very basic output."""
            logger.debug("Executing minimal solve fallback")
            try:
                prediction, metadata = self.mit_ttt_strategy.solve_task(
                    arc_task,
                    use_self_consistency=False,
                    temperature=0.0,
                    max_tokens=50,
                )
                return prediction, metadata
            except Exception as e:
                logger.warning(f"Minimal solve failed: {e}, returning input copy")
                return arc_task.test_input, {"success": False, "error": str(e)}

        # Create fallback function mapping
        fallback_functions = {
            FallbackLevel.FAST_SAMPLING: fast_solve_function,
            FallbackLevel.DETERMINISTIC: deterministic_solve_function,
            FallbackLevel.SHORT_GENERATION: deterministic_solve_function,  # Same as deterministic
            FallbackLevel.MINIMAL_OUTPUT: minimal_solve_function,
        }

        # Execute with progressive fallbacks
        result = self.progressive_inference_engine.execute_with_fallbacks(
            primary_solve_function,
            fallback_functions,
            task
        )

        # If the output is a tuple (prediction, metadata), extract the prediction
        if isinstance(result.output, tuple) and len(result.output) == 2:
            prediction, solve_metadata = result.output
            result.output = prediction

            # Merge metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(solve_metadata)

        logger.info(f"Progressive inference completed: level={result.fallback_level.value}, success={result.success}, time={result.execution_time:.2f}s")
        return result

    @contextmanager
    def _inference_timeout_context(self):
        """Context manager to enforce inference timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Inference exceeded {self.config.max_inference_time}s limit")

        # Set up timeout signal (Unix systems only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.max_inference_time))

            try:
                yield
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # For Windows or systems without SIGALRM, just yield without timeout
            logger.warning("Signal-based timeout not available on this platform")
            yield

    def _generate_prediction(
        self, input_grid: list[list[int]], adaptation: TTTAdaptation
    ) -> list[list[int]]:
        """Generate prediction using MIT TTT methodology (handled by strategy)."""
        # This method is now handled by the MIT_TTTStrategy
        # Return fallback grid if called directly
        return input_grid

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                return max(allocated, reserved)
        except Exception:
            pass
        return 0.0

    def _validate_memory_constraints(self) -> bool:
        """Validate that 8B model fits within memory constraints."""
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                current_usage = self._estimate_gpu_memory()
                available = total_memory - current_usage

                # 8B model with 4-bit quantization should use ~6-8GB
                estimated_usage = 8000  # Conservative estimate in MB
                return available >= estimated_usage
        except Exception:
            pass
        return True  # Assume valid if cannot check

    # StrategyPort interface implementation (Story 3.1)

    async def solve_task(self, task: ARCTask) -> StrategyOutput:
        """
        Solve ARC task using enhanced TTT with leave-one-out, self-consistency, and LoRA optimization.

        Implements StrategyPort interface for ensemble integration.

        Args:
            task: ARCTask to solve

        Returns:
            StrategyOutput with predictions, confidence, and metadata
        """
        start_time = time.time()
        strategy_id = f"ttt_{task.task_id}"

        try:
            # Register with timing coordinator if available
            if self.timing_coordinator:
                await self.timing_coordinator.register_strategy(
                    strategy_id,
                    timeout_ms=int(self.config.max_inference_time * 1000)
                )

            # Record start time in metrics
            self.metrics_collector.record_solve_duration(
                strategy="test_time_training",
                duration_seconds=0.0,
                task_type="start"
            )

            # Execute TTT solve with enhanced components
            solution = self.solve(task)

            # Convert ARCTaskSolution to StrategyOutput
            predicted_output = solution.predictions[0] if solution.predictions else task.test_input

            # Convert to numpy array if needed
            if not isinstance(predicted_output, np.ndarray):
                predicted_output = np.array(predicted_output, dtype=np.int8)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Extract per-pixel confidence from metadata if available
            per_pixel_confidence = solution.metadata.get('per_pixel_confidence')
            if per_pixel_confidence is not None and not isinstance(per_pixel_confidence, np.ndarray):
                per_pixel_confidence = np.array(per_pixel_confidence, dtype=np.float32)

            # Create StrategyOutput
            output = StrategyOutput(
                strategy_type=StrategyType.TEST_TIME_TRAINING,
                predicted_output=predicted_output,
                confidence_score=solution.confidence_score,
                per_pixel_confidence=per_pixel_confidence,
                reasoning_trace=solution.metadata.get('reasoning_trace', 'Enhanced TTT with leave-one-out, self-consistency, LoRA optimization'),
                resource_usage=solution.resource_usage,
                execution_time_ms=execution_time_ms
            )

            # Record metrics
            self.metrics_collector.record_solve_duration(
                strategy="test_time_training",
                duration_seconds=execution_time_ms / 1000.0,
                task_type="complete"
            )
            self.metrics_collector.record_confidence_score(
                strategy="test_time_training",
                confidence=solution.confidence_score
            )

            # Signal success to coordinator if available
            if self.timing_coordinator:
                await self.timing_coordinator.signal_success(
                    strategy_id,
                    confidence=solution.confidence_score,
                    metadata={"execution_time_ms": execution_time_ms}
                )

            return output

        except Exception as e:
            logger.error(f"Error in solve_task for {task.task_id}: {e}")

            # Record error in metrics
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.metrics_collector.record_solve_duration(
                strategy="test_time_training",
                duration_seconds=execution_time_ms / 1000.0,
                task_type="error"
            )

            # Return fallback output
            return StrategyOutput(
                strategy_type=StrategyType.TEST_TIME_TRAINING,
                predicted_output=np.array(task.test_input, dtype=np.int8),
                confidence_score=0.0,
                reasoning_trace=f"Error: {str(e)}",
                resource_usage=ResourceUsage(
                    api_calls=0,
                    total_tokens=0,
                    estimated_cost=0.0
                ),
                execution_time_ms=execution_time_ms
            )

        finally:
            # Unregister from coordinator
            if self.timing_coordinator:
                from src.domain.ports.timing import TerminationReason
                await self.timing_coordinator.unregister_strategy(
                    strategy_id,
                    TerminationReason.SUCCESS_SIGNAL
                )

    def get_confidence_estimate(self, task: ARCTask) -> float:
        """
        Quick confidence estimate for routing (<100ms).

        Implements StrategyPort interface.

        Args:
            task: ARCTask to estimate

        Returns:
            Estimated confidence (0.0-1.0)
        """
        # Heuristics for TTT confidence based on task characteristics
        num_train = len(task.train_examples)

        # Calculate average grid size
        grid_sizes = []
        for example in task.train_examples:
            if example.get("input"):
                grid = example["input"]
                grid_sizes.append(len(grid) * len(grid[0]) if grid else 0)

        avg_grid_size = sum(grid_sizes) / len(grid_sizes) if grid_sizes else 0

        # TTT performs better with more training examples
        train_factor = min(num_train / 5.0, 1.0)  # Cap at 5 examples

        # TTT performs better with smaller grids (easier to adapt)
        if avg_grid_size < 100:
            size_factor = 1.0
        elif avg_grid_size < 300:
            size_factor = 0.8
        else:
            size_factor = 0.6

        # Base confidence for enhanced TTT
        base_confidence = 0.70  # Higher than baseline (0.58) due to enhancements

        estimated_confidence = base_confidence * train_factor * size_factor

        # Record estimate in metrics
        self.metrics_collector.record_confidence_score(
            strategy="test_time_training",
            confidence=estimated_confidence,
            task_type="estimate"
        )

        return estimated_confidence

    def get_resource_estimate(self, task: ARCTask) -> ResourceUsage:
        """
        Estimate resource requirements (<50ms).

        Implements StrategyPort interface.

        Args:
            task: ARCTask to estimate

        Returns:
            ResourceUsage with estimated requirements
        """
        from datetime import datetime

        # Estimate based on task complexity
        num_train = len(task.train_examples)

        # Estimate CPU time: base + per-example adaptation
        # Enhanced TTT: ~120s base + ~30s per example (with optimizations)
        estimated_cpu_seconds = 120.0 + (num_train * 30.0)

        # Estimate memory: 8B model with 4-bit quantization ~6-8GB
        estimated_memory_mb = 8000.0
        estimated_gpu_memory_mb = 8000.0

        # Record estimate in metrics
        self.metrics_collector.record_resource_usage(
            strategy="test_time_training",
            resource_type="cpu_seconds",
            value=estimated_cpu_seconds,
            task_id=task.task_id
        )
        self.metrics_collector.record_resource_usage(
            strategy="test_time_training",
            resource_type="memory_mb",
            value=estimated_memory_mb,
            task_id=task.task_id
        )

        # Return ResourceUsage with estimated values
        # Note: ResourceUsage expects actual usage, but we use it for estimates
        return ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            cpu_seconds=estimated_cpu_seconds,
            memory_mb=estimated_memory_mb,
            gpu_memory_mb=estimated_gpu_memory_mb,
            api_calls={},  # TTT uses local model
            total_tokens=0,
            estimated_cost=0.0,
            timestamp=datetime.now()
        )

    def set_timing_coordinator(self, coordinator: TimingCoordinator) -> None:
        """
        Set timing coordinator for ensemble integration.

        Args:
            coordinator: TimingCoordinator instance
        """
        self.timing_coordinator = coordinator
        logger.info("TimingCoordinator integration enabled")

    def cleanup(self) -> None:
        """Clean up resources and free memory with comprehensive error handling component cleanup."""
        logger.info("Starting comprehensive TTT Adapter cleanup")

        try:
            # Stop health monitoring
            if hasattr(self, 'health_monitor'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.health_monitor.stop_monitoring())
                    else:
                        loop.run_until_complete(self.health_monitor.stop_monitoring())
                except Exception:
                    pass

            # Clean up optimized model wrapper
            if self.optimized_model_wrapper is not None:
                if hasattr(self.optimized_model_wrapper, 'memory_monitor'):
                    self.optimized_model_wrapper.memory_monitor.stop_monitoring_thread()
                del self.optimized_model_wrapper
                self.optimized_model_wrapper = None

            # Clean up MIT TTT strategy
            if hasattr(self, 'mit_ttt_strategy'):
                self.mit_ttt_strategy.cleanup()

            # Clear adaptations
            self.adaptations.clear()

            # Generate final error report summary
            if hasattr(self, 'error_reporter'):
                error_summary = self.error_reporter.get_error_summary()
                if error_summary.get('total_errors', 0) > 0:
                    logger.info(f"Session error summary: {error_summary}")

            # Log circuit breaker statistics
            if hasattr(self, 'model_loading_breaker'):
                logger.info(f"Model loading circuit breaker stats: {self.model_loading_breaker.get_stats()}")
            if hasattr(self, 'training_breaker'):
                logger.info(f"Training circuit breaker stats: {self.training_breaker.get_stats()}")
            if hasattr(self, 'inference_breaker'):
                logger.info(f"Inference circuit breaker stats: {self.inference_breaker.get_stats()}")

            # Log fallback strategy statistics
            if hasattr(self, 'training_fallback'):
                logger.info(f"Training fallback stats: {self.training_fallback.get_stats()}")
            if hasattr(self, 'inference_fallback'):
                logger.info(f"Inference fallback stats: {self.inference_fallback.get_stats()}")

            # Log retry strategy statistics
            if hasattr(self, 'retry_strategy'):
                logger.info(f"Retry strategy stats: {self.retry_strategy.get_recovery_stats()}")

            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass

        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

        logger.info("TTT Adapter cleanup complete with comprehensive error handling")

    def _setup_error_fallbacks(self) -> None:
        """Setup error-specific fallback strategies."""

        # Training fallbacks
        def reduce_batch_size_fallback(*args, **kwargs):
            """Fallback that reduces batch size for training."""
            kwargs["batch_size"] = max(1, kwargs.get("batch_size", self.config.batch_size) // 2)
            return self.mit_ttt_strategy.solve_task(*args, **kwargs)

        def disable_augmentation_fallback(*args, **kwargs):
            """Fallback that disables augmentation."""
            kwargs["use_augmentation"] = False
            return self.mit_ttt_strategy.solve_task(*args, **kwargs)

        def simple_training_fallback(*args, **kwargs):
            """Fallback with minimal training configuration."""
            kwargs.update({"per_instance_epochs": 1, "batch_size": 1, "use_augmentation": False})
            return self.mit_ttt_strategy.solve_task(*args, **kwargs)

        # Add OOM-specific fallbacks
        self.training_fallback.add_fallback(reduce_batch_size_fallback, priority=20)
        self.training_fallback.add_fallback(disable_augmentation_fallback, priority=15)
        self.training_fallback.add_fallback(simple_training_fallback, priority=10)

        # Inference fallbacks
        def fast_inference_fallback(*args, **kwargs):
            """Fast inference without self-consistency."""
            kwargs["use_self_consistency"] = False
            kwargs["temperature"] = 0.0
            return self.mit_ttt_strategy.solve_task(*args, **kwargs)

        def minimal_inference_fallback(*args, **kwargs):
            """Minimal inference with very short generation."""
            kwargs.update({"use_self_consistency": False, "max_tokens": 100, "temperature": 0.0})
            return self.mit_ttt_strategy.solve_task(*args, **kwargs)

        def emergency_fallback(*args, **kwargs):
            """Emergency fallback that returns input copy."""
            task = args[0] if args else kwargs.get('task')
            if task:
                return task.test_input, {"success": False, "fallback_used": True, "fallback_level": "emergency"}
            return None, {"success": False, "fallback_used": True, "fallback_level": "emergency"}

        self.inference_fallback.add_fallback(fast_inference_fallback, priority=20)
        self.inference_fallback.add_fallback(minimal_inference_fallback, priority=15)
        self.inference_fallback.add_fallback(emergency_fallback, priority=5)

    def _setup_health_checks(self) -> None:
        """Setup health monitoring checks for critical components."""
        def check_model_health() -> bool:
            """Check if model is loaded and functional."""
            try:
                return (hasattr(self, 'mit_ttt_strategy') and
                       hasattr(self.mit_ttt_strategy, 'trainer') and
                       self.mit_ttt_strategy.trainer is not None)
            except Exception:
                return False

        def check_gpu_health() -> bool:
            """Check GPU availability and memory."""
            try:
                import torch
                if not torch.cuda.is_available():
                    return True  # CPU is fine
                # Check if we can allocate a small tensor
                test_tensor = torch.randn(10, 10, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception:
                return False

        def check_checkpoint_dir() -> bool:
            """Check if checkpoint directory is accessible."""
            try:
                return self.config.checkpoint_dir.exists() and self.config.checkpoint_dir.is_dir()
            except Exception:
                return False

        # Add health checks
        self.health_monitor.add_health_check(
            "model", check_model_health, recovery_func=self.initialize_model, critical=True
        )
        self.health_monitor.add_health_check(
            "gpu", check_gpu_health, critical=False
        )
        self.health_monitor.add_health_check(
            "checkpoint_dir", check_checkpoint_dir, critical=False
        )

        # Start monitoring
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.health_monitor.start_monitoring())
            else:
                loop.run_until_complete(self.health_monitor.start_monitoring())
        except Exception:
            logger.warning("Could not start health monitoring (async event loop not available)")

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics including error handling metrics."""
        stats = {
            "memory_optimization_stats": self.get_memory_optimization_stats(),
            "adaptation_count": len(self.adaptations),
            "config": {
                "model_name": self.config.model_name,
                "batch_size": self.config.batch_size,
                "mixed_precision": self.config.mixed_precision,
                "gradient_checkpointing": self.config.gradient_checkpointing,
            }
        }

        # Add error handling stats if available
        if hasattr(self, 'error_reporter'):
            stats["error_summary"] = self.error_reporter.get_error_summary()

        if hasattr(self, 'oom_handler'):
            stats["oom_stats"] = {
                "batch_size_history": self.oom_handler.batch_size_history,
                "cached_batch_sizes": self.oom_handler.batch_size_cache,
            }

        if hasattr(self, 'retry_strategy'):
            stats["retry_stats"] = self.retry_strategy.get_recovery_stats()

        if hasattr(self, 'training_fallback'):
            stats["training_fallback_stats"] = self.training_fallback.get_stats()

        if hasattr(self, 'inference_fallback'):
            stats["inference_fallback_stats"] = self.inference_fallback.get_stats()

        # Circuit breaker stats
        circuit_breaker_stats = {}
        if hasattr(self, 'model_loading_breaker'):
            circuit_breaker_stats["model_loading"] = self.model_loading_breaker.get_stats()
        if hasattr(self, 'training_breaker'):
            circuit_breaker_stats["training"] = self.training_breaker.get_stats()
        if hasattr(self, 'inference_breaker'):
            circuit_breaker_stats["inference"] = self.inference_breaker.get_stats()

        if circuit_breaker_stats:
            stats["circuit_breaker_stats"] = circuit_breaker_stats

        # Health monitoring stats
        if hasattr(self, 'health_monitor'):
            stats["health_status"] = self.health_monitor.get_health_status()

        return stats
