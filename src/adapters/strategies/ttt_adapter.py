"""
MIT Test-Time Training (TTT) adapter for ARC tasks.

This module provides integration with the actual MIT TTT methodology,
implementing per-instance adaptation, self-consistency, and data augmentation
as described in the MIT research paper.
"""
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from src.domain.models import (
    ARCTask,
    ARCTaskSolution,
    ResourceUsage,
    StrategyType,
    TTTAdaptation,
)
from src.utils.ttt_data_conversion import (
    TTTDataConverter, 
    AugmentationType
)
from src.utils.ttt_methodology import (
    MIT_TTTStrategy,
    TTTTrainingConfig
)


@dataclass
class TTTConfig:
    """Configuration for MIT Test-Time Training adapter."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    device: str = "auto"
    quantization: bool = True
    mixed_precision: bool = True
    
    # MIT TTT specific parameters
    lora_rank: int = 64  # MIT uses rank 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
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
    
    # Memory and performance
    max_length: int = 2048
    gradient_checkpointing: bool = True
    memory_limit_mb: float = 10240
    max_training_time: float = 300.0
    
    # Paths
    checkpoint_dir: Path = Path("data/models/ttt")
    cache_dir: Path = Path("data/cache/ttt")
    config_path: Optional[Path] = Path("configs/strategies/ttt.yaml")
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "TTTConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Map YAML structure to config fields
        kwargs = {}
        
        if 'model' in config_data:
            model_config = config_data['model']
            kwargs.update({
                'model_name': model_config.get('name', "meta-llama/Llama-3.2-1B"),
                'device': model_config.get('device', "auto"),
                'quantization': model_config.get('quantization', True),
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
                'memory_limit_mb': training_config.get('memory_limit_mb', 10240),
                'max_training_time': training_config.get('max_training_time', 300.0),
                'max_examples': training_config.get('max_examples', 5)
            })
        
        if 'lora' in config_data:
            lora_config = config_data['lora']
            kwargs.update({
                'lora_rank': lora_config.get('rank', 64),
                'lora_alpha': lora_config.get('alpha', 16),
                'lora_dropout': lora_config.get('dropout', 0.1)
            })
        
        if 'inference' in config_data:
            inference_config = config_data['inference']
            kwargs.update({
                'temperature': inference_config.get('temperature', 0.0)
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


class TTTAdapter:
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
        self.adaptations: Dict[str, TTTAdaptation] = {}
        
        # Initialize model components (loaded on demand)
        self.model = None
        self.tokenizer = None
        
        # Setup device
        self.device = self._setup_device()
        
        # Ensure directories exist
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def initialize_model(self) -> None:
        """Initialize MIT TTT strategy (handled internally)."""
        # Model initialization is handled by MIT_TTTStrategy
        pass

    def adapt_to_task(self, task: ARCTask) -> TTTAdaptation:
        """
        Adapt model to specific task using MIT TTT methodology.
        
        Args:
            task: ARC task to adapt to
            
        Returns:
            TTTAdaptation containing adapted model information
        """
        adaptation_id = f"mit_ttt_{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        adaptation_start = time.time()
        
        try:
            # Use MIT TTT strategy for adaptation
            prediction, metadata = self.mit_ttt_strategy.solve_task(task, use_self_consistency=True)
            
            adaptation_time = time.time() - adaptation_start
            
            # Extract adaptation metrics from metadata
            adaptation_result = metadata.get('adaptation_result')
            training_metrics = adaptation_result.training_metrics if adaptation_result else {}
            
            # Create TTT adaptation record
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
                    "mit_ttt_success": metadata.get("success", False)
                },
                created_at=datetime.now(),
            )
            
            # Store adaptation
            self.adaptations[task.task_id] = adaptation
            
            return adaptation
            
        except Exception as e:
            adaptation_time = time.time() - adaptation_start
            
            # Create failed adaptation record
            adaptation = TTTAdaptation(
                adaptation_id=adaptation_id,
                task_id=task.task_id,
                base_model_checkpoint=str(self.config.checkpoint_dir / "base_model.pt"),
                adapted_weights_path=None,
                training_examples=[],
                adaptation_metrics={
                    "adaptation_time": adaptation_time,
                    "error": str(e),
                    "mit_ttt_success": False
                },
                created_at=datetime.now(),
            )
            
            self.adaptations[task.task_id] = adaptation
            return adaptation

    def solve(self, task: ARCTask) -> ARCTaskSolution:
        """
        Solve ARC task using MIT TTT methodology.
        
        Args:
            task: ARC task to solve
            
        Returns:
            Solution with predictions and metadata
        """
        start_time = datetime.now()

        try:
            # Use MIT TTT strategy to solve the task
            prediction, metadata = self.mit_ttt_strategy.solve_task(
                task, 
                use_self_consistency=True
            )
            
            predictions = [prediction]
            success = metadata.get('success', False)
            confidence = metadata.get('confidence', 0.0)
            
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
            
        except Exception as e:
            # Fallback to input grid on error
            prediction = task.test_input
            predictions = [prediction]
            success = False
            confidence = 0.0
            metadata = {
                'success': False,
                'error': str(e),
                'fallback_used': True
            }
        
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

    def _prepare_training_examples(self, task: ARCTask) -> List[Dict[str, Any]]:
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

    def _generate_prediction(
        self, input_grid: List[List[int]], adaptation: TTTAdaptation
    ) -> List[List[int]]:
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
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0.0

    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        # Clean up MIT TTT strategy
        if hasattr(self, 'mit_ttt_strategy'):
            self.mit_ttt_strategy.cleanup()
        
        # Clear adaptations
        self.adaptations.clear()
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass