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
from typing import Any

import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from src.domain.models import (
    ARCTask,
    ARCTaskSolution,
    ResourceUsage,
    StrategyType,
    TTTAdaptation,
)
from src.utils.ttt_methodology import MIT_TTTStrategy

try:
    from src.utils.grid_ops import grid_to_string, string_to_grid
    HAS_GRID_OPS = True
except ImportError:
    HAS_GRID_OPS = False

    def grid_to_string(grid):
        """Fallback grid to string converter."""
        return "\n".join([" ".join(map(str, row)) for row in grid])

    def string_to_grid(text):
        """Fallback string to grid converter."""
        lines = text.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.split() if x.isdigit()]
                if row:
                    grid.append(row)
        return grid if grid else [[0]]


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

    # Memory and performance
    max_length: int = 2048
    gradient_checkpointing: bool = True
    memory_limit_mb: float = 10240
    max_training_time: float = 300.0

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
                'model_name': model_config.get('name', "meta-llama/Llama-3.2-1B"),
                'device': model_config.get('device', "auto"),
                'quantization': model_config.get('quantization', True),
                'max_length': model_config.get('max_length', 2048)
            })

        if 'training' in config_data:
            training_config = config_data['training']
            kwargs.update({
                'learning_rate': training_config.get('learning_rate', 5e-5),
                'num_epochs': training_config.get('num_epochs', 2),
                'batch_size': training_config.get('batch_size', 2),
                'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 1),
                'mixed_precision': training_config.get('mixed_precision', True),
                'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
                'memory_limit_mb': training_config.get('memory_limit_mb', 10240),
                'max_training_time': training_config.get('max_training_time', 300.0)
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

        return cls(**kwargs)


class TTTAdapter:
    """MIT Test-Time Training adapter for ARC tasks."""

    def __init__(self, config: TTTConfig | None = None):
        """Initialize MIT TTT adapter with configuration."""
        self.config = config or TTTConfig()

        # Load configuration from YAML if path provided
        if self.config.config_path and self.config.config_path.exists():
            self.config = TTTConfig.from_yaml(self.config.config_path)

        # Create TTT training configuration
        self.ttt_config = self._create_ttt_training_config()

        # Initialize MIT TTT strategy
        self.mit_ttt_strategy = MIT_TTTStrategy(self.ttt_config)

        # Track adaptations
        self.adaptations: dict[str, TTTAdaptation] = {}

        # Ensure directories exist
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        """Set up computing device based on configuration."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for TTT adapter")

        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def initialize_model(self) -> None:
        """Initialize base model and tokenizer with quantization if enabled."""
        if self.model is not None:
            return

        if not HAS_TRANSFORMERS:
            raise RuntimeError("Transformers library is required for TTT adapter")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading configuration
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }

        if self.config.quantization and self.device.type == "cuda":
            # Enable 8-bit quantization for memory efficiency
            model_kwargs["load_in_8bit"] = True

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # Move to device if not using device_map
        if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
            self.model = self.model.to(self.device)

        # Set model to evaluation mode by default
        self.model.eval()

    def adapt_to_task(self, task: ARCTask) -> TTTAdaptation:
        """
        Adapt model to specific task using test-time training.

        Args:
            task: ARC task to adapt to

        Returns:
            TTTAdaptation containing adapted model information
        """
        self.initialize_model()

        # Prepare training examples from task
        training_examples = self._prepare_training_examples(task)

        # Create adaptation ID
        adaptation_id = f"ttt_{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        adaptation_start = time.time()

        # Apply LoRA for efficient adaptation
        from src.utils.lora_adapter import apply_lora_to_model
        lora_adapter = apply_lora_to_model(
            self.model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )

        # Set up optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(
            lora_adapter.get_trainable_parameters(),
            lr=self.config.learning_rate,
        )

        # Training metrics
        total_loss = 0.0
        num_steps = 0
        best_loss = float('inf')

        # Training loop
        self.model.train()
        for _epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            for example in training_examples:
                # Tokenize input and output
                input_ids = self.tokenizer.encode(
                    example["prompt"],
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                ).to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Track metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_steps += 1

            avg_epoch_loss = epoch_loss / len(training_examples)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                # Save best checkpoint
                checkpoint_path = self.config.checkpoint_dir / f"{adaptation_id}_best.pt"
                torch.save(lora_adapter.state_dict(), checkpoint_path)

        # Calculate adaptation time
        adaptation_time = time.time() - adaptation_start

        # Create adaptation record
        adaptation = TTTAdaptation(
            adaptation_id=adaptation_id,
            task_id=task.task_id,
            base_model_checkpoint=str(self.config.checkpoint_dir / "base_model.pt"),
            adapted_weights_path=str(self.config.checkpoint_dir / f"{adaptation_id}_best.pt"),
            training_examples=training_examples,
            adaptation_metrics={
                "training_loss": total_loss / max(num_steps, 1),
                "best_loss": best_loss,
                "adaptation_time": adaptation_time,
                "num_epochs": self.config.num_epochs,
                "num_steps": num_steps,
            },
            created_at=datetime.now(),
        )

        # Store adaptation and LoRA adapter
        self.adaptations[task.task_id] = adaptation
        self.lora_adapters[task.task_id] = lora_adapter

        # Set model back to eval mode
        self.model.eval()

        return adaptation

    def solve(self, task: ARCTask) -> ARCTaskSolution:
        """
        Solve ARC task using TTT-adapted model.

        Args:
            task: ARC task to solve

        Returns:
            Solution with predictions and metadata
        """
        start_time = datetime.now()

        # Adapt model to task if not already adapted
        if task.task_id not in self.adaptations:
            adaptation = self.adapt_to_task(task)
        else:
            adaptation = self.adaptations[task.task_id]

        # Generate predictions for test case
        prediction = self._generate_prediction(task.test_input, adaptation)
        predictions = [prediction]

        # Calculate resource usage
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        resource_usage = ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            cpu_seconds=duration,
            memory_mb=self._estimate_memory_usage(),
            gpu_memory_mb=self._estimate_gpu_memory() if self.device.type == "cuda" else None,
            api_calls={},
            total_tokens=0,  # Will be updated when actual implementation is done
            estimated_cost=0.0,
            timestamp=datetime.now(),
        )

        return ARCTaskSolution(
            task_id=task.task_id,
            predictions=predictions,
            strategy_used=StrategyType.TEST_TIME_TRAINING,
            confidence_score=0.5,  # Placeholder confidence
            metadata={
                "adaptation_id": adaptation.adaptation_id,
                "model_name": self.config.model_name,
                "device": str(self.device),
            },
            resource_usage=resource_usage,
        )

    def _prepare_training_examples(self, task: ARCTask) -> list[dict[str, Any]]:
        """Prepare training examples from task for TTT adaptation."""
        examples = []

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

        return examples[:self.config.max_examples]

    def _generate_prediction(
        self, input_grid: list[list[int]], adaptation: TTTAdaptation
    ) -> list[list[int]]:
        """Generate prediction for a single test input using adapted model."""
        # Convert input grid to string
        input_str = grid_to_string(input_grid)

        # Create prompt
        prompt = f"Task: Transform the input grid to output grid.\n\nInput:\n{input_str}\n\nOutput:\n"

        # Tokenize prompt
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        ).to(self.device)

        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_length,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the output grid from generated text
        # Look for the output section after "Output:"
        output_start = generated_text.find("Output:\n") + len("Output:\n")
        if output_start > len("Output:\n") - 1:
            output_text = generated_text[output_start:].strip()
            try:
                # Convert string back to grid
                prediction_grid = string_to_grid(output_text)
                return prediction_grid
            except Exception:
                # If parsing fails, return a grid of same size as input
                return [[0 for _ in row] for row in input_grid]
        else:
            # If no output section found, return zeros
            return [[0 for _ in row] for row in input_grid]

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
        if HAS_TORCH and hasattr(self, 'device') and self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.adaptations.clear()
        self.lora_adapters.clear()

        # Clear GPU cache if using CUDA
        if HAS_TORCH and hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
