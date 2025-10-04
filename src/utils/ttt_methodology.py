"""
MIT Test-Time Training Methodology Implementation

This module implements the core TTT methodology from MIT research,
including per-instance adaptation, self-consistency, and memory-efficient training.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.auth_config import get_model_access_info, setup_hf_auth, suggest_public_model
from src.utils.lora_adapter import LoRAAdapter, LoRAConfig
from src.utils.ttt_data_conversion import AugmentationType, TTTDataConverter, TTTTask

logger = logging.getLogger(__name__)


@dataclass
class TTTTrainingConfig:
    """Configuration for MIT TTT training methodology."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    device: str = "auto"
    quantization: bool = True
    mixed_precision: bool = True

    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = None

    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Per-instance training
    per_instance_epochs: int = 1
    per_instance_lr: float = 1e-4
    max_training_time: float = 300.0  # 5 minutes per task

    # Self-consistency
    permute_n: int = 1
    temperature: float = 0.0
    num_return_sequences: int = 1

    # Memory management
    max_sequence_length: int = 2048
    gradient_checkpointing: bool = True
    memory_limit_mb: float = 10240  # 10GB

    # Augmentation
    augmentation_types: list[AugmentationType] = None

    # Paths
    cache_dir: Path = Path("data/cache/ttt")
    checkpoint_dir: Path = Path("data/models/ttt")

    def __post_init__(self):
        """Post-initialization setup."""
        if self.lora_target_modules is None:
            # Default target modules supporting both GPT-2 (Conv1D) and Llama (Linear)
            self.lora_target_modules = [
                "c_attn", "c_proj",  # GPT-2 Conv1D modules
                "q_proj", "v_proj", "k_proj", "o_proj",  # Llama Linear modules
                "gate_proj", "up_proj", "down_proj"
            ]

        if self.augmentation_types is None:
            self.augmentation_types = [AugmentationType.BASIC]

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TTTAdaptationResult:
    """Result of TTT adaptation process."""

    adaptation_id: str
    task_id: str
    training_metrics: dict[str, float]
    adaptation_time: float
    memory_usage: dict[str, float]
    adapter_path: str | None = None
    success: bool = True
    error_message: str | None = None


class SelfConsistencyVoter:
    """Implements self-consistency voting for multiple predictions."""

    def __init__(self, temperature: float = 0.0):
        """Initialize voter with temperature for diversity."""
        self.temperature = temperature

    def vote_predictions(
        self,
        predictions: list[list[list[int]]],
        confidence_scores: list[float] | None = None
    ) -> tuple[list[list[int]], float]:
        """
        Vote on multiple predictions using majority voting.

        Args:
            predictions: List of predicted grids
            confidence_scores: Optional confidence scores for weighting

        Returns:
            Tuple of (best_prediction, confidence_score)
        """
        if not predictions:
            return [[0]], 0.0

        if len(predictions) == 1:
            return predictions[0], 1.0

        # Convert predictions to strings for comparison
        pred_strings = [str(pred) for pred in predictions]

        # Count occurrences
        vote_counts = {}
        for i, pred_str in enumerate(pred_strings):
            weight = confidence_scores[i] if confidence_scores else 1.0
            vote_counts[pred_str] = vote_counts.get(pred_str, 0) + weight

        # Find most voted prediction
        best_pred_str = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        best_prediction = predictions[pred_strings.index(best_pred_str)]

        # Calculate confidence as agreement ratio
        total_votes = sum(vote_counts.values())
        max_votes = vote_counts[best_pred_str]
        confidence = max_votes / total_votes if total_votes > 0 else 0.0

        return best_prediction, confidence


class TTTTrainer:
    """MIT TTT methodology trainer for per-instance adaptation."""

    def __init__(self, config: TTTTrainingConfig):
        """Initialize TTT trainer."""
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.base_adapter = None
        self.data_converter = TTTDataConverter(use_gpt_format=True)
        self.voter = SelfConsistencyVoter(temperature=config.temperature)

        # Training state
        self.current_adapter = None
        self.training_history = []

        logger.info(f"Initialized TTT trainer with device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def initialize_model(self) -> None:
        """Initialize base model and tokenizer."""
        if self.model is not None:
            return

        # Set up authentication and check access
        auth_setup = setup_hf_auth()
        if auth_setup:
            logger.info("HuggingFace authentication configured")

        access_info = get_model_access_info(self.config.model_name)
        model_name = self.config.model_name

        if not access_info["can_access"]:
            logger.warning(f"Cannot access {model_name} - using public alternative")
            model_name = suggest_public_model(model_name)
            logger.info(f"Using model: {model_name}")
        else:
            logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading configuration
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.mixed_precision else torch.float32,
        }

        if self.config.quantization and self.device.type == "cuda":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Move to device if not using device_map
        if model_kwargs["device_map"] is None:
            self.model = self.model.to(self.device)

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Create base LoRA adapter
        lora_config = LoRAConfig(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules
        )

        self.base_adapter = LoRAAdapter(self.model, lora_config)

        logger.info("Model initialization complete")
        self.base_adapter.print_trainable_parameters()

    def _prepare_training_batch(
        self,
        prompts: list[str],
        max_length: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Prepare training batch from prompts."""
        if max_length is None:
            max_length = self.config.max_sequence_length

        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        batch = {k: v.to(self.device) for k, v in encoded.items()}

        # Labels are same as input_ids for causal LM
        batch["labels"] = batch["input_ids"].clone()

        return batch

    def _train_on_prompts(
        self,
        prompts: list[str],
        adapter: LoRAAdapter,
        learning_rate: float,
        num_epochs: int
    ) -> dict[str, float]:
        """Train adapter on list of prompts."""
        # Setup optimizer
        optimizer = AdamW(
            adapter.get_trainable_parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Setup scheduler
        total_steps = (len(prompts) // self.config.batch_size + 1) * num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            eta_min=learning_rate * 0.1
        )

        # Training metrics
        metrics = {
            "total_loss": 0.0,
            "num_steps": 0,
            "avg_loss": 0.0
        }

        self.model.train()

        for _epoch in range(num_epochs):
            # Process prompts in batches
            for i in range(0, len(prompts), self.config.batch_size):
                batch_prompts = prompts[i:i + self.config.batch_size]
                batch = self._prepare_training_batch(batch_prompts)

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Scale loss for gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights
                if ((metrics["num_steps"] + 1) % self.config.gradient_accumulation_steps == 0):
                    torch.nn.utils.clip_grad_norm_(
                        adapter.get_trainable_parameters(),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Track metrics
                metrics["total_loss"] += loss.item()
                metrics["num_steps"] += 1

        # Calculate average loss
        if metrics["num_steps"] > 0:
            metrics["avg_loss"] = metrics["total_loss"] / metrics["num_steps"]

        self.model.eval()
        return metrics

    def adapt_to_task(
        self,
        ttt_task: TTTTask,
        adaptation_id: str | None = None
    ) -> TTTAdaptationResult:
        """
        Perform per-instance adaptation using MIT TTT methodology.

        Args:
            ttt_task: TTT-formatted task
            adaptation_id: Optional adaptation identifier

        Returns:
            TTT adaptation result
        """
        if adaptation_id is None:
            adaptation_id = f"ttt_{ttt_task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        start_time = time.time()

        try:
            self.initialize_model()

            # Create per-instance adapter
            lora_config = LoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules
            )

            instance_adapter = LoRAAdapter(self.model, lora_config)

            # Prepare training data from leave-one-out splits
            all_training_prompts = []

            for split_data in ttt_task.leave_one_out_splits:
                # Create training prompts for this split
                split_prompts = []

                for example in split_data:
                    # Create prompt for this example
                    other_examples = [
                        (e.input_grid, e.output_grid)
                        for e in split_data
                        if e != example
                    ]

                    if other_examples:  # Only if we have other examples
                        prompt = self.data_converter.representer.create_task_prompt(
                            other_examples,
                            example.input_grid
                        )
                        prompt += self.data_converter.representer.grid_to_text(example.output_grid)
                        split_prompts.append(prompt)

                all_training_prompts.extend(split_prompts)

            # Remove duplicates while preserving order
            seen = set()
            unique_prompts = []
            for prompt in all_training_prompts:
                if prompt not in seen:
                    seen.add(prompt)
                    unique_prompts.append(prompt)

            # Train adapter on prepared prompts
            training_metrics = self._train_on_prompts(
                unique_prompts,
                instance_adapter,
                self.config.per_instance_lr,
                self.config.per_instance_epochs
            )

            # Save adapter
            adapter_path = self.config.checkpoint_dir / f"{adaptation_id}.pt"
            instance_adapter.save_adapter(str(adapter_path))

            # Store current adapter
            self.current_adapter = instance_adapter

            adaptation_time = time.time() - start_time

            # Estimate memory usage
            memory_usage = {
                "peak_memory_mb": self._get_peak_memory(),
                "current_memory_mb": self._get_current_memory()
            }

            result = TTTAdaptationResult(
                adaptation_id=adaptation_id,
                task_id=ttt_task.task_id,
                training_metrics=training_metrics,
                adaptation_time=adaptation_time,
                memory_usage=memory_usage,
                adapter_path=str(adapter_path),
                success=True
            )

            logger.info(f"Successfully adapted to task {ttt_task.task_id} in {adaptation_time:.2f}s")
            return result

        except Exception as e:
            adaptation_time = time.time() - start_time
            error_msg = f"Adaptation failed: {str(e)}"
            logger.error(error_msg)

            return TTTAdaptationResult(
                adaptation_id=adaptation_id,
                task_id=ttt_task.task_id,
                training_metrics={},
                adaptation_time=adaptation_time,
                memory_usage={},
                success=False,
                error_message=error_msg
            )

    def generate_prediction(
        self,
        ttt_task: TTTTask,
        use_self_consistency: bool = True
    ) -> tuple[list[list[int]], float]:
        """
        Generate prediction using adapted model.

        Args:
            ttt_task: TTT-formatted task
            use_self_consistency: Whether to use self-consistency voting

        Returns:
            Tuple of (prediction_grid, confidence_score)
        """
        self.initialize_model()

        predictions = []
        confidence_scores = []

        # Generate multiple predictions for self-consistency
        num_predictions = self.config.permute_n if use_self_consistency else 1

        for _ in range(num_predictions):
            try:
                # Create inference prompt
                inference_prompt = self.data_converter.create_inference_prompt(ttt_task)

                # Tokenize prompt
                inputs = self.tokenizer.encode(
                    inference_prompt,
                    return_tensors="pt",
                    max_length=self.config.max_sequence_length,
                    truncation=True
                ).to(self.device)

                # Generate prediction
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=500,
                        temperature=self.config.temperature,
                        do_sample=self.config.temperature > 0,
                        num_return_sequences=self.config.num_return_sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )

                # Decode generated text
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # Extract prediction grid
                prediction_grid = self._extract_prediction_grid(
                    generated_text,
                    ttt_task.test_input
                )

                predictions.append(prediction_grid)
                confidence_scores.append(1.0)  # Placeholder confidence

            except Exception as e:
                logger.warning(f"Failed to generate prediction: {e}")
                # Fallback to input grid
                predictions.append(ttt_task.test_input)
                confidence_scores.append(0.1)

        # Use self-consistency voting if multiple predictions
        if len(predictions) > 1 and use_self_consistency:
            final_prediction, confidence = self.voter.vote_predictions(
                predictions,
                confidence_scores
            )
        else:
            final_prediction = predictions[0] if predictions else ttt_task.test_input
            confidence = confidence_scores[0] if confidence_scores else 0.1

        return final_prediction, confidence

    def _extract_prediction_grid(
        self,
        generated_text: str,
        fallback_grid: list[list[int]]
    ) -> list[list[int]]:
        """Extract prediction grid from generated text."""
        try:
            # Look for output section
            if "Test output:" in generated_text:
                output_start = generated_text.find("Test output:") + len("Test output:")
                output_text = generated_text[output_start:].strip()
            elif "Output:" in generated_text:
                output_start = generated_text.find("Output:") + len("Output:")
                output_text = generated_text[output_start:].strip()
            else:
                output_text = generated_text.strip()

            # Try to parse as Python literal
            lines = output_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    try:
                        grid = eval(line)
                        if isinstance(grid, list) and len(grid) > 0:
                            return grid
                    except Exception:
                        continue

            # Fallback: try to parse entire output
            try:
                grid = eval(output_text)
                if isinstance(grid, list) and len(grid) > 0:
                    return grid
            except Exception:
                pass

            # Final fallback
            return fallback_grid

        except Exception:
            return fallback_grid

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.base_adapter is not None:
            del self.base_adapter
            self.base_adapter = None

        if self.current_adapter is not None:
            del self.current_adapter
            self.current_adapter = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("TTT trainer cleanup complete")


class MIT_TTTStrategy:
    """Complete MIT TTT strategy implementation."""

    def __init__(self, config: TTTTrainingConfig):
        """Initialize MIT TTT strategy."""
        self.config = config
        self.trainer = TTTTrainer(config)
        self.data_converter = TTTDataConverter(use_gpt_format=True)

    def solve_task(
        self,
        arc_task,  # ARCTask type
        use_self_consistency: bool = True
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """
        Solve ARC task using complete MIT TTT methodology.

        Args:
            arc_task: ARC task to solve
            use_self_consistency: Whether to use self-consistency voting

        Returns:
            Tuple of (prediction, metadata)
        """
        start_time = time.time()

        # Convert to TTT format
        ttt_task = self.data_converter.convert_arc_task(
            arc_task,
            self.config.augmentation_types
        )

        # Perform per-instance adaptation
        adaptation_result = self.trainer.adapt_to_task(ttt_task)

        if not adaptation_result.success:
            # Return fallback prediction
            return arc_task.test_input, {
                "success": False,
                "error": adaptation_result.error_message,
                "fallback_used": True
            }

        # Generate prediction with adapted model
        prediction, confidence = self.trainer.generate_prediction(
            ttt_task,
            use_self_consistency
        )

        total_time = time.time() - start_time

        metadata = {
            "success": True,
            "adaptation_result": adaptation_result,
            "confidence": confidence,
            "total_time": total_time,
            "ttt_methodology": "MIT_TTT",
            "permutations": self.config.permute_n,
            "augmentations": [t.value for t in self.config.augmentation_types]
        }

        return prediction, metadata

    def cleanup(self) -> None:
        """Clean up strategy resources."""
        self.trainer.cleanup()
