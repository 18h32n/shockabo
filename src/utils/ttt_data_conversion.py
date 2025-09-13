"""
MIT TTT Data Format Conversion Utilities

This module provides data format conversion between ARC data models and MIT TTT format,
implementing the exact data preprocessing approach from the MIT TTT research.
"""
import json
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.domain.models import ARCTask


class AugmentationType(Enum):
    """Types of augmentations supported by MIT TTT."""
    BASIC = "basic"  # Rotation, flip, transpose
    SIZE = "size"    # Grid size variations
    CHAIN = "chain"  # Chained transformations
    REPEAT = "repeat"  # Repeated patterns


@dataclass
class TTTExample:
    """Single training example in TTT format."""
    input_grid: List[List[int]]
    output_grid: List[List[int]]
    text_representation: str
    metadata: Dict[str, Any]


@dataclass
class TTTTask:
    """Task formatted for MIT TTT processing."""
    task_id: str
    examples: List[TTTExample]
    test_input: List[List[int]]
    augmented_examples: List[TTTExample]
    leave_one_out_splits: List[List[TTTExample]]
    metadata: Dict[str, Any]


class TextTaskRepresenter:
    """Converts ARC grids to text representation for LLM processing."""
    
    def __init__(self, use_gpt_format: bool = True):
        """
        Initialize text representer.
        
        Args:
            use_gpt_format: Whether to use GPT-style message formatting
        """
        self.use_gpt_format = use_gpt_format
    
    def grid_to_text(self, grid: List[List[int]]) -> str:
        """
        Convert grid to text representation.
        
        Args:
            grid: 2D grid with integer values (0-9)
            
        Returns:
            String representation of grid in Python list syntax
        """
        return str(grid)
    
    def create_example_text(
        self, 
        input_grid: List[List[int]], 
        output_grid: List[List[int]]
    ) -> str:
        """
        Create text representation of input-output example.
        
        Args:
            input_grid: Input grid
            output_grid: Expected output grid
            
        Returns:
            Formatted text representation
        """
        input_text = self.grid_to_text(input_grid)
        output_text = self.grid_to_text(output_grid)
        return f"{input_text} -> {output_text}"
    
    def create_task_prompt(
        self, 
        examples: List[Tuple[List[List[int]], List[List[int]]]], 
        test_input: List[List[int]]
    ) -> str:
        """
        Create complete task prompt in MIT TTT format.
        
        Args:
            examples: List of (input, output) example pairs
            test_input: Test input grid to predict
            
        Returns:
            Complete prompt text
        """
        example_texts = []
        for input_grid, output_grid in examples:
            example_texts.append(self.create_example_text(input_grid, output_grid))
        
        examples_str = "\n\n".join(example_texts)
        test_input_str = self.grid_to_text(test_input)
        
        prompt = f"""Transform the input grid according to the pattern shown in the examples.

Examples:
{examples_str}

Test input:
{test_input_str}

Test output:
"""
        
        if self.use_gpt_format:
            # Wrap in GPT-style message format with metadata
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"""Date: {current_time}
Role: assistant
Task: ARC Pattern Recognition

{prompt}"""
        
        return prompt


class AugmentationEngine:
    """Implements MIT TTT augmentation strategies."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize augmentation engine."""
        if random_seed is not None:
            random.seed(random_seed)
    
    def rotate_grid(self, grid: List[List[int]], times: int = 1) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise."""
        result = grid
        for _ in range(times % 4):
            rows, cols = len(result), len(result[0])
            result = [[result[rows - 1 - j][i] for j in range(rows)] for i in range(cols)]
        return result
    
    def flip_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid horizontally."""
        return [row[::-1] for row in grid]
    
    def flip_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid vertically."""
        return grid[::-1]
    
    def transpose_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Transpose grid (swap rows and columns)."""
        if not grid or not grid[0]:
            return grid
        return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
    
    def basic_augmentations(
        self, 
        input_grid: List[List[int]], 
        output_grid: List[List[int]]
    ) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """
        Apply basic augmentations (rotation, flip, transpose).
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            
        Returns:
            List of augmented (input, output) pairs
        """
        augmented = []
        
        # Original
        augmented.append((input_grid, output_grid))
        
        # Rotations
        for rot in [1, 2, 3]:
            aug_input = self.rotate_grid(input_grid, rot)
            aug_output = self.rotate_grid(output_grid, rot)
            augmented.append((aug_input, aug_output))
        
        # Flips
        aug_input = self.flip_horizontal(input_grid)
        aug_output = self.flip_horizontal(output_grid)
        augmented.append((aug_input, aug_output))
        
        aug_input = self.flip_vertical(input_grid)
        aug_output = self.flip_vertical(output_grid)
        augmented.append((aug_input, aug_output))
        
        # Transpose
        aug_input = self.transpose_grid(input_grid)
        aug_output = self.transpose_grid(output_grid)
        augmented.append((aug_input, aug_output))
        
        return augmented
    
    def size_augmentations(
        self, 
        input_grid: List[List[int]], 
        output_grid: List[List[int]],
        max_size: int = 30
    ) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """
        Apply size-based augmentations (padding, cropping).
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            max_size: Maximum grid size
            
        Returns:
            List of augmented (input, output) pairs
        """
        augmented = [(input_grid, output_grid)]
        
        rows, cols = len(input_grid), len(input_grid[0])
        
        # Padding augmentation
        if rows < max_size and cols < max_size:
            pad_rows = min(2, max_size - rows)
            pad_cols = min(2, max_size - cols)
            
            # Pad with zeros
            padded_input = input_grid[:]
            padded_output = output_grid[:]
            
            for _ in range(pad_rows):
                padded_input.append([0] * cols)
                if len(padded_output) < len(output_grid) + pad_rows:
                    padded_output.append([0] * len(output_grid[0]))
            
            for row_idx in range(len(padded_input)):
                if row_idx < len(padded_input):
                    padded_input[row_idx].extend([0] * pad_cols)
                if row_idx < len(padded_output):
                    padded_output[row_idx].extend([0] * pad_cols)
            
            augmented.append((padded_input, padded_output))
        
        return augmented
    
    def chain_augmentations(
        self, 
        input_grid: List[List[int]], 
        output_grid: List[List[int]],
        max_chains: int = 3
    ) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """
        Apply chained augmentations (combinations of basic transforms).
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            max_chains: Maximum number of chained operations
            
        Returns:
            List of augmented (input, output) pairs
        """
        augmented = [(input_grid, output_grid)]
        
        # Chain rotation + flip
        aug_input = self.flip_horizontal(self.rotate_grid(input_grid, 1))
        aug_output = self.flip_horizontal(self.rotate_grid(output_grid, 1))
        augmented.append((aug_input, aug_output))
        
        # Chain rotation + transpose
        aug_input = self.transpose_grid(self.rotate_grid(input_grid, 2))
        aug_output = self.transpose_grid(self.rotate_grid(output_grid, 2))
        augmented.append((aug_input, aug_output))
        
        return augmented[:max_chains + 1]
    
    def get_augmenters(self, augmentation_types: List[AugmentationType]) -> List[str]:
        """
        Get list of augmentation function names.
        
        Args:
            augmentation_types: Types of augmentations to include
            
        Returns:
            List of augmentation method names
        """
        augmenters = []
        
        if AugmentationType.BASIC in augmentation_types:
            augmenters.append("basic_augmentations")
        
        if AugmentationType.SIZE in augmentation_types:
            augmenters.append("size_augmentations")
        
        if AugmentationType.CHAIN in augmentation_types:
            augmenters.append("chain_augmentations")
        
        return augmenters


class TTTDataConverter:
    """Main converter for ARC to MIT TTT format."""
    
    def __init__(
        self, 
        use_gpt_format: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize TTT data converter.
        
        Args:
            use_gpt_format: Whether to use GPT-style message formatting
            random_seed: Random seed for reproducible augmentations
        """
        self.representer = TextTaskRepresenter(use_gpt_format)
        self.augmenter = AugmentationEngine(random_seed)
    
    def convert_arc_task(
        self, 
        arc_task: ARCTask,
        augmentation_types: Optional[List[AugmentationType]] = None
    ) -> TTTTask:
        """
        Convert ARC task to TTT format.
        
        Args:
            arc_task: ARC task to convert
            augmentation_types: Types of augmentations to apply
            
        Returns:
            TTT-formatted task
        """
        if augmentation_types is None:
            augmentation_types = [AugmentationType.BASIC]
        
        # Convert training examples
        examples = []
        for train_example in arc_task.train_examples:
            text_repr = self.representer.create_example_text(
                train_example["input"], 
                train_example["output"]
            )
            
            example = TTTExample(
                input_grid=train_example["input"],
                output_grid=train_example["output"],
                text_representation=text_repr,
                metadata={"original": True}
            )
            examples.append(example)
        
        # Generate augmented examples
        augmented_examples = []
        for train_example in arc_task.train_examples:
            input_grid = train_example["input"]
            output_grid = train_example["output"]
            
            for aug_type in augmentation_types:
                if aug_type == AugmentationType.BASIC:
                    aug_pairs = self.augmenter.basic_augmentations(input_grid, output_grid)
                elif aug_type == AugmentationType.SIZE:
                    aug_pairs = self.augmenter.size_augmentations(input_grid, output_grid)
                elif aug_type == AugmentationType.CHAIN:
                    aug_pairs = self.augmenter.chain_augmentations(input_grid, output_grid)
                else:
                    continue
                
                for aug_input, aug_output in aug_pairs[1:]:  # Skip original
                    text_repr = self.representer.create_example_text(aug_input, aug_output)
                    
                    aug_example = TTTExample(
                        input_grid=aug_input,
                        output_grid=aug_output,
                        text_representation=text_repr,
                        metadata={"original": False, "augmentation_type": aug_type.value}
                    )
                    augmented_examples.append(aug_example)
        
        # Create leave-one-out splits for per-instance training
        leave_one_out_splits = []
        all_examples = examples + augmented_examples
        
        for i in range(len(examples)):  # Only for original examples
            # Create training set excluding example i
            training_set = []
            for j, example in enumerate(all_examples):
                if j != i or not example.metadata.get("original", False):
                    training_set.append(example)
            
            leave_one_out_splits.append(training_set)
        
        # Create TTT task
        ttt_task = TTTTask(
            task_id=arc_task.task_id,
            examples=examples,
            test_input=arc_task.test_input,
            augmented_examples=augmented_examples,
            leave_one_out_splits=leave_one_out_splits,
            metadata={
                "original_task": arc_task.task_id,
                "num_train_examples": len(examples),
                "num_augmented_examples": len(augmented_examples),
                "augmentation_types": [t.value for t in augmentation_types],
                "conversion_timestamp": datetime.now().isoformat()
            }
        )
        
        return ttt_task
    
    def create_training_prompts(
        self, 
        ttt_task: TTTTask, 
        split_index: int = 0
    ) -> List[str]:
        """
        Create training prompts for test-time training.
        
        Args:
            ttt_task: TTT-formatted task
            split_index: Which leave-one-out split to use
            
        Returns:
            List of training prompts
        """
        if split_index >= len(ttt_task.leave_one_out_splits):
            split_index = 0
        
        training_examples = ttt_task.leave_one_out_splits[split_index]
        prompts = []
        
        for example in training_examples:
            # Create prompt with this example as target
            other_examples = [
                (e.input_grid, e.output_grid) 
                for e in training_examples 
                if e != example
            ]
            
            prompt = self.representer.create_task_prompt(
                other_examples, 
                example.input_grid
            )
            prompt += self.representer.grid_to_text(example.output_grid)
            
            prompts.append(prompt)
        
        return prompts
    
    def create_inference_prompt(self, ttt_task: TTTTask) -> str:
        """
        Create inference prompt for test input prediction.
        
        Args:
            ttt_task: TTT-formatted task
            
        Returns:
            Inference prompt
        """
        example_pairs = [
            (example.input_grid, example.output_grid) 
            for example in ttt_task.examples
        ]
        
        return self.representer.create_task_prompt(
            example_pairs, 
            ttt_task.test_input
        )
    
    def process_task(
        self, 
        arc_task: ARCTask,
        permute_n: int = 1,
        augmentation_types: Optional[List[AugmentationType]] = None
    ) -> Dict[str, Any]:
        """
        Complete MIT TTT processing for a single task.
        
        Args:
            arc_task: ARC task to process
            permute_n: Number of permutations for self-consistency
            augmentation_types: Types of augmentations to apply
            
        Returns:
            Dictionary with all TTT processing data
        """
        if augmentation_types is None:
            augmentation_types = [AugmentationType.BASIC]
        
        # Convert to TTT format
        ttt_task = self.convert_arc_task(arc_task, augmentation_types)
        
        # Generate multiple permutations for self-consistency
        permutations = []
        for perm in range(permute_n):
            # Create training prompts for each leave-one-out split
            training_data = []
            for split_idx in range(len(ttt_task.leave_one_out_splits)):
                prompts = self.create_training_prompts(ttt_task, split_idx)
                training_data.append({
                    "split_index": split_idx,
                    "prompts": prompts
                })
            
            # Create inference prompt
            inference_prompt = self.create_inference_prompt(ttt_task)
            
            permutations.append({
                "permutation_id": perm,
                "training_data": training_data,
                "inference_prompt": inference_prompt
            })
        
        return {
            "task_id": arc_task.task_id,
            "ttt_task": ttt_task,
            "permutations": permutations,
            "processing_metadata": {
                "permute_n": permute_n,
                "augmentation_types": [t.value for t in augmentation_types],
                "processed_at": datetime.now().isoformat()
            }
        }


def string_to_grid(grid_str: str) -> List[List[int]]:
    """
    Convert string representation back to grid.
    
    Args:
        grid_str: String representation of grid
        
    Returns:
        2D grid with integer values
    """
    try:
        # Parse as Python literal
        return eval(grid_str)
    except:
        # Fallback parsing
        lines = grid_str.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():
                # Try to parse as list
                try:
                    row = eval(line.strip())
                    if isinstance(row, list):
                        grid.append([int(x) for x in row])
                except:
                    continue
        return grid if grid else [[0]]