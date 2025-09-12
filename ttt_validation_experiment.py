#!/usr/bin/env python3
"""
TTT (Test-Time Training) Validation Experiment for ARC Prize 2025
This script validates the core assumption that TTT can achieve 60%+ accuracy
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class ARCTask:
    """Represents a single ARC task"""
    task_id: str
    train_inputs: list[np.ndarray]
    train_outputs: list[np.ndarray]
    test_inputs: list[np.ndarray]
    test_outputs: list[np.ndarray]

class SimpleARCModel(nn.Module):
    """
    Simplified neural model for ARC tasks
    This is a baseline - real implementation would use transformers
    """
    def __init__(self, max_grid_size=30, hidden_dim=512):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim

        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(max_grid_size * max_grid_size * 10, hidden_dim),  # 10 colors
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_grid_size * max_grid_size * 10),
        )

    def forward(self, x):
        # Flatten and encode
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Pad to max size
        if x_flat.shape[1] < self.max_grid_size * self.max_grid_size * 10:
            padding = torch.zeros(
                batch_size,
                self.max_grid_size * self.max_grid_size * 10 - x_flat.shape[1],
                device=x.device
            )
            x_flat = torch.cat([x_flat, padding], dim=1)

        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)

        return decoded

class TTTValidator:
    """
    Validates Test-Time Training approach on ARC tasks
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SimpleARCModel().to(device)
        self.base_model_state = None

    def load_arc_data(self, data_path: Path, max_tasks: int = 100) -> list[ARCTask]:
        """Load ARC tasks from JSON files"""
        tasks = []

        # Try to load evaluation data
        eval_dir = data_path / "evaluation"
        if eval_dir.exists():
            json_files = list(eval_dir.glob("*.json"))[:max_tasks]

            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    task = ARCTask(
                        task_id=json_file.stem,
                        train_inputs=[np.array(ex['input']) for ex in data['train']],
                        train_outputs=[np.array(ex['output']) for ex in data['train']],
                        test_inputs=[np.array(ex['input']) for ex in data['test']],
                        test_outputs=[np.array(ex['output']) for ex in data['test']]
                    )
                    tasks.append(task)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(tasks)} ARC tasks")
        return tasks

    def grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convert ARC grid to one-hot tensor"""
        h, w = grid.shape
        one_hot = torch.zeros(10, h, w)  # 10 possible colors
        for i in range(h):
            for j in range(w):
                if grid[i, j] < 10:
                    one_hot[int(grid[i, j]), i, j] = 1
        return one_hot.float()

    def tensor_to_grid(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to ARC grid"""
        # Reshape and get argmax
        if len(tensor.shape) == 1:
            # Reshape to 2D grid
            size = int(np.sqrt(tensor.shape[0] // 10))
            tensor = tensor.view(10, size, size)

        grid = torch.argmax(tensor, dim=0).cpu().numpy()
        return grid

    def test_time_train(self, task: ARCTask, max_steps: int = 100, lr: float = 0.001) -> float:
        """
        Perform test-time training on a single task
        Returns accuracy on test examples
        """
        # Reset model to base state
        if self.base_model_state is not None:
            self.model.load_state_dict(self.base_model_state.copy())

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # Use MSE for simplicity

        # Prepare training data from task examples
        train_x = []
        train_y = []

        for inp, out in zip(task.train_inputs, task.train_outputs, strict=False):
            train_x.append(self.grid_to_tensor(inp))
            train_y.append(self.grid_to_tensor(out))

        if not train_x:
            return 0.0

        # TTT: Adapt model on this specific task
        if max_steps > 0:
            self.model.train()
            for _step in range(max_steps):
                total_loss = 0

                for x, y in zip(train_x, train_y, strict=False):
                    x = x.unsqueeze(0).to(self.device)
                    # Flatten y to match model output
                    y_flat = y.view(1, -1).to(self.device)

                    # Pad y if needed
                    if y_flat.shape[1] < self.model.max_grid_size * self.model.max_grid_size * 10:
                        padding = torch.zeros(
                            1,
                            self.model.max_grid_size * self.model.max_grid_size * 10 - y_flat.shape[1],
                            device=self.device
                        )
                        y_flat = torch.cat([y_flat, padding], dim=1)

                    optimizer.zero_grad()
                    pred = self.model(x)

                    # Calculate loss
                    loss = criterion(pred, y_flat)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Early stopping if loss is low
                if total_loss / len(train_x) < 0.01:
                    break

        # Evaluate on test examples
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for test_inp, test_out in zip(task.test_inputs, task.test_outputs, strict=False):
                x = self.grid_to_tensor(test_inp).unsqueeze(0).to(self.device)
                pred = self.model(x)
                pred_grid = self.tensor_to_grid(pred[0])

                # Check if prediction matches output
                if pred_grid.shape == test_out.shape:
                    if np.array_equal(pred_grid, test_out):
                        correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def run_validation(self, tasks: list[ARCTask]) -> dict:
        """
        Run validation experiment on multiple tasks
        """
        # Save initial model state
        self.base_model_state = self.model.state_dict().copy()

        results = {
            'task_accuracies': [],
            'baseline_accuracies': [],
            'ttt_improvements': [],
            'times': []
        }

        print("\nRunning TTT Validation Experiment")
        print("=" * 50)

        for task in tqdm(tasks, desc="Processing tasks"):
            start_time = time.time()

            # Get baseline (zero-shot) accuracy
            baseline_acc = self.test_time_train(task, max_steps=0)

            # Get TTT accuracy
            ttt_acc = self.test_time_train(task, max_steps=50)

            elapsed = time.time() - start_time

            results['baseline_accuracies'].append(baseline_acc)
            results['task_accuracies'].append(ttt_acc)
            results['ttt_improvements'].append(ttt_acc - baseline_acc)
            results['times'].append(elapsed)

        # Calculate summary statistics
        results['summary'] = {
            'mean_baseline_accuracy': np.mean(results['baseline_accuracies']),
            'mean_ttt_accuracy': np.mean(results['task_accuracies']),
            'mean_improvement': np.mean(results['ttt_improvements']),
            'tasks_improved': sum(1 for x in results['ttt_improvements'] if x > 0),
            'tasks_above_60pct': sum(1 for x in results['task_accuracies'] if x >= 0.6),
            'mean_time_per_task': np.mean(results['times']),
            'total_tasks': len(tasks)
        }

        return results

def main():
    """Main validation experiment"""
    print("ARC Prize 2025 - TTT Validation Experiment")
    print("=" * 50)

    # Initialize validator
    validator = TTTValidator()

    # Load ARC data
    data_path = Path("data/arc-agi")
    tasks = validator.load_arc_data(data_path, max_tasks=20)  # Start with 20 for quick test

    if not tasks:
        print("No tasks loaded. Creating synthetic tasks for demonstration...")
        # Create synthetic tasks for testing
        tasks = []
        for i in range(5):
            task = ARCTask(
                task_id=f"synthetic_{i}",
                train_inputs=[np.random.randint(0, 3, (3, 3)) for _ in range(3)],
                train_outputs=[np.random.randint(0, 3, (3, 3)) for _ in range(3)],
                test_inputs=[np.random.randint(0, 3, (3, 3)) for _ in range(1)],
                test_outputs=[np.random.randint(0, 3, (3, 3)) for _ in range(1)]
            )
            tasks.append(task)

    # Run validation
    results = validator.run_validation(tasks)

    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    summary = results['summary']
    print(f"Total tasks evaluated: {summary['total_tasks']}")
    print(f"Mean baseline accuracy: {summary['mean_baseline_accuracy']:.2%}")
    print(f"Mean TTT accuracy: {summary['mean_ttt_accuracy']:.2%}")
    print(f"Mean improvement: {summary['mean_improvement']:.2%}")
    print(f"Tasks improved by TTT: {summary['tasks_improved']}/{summary['total_tasks']}")
    print(f"Tasks achieving ≥60% accuracy: {summary['tasks_above_60pct']}/{summary['total_tasks']}")
    print(f"Mean time per task: {summary['mean_time_per_task']:.2f}s")

    # Validation decision
    print("\n" + "=" * 50)
    print("VALIDATION DECISION")
    print("=" * 50)

    if summary['mean_ttt_accuracy'] >= 0.6:
        print("✓ ASSUMPTION VALIDATED: TTT can achieve 60%+ accuracy")
        print("  Recommendation: Proceed with full TTT implementation")
    elif summary['mean_ttt_accuracy'] >= 0.4:
        print("⚠ PARTIAL VALIDATION: TTT shows promise but needs optimization")
        print("  Recommendation: Investigate advanced TTT techniques")
    else:
        print("✗ ASSUMPTION CHALLENGED: TTT alone may not reach 60%")
        print("  Recommendation: Consider hybrid approaches or novel architectures")

    # Save results
    with open("ttt_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print("\nDetailed results saved to ttt_validation_results.json")

if __name__ == "__main__":
    main()
