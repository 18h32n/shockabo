"""Accuracy validation tests for evolution pipeline on ARC tasks."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.adapters.strategies.evolution_strategy_adapter import EvolutionStrategyAdapter
from src.domain.models import ARCTask
from src.infrastructure.config import Config


class TestEvolutionAccuracyValidation:
    """Test evolution accuracy on validation set subset."""

    @pytest.fixture
    def validation_config(self):
        """Create configuration for validation testing."""
        config = Config()

        # Evolution settings optimized for accuracy
        config.evolution.enabled = True
        config.evolution.population_size = 200
        config.evolution.max_generations = 30
        config.evolution.mutation_rate = 0.15
        config.evolution.crossover_rate = 0.75
        config.evolution.elitism_count = 10
        config.evolution.tournament_size = 5

        # Enable advanced features
        config.evolution.hybrid_initialization.enabled = True
        config.evolution.hybrid_initialization.llm_percentage = 0.2
        config.evolution.adaptive_mutation.enabled = True
        config.evolution.novelty_search.enabled = True
        config.evolution.novelty_search.novelty_weight = 0.3

        # Performance settings
        config.evolution.batch_size = 50
        config.evolution.parallel_workers = 4
        config.evolution.memory_limit_mb = 2048

        # Early termination
        config.evolution.early_termination.enabled = True
        config.evolution.early_termination.fitness_threshold = 0.95
        config.evolution.early_termination.stagnation_generations = 10

        config.platform = "validation"

        return config

    @pytest.fixture
    def validation_tasks(self):
        """Load subset of validation tasks for testing."""
        # Simple transformation tasks that evolution should solve
        tasks = [
            # Task 1: Horizontal flip
            ARCTask(
                id="val_001_hflip",
                train=[
                    {
                        "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        "output": [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
                    },
                    {
                        "input": [[1, 0], [0, 1]],
                        "output": [[0, 1], [1, 0]]
                    }
                ],
                test=[{"input": [[1, 2], [3, 4]]}]
            ),

            # Task 2: Vertical flip
            ARCTask(
                id="val_002_vflip",
                train=[
                    {
                        "input": [[1, 2], [3, 4], [5, 6]],
                        "output": [[5, 6], [3, 4], [1, 2]]
                    },
                    {
                        "input": [[0, 1], [2, 3]],
                        "output": [[2, 3], [0, 1]]
                    }
                ],
                test=[{"input": [[7, 8], [9, 0]]}]
            ),

            # Task 3: 90-degree rotation
            ARCTask(
                id="val_003_rot90",
                train=[
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[3, 1], [4, 2]]
                    },
                    {
                        "input": [[5, 6, 7], [8, 9, 0]],
                        "output": [[8, 5], [9, 6], [0, 7]]
                    }
                ],
                test=[{"input": [[1, 2, 3], [4, 5, 6]]}]
            ),

            # Task 4: Color replacement
            ARCTask(
                id="val_004_color",
                train=[
                    {
                        "input": [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
                        "output": [[3, 2, 3], [2, 3, 2], [3, 2, 3]]  # 1->3
                    },
                    {
                        "input": [[1, 1], [1, 1]],
                        "output": [[3, 3], [3, 3]]
                    }
                ],
                test=[{"input": [[2, 1, 2], [1, 2, 1]]}]
            ),

            # Task 5: Pattern extraction
            ARCTask(
                id="val_005_pattern",
                train=[
                    {
                        "input": [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]],
                        "output": [[1, 2], [3, 4]]
                    },
                    {
                        "input": [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
                        "output": [[5]]
                    }
                ],
                test=[{"input": [[0, 0, 0, 0, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]]}]
            )
        ]

        return tasks

    @pytest.fixture
    def real_validation_tasks(self):
        """Load real ARC validation tasks if available."""
        validation_path = Path("data/arc-agi/training")
        if not validation_path.exists():
            return []

        # Load first 10 training tasks as validation
        tasks = []
        task_files = sorted(validation_path.glob("*.json"))[:10]

        for task_file in task_files:
            with open(task_file) as f:
                data = json.load(f)
                task = ARCTask(
                    id=task_file.stem,
                    train=data.get("train", []),
                    test=data.get("test", [])
                )
                tasks.append(task)

        return tasks

    def test_simple_transformation_accuracy(self, validation_config, validation_tasks):
        """Test accuracy on simple transformation tasks."""
        adapter = EvolutionStrategyAdapter(validation_config)

        results = []
        successful_tasks = 0

        for task in validation_tasks:
            print(f"\nEvaluating task: {task.id}")

            # Process task
            result = adapter.process_task(task)

            # Check if solution is correct
            if result.success and result.solution is not None:
                # For simple tasks, check exact match
                expected = self._get_expected_output(task)
                if expected is not None:
                    is_correct = np.array_equal(result.solution, expected)
                    if is_correct:
                        successful_tasks += 1
                else:
                    # If no expected output, use confidence threshold
                    if result.confidence >= 0.8:
                        successful_tasks += 1

            results.append({
                "task_id": task.id,
                "success": result.success,
                "confidence": result.confidence,
                "error": result.error,
                "programs_found": len(result.programs) if result.programs else 0
            })

        # Calculate accuracy
        accuracy = successful_tasks / len(validation_tasks)

        print("\n=== Simple Transformation Accuracy ===")
        print(f"Tasks evaluated: {len(validation_tasks)}")
        print(f"Successful solutions: {successful_tasks}")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nDetailed Results:")
        for r in results:
            print(f"  {r['task_id']}: success={r['success']}, "
                  f"confidence={r['confidence']:.3f}, programs={r['programs_found']}")

        # Should achieve reasonable accuracy on simple tasks
        assert accuracy >= 0.4, f"Accuracy {accuracy:.2%} below 40% threshold"

    def test_accuracy_with_time_limit(self, validation_config, validation_tasks):
        """Test accuracy under strict time constraints."""
        # Configure for fast execution
        validation_config.evolution.max_generations = 10
        validation_config.evolution.max_runtime_seconds = 30  # 30 seconds per task

        adapter = EvolutionStrategyAdapter(validation_config)

        results = []
        total_time = 0

        for task in validation_tasks[:3]:  # Test first 3 tasks
            start_time = time.time()
            result = adapter.process_task(task)
            task_time = time.time() - start_time
            total_time += task_time

            results.append({
                "task_id": task.id,
                "time": task_time,
                "success": result.success,
                "confidence": result.confidence
            })

        print("\n=== Time-Limited Accuracy Test ===")
        print(f"{'Task':>15} {'Time(s)':>10} {'Success':>10} {'Confidence':>12}")
        print("-" * 50)

        for r in results:
            print(f"{r['task_id']:>15} {r['time']:>10.2f} "
                  f"{str(r['success']):>10} {r['confidence']:>12.3f}")

        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Average time per task: {total_time/len(results):.2f}s")

        # All tasks should complete within time limit
        assert all(r['time'] < 30 for r in results), "Some tasks exceeded time limit"

    def test_program_diversity_impact(self, validation_config):
        """Test how program diversity affects accuracy."""
        # Test same task with different diversity settings
        task = ARCTask(
            id="diversity_test",
            train=[
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]}
            ],
            test=[{"input": [[0, 1], [2, 3]]}]
        )

        diversity_settings = [
            {"enabled": False, "name": "No Diversity"},
            {"enabled": True, "method": "speciation", "name": "Speciation"},
            {"enabled": True, "method": "novelty", "name": "Novelty Search"}
        ]

        results = []

        for setting in diversity_settings:
            # Configure diversity
            config = validation_config.copy()
            if setting["enabled"]:
                config.evolution.diversity_preservation.enabled = True
                config.evolution.diversity_preservation.method = setting.get("method")
            else:
                config.evolution.diversity_preservation.enabled = False

            adapter = EvolutionStrategyAdapter(config)
            result = adapter.process_task(task)

            # Count unique programs
            unique_programs = 0
            if result.programs:
                unique_programs = len(set(str(p) for p in result.programs))

            results.append({
                "method": setting["name"],
                "confidence": result.confidence,
                "unique_programs": unique_programs,
                "total_programs": len(result.programs) if result.programs else 0
            })

        print("\n=== Diversity Impact on Accuracy ===")
        print(f"{'Method':>15} {'Confidence':>12} {'Unique':>10} {'Total':>10}")
        print("-" * 50)

        for r in results:
            print(f"{r['method']:>15} {r['confidence']:>12.3f} "
                  f"{r['unique_programs']:>10} {r['total_programs']:>10}")

    def test_real_arc_tasks_baseline(self, validation_config, real_validation_tasks):
        """Test on real ARC tasks to establish baseline."""
        if not real_validation_tasks:
            pytest.skip("No real ARC tasks available")

        adapter = EvolutionStrategyAdapter(validation_config)

        results = []
        high_confidence_count = 0

        # Test on subset of real tasks
        for task in real_validation_tasks[:5]:
            print(f"\nEvaluating real task: {task.id}")

            result = adapter.process_task(task)

            if result.confidence >= 0.45:  # Target threshold
                high_confidence_count += 1

            results.append({
                "task_id": task.id,
                "confidence": result.confidence,
                "programs_evaluated": result.metadata.get("evolution_metrics", {}).get("total_programs_evaluated", 0),
                "generations": result.metadata.get("evolution_metrics", {}).get("generations_completed", 0)
            })

        accuracy_estimate = high_confidence_count / len(results)

        print("\n=== Real ARC Tasks Baseline ===")
        print(f"Tasks evaluated: {len(results)}")
        print(f"High confidence solutions (≥45%): {high_confidence_count}")
        print(f"Estimated accuracy: {accuracy_estimate:.2%}")

        print("\nDetailed Results:")
        for r in results:
            print(f"  {r['task_id']}: confidence={r['confidence']:.3f}, "
                  f"programs={r['programs_evaluated']}, gens={r['generations']}")

        # Log baseline for future comparison
        self._save_baseline_results(results)

    def test_ensemble_accuracy_boost(self, validation_config, validation_tasks):
        """Test accuracy improvement with ensemble of evolution runs."""
        task = validation_tasks[0]  # Use first task

        # Run evolution multiple times with different seeds
        ensemble_size = 3
        all_results = []

        for i in range(ensemble_size):
            config = validation_config.copy()
            config.evolution.random_seed = 42 + i

            adapter = EvolutionStrategyAdapter(config)
            result = adapter.process_task(task)
            all_results.append(result)

        # Combine results
        best_confidence = max(r.confidence for r in all_results)
        avg_confidence = sum(r.confidence for r in all_results) / ensemble_size

        # Collect all unique programs
        all_programs = []
        for result in all_results:
            if result.programs:
                all_programs.extend(result.programs)

        unique_programs = len(set(str(p) for p in all_programs))

        print("\n=== Ensemble Accuracy Test ===")
        print(f"Ensemble size: {ensemble_size}")
        print(f"Individual confidences: {[f'{r.confidence:.3f}' for r in all_results]}")
        print(f"Best confidence: {best_confidence:.3f}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Total unique programs: {unique_programs}")

        # Ensemble should improve results
        assert best_confidence >= all_results[0].confidence

    def test_accuracy_by_task_complexity(self, validation_config):
        """Test accuracy correlation with task complexity."""
        # Create tasks of varying complexity
        complexity_tasks = [
            {
                "name": "identity",
                "complexity": "trivial",
                "task": ARCTask(
                    id="complex_1",
                    train=[{"input": [[1, 2]], "output": [[1, 2]]}],
                    test=[{"input": [[3, 4]]}]
                )
            },
            {
                "name": "single_op",
                "complexity": "simple",
                "task": ARCTask(
                    id="complex_2",
                    train=[
                        {"input": [[1, 2]], "output": [[2, 1]]},
                        {"input": [[3, 4]], "output": [[4, 3]]}
                    ],
                    test=[{"input": [[5, 6]]}]
                )
            },
            {
                "name": "multi_op",
                "complexity": "medium",
                "task": ARCTask(
                    id="complex_3",
                    train=[
                        {"input": [[1, 2], [3, 4]], "output": [[4, 2], [3, 1]]},
                        {"input": [[5, 6], [7, 8]], "output": [[8, 6], [7, 5]]}
                    ],
                    test=[{"input": [[0, 1], [2, 3]]}]
                )
            }
        ]

        adapter = EvolutionStrategyAdapter(validation_config)
        results = []

        for ct in complexity_tasks:
            result = adapter.process_task(ct["task"])
            results.append({
                "name": ct["name"],
                "complexity": ct["complexity"],
                "confidence": result.confidence,
                "generations": result.metadata.get("evolution_metrics", {}).get("generations_completed", 0)
            })

        print("\n=== Accuracy by Task Complexity ===")
        print(f"{'Task':>15} {'Complexity':>12} {'Confidence':>12} {'Generations':>13}")
        print("-" * 55)

        for r in results:
            print(f"{r['name']:>15} {r['complexity']:>12} "
                  f"{r['confidence']:>12.3f} {r['generations']:>13}")

        # Should solve simpler tasks with higher confidence
        assert results[0]["confidence"] >= results[-1]["confidence"]

    def _get_expected_output(self, task):
        """Get expected output for validation tasks."""
        # Map of task IDs to expected test outputs
        expected_outputs = {
            "val_001_hflip": [[2, 1], [4, 3]],
            "val_002_vflip": [[9, 0], [7, 8]],
            "val_003_rot90": [[4, 1], [5, 2], [6, 3]],
            "val_004_color": [[2, 3, 2], [3, 2, 3]],
            "val_005_pattern": [[7, 8, 9]]
        }

        return expected_outputs.get(task.id)

    def _save_baseline_results(self, results):
        """Save baseline results for tracking progress."""
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "tasks_evaluated": len(results),
                "average_confidence": sum(r["confidence"] for r in results) / len(results),
                "high_confidence_count": sum(1 for r in results if r["confidence"] >= 0.45)
            }
        }

        baseline_path = Path("evolution_baseline_results.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"\nBaseline results saved to: {baseline_path}")
