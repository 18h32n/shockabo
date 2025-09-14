#!/usr/bin/env python3
"""
Real ARC Dataset Validation Runner

Generates comprehensive performance metrics using the real ARC Prize 2025 dataset
without requiring full TTT implementation. Focuses on data loading performance,
memory usage, and pipeline validation.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from adapters.repositories.arc_data_repository import ARCDataRepository
from utils.performance_validator import PerformanceValidator


class RealDatasetValidator:
    """Validates ARC dataset loading and basic performance metrics."""

    def __init__(self, output_dir: str = "validation_results"):
        """Initialize validator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize repository with real dataset
        self.data_repository = ARCDataRepository(use_real_dataset=True)
        self.performance_validator = PerformanceValidator()

        # Results tracking
        self.results = {
            "validation_start": datetime.now().isoformat(),
            "dataset_type": "real_arc_prize_2025",
            "validation_passed": False,
            "datasets": {},
            "performance_metrics": {},
            "memory_analysis": {},
            "data_integrity": {},
            "sample_analysis": []
        }

    def validate_dataset_sources(self) -> dict[str, Any]:
        """Validate all available dataset sources."""
        logger.info("Validating dataset sources...")

        sources = ["training", "evaluation", "test"]
        dataset_info = {}

        for source in sources:
            logger.info(f"Analyzing {source} dataset...")

            start_time = time.time()
            task_ids = self.data_repository.get_task_ids(source)
            load_time = time.time() - start_time

            # Load sample tasks to analyze structure
            sample_size = min(10, len(task_ids))
            sample_tasks = {}
            if sample_size > 0:
                sample_dict = self.data_repository.load_all_tasks(source, limit=sample_size)
                sample_tasks = sample_dict

            # Analyze tasks
            has_solutions = 0
            total_train_examples = 0
            grid_sizes = []
            memory_usage = 0

            for task in sample_tasks.values():
                if task.test_output is not None:
                    has_solutions += 1
                total_train_examples += len(task.train_examples)

                # Analyze grid dimensions
                dimensions = task.get_grid_dimensions()
                grid_sizes.extend(dimensions.get("train_input", []))
                grid_sizes.extend(dimensions.get("test_input", []))

                # Estimate memory
                memory_usage += task.get_memory_usage_estimate()

            dataset_info[source] = {
                "total_tasks": len(task_ids),
                "sample_size": sample_size,
                "tasks_with_solutions": has_solutions,
                "solution_rate": has_solutions / sample_size if sample_size > 0 else 0,
                "avg_train_examples": total_train_examples / sample_size if sample_size > 0 else 0,
                "grid_size_range": {
                    "min": min(grid_sizes) if grid_sizes else (0, 0),
                    "max": max(grid_sizes) if grid_sizes else (0, 0),
                    "avg": (sum(s[0] for s in grid_sizes) / len(grid_sizes) if grid_sizes else 0,
                           sum(s[1] for s in grid_sizes) / len(grid_sizes) if grid_sizes else 0)
                },
                "estimated_memory_mb": memory_usage / (1024 * 1024),
                "load_time_seconds": load_time
            }

            logger.info(f"{source}: {len(task_ids)} tasks, {has_solutions}/{sample_size} with solutions")

        return dataset_info

    def performance_benchmark(self, source: str = "training", num_tasks: int = 100) -> dict[str, Any]:
        """Run performance benchmarks on dataset loading."""
        logger.info(f"Running performance benchmark: {num_tasks} tasks from {source}")

        # Benchmark 1: Individual task loading
        task_ids = self.data_repository.get_task_ids(source)
        if not task_ids:
            return {"error": f"No tasks found in {source} dataset"}

        test_ids = task_ids[:min(num_tasks, len(task_ids))]

        individual_times = []
        for task_id in test_ids[:10]:  # Test first 10 individually
            start_time = time.time()
            task = self.data_repository.load_task(task_id, source)
            load_time = time.time() - start_time
            individual_times.append(load_time)
            if task is None:
                logger.warning(f"Failed to load task {task_id}")

        # Benchmark 2: Batch loading
        start_time = time.time()
        batch_tasks = self.data_repository.load_all_tasks(source, limit=num_tasks)
        batch_time = time.time() - start_time

        # Memory analysis
        total_memory = sum(task.get_memory_usage_estimate() for task in batch_tasks.values())

        performance_metrics = {
            "individual_loading": {
                "sample_size": len(individual_times),
                "avg_time_seconds": sum(individual_times) / len(individual_times) if individual_times else 0,
                "min_time_seconds": min(individual_times) if individual_times else 0,
                "max_time_seconds": max(individual_times) if individual_times else 0
            },
            "batch_loading": {
                "tasks_loaded": len(batch_tasks),
                "total_time_seconds": batch_time,
                "avg_time_per_task": batch_time / len(batch_tasks) if batch_tasks else 0,
                "tasks_per_second": len(batch_tasks) / batch_time if batch_time > 0 else 0
            },
            "memory_analysis": {
                "total_estimated_mb": total_memory / (1024 * 1024),
                "avg_per_task_kb": (total_memory / len(batch_tasks)) / 1024 if batch_tasks else 0,
                "memory_efficiency": "high" if total_memory < 100 * 1024 * 1024 else "medium"  # < 100MB is good
            }
        }

        return performance_metrics

    def validate_data_integrity(self, source: str = "training", sample_size: int = 50) -> dict[str, Any]:
        """Validate data integrity and format consistency."""
        logger.info(f"Validating data integrity: {sample_size} tasks from {source}")

        # Load sample tasks
        sample_tasks = self.data_repository.load_all_tasks(source, limit=sample_size)

        integrity_results = {
            "total_checked": len(sample_tasks),
            "valid_tasks": 0,
            "invalid_tasks": [],
            "format_issues": [],
            "grid_issues": [],
            "structure_analysis": {
                "consistent_format": True,
                "issues_found": []
            }
        }

        for task_id, task in sample_tasks.items():
            try:
                # Basic structure validation
                if not task.train_examples:
                    integrity_results["format_issues"].append(f"{task_id}: No training examples")
                    continue

                if not task.test_input:
                    integrity_results["format_issues"].append(f"{task_id}: No test input")
                    continue

                # Validate grid consistency
                for i, example in enumerate(task.train_examples):
                    if "input" not in example:
                        integrity_results["grid_issues"].append(f"{task_id}: Missing input in train example {i}")
                        continue

                    input_grid = example["input"]
                    if not input_grid or not isinstance(input_grid, list):
                        integrity_results["grid_issues"].append(f"{task_id}: Invalid input grid in train example {i}")
                        continue

                    # Check grid is rectangular
                    if len({len(row) for row in input_grid}) > 1:
                        integrity_results["grid_issues"].append(f"{task_id}: Non-rectangular input grid in train example {i}")

                # Test input validation
                if not isinstance(task.test_input, list) or not task.test_input:
                    integrity_results["grid_issues"].append(f"{task_id}: Invalid test input")
                    continue

                if len({len(row) for row in task.test_input}) > 1:
                    integrity_results["grid_issues"].append(f"{task_id}: Non-rectangular test input")

                integrity_results["valid_tasks"] += 1

            except Exception as e:
                integrity_results["invalid_tasks"].append(f"{task_id}: {str(e)}")

        # Calculate integrity score
        integrity_score = integrity_results["valid_tasks"] / integrity_results["total_checked"] if integrity_results["total_checked"] > 0 else 0
        integrity_results["integrity_score"] = integrity_score
        integrity_results["integrity_rating"] = "excellent" if integrity_score > 0.95 else "good" if integrity_score > 0.9 else "needs_attention"

        return integrity_results

    def analyze_task_complexity(self, source: str = "training", sample_size: int = 100) -> dict[str, Any]:
        """Analyze task complexity and characteristics."""
        logger.info(f"Analyzing task complexity: {sample_size} tasks from {source}")

        sample_tasks = self.data_repository.load_all_tasks(source, limit=sample_size)

        complexities = {
            "grid_sizes": [],
            "train_example_counts": [],
            "color_counts": [],
            "pattern_diversity": [],
            "memory_requirements": []
        }

        for task in sample_tasks.values():
            # Grid size analysis
            dimensions = task.get_grid_dimensions()
            for input_dim in dimensions.get("train_input", []):
                complexities["grid_sizes"].append(input_dim[0] * input_dim[1])

            # Training examples
            complexities["train_example_counts"].append(len(task.train_examples))

            # Color diversity (unique values in grids)
            colors = set()
            for example in task.train_examples:
                for row in example.get("input", []):
                    colors.update(row)
            complexities["color_counts"].append(len(colors))

            # Memory requirements
            complexities["memory_requirements"].append(task.get_memory_usage_estimate())

        # Statistical analysis
        def analyze_metric(values, name):
            if not values:
                return {"error": f"No {name} data"}
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }

        analysis = {
            "sample_size": len(sample_tasks),
            "grid_complexity": analyze_metric(complexities["grid_sizes"], "grid sizes"),
            "training_complexity": analyze_metric(complexities["train_example_counts"], "training examples"),
            "color_complexity": analyze_metric(complexities["color_counts"], "colors"),
            "memory_requirements": analyze_metric(complexities["memory_requirements"], "memory"),
            "complexity_rating": "mixed",  # Could be calculated based on distributions
            "recommendations": []
        }

        # Add recommendations based on analysis
        avg_grid_size = analysis["grid_complexity"].get("avg", 0)
        if avg_grid_size > 400:  # 20x20 grids
            analysis["recommendations"].append("Consider grid size optimization for large tasks")

        avg_memory = analysis["memory_requirements"].get("avg", 0)
        if avg_memory > 10000:  # > 10KB per task
            analysis["recommendations"].append("Memory usage optimization may be beneficial")

        return analysis

    def run_comprehensive_validation(self, args) -> dict[str, Any]:
        """Run comprehensive validation with real dataset."""
        logger.info("Starting comprehensive real dataset validation...")

        # 1. Dataset source validation
        logger.info("Step 1: Validating dataset sources...")
        self.results["datasets"] = self.validate_dataset_sources()

        # 2. Performance benchmarking
        logger.info("Step 2: Running performance benchmarks...")
        perf_source = args.subset if args.subset in ["training", "evaluation"] else "training"
        self.results["performance_metrics"] = self.performance_benchmark(perf_source, args.num_tasks)

        # 3. Data integrity validation
        logger.info("Step 3: Validating data integrity...")
        self.results["data_integrity"] = self.validate_data_integrity(perf_source, min(50, args.num_tasks))

        # 4. Complexity analysis
        logger.info("Step 4: Analyzing task complexity...")
        self.results["memory_analysis"] = self.analyze_task_complexity(perf_source, args.num_tasks)

        # 5. Generate summary
        self.results["validation_end"] = datetime.now().isoformat()
        self.results["validation_passed"] = self._assess_validation_success()

        # 6. Save results
        self._save_results()

        return self.results

    def _assess_validation_success(self) -> bool:
        """Assess if validation passed based on criteria."""
        criteria = {
            "datasets_available": len(self.results["datasets"]) >= 2,  # At least training + evaluation
            "data_integrity_good": self.results["data_integrity"].get("integrity_score", 0) > 0.9,
            "performance_acceptable": self.results["performance_metrics"].get("batch_loading", {}).get("tasks_per_second", 0) > 10,
            "memory_reasonable": self.results["memory_analysis"].get("memory_requirements", {}).get("avg", 0) < 50000  # < 50KB per task
        }

        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)

        logger.info(f"Validation criteria: {passed_criteria}/{total_criteria} passed")
        for criterion, passed in criteria.items():
            logger.info(f"  {criterion}: {'✓' if passed else '✗'}")

        return passed_criteria >= 3  # At least 3/4 criteria must pass

    def _save_results(self):
        """Save validation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed JSON results
        results_file = self.output_dir / f"real_dataset_validation_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Detailed results saved to: {results_file}")

        # Save human-readable report
        report_file = self.output_dir / f"real_dataset_validation_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            self._write_readable_report(f)

        logger.info(f"Human-readable report saved to: {report_file}")

    def _write_readable_report(self, f):
        """Write human-readable validation report."""
        f.write("REAL ARC DATASET VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Validation Date: {self.results['validation_start']}\n")
        f.write(f"Dataset Type: {self.results['dataset_type']}\n")
        f.write(f"Overall Result: {'PASSED' if self.results['validation_passed'] else 'FAILED'}\n\n")

        # Dataset overview
        f.write("DATASET OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        for source, info in self.results["datasets"].items():
            f.write(f"{source.capitalize()}: {info['total_tasks']} tasks, {info['solution_rate']:.0%} with solutions\n")
            f.write(f"  - Avg training examples: {info['avg_train_examples']:.1f}\n")
            f.write(f"  - Load time: {info['load_time_seconds']:.3f}s\n")
            f.write(f"  - Est. memory: {info['estimated_memory_mb']:.2f}MB\n\n")

        # Performance metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 20 + "\n")
        perf = self.results.get("performance_metrics", {})
        batch = perf.get("batch_loading", {})
        f.write(f"Batch loading: {batch.get('tasks_loaded', 0)} tasks in {batch.get('total_time_seconds', 0):.2f}s\n")
        f.write(f"Loading rate: {batch.get('tasks_per_second', 0):.1f} tasks/second\n")
        f.write(f"Avg time per task: {batch.get('avg_time_per_task', 0):.4f}s\n\n")

        # Data integrity
        f.write("DATA INTEGRITY:\n")
        f.write("-" * 20 + "\n")
        integrity = self.results.get("data_integrity", {})
        f.write(f"Tasks checked: {integrity.get('total_checked', 0)}\n")
        f.write(f"Valid tasks: {integrity.get('valid_tasks', 0)}\n")
        f.write(f"Integrity score: {integrity.get('integrity_score', 0):.1%}\n")
        f.write(f"Rating: {integrity.get('integrity_rating', 'unknown')}\n\n")

        # Issues found
        format_issues = integrity.get("format_issues", [])
        grid_issues = integrity.get("grid_issues", [])
        if format_issues or grid_issues:
            f.write("ISSUES FOUND:\n")
            f.write("-" * 20 + "\n")
            for issue in (format_issues + grid_issues)[:10]:  # Show first 10 issues
                f.write(f"  - {issue}\n")
            if len(format_issues + grid_issues) > 10:
                f.write(f"  ... and {len(format_issues + grid_issues) - 10} more issues\n")
            f.write("\n")

        # Complexity analysis
        f.write("COMPLEXITY ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        memory = self.results.get("memory_analysis", {})
        grid_comp = memory.get("grid_complexity", {})
        mem_req = memory.get("memory_requirements", {})
        f.write(f"Grid complexity: {grid_comp.get('min', 0)}-{grid_comp.get('max', 0)} cells (avg: {grid_comp.get('avg', 0):.1f})\n")
        f.write(f"Memory per task: {mem_req.get('avg', 0):.0f} bytes (max: {mem_req.get('max', 0):.0f})\n")

        recommendations = memory.get("recommendations", [])
        if recommendations:
            f.write("\nRECOMMENDATIONS:\n")
            for rec in recommendations:
                f.write(f"  - {rec}\n")

def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate real ARC dataset performance and integrity")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to analyze")
    parser.add_argument("--subset", default="training", choices=["training", "evaluation", "test"],
                       help="Dataset subset to focus on")
    parser.add_argument("--output-dir", default="validation_results", help="Output directory")

    args = parser.parse_args()

    # Create validator and run comprehensive validation
    validator = RealDatasetValidator(output_dir=args.output_dir)
    results = validator.run_comprehensive_validation(args)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Overall result: {'PASSED' if results['validation_passed'] else 'FAILED'}")
    logger.info(f"Datasets analyzed: {len(results['datasets'])}")

    perf = results.get("performance_metrics", {}).get("batch_loading", {})
    logger.info(f"Loading performance: {perf.get('tasks_per_second', 0):.1f} tasks/second")

    integrity = results.get("data_integrity", {})
    logger.info(f"Data integrity: {integrity.get('integrity_score', 0):.1%}")

    logger.info("=" * 50)

    # Exit with appropriate code
    exit(0 if results["validation_passed"] else 1)

if __name__ == "__main__":
    main()
