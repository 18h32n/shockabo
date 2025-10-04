"""Script to analyze evolution runs and generate comprehensive reports."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.strategies.evolution_analysis import EvolutionAnalyzer
from src.adapters.strategies.evolution_strategy_adapter import EvolutionStrategyAdapter
from src.domain.models import ARCTask
from src.infrastructure.config import Config


def load_evolution_results(results_dir: str) -> list[dict[str, Any]]:
    """Load evolution results from directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return results

    # Load all JSON result files
    for result_file in results_path.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def run_sample_evolution_for_analysis():
    """Run sample evolution for demonstration."""
    print("Running sample evolution for analysis...")

    # Configure evolution
    config = Config()
    config.evolution.enabled = True
    config.evolution.population_size = 50
    config.evolution.max_generations = 10
    config.evolution.mutation_rate = 0.2
    config.evolution.crossover_rate = 0.7

    # Create sample tasks
    tasks = [
        ARCTask(
            id="sample_1",
            train=[
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]}
            ],
            test=[{"input": [[0, 1], [2, 3]]}]
        ),
        ARCTask(
            id="sample_2",
            train=[
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
            ],
            test=[{"input": [[3, 0], [0, 3]]}]
        )
    ]

    # Run evolution
    adapter = EvolutionStrategyAdapter(config)
    results = []

    for task in tasks:
        print(f"  Evolving solution for {task.id}...")
        result = adapter.process_task(task)

        # Convert to analysis format
        evolution_data = {
            "task_id": task.id,
            "best_individual": {
                "fitness": result.confidence,
                "operations": result.programs[0] if result.programs else []
            },
            "population": {"individuals": []},
            "generations": result.metadata.get("evolution_metrics", {}).get("generations_completed", 10),
            "convergence_history": [i * 0.1 for i in range(10)],  # Mock data
            "metrics": result.metadata.get("evolution_metrics", {})
        }

        results.append(evolution_data)

    return results


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(description="Analyze evolution runs")
    parser.add_argument("--results-dir", type=str, help="Directory containing evolution results")
    parser.add_argument("--output-dir", type=str, default="evolution_analysis",
                       help="Output directory for analysis")
    parser.add_argument("--demo", action="store_true", help="Run demo analysis with sample data")
    parser.add_argument("--format", choices=["csv", "excel"], default="csv",
                       help="Export format for metrics")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = EvolutionAnalyzer(output_dir=args.output_dir)

    # Get evolution results
    if args.demo:
        evolution_results = run_sample_evolution_for_analysis()
    elif args.results_dir:
        evolution_results = load_evolution_results(args.results_dir)
    else:
        print("Please specify --results-dir or use --demo")
        return

    if not evolution_results:
        print("No evolution results found")
        return

    print(f"\nAnalyzing {len(evolution_results)} evolution runs...")

    # Analyze each run
    analyses = []
    for result in evolution_results:
        analysis = analyzer.analyze_evolution_run(
            result,
            task_id=result.get("task_id", "unknown")
        )
        analyses.append(analysis)

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_comprehensive_report(analyses)

    # Print summary
    print("\n=== Evolution Analysis Summary ===")
    print(f"Total runs analyzed: {report['summary']['total_runs']}")
    print(f"Average generations: {report['summary']['average_generations']:.1f}")
    print(f"Average best fitness: {report['summary']['average_best_fitness']:.3f}")
    print(f"Successful runs (≥45%): {report['summary']['successful_runs']}")
    print(f"Total programs evaluated: {report['summary']['average_programs_evaluated']:.0f}")

    print("\n=== Performance Metrics ===")
    print(f"Convergence rate: {report['performance_metrics']['convergence_rates']['mean']:.3f} ± "
          f"{report['performance_metrics']['convergence_rates']['std']:.3f}")
    print(f"Population diversity: {report['performance_metrics']['diversity_maintenance']['mean']:.3f} ± "
          f"{report['performance_metrics']['diversity_maintenance']['std']:.3f}")
    print(f"Mutation success: {report['performance_metrics']['operator_effectiveness']['mutation_success']:.2%}")
    print(f"Crossover success: {report['performance_metrics']['operator_effectiveness']['crossover_success']:.2%}")

    print("\n=== Resource Usage ===")
    print(f"Average memory: {report['resource_usage']['average_memory_mb']:.1f} MB")
    print(f"Programs/second: {report['resource_usage']['programs_per_second']:.1f}")

    # Generate plots
    print("\nGenerating analysis plots...")
    analyzer.plot_fitness_progression(analyses)
    analyzer.plot_diversity_analysis(analyses)
    analyzer.plot_operator_effectiveness(analyses)

    # Export detailed metrics
    print(f"\nExporting detailed metrics in {args.format} format...")
    export_path = analyzer.export_detailed_metrics(analyses, format=args.format)
    print(f"Metrics exported to: {export_path}")

    # Generate optimization suggestions
    print("\n=== Optimization Suggestions ===")
    suggestions = analyzer.generate_optimization_suggestions(analyses)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

    # Analyze genealogy if available
    if evolution_results and "genealogy" in evolution_results[0]:
        print("\nAnalyzing genealogy information...")
        genealogy_metrics = analyzer.analyze_genealogy(evolution_results[0]["genealogy"])
        print(f"Total individuals: {genealogy_metrics['total_individuals']}")
        print(f"Max genealogy depth: {genealogy_metrics['max_genealogy_depth']}")

    print(f"\nAnalysis complete! Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
