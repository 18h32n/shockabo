"""Evolution analysis tools for understanding and improving evolution performance."""

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class EvolutionRunAnalysis:
    """Analysis results for a single evolution run."""
    task_id: str
    generations: int
    best_fitness: float
    average_fitness: float
    population_diversity: float
    convergence_rate: float
    mutation_success_rate: float
    crossover_success_rate: float
    total_programs_evaluated: int
    unique_programs_ratio: float
    time_taken: float
    memory_peak_mb: float
    successful_operations: list[str]
    fitness_progression: list[float]
    diversity_progression: list[float]


class EvolutionAnalyzer:
    """Comprehensive analysis tools for evolution runs."""

    def __init__(self, output_dir: str = "evolution_analysis"):
        """Initialize analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.genealogy_dir = self.output_dir / "genealogy"
        self.genealogy_dir.mkdir(exist_ok=True)

    def analyze_evolution_run(self, evolution_result: dict[str, Any],
                            task_id: str) -> EvolutionRunAnalysis:
        """Analyze a single evolution run."""
        # Extract metrics
        best_individual = evolution_result.get("best_individual")
        population = evolution_result.get("population", {})
        convergence_history = evolution_result.get("convergence_history", [])
        metrics = evolution_result.get("metrics", {})

        # Calculate analysis metrics
        best_fitness = best_individual.fitness if best_individual else 0.0

        # Population statistics
        if hasattr(population, 'individuals') and population.individuals:
            fitnesses = [ind.fitness for ind in population.individuals]
            average_fitness = statistics.mean(fitnesses) if fitnesses else 0.0

            # Diversity as unique program ratio
            unique_programs = len({str(ind.operations)
                                    for ind in population.individuals})
            population_diversity = unique_programs / len(population.individuals)
        else:
            average_fitness = 0.0
            population_diversity = 0.0

        # Convergence rate
        convergence_rate = self._calculate_convergence_rate(convergence_history)

        # Operation success analysis
        successful_ops = self._analyze_successful_operations(evolution_result)

        # Create analysis object
        analysis = EvolutionRunAnalysis(
            task_id=task_id,
            generations=evolution_result.get("generations", 0),
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            population_diversity=population_diversity,
            convergence_rate=convergence_rate,
            mutation_success_rate=metrics.get("mutation_success_rate", 0.0),
            crossover_success_rate=metrics.get("crossover_success_rate", 0.0),
            total_programs_evaluated=metrics.get("total_programs_evaluated", 0),
            unique_programs_ratio=metrics.get("unique_programs_ratio", population_diversity),
            time_taken=metrics.get("total_time", 0.0),
            memory_peak_mb=metrics.get("memory_peak_mb", 0.0),
            successful_operations=successful_ops,
            fitness_progression=convergence_history,
            diversity_progression=evolution_result.get("diversity_history", [])
        )

        return analysis

    def generate_comprehensive_report(self, analyses: list[EvolutionRunAnalysis]) -> dict[str, Any]:
        """Generate comprehensive report from multiple analyses."""
        if not analyses:
            return {"error": "No analyses provided"}

        report = {
            "summary": {
                "total_runs": len(analyses),
                "average_generations": statistics.mean(a.generations for a in analyses),
                "average_best_fitness": statistics.mean(a.best_fitness for a in analyses),
                "average_programs_evaluated": statistics.mean(a.total_programs_evaluated for a in analyses),
                "total_time": sum(a.time_taken for a in analyses),
                "successful_runs": sum(1 for a in analyses if a.best_fitness >= 0.45)
            },
            "performance_metrics": {
                "convergence_rates": {
                    "mean": statistics.mean(a.convergence_rate for a in analyses),
                    "std": statistics.stdev(a.convergence_rate for a in analyses) if len(analyses) > 1 else 0
                },
                "diversity_maintenance": {
                    "mean": statistics.mean(a.population_diversity for a in analyses),
                    "std": statistics.stdev(a.population_diversity for a in analyses) if len(analyses) > 1 else 0
                },
                "operator_effectiveness": {
                    "mutation_success": statistics.mean(a.mutation_success_rate for a in analyses),
                    "crossover_success": statistics.mean(a.crossover_success_rate for a in analyses)
                }
            },
            "resource_usage": {
                "average_memory_mb": statistics.mean(a.memory_peak_mb for a in analyses),
                "max_memory_mb": max(a.memory_peak_mb for a in analyses),
                "programs_per_second": sum(a.total_programs_evaluated for a in analyses) /
                                      sum(a.time_taken for a in analyses) if sum(a.time_taken for a in analyses) > 0 else 0
            },
            "successful_operations": self._aggregate_successful_operations(analyses),
            "recommendations": self._generate_recommendations(analyses)
        }

        # Save report
        report_path = self.reports_dir / f"evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def plot_fitness_progression(self, analyses: list[EvolutionRunAnalysis], save: bool = True):
        """Plot fitness progression across generations."""
        plt.figure(figsize=(12, 8))

        for analysis in analyses:
            if analysis.fitness_progression:
                plt.plot(analysis.fitness_progression,
                        label=f"{analysis.task_id} (final: {analysis.best_fitness:.3f})",
                        alpha=0.7)

        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Progression Across Evolution Runs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        if save:
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'fitness_progression.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diversity_analysis(self, analyses: list[EvolutionRunAnalysis], save: bool = True):
        """Plot diversity metrics analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Diversity distribution
        diversities = [a.population_diversity for a in analyses]
        ax1.hist(diversities, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(statistics.mean(diversities), color='red', linestyle='--',
                   label=f'Mean: {statistics.mean(diversities):.3f}')
        ax1.set_xlabel('Population Diversity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Population Diversity Distribution')
        ax1.legend()

        # Diversity vs Fitness
        fitnesses = [a.best_fitness for a in analyses]
        ax2.scatter(diversities, fitnesses, alpha=0.6)

        # Add trend line
        z = np.polyfit(diversities, fitnesses, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(diversities), p(sorted(diversities)), "r--", alpha=0.8)

        ax2.set_xlabel('Population Diversity')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('Diversity vs Performance')
        ax2.grid(True, alpha=0.3)

        if save:
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'diversity_analysis.png', dpi=300)
        plt.close()

    def plot_operator_effectiveness(self, analyses: list[EvolutionRunAnalysis], save: bool = True):
        """Plot operator effectiveness analysis."""
        # Aggregate operation counts
        operation_counts = defaultdict(int)
        operation_fitness = defaultdict(list)

        for analysis in analyses:
            for op in analysis.successful_operations:
                operation_counts[op] += 1
                operation_fitness[op].append(analysis.best_fitness)

        # Sort by frequency
        sorted_ops = sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)[:15]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Operation frequency
        ops, counts = zip(*sorted_ops, strict=False)
        ax1.bar(ops, counts)
        ax1.set_xlabel('Operation')
        ax1.set_ylabel('Frequency in Successful Programs')
        ax1.set_title('Most Common Operations in High-Fitness Programs')
        ax1.tick_params(axis='x', rotation=45)

        # Operation effectiveness
        op_effectiveness = []
        for op, _ in sorted_ops:
            if op in operation_fitness and operation_fitness[op]:
                avg_fitness = statistics.mean(operation_fitness[op])
                op_effectiveness.append((op, avg_fitness))

        if op_effectiveness:
            ops, fitnesses = zip(*op_effectiveness, strict=False)
            ax2.bar(ops, fitnesses)
            ax2.set_xlabel('Operation')
            ax2.set_ylabel('Average Fitness')
            ax2.set_title('Average Fitness by Operation')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(0.45, color='red', linestyle='--', label='Target Fitness')
            ax2.legend()

        if save:
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'operator_effectiveness.png', dpi=300)
        plt.close()

    def analyze_genealogy(self, genealogy_data: dict[str, Any], save: bool = True):
        """Analyze and visualize genealogy information."""
        # Create genealogy graph
        G = nx.DiGraph()

        # Add nodes and edges
        for individual_id, data in genealogy_data.items():
            G.add_node(individual_id, fitness=data.get('fitness', 0))

            for parent_id in data.get('parents', []):
                G.add_edge(parent_id, individual_id)

        # Calculate genealogy metrics
        metrics = {
            "total_individuals": G.number_of_nodes(),
            "total_relationships": G.number_of_edges(),
            "max_genealogy_depth": nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else 0,
            "average_offspring": statistics.mean(G.out_degree(n) for n in G.nodes()) if G.nodes() else 0,
            "most_influential": self._find_most_influential_ancestors(G)
        }

        if save:
            # Save genealogy metrics
            genealogy_report = self.genealogy_dir / f"genealogy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(genealogy_report, 'w') as f:
                json.dump(metrics, f, indent=2)

            # Create genealogy visualization if graph is small enough
            if G.number_of_nodes() < 100:
                self._visualize_genealogy_graph(G)

        return metrics

    def generate_optimization_suggestions(self, analyses: list[EvolutionRunAnalysis]) -> list[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []

        # Analyze convergence patterns
        avg_convergence = statistics.mean(a.convergence_rate for a in analyses)
        if avg_convergence < 0.1:
            suggestions.append("Low convergence rate detected. Consider increasing mutation rate or population size.")

        # Analyze diversity
        avg_diversity = statistics.mean(a.population_diversity for a in analyses)
        if avg_diversity < 0.3:
            suggestions.append("Low population diversity. Enable diversity preservation mechanisms.")
        elif avg_diversity > 0.8:
            suggestions.append("Very high diversity may slow convergence. Consider reducing novelty weight.")

        # Analyze operator effectiveness
        avg_mutation_success = statistics.mean(a.mutation_success_rate for a in analyses)
        avg_crossover_success = statistics.mean(a.crossover_success_rate for a in analyses)

        if avg_mutation_success < 0.2:
            suggestions.append("Low mutation success rate. Consider adjusting mutation operators or rates.")

        if avg_crossover_success < 0.3:
            suggestions.append("Low crossover success rate. Review crossover operators or increase tournament size.")

        # Resource usage
        avg_programs = statistics.mean(a.total_programs_evaluated for a in analyses)
        if avg_programs < 500:
            suggestions.append("Not reaching 500+ program target. Increase population size or generations.")

        # Time efficiency
        avg_time = statistics.mean(a.time_taken for a in analyses)
        if avg_time > 240:  # 4 minutes
            suggestions.append("Evolution taking too long. Consider parallel evaluation or batch size optimization.")

        return suggestions

    def export_detailed_metrics(self, analyses: list[EvolutionRunAnalysis], format: str = "csv"):
        """Export detailed metrics in specified format."""
        # Create DataFrame
        data = []
        for a in analyses:
            data.append({
                "task_id": a.task_id,
                "generations": a.generations,
                "best_fitness": a.best_fitness,
                "avg_fitness": a.average_fitness,
                "diversity": a.population_diversity,
                "convergence_rate": a.convergence_rate,
                "mutation_success": a.mutation_success_rate,
                "crossover_success": a.crossover_success_rate,
                "programs_evaluated": a.total_programs_evaluated,
                "unique_ratio": a.unique_programs_ratio,
                "time_seconds": a.time_taken,
                "memory_mb": a.memory_peak_mb
            })

        df = pd.DataFrame(data)

        # Export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if format == "csv":
            export_path = self.reports_dir / f"evolution_metrics_{timestamp}.csv"
            df.to_csv(export_path, index=False)
        elif format == "excel":
            export_path = self.reports_dir / f"evolution_metrics_{timestamp}.xlsx"
            df.to_excel(export_path, index=False)

        # Add summary statistics
        summary_path = self.reports_dir / f"evolution_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("Evolution Pipeline Summary Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.describe().to_string())
            f.write("\n\n")
            f.write("Correlation Matrix:\n")
            f.write(df.corr().to_string())

        return export_path

    def _calculate_convergence_rate(self, fitness_history: list[float]) -> float:
        """Calculate convergence rate from fitness history."""
        if len(fitness_history) < 2:
            return 0.0

        # Calculate improvement per generation
        improvements = []
        for i in range(1, len(fitness_history)):
            improvement = fitness_history[i] - fitness_history[i-1]
            improvements.append(improvement)

        # Average improvement rate
        return statistics.mean(improvements) if improvements else 0.0

    def _analyze_successful_operations(self, evolution_result: dict[str, Any]) -> list[str]:
        """Extract operations from successful individuals."""
        operations = []

        # Get high-fitness individuals
        population = evolution_result.get("population", {})
        if hasattr(population, 'individuals'):
            high_fitness_individuals = [ind for ind in population.individuals
                                      if ind.fitness >= 0.8]

            for ind in high_fitness_individuals:
                for op in ind.operations:
                    if isinstance(op, dict) and 'op' in op:
                        operations.append(op['op'])

        return operations

    def _aggregate_successful_operations(self, analyses: list[EvolutionRunAnalysis]) -> dict[str, int]:
        """Aggregate successful operations across analyses."""
        operation_counts = Counter()

        for analysis in analyses:
            operation_counts.update(analysis.successful_operations)

        return dict(operation_counts.most_common(20))

    def _generate_recommendations(self, analyses: list[EvolutionRunAnalysis]) -> list[str]:
        """Generate recommendations based on analyses."""
        recommendations = self.generate_optimization_suggestions(analyses)

        # Add specific recommendations based on patterns
        high_performing = [a for a in analyses if a.best_fitness >= 0.8]
        if high_performing:
            # Analyze what made them successful
            successful_gens = statistics.mean(a.generations for a in high_performing)
            recommendations.append(f"High-performing runs averaged {successful_gens:.0f} generations.")

        return recommendations

    def _find_most_influential_ancestors(self, G: nx.DiGraph, top_n: int = 5) -> list[tuple[str, int]]:
        """Find most influential ancestors in genealogy."""
        influence_scores = {}

        for node in G.nodes():
            # Count descendants
            descendants = nx.descendants(G, node)
            influence_scores[node] = len(descendants)

        # Sort by influence
        sorted_ancestors = sorted(influence_scores.items(),
                                key=lambda x: x[1], reverse=True)

        return sorted_ancestors[:top_n]

    def _visualize_genealogy_graph(self, G: nx.DiGraph):
        """Visualize genealogy graph."""
        plt.figure(figsize=(14, 10))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Node colors by fitness
        node_colors = []
        for node in G.nodes():
            fitness = G.nodes[node].get('fitness', 0)
            node_colors.append(fitness)

        # Draw graph
        nx.draw(G, pos,
               node_color=node_colors,
               cmap='viridis',
               node_size=50,
               edge_color='gray',
               alpha=0.6,
               arrows=True,
               arrowsize=10)

        plt.title('Evolution Genealogy Visualization')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'),
                    label='Fitness', orientation='horizontal')

        plt.savefig(self.genealogy_dir / 'genealogy_graph.png', dpi=300, bbox_inches='tight')
        plt.close()
