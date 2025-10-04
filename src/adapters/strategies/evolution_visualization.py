"""
Visualization tools for genetic algorithm evolution.

This module provides utilities for visualizing evolution progress,
fitness landscapes, and population diversity.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.adapters.strategies.evolution_engine import Population


class EvolutionVisualizer:
    """
    Visualizer for genetic algorithm evolution progress.

    Generates data that can be used with various plotting libraries
    or exported for analysis.
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualization data
        """
        self.output_dir = output_dir or Path("evolution_viz")
        self.output_dir.mkdir(exist_ok=True)

        self.generation_data: list[dict] = []
        self.individual_history: dict[str, list[float]] = {}
        self.diversity_metrics: list[dict] = []
        self.genealogy_data: dict[str, dict] = {}  # Individual ID -> genealogy info
        self.mutation_success_patterns: dict[str, int] = {}  # Pattern -> success count
        self.crossover_success_patterns: dict[tuple, int] = {}  # (parent1_fitness, parent2_fitness) -> success count

    def record_generation(self, population: Population) -> None:
        """
        Record data for a generation.

        Args:
            population: Current population state
        """
        # Basic statistics
        fitness_values = [ind.fitness for ind in population.individuals]

        generation_stats = {
            "generation": population.generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": population.size(),
            "best_fitness": max(fitness_values) if fitness_values else 0.0,
            "average_fitness": np.mean(fitness_values) if fitness_values else 0.0,
            "fitness_std": np.std(fitness_values) if fitness_values else 0.0,
            "min_fitness": min(fitness_values) if fitness_values else 0.0,
            "fitness_quartiles": self._calculate_quartiles(fitness_values),
            "age_distribution": self._get_age_distribution(population),
            "diversity_metrics": population.diversity_metrics.copy()
        }

        self.generation_data.append(generation_stats)

        # Track top individuals
        elite = population.get_elite(10)
        for ind in elite:
            if ind.id not in self.individual_history:
                self.individual_history[ind.id] = []
            self.individual_history[ind.id].append({
                "generation": population.generation,
                "fitness": ind.fitness,
                "age": ind.age,
                "program_length": ind.program_length()
            })

    def _calculate_quartiles(self, values: list[float]) -> dict[str, float]:
        """Calculate quartiles for a list of values."""
        if not values:
            return {"q1": 0.0, "q2": 0.0, "q3": 0.0}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "q1": sorted_values[n // 4],
            "q2": sorted_values[n // 2],
            "q3": sorted_values[3 * n // 4]
        }

    def _get_age_distribution(self, population: Population) -> dict[int, int]:
        """Get distribution of individual ages."""
        age_counts = {}
        for ind in population.individuals:
            age_counts[ind.age] = age_counts.get(ind.age, 0) + 1
        return age_counts

    def record_genealogy(self, population: Population) -> None:
        """
        Record genealogy information for the current generation.

        Args:
            population: Current population state
        """
        for individual in population.individuals:
            if individual.id not in self.genealogy_data:
                self.genealogy_data[individual.id] = {
                    'id': individual.id,
                    'generation_created': individual.metadata.get('generation', 0),
                    'creation_method': individual.metadata.get('creation_method', 'unknown'),
                    'parent_ids': list(individual.parent_ids),
                    'lineage': individual.metadata.get('lineage', []),
                    'mutation_history': individual.metadata.get('mutation_history', []),
                    'fitness_history': [],
                    'children': []
                }

            # Update fitness history
            self.genealogy_data[individual.id]['fitness_history'].append({
                'generation': population.generation,
                'fitness': individual.fitness
            })

            # Track successful patterns
            if individual.fitness > 0.5:  # Consider it successful if fitness > 0.5
                # Track mutation patterns
                if 'mutation_type' in individual.metadata:
                    pattern = individual.metadata['mutation_type']
                    self.mutation_success_patterns[pattern] = self.mutation_success_patterns.get(pattern, 0) + 1

                # Track crossover patterns
                if 'parent1_fitness' in individual.metadata and 'parent2_fitness' in individual.metadata:
                    parent_fitnesses = (
                        round(individual.metadata['parent1_fitness'], 1),
                        round(individual.metadata['parent2_fitness'], 1)
                    )
                    self.crossover_success_patterns[parent_fitnesses] = \
                        self.crossover_success_patterns.get(parent_fitnesses, 0) + 1

            # Update parent-child relationships
            for parent_id in individual.parent_ids:
                if parent_id in self.genealogy_data:
                    if individual.id not in self.genealogy_data[parent_id]['children']:
                        self.genealogy_data[parent_id]['children'].append(individual.id)

    def generate_genealogy_tree(self, individual_id: str, max_depth: int = 5) -> dict:
        """
        Generate a genealogy tree for a specific individual.

        Args:
            individual_id: ID of the individual to trace
            max_depth: Maximum depth to trace back

        Returns:
            Tree structure representing genealogy
        """
        if individual_id not in self.genealogy_data:
            return {}

        def build_tree(ind_id: str, depth: int = 0) -> dict:
            if depth >= max_depth or ind_id not in self.genealogy_data:
                return {'id': ind_id, 'children': []}

            data = self.genealogy_data[ind_id]
            tree = {
                'id': ind_id,
                'fitness': data['fitness_history'][-1]['fitness'] if data['fitness_history'] else 0,
                'generation': data['generation_created'],
                'creation_method': data['creation_method'],
                'parents': []
            }

            # Add parent trees
            for parent_id in data['parent_ids']:
                parent_tree = build_tree(parent_id, depth + 1)
                tree['parents'].append(parent_tree)

            return tree

        return build_tree(individual_id)

    def export_genealogy_data(self, filename: str | None = None) -> Path:
        """
        Export genealogy data to JSON.

        Args:
            filename: Output filename (default: genealogy_data.json)

        Returns:
            Path to exported file
        """
        filename = filename or "genealogy_data.json"
        filepath = self.output_dir / filename

        # Prepare exportable data
        export_data = {
            'genealogy': self.genealogy_data,
            'mutation_patterns': self.mutation_success_patterns,
            'crossover_patterns': {
                f"{k[0]},{k[1]}": v for k, v in self.crossover_success_patterns.items()
            },
            'summary': {
                'total_individuals': len(self.genealogy_data),
                'successful_mutations': sum(self.mutation_success_patterns.values()),
                'successful_crossovers': sum(self.crossover_success_patterns.values())
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        return filepath

    def generate_genealogy_html(self, top_n: int = 10, filename: str | None = None) -> Path:
        """
        Generate HTML visualization of genealogy data.

        Args:
            top_n: Number of top individuals to show genealogy for
            filename: Output filename (default: genealogy_viz.html)

        Returns:
            Path to HTML file
        """
        filename = filename or "genealogy_viz.html"
        filepath = self.output_dir / filename

        # Get top performing individuals
        top_individuals = []
        for ind_id, data in self.genealogy_data.items():
            if data['fitness_history']:
                final_fitness = data['fitness_history'][-1]['fitness']
                top_individuals.append((ind_id, final_fitness))

        top_individuals.sort(key=lambda x: x[1], reverse=True)
        top_individuals = top_individuals[:top_n]

        # Prepare data for visualization
        genealogy_trees = []
        for ind_id, fitness in top_individuals:
            tree = self.generate_genealogy_tree(ind_id, max_depth=5)
            genealogy_trees.append({
                'id': ind_id,
                'fitness': fitness,
                'tree': json.dumps(tree)
            })

        # Prepare mutation pattern data
        mutation_labels = list(self.mutation_success_patterns.keys())
        mutation_values = [self.mutation_success_patterns[k] for k in mutation_labels]

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Genealogy Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 30px 0; }}
        .individual {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
        .tree {{ margin: 10px 0; }}
        .node {{ cursor: pointer; }}
        .node circle {{ fill: #fff; stroke: steelblue; stroke-width: 3px; }}
        .node text {{ font: 12px sans-serif; }}
        .link {{ fill: none; stroke: #ccc; stroke-width: 2px; }}
    </style>
</head>
<body>
    <h1>Genealogy Analysis</h1>

    <div class="section">
        <h2>Mutation Pattern Success</h2>
        <div id="mutation-patterns"></div>
    </div>

    <div class="section">
        <h2>Top Individuals Genealogy</h2>
        {"".join(f'''
        <div class="individual">
            <h3>Individual {tree['id']} (Fitness: {tree['fitness']:.3f})</h3>
            <div id="tree-{i}" class="tree"></div>
        </div>
        ''' for i, tree in enumerate(genealogy_trees))}
    </div>

    <script>
        // Mutation patterns pie chart
        var mutationData = [{{
            values: {mutation_values},
            labels: {mutation_labels},
            type: 'pie'
        }}];

        var mutationLayout = {{
            title: 'Successful Mutation Types',
            height: 400,
            width: 500
        }};

        Plotly.newPlot('mutation-patterns', mutationData, mutationLayout);

        // Genealogy trees
        {chr(10).join(f'''
        (function() {{
            var treeData = {tree['tree']};
            drawTree('tree-{i}', treeData);
        }})();
        ''' for i, tree in enumerate(genealogy_trees))}

        function drawTree(containerId, data) {{
            var margin = {{top: 20, right: 120, bottom: 20, left: 120}},
                width = 960 - margin.right - margin.left,
                height = 300 - margin.top - margin.bottom;

            var tree = d3.tree()
                .size([height, width]);

            var svg = d3.select("#" + containerId).append("svg")
                .attr("width", width + margin.right + margin.left)
                .attr("height", height + margin.top + margin.bottom)
              .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            // Convert to hierarchy
            function convertToHierarchy(node) {{
                return {{
                    name: node.id.substring(0, 8) + "... (F:" + (node.fitness || 0).toFixed(2) + ")",
                    children: node.parents ? node.parents.map(convertToHierarchy) : []
                }};
            }}

            var root = d3.hierarchy(convertToHierarchy(data));
            tree(root);

            // Links
            svg.selectAll(".link")
                .data(root.links())
              .enter().append("path")
                .attr("class", "link")
                .attr("d", d3.linkHorizontal()
                    .x(function(d) {{ return d.y; }})
                    .y(function(d) {{ return d.x; }}));

            // Nodes
            var node = svg.selectAll(".node")
                .data(root.descendants())
              .enter().append("g")
                .attr("class", "node")
                .attr("transform", function(d) {{
                    return "translate(" + d.y + "," + d.x + ")";
                }});

            node.append("circle")
                .attr("r", 5);

            node.append("text")
                .attr("dy", ".35em")
                .attr("x", function(d) {{ return d.children ? -13 : 13; }})
                .style("text-anchor", function(d) {{
                    return d.children ? "end" : "start";
                }})
                .text(function(d) {{ return d.data.name; }});
        }}
    </script>
</body>
</html>
"""

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def export_generation_data(self, filename: str | None = None) -> Path:
        """
        Export generation data to JSON.

        Args:
            filename: Output filename (default: generation_data.json)

        Returns:
            Path to exported file
        """
        filename = filename or "generation_data.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.generation_data, f, indent=2)

        return filepath

    def export_fitness_history(self, filename: str | None = None) -> Path:
        """
        Export fitness history data for plotting.

        Args:
            filename: Output filename (default: fitness_history.csv)

        Returns:
            Path to exported file
        """
        filename = filename or "fitness_history.csv"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write("generation,best_fitness,avg_fitness,min_fitness,q1,q2,q3\n")

            for data in self.generation_data:
                f.write(f"{data['generation']},")
                f.write(f"{data['best_fitness']},")
                f.write(f"{data['average_fitness']},")
                f.write(f"{data['min_fitness']},")
                f.write(f"{data['fitness_quartiles']['q1']},")
                f.write(f"{data['fitness_quartiles']['q2']},")
                f.write(f"{data['fitness_quartiles']['q3']}\n")

        return filepath

    def generate_html_report(self, filename: str | None = None) -> Path:
        """
        Generate HTML report with embedded visualizations.

        Args:
            filename: Output filename (default: evolution_report.html)

        Returns:
            Path to HTML report
        """
        filename = filename or "evolution_report.html"
        filepath = self.output_dir / filename

        # Prepare data for JavaScript visualization
        generations = [d['generation'] for d in self.generation_data]
        best_fitness = [d['best_fitness'] for d in self.generation_data]
        avg_fitness = [d['average_fitness'] for d in self.generation_data]

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evolution Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px;
                   background: #f0f0f0; border-radius: 5px; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Genetic Algorithm Evolution Report</h1>

    <div class="metrics">
        <div class="metric">
            <strong>Total Generations:</strong> {len(generations)}
        </div>
        <div class="metric">
            <strong>Final Best Fitness:</strong> {best_fitness[-1] if best_fitness else 0:.3f}
        </div>
        <div class="metric">
            <strong>Final Avg Fitness:</strong> {avg_fitness[-1] if avg_fitness else 0:.3f}
        </div>
    </div>

    <div id="fitness-plot" class="chart"></div>
    <div id="diversity-plot" class="chart"></div>
    <div id="age-distribution" class="chart"></div>

    <script>
        // Fitness over time
        var trace1 = {{
            x: {generations},
            y: {best_fitness},
            type: 'scatter',
            name: 'Best Fitness'
        }};

        var trace2 = {{
            x: {generations},
            y: {avg_fitness},
            type: 'scatter',
            name: 'Average Fitness'
        }};

        var layout1 = {{
            title: 'Fitness Evolution',
            xaxis: {{ title: 'Generation' }},
            yaxis: {{ title: 'Fitness' }}
        }};

        Plotly.newPlot('fitness-plot', [trace1, trace2], layout1);

        // Additional visualizations can be added here
    </script>
</body>
</html>
"""

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def analyze_convergence(self) -> dict[str, any]:
        """
        Analyze convergence characteristics.

        Returns:
            Dictionary with convergence metrics
        """
        if not self.generation_data:
            return {}

        best_fitness_values = [d['best_fitness'] for d in self.generation_data]

        # Find when fitness plateaued
        plateau_generation = None
        plateau_threshold = 0.001
        window_size = 10

        if len(best_fitness_values) > window_size:
            for i in range(window_size, len(best_fitness_values)):
                window = best_fitness_values[i-window_size:i]
                if max(window) - min(window) < plateau_threshold:
                    plateau_generation = i - window_size
                    break

        # Calculate improvement rate
        if len(best_fitness_values) > 1:
            total_improvement = best_fitness_values[-1] - best_fitness_values[0]
            improvement_rate = total_improvement / len(best_fitness_values)
        else:
            improvement_rate = 0.0

        return {
            "total_generations": len(self.generation_data),
            "final_best_fitness": best_fitness_values[-1] if best_fitness_values else 0.0,
            "plateau_generation": plateau_generation,
            "improvement_rate": improvement_rate,
            "fitness_range": (min(best_fitness_values), max(best_fitness_values))
                            if best_fitness_values else (0.0, 0.0)
        }


class PopulationSnapshot:
    """
    Capture and analyze population state at specific generation.
    """

    def __init__(self, population: Population):
        """Initialize snapshot from population."""
        self.generation = population.generation
        self.timestamp = datetime.now()
        self.individuals = [
            {
                "id": ind.id,
                "fitness": ind.fitness,
                "age": ind.age,
                "program_length": ind.program_length(),
                "parent_ids": list(ind.parent_ids),
                "species_id": ind.species_id,
                "novelty_score": ind.novelty_score
            }
            for ind in population.individuals
        ]
        self.diversity_metrics = population.diversity_metrics.copy()
        self.species = {
            species_id: len(individuals)
            for species_id, individuals in population.species.items()
        }

    def analyze_diversity(self) -> dict[str, float]:
        """
        Analyze population diversity.

        Returns:
            Dictionary with diversity metrics
        """
        fitness_values = [ind['fitness'] for ind in self.individuals]
        program_lengths = [ind['program_length'] for ind in self.individuals]

        # Fitness diversity
        fitness_diversity = np.std(fitness_values) if fitness_values else 0.0

        # Program length diversity
        length_diversity = np.std(program_lengths) if program_lengths else 0.0

        # Age diversity
        ages = [ind['age'] for ind in self.individuals]
        age_diversity = len(set(ages)) / len(ages) if ages else 0.0

        # Species diversity (if available)
        num_species = len(self.species)
        species_evenness = 0.0
        if num_species > 1:
            species_sizes = list(self.species.values())
            total = sum(species_sizes)
            species_probs = [s/total for s in species_sizes]
            species_evenness = -sum(p * np.log(p) for p in species_probs if p > 0)

        return {
            "fitness_diversity": fitness_diversity,
            "length_diversity": length_diversity,
            "age_diversity": age_diversity,
            "num_species": num_species,
            "species_evenness": species_evenness
        }

    def export_genealogy(self) -> dict[str, list[str]]:
        """
        Export parent-child relationships.

        Returns:
            Dictionary mapping individual IDs to parent IDs
        """
        genealogy = {}
        for ind in self.individuals:
            genealogy[ind['id']] = ind['parent_ids']
        return genealogy


def create_evolution_monitor(visualizer: EvolutionVisualizer):
    """
    Create a callback function for monitoring evolution.

    Args:
        visualizer: EvolutionVisualizer instance

    Returns:
        Callback function for use with evolution engine
    """
    def monitor_callback(population: Population):
        visualizer.record_generation(population)

        # Print progress
        if population.generation % 10 == 0:
            stats = population.diversity_metrics
            print(f"Generation {population.generation}:")
            print(f"  Best fitness: {population.best_individual.fitness:.3f}"
                  if population.best_individual else "  No best individual")
            print(f"  Avg fitness: {population.average_fitness():.3f}")
            print(f"  Unique programs: {stats.get('unique_programs', 0):.1%}")
            print(f"  Avg age: {stats.get('average_age', 0):.1f}")

    return monitor_callback


def plot_fitness_landscape(population: Population,
                          output_file: Path | None = None) -> None:
    """
    Generate fitness landscape visualization data.

    Args:
        population: Current population
        output_file: Optional file to save data
    """
    # Extract features for 2D projection
    features = []
    fitness_values = []

    for ind in population.individuals:
        # Simple features: program length and operation diversity
        program_length = ind.program_length()
        unique_ops = len({op.get_name() for op in ind.operations})

        features.append([program_length, unique_ops])
        fitness_values.append(ind.fitness)

    landscape_data = {
        "features": features,
        "fitness": fitness_values,
        "generation": population.generation
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(landscape_data, f, indent=2)

    return landscape_data
