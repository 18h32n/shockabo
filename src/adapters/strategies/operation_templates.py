"""
Operation templates for genetic algorithm population initialization.

This module provides templates and generators for creating diverse initial
populations of DSL programs for the evolution engine.
"""

from __future__ import annotations

import random
from typing import Any


class OperationTemplate:
    """Base class for operation templates."""

    def generate(self) -> list[dict[str, Any]]:
        """Generate a list of operation dictionaries."""
        raise NotImplementedError


class BasicTransformTemplate(OperationTemplate):
    """Template for basic transformation operations."""

    def generate(self) -> list[dict[str, Any]]:
        """Generate basic transformation sequence."""
        operations = []

        # Random rotation
        if random.random() < 0.3:
            operations.append({
                "name": "rotate",
                "parameters": {
                    "angle": random.choice([90, 180, 270])
                }
            })

        # Random flip
        if random.random() < 0.3:
            operations.append({
                "name": "flip",
                "parameters": {
                    "direction": random.choice(["horizontal", "vertical"])
                }
            })

        # Random translation
        if random.random() < 0.2:
            operations.append({
                "name": "translate",
                "parameters": {
                    "dx": random.randint(-2, 2),
                    "dy": random.randint(-2, 2)
                }
            })

        return operations


class ColorMappingTemplate(OperationTemplate):
    """Template for color mapping operations."""

    def generate(self) -> list[dict[str, Any]]:
        """Generate color mapping sequence."""
        operations = []

        # Fill background
        if random.random() < 0.3:
            operations.append({
                "name": "fill_background",
                "parameters": {
                    "target_color": random.randint(0, 9)
                }
            })

        # Replace colors
        num_replacements = random.randint(1, 3)
        for _ in range(num_replacements):
            if random.random() < 0.5:
                operations.append({
                    "name": "replace_color",
                    "parameters": {
                        "source_color": random.randint(0, 9),
                        "target_color": random.randint(0, 9)
                    }
                })

        return operations


class PatternDetectionTemplate(OperationTemplate):
    """Template for pattern detection and manipulation."""

    def generate(self) -> list[dict[str, Any]]:
        """Generate pattern detection sequence."""
        operations = []

        # Find patterns
        if random.random() < 0.4:
            operations.append({
                "name": "find_pattern",
                "parameters": {
                    "min_size": random.randint(2, 4),
                    "max_size": random.randint(4, 8)
                }
            })

        # Extract objects
        if random.random() < 0.3:
            operations.append({
                "name": "extract_objects",
                "parameters": {
                    "background_color": 0,
                    "min_size": random.randint(1, 3)
                }
            })

        return operations


class SymmetryTemplate(OperationTemplate):
    """Template for symmetry operations."""

    def generate(self) -> list[dict[str, Any]]:
        """Generate symmetry-based sequence."""
        operations = []

        symmetry_type = random.choice(["horizontal", "vertical", "rotational"])

        if symmetry_type == "horizontal":
            operations.append({
                "name": "make_symmetric",
                "parameters": {
                    "axis": "horizontal",
                    "mode": random.choice(["average", "left", "right"])
                }
            })
        elif symmetry_type == "vertical":
            operations.append({
                "name": "make_symmetric",
                "parameters": {
                    "axis": "vertical",
                    "mode": random.choice(["average", "top", "bottom"])
                }
            })
        else:
            operations.append({
                "name": "rotate",
                "parameters": {"angle": 90}
            })
            operations.append({
                "name": "overlay",
                "parameters": {"mode": "or"}
            })

        return operations


class CompositeTemplate(OperationTemplate):
    """Template that combines multiple template types."""

    def __init__(self):
        self.templates = [
            BasicTransformTemplate(),
            ColorMappingTemplate(),
            PatternDetectionTemplate(),
            SymmetryTemplate()
        ]

    def generate(self) -> list[dict[str, Any]]:
        """Generate composite sequence from multiple templates."""
        operations = []

        # Select 2-3 templates to combine
        num_templates = random.randint(2, 3)
        selected_templates = random.sample(self.templates, num_templates)

        for template in selected_templates:
            operations.extend(template.generate())

        return operations


class OperationTemplateGenerator:
    """
    Generates diverse operation templates for population initialization.

    Supports different initialization strategies including random,
    template-based, and hybrid approaches.
    """

    def __init__(self):
        """Initialize template generator."""
        self.basic_operations = [
            "rotate", "flip", "translate", "scale",
            "fill_background", "replace_color", "map_colors",
            "find_pattern", "extract_objects", "apply_pattern",
            "make_symmetric", "detect_symmetry",
            "connect_components", "separate_objects",
            "find_edges", "trace_contours"
        ]

        self.templates = {
            "transform": BasicTransformTemplate(),
            "color": ColorMappingTemplate(),
            "pattern": PatternDetectionTemplate(),
            "symmetry": SymmetryTemplate(),
            "composite": CompositeTemplate()
        }

    def generate_random_program(self, min_length: int = 2, max_length: int = 8) -> list[dict[str, Any]]:
        """
        Generate a completely random program.

        Args:
            min_length: Minimum number of operations
            max_length: Maximum number of operations

        Returns:
            List of operation dictionaries
        """
        length = random.randint(min_length, max_length)
        operations = []

        for _ in range(length):
            op_name = random.choice(self.basic_operations)
            parameters = self._generate_random_parameters(op_name)

            operations.append({
                "name": op_name,
                "parameters": parameters
            })

        return operations

    def generate_from_template(self, template_type: str | None = None) -> list[dict[str, Any]]:
        """
        Generate program from a specific template.

        Args:
            template_type: Type of template to use (None for random)

        Returns:
            List of operation dictionaries
        """
        if template_type is None:
            template_type = random.choice(list(self.templates.keys()))

        if template_type in self.templates:
            return self.templates[template_type].generate()
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def generate_hybrid_program(self, template_ratio: float = 0.5) -> list[dict[str, Any]]:
        """
        Generate hybrid program combining template and random operations.

        Args:
            template_ratio: Ratio of template-based vs random operations

        Returns:
            List of operation dictionaries
        """
        operations = []

        # Start with template
        if random.random() < template_ratio:
            operations.extend(self.generate_from_template())

        # Add random operations
        num_random = random.randint(1, 3)
        for _ in range(num_random):
            if random.random() > template_ratio:
                op_name = random.choice(self.basic_operations)
                parameters = self._generate_random_parameters(op_name)

                operations.append({
                    "name": op_name,
                    "parameters": parameters
                })

        return operations

    def generate_diverse_population(self, size: int,
                                  random_ratio: float = 0.3,
                                  template_ratio: float = 0.5) -> list[list[dict[str, Any]]]:
        """
        Generate a diverse population of programs.

        Args:
            size: Population size
            random_ratio: Ratio of purely random programs
            template_ratio: Ratio of template-based programs

        Returns:
            List of programs (each program is a list of operations)
        """
        population = []

        # Calculate counts
        num_random = int(size * random_ratio)
        num_template = int(size * template_ratio)
        num_hybrid = size - num_random - num_template

        # Generate random programs
        for _ in range(num_random):
            population.append(self.generate_random_program())

        # Generate template-based programs
        for _ in range(num_template):
            population.append(self.generate_from_template())

        # Generate hybrid programs
        for _ in range(num_hybrid):
            population.append(self.generate_hybrid_program())

        # Shuffle to mix types
        random.shuffle(population)

        return population

    def _generate_random_parameters(self, operation_name: str) -> dict[str, Any]:
        """Generate random parameters for an operation."""
        # This is a simplified parameter generator
        # In practice, this would be more sophisticated based on operation schemas

        params = {}

        if operation_name in ["rotate"]:
            params["angle"] = random.choice([90, 180, 270])
        elif operation_name in ["flip"]:
            params["direction"] = random.choice(["horizontal", "vertical"])
        elif operation_name in ["translate"]:
            params["dx"] = random.randint(-3, 3)
            params["dy"] = random.randint(-3, 3)
        elif operation_name in ["scale"]:
            params["factor"] = random.choice([0.5, 2.0])
        elif operation_name in ["fill_background", "replace_color"]:
            params["target_color"] = random.randint(0, 9)
            if operation_name == "replace_color":
                params["source_color"] = random.randint(0, 9)
        elif operation_name in ["find_pattern", "extract_objects"]:
            params["min_size"] = random.randint(2, 4)
            params["max_size"] = random.randint(4, 8)
        elif operation_name in ["make_symmetric"]:
            params["axis"] = random.choice(["horizontal", "vertical"])

        return params


def create_seed_programs() -> list[list[dict[str, Any]]]:
    """
    Create a set of hand-crafted seed programs.

    These represent known-good program patterns that can serve
    as high-quality starting points for evolution.

    Returns:
        List of seed programs
    """
    seeds = []

    # Rotation and color mapping
    seeds.append([
        {"name": "rotate", "parameters": {"angle": 90}},
        {"name": "replace_color", "parameters": {"source_color": 0, "target_color": 1}}
    ])

    # Symmetry detection and completion
    seeds.append([
        {"name": "detect_symmetry", "parameters": {"axis": "vertical"}},
        {"name": "make_symmetric", "parameters": {"axis": "vertical", "mode": "average"}}
    ])

    # Pattern extraction and application
    seeds.append([
        {"name": "extract_objects", "parameters": {"background_color": 0, "min_size": 2}},
        {"name": "find_pattern", "parameters": {"min_size": 2, "max_size": 5}},
        {"name": "apply_pattern", "parameters": {"mode": "tile"}}
    ])

    # Edge detection and filling
    seeds.append([
        {"name": "find_edges", "parameters": {"mode": "outer"}},
        {"name": "fill_enclosed", "parameters": {"fill_color": 3}}
    ])

    # Object manipulation
    seeds.append([
        {"name": "separate_objects", "parameters": {}},
        {"name": "scale", "parameters": {"factor": 2.0}},
        {"name": "overlay", "parameters": {"mode": "or"}}
    ])

    return seeds
