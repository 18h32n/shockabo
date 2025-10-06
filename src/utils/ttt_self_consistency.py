"""
Self-Consistency Validation for TTT Strategy

Implements self-consistency validation using permutations and majority voting
to reduce prediction variance and improve accuracy.
Based on self-consistency research for improved test-time performance.
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency validation."""

    permute_n: int = 3
    consensus_threshold: float = 0.6
    enable_geometric: bool = True
    enable_color_remap: bool = False
    max_permutations: int = 8


@dataclass
class PermutationResult:
    """Result of applying a permutation."""

    permutation_type: str
    permuted_input: np.ndarray
    predicted_output: np.ndarray | None = None
    reverse_function: Any = None


class SelfConsistencyValidator:
    """
    Implement self-consistency validation for TTT predictions.

    This implements permutation-based self-consistency where:
    - Generate multiple permutations of input (rotations, flips, color remap)
    - Run inference on each permuted input
    - Reverse permutations to align outputs
    - Use majority voting to select final prediction
    - Calculate confidence based on agreement rate

    Expected improvement: 2-4% accuracy gain from reducing prediction variance.
    """

    def __init__(self, config: SelfConsistencyConfig | None = None):
        """
        Initialize self-consistency validator.

        Args:
            config: Configuration for self-consistency validation
        """
        self.config = config or SelfConsistencyConfig()
        self.permutation_cache: dict[str, list[PermutationResult]] = {}

    def generate_permutations(
        self, input_grid: np.ndarray, grid_id: str | None = None
    ) -> list[PermutationResult]:
        """
        Generate permutations of input grid.

        Args:
            input_grid: Input grid to permute (numpy array)
            grid_id: Optional identifier for caching

        Returns:
            List of permutation results with reverse functions
        """
        if grid_id and grid_id in self.permutation_cache:
            return self.permutation_cache[grid_id]

        permutations = []

        if self.config.enable_geometric:
            permutations.extend(self._generate_geometric_permutations(input_grid))

        if self.config.enable_color_remap:
            permutations.extend(self._generate_color_permutations(input_grid))

        permutations = permutations[: self.config.max_permutations]

        if grid_id:
            self.permutation_cache[grid_id] = permutations

        logger.debug(f"Generated {len(permutations)} permutations for grid")
        return permutations

    def _generate_geometric_permutations(
        self, grid: np.ndarray
    ) -> list[PermutationResult]:
        """
        Generate geometric permutations (rotations and flips).

        Args:
            grid: Input grid

        Returns:
            List of geometric permutations
        """
        permutations = []

        permutations.append(
            PermutationResult(
                permutation_type="identity",
                permuted_input=grid.copy(),
                reverse_function=lambda x: x,
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="rotate_90",
                permuted_input=np.rot90(grid, k=1),
                reverse_function=lambda x: np.rot90(x, k=-1),
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="rotate_180",
                permuted_input=np.rot90(grid, k=2),
                reverse_function=lambda x: np.rot90(x, k=-2),
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="rotate_270",
                permuted_input=np.rot90(grid, k=3),
                reverse_function=lambda x: np.rot90(x, k=-3),
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="flip_horizontal",
                permuted_input=np.fliplr(grid),
                reverse_function=lambda x: np.fliplr(x),
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="flip_vertical",
                permuted_input=np.flipud(grid),
                reverse_function=lambda x: np.flipud(x),
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="transpose",
                permuted_input=np.transpose(grid),
                reverse_function=lambda x: np.transpose(x),
            )
        )

        permutations.append(
            PermutationResult(
                permutation_type="anti_transpose",
                permuted_input=np.rot90(np.transpose(grid), k=2),
                reverse_function=lambda x: np.transpose(np.rot90(x, k=2)),
            )
        )

        return permutations[: self.config.permute_n]

    def _generate_color_permutations(
        self, grid: np.ndarray
    ) -> list[PermutationResult]:
        """
        Generate color remapping permutations.

        Args:
            grid: Input grid

        Returns:
            List of color permutations
        """
        permutations = []

        unique_colors = np.unique(grid)
        if len(unique_colors) <= 1:
            return permutations

        color_map = {c: i for i, c in enumerate(unique_colors)}
        reverse_map = {i: c for c, i in color_map.items()}

        remapped = grid.copy()
        for color, new_color in color_map.items():
            remapped[grid == color] = new_color

        def reverse_color_remap(x):
            result = x.copy()
            for new_color, color in reverse_map.items():
                result[x == new_color] = color
            return result

        permutations.append(
            PermutationResult(
                permutation_type="color_remap",
                permuted_input=remapped,
                reverse_function=reverse_color_remap,
            )
        )

        return permutations

    def aggregate_predictions(
        self,
        permutation_results: list[PermutationResult],
        consensus_threshold: float | None = None,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        Aggregate predictions using majority voting.

        Args:
            permutation_results: List of permutation results with predictions
            consensus_threshold: Minimum agreement threshold (default from config)

        Returns:
            Tuple of (best_prediction, confidence_score, metadata)
        """
        if consensus_threshold is None:
            consensus_threshold = self.config.consensus_threshold

        aligned_predictions = []
        for result in permutation_results:
            if result.predicted_output is not None and result.reverse_function is not None:
                reversed_pred = result.reverse_function(result.predicted_output)
                aligned_predictions.append(reversed_pred)

        if not aligned_predictions:
            return (
                np.array([[0]]),
                0.0,
                {"error": "No predictions to aggregate", "num_predictions": 0},
            )

        votes = self._vote_predictions(aligned_predictions)
        best_prediction = votes["best_prediction"]
        confidence = votes["confidence"]

        metadata = {
            "num_predictions": len(aligned_predictions),
            "vote_counts": votes["vote_counts"],
            "agreement_rate": votes["agreement_rate"],
            "consensus_met": confidence >= consensus_threshold,
            "consensus_threshold": consensus_threshold,
        }

        logger.debug(
            f"Aggregated {len(aligned_predictions)} predictions with "
            f"confidence {confidence:.3f}"
        )

        return best_prediction, confidence, metadata

    def _vote_predictions(
        self, predictions: list[np.ndarray]
    ) -> dict[str, Any]:
        """
        Perform majority voting on predictions.

        Args:
            predictions: List of aligned predictions

        Returns:
            Dictionary with voting results
        """
        if len(predictions) == 1:
            return {
                "best_prediction": predictions[0],
                "confidence": 1.0,
                "vote_counts": {str(predictions[0].tolist()): 1},
                "agreement_rate": 1.0,
            }

        pred_strings = [str(pred.tolist()) for pred in predictions]

        vote_counts = {}
        for pred_str in pred_strings:
            vote_counts[pred_str] = vote_counts.get(pred_str, 0) + 1

        max_votes = max(vote_counts.values())
        best_pred_str = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        best_prediction = predictions[pred_strings.index(best_pred_str)]

        confidence = max_votes / len(predictions)
        agreement_rate = max_votes / len(predictions)

        return {
            "best_prediction": best_prediction,
            "confidence": confidence,
            "vote_counts": vote_counts,
            "agreement_rate": agreement_rate,
        }

    def calculate_per_pixel_confidence(
        self, permutation_results: list[PermutationResult]
    ) -> np.ndarray | None:
        """
        Calculate per-pixel confidence scores.

        Args:
            permutation_results: List of permutation results with predictions

        Returns:
            Per-pixel confidence map (same shape as predictions), or None
        """
        aligned_predictions = []
        for result in permutation_results:
            if result.predicted_output is not None and result.reverse_function is not None:
                reversed_pred = result.reverse_function(result.predicted_output)
                aligned_predictions.append(reversed_pred)

        if not aligned_predictions:
            return None

        if len({pred.shape for pred in aligned_predictions}) > 1:
            logger.warning("Predictions have different shapes, cannot compute per-pixel confidence")
            return None

        shape = aligned_predictions[0].shape
        confidence_map = np.zeros(shape, dtype=float)

        for i in range(shape[0]):
            for j in range(shape[1]):
                pixel_values = [pred[i, j] for pred in aligned_predictions]
                value_counts = {}
                for val in pixel_values:
                    value_counts[val] = value_counts.get(val, 0) + 1
                max_votes = max(value_counts.values())
                confidence_map[i, j] = max_votes / len(pixel_values)

        return confidence_map

    def clear_cache(self) -> None:
        """Clear permutation cache."""
        self.permutation_cache.clear()
        logger.debug("Cleared permutation cache")
