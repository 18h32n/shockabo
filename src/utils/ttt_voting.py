"""
Self-Consistency and Augmentation Voting for MIT TTT

This module implements voting mechanisms for aggregating predictions from multiple
augmented views and permutations, following MIT TTT research methodology.
"""
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionCandidate:
    """Single prediction candidate with metadata."""

    prediction: list[list[int]]
    confidence: float
    source_type: str  # 'original', 'augmented', 'permutation'
    augmentation_info: dict[str, Any]
    generation_metadata: dict[str, Any]


@dataclass
class VotingResult:
    """Result of voting process."""

    best_prediction: list[list[int]]
    confidence_score: float
    vote_distribution: dict[str, int]
    agreement_ratio: float
    total_candidates: int
    voting_method: str
    metadata: dict[str, Any]


class GridHasher:
    """Utility for hashing grid representations for voting."""

    @staticmethod
    def grid_to_string(grid: list[list[int]]) -> str:
        """Convert grid to canonical string representation."""
        return str(grid)

    @staticmethod
    def grid_to_hash(grid: list[list[int]]) -> str:
        """Convert grid to hash for efficient comparison."""
        grid_str = GridHasher.grid_to_string(grid)
        return hashlib.md5(grid_str.encode()).hexdigest()

    @staticmethod
    def grids_are_equal(grid1: list[list[int]], grid2: list[list[int]]) -> bool:
        """Check if two grids are equal."""
        if len(grid1) != len(grid2):
            return False

        for row1, row2 in zip(grid1, grid2, strict=False):
            if len(row1) != len(row2):
                return False
            for val1, val2 in zip(row1, row2, strict=False):
                if val1 != val2:
                    return False

        return True


class SelfConsistencyVoter:
    """
    Implements self-consistency voting mechanism from MIT TTT research.
    
    Self-consistency aggregates predictions from multiple model runs with different
    augmentations/permutations to improve reliability.
    """

    def __init__(
        self,
        min_agreement_threshold: float = 0.5,
        confidence_weighting: bool = True,
        augmentation_weighting: bool = True
    ):
        """
        Initialize self-consistency voter.
        
        Args:
            min_agreement_threshold: Minimum agreement ratio to accept result
            confidence_weighting: Whether to weight votes by confidence
            augmentation_weighting: Whether to weight by augmentation quality
        """
        self.min_agreement_threshold = min_agreement_threshold
        self.confidence_weighting = confidence_weighting
        self.augmentation_weighting = augmentation_weighting
        self.hasher = GridHasher()

    def vote_predictions(
        self,
        candidates: list[PredictionCandidate]
    ) -> VotingResult:
        """
        Vote on multiple prediction candidates using self-consistency.
        
        Args:
            candidates: List of prediction candidates
            
        Returns:
            VotingResult with best prediction and metadata
        """
        if not candidates:
            return VotingResult(
                best_prediction=[[0]],
                confidence_score=0.0,
                vote_distribution={},
                agreement_ratio=0.0,
                total_candidates=0,
                voting_method="self_consistency",
                metadata={"error": "No candidates provided"}
            )

        if len(candidates) == 1:
            return VotingResult(
                best_prediction=candidates[0].prediction,
                confidence_score=candidates[0].confidence,
                vote_distribution={self.hasher.grid_to_hash(candidates[0].prediction): 1},
                agreement_ratio=1.0,
                total_candidates=1,
                voting_method="self_consistency",
                metadata={"single_candidate": True}
            )

        # Group candidates by prediction hash
        vote_groups = defaultdict(list)
        for candidate in candidates:
            pred_hash = self.hasher.grid_to_hash(candidate.prediction)
            vote_groups[pred_hash].append(candidate)

        # Calculate weighted votes for each group
        group_scores = {}
        for pred_hash, group_candidates in vote_groups.items():
            total_weight = 0.0
            total_confidence = 0.0

            for candidate in group_candidates:
                weight = self._calculate_candidate_weight(candidate)
                total_weight += weight
                total_confidence += candidate.confidence * weight

            avg_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
            group_scores[pred_hash] = {
                'weight': total_weight,
                'confidence': avg_confidence,
                'count': len(group_candidates),
                'candidates': group_candidates
            }

        # Find best prediction by weighted vote
        best_hash = max(group_scores.keys(), key=lambda h: group_scores[h]['weight'])
        best_group = group_scores[best_hash]
        best_prediction = best_group['candidates'][0].prediction

        # Calculate agreement ratio
        total_weight = sum(group['weight'] for group in group_scores.values())
        agreement_ratio = best_group['weight'] / total_weight if total_weight > 0 else 0.0

        # Create vote distribution for transparency
        vote_distribution = {
            pred_hash: group['count'] for pred_hash, group in group_scores.items()
        }

        # Determine final confidence
        if agreement_ratio >= self.min_agreement_threshold:
            confidence_score = best_group['confidence'] * agreement_ratio
        else:
            confidence_score = best_group['confidence'] * 0.5  # Penalize low agreement

        return VotingResult(
            best_prediction=best_prediction,
            confidence_score=confidence_score,
            vote_distribution=vote_distribution,
            agreement_ratio=agreement_ratio,
            total_candidates=len(candidates),
            voting_method="self_consistency",
            metadata={
                "weighted_scores": {h: g['weight'] for h, g in group_scores.items()},
                "confidence_scores": {h: g['confidence'] for h, g in group_scores.items()},
                "agreement_meets_threshold": agreement_ratio >= self.min_agreement_threshold
            }
        )

    def _calculate_candidate_weight(self, candidate: PredictionCandidate) -> float:
        """Calculate weight for a candidate based on various factors."""
        weight = 1.0

        # Confidence weighting
        if self.confidence_weighting:
            weight *= max(0.1, candidate.confidence)  # Minimum weight of 0.1

        # Augmentation type weighting
        if self.augmentation_weighting:
            aug_info = candidate.augmentation_info

            # Original examples get higher weight
            if aug_info.get('original', False):
                weight *= 1.2

            # Different augmentation types get different weights
            aug_type = aug_info.get('augmentation_type', 'original')
            aug_weights = {
                'original': 1.0,
                'basic': 1.0,      # Rotation, flip - reliable
                'size': 0.8,       # Size changes - less reliable
                'chain': 0.9,      # Chained transforms - moderate reliability
                'repeat': 0.7      # Repetition patterns - less reliable
            }
            weight *= aug_weights.get(aug_type, 1.0)

        return weight


class AugmentationVoter:
    """
    Implements augmentation-aware voting for TTT predictions.
    
    This voter specifically handles different types of augmentations and their
    reliability for final prediction aggregation.
    """

    def __init__(
        self,
        augmentation_weights: dict[str, float] | None = None,
        diversity_bonus: bool = True,
        consistency_penalty: bool = True
    ):
        """
        Initialize augmentation voter.
        
        Args:
            augmentation_weights: Custom weights for augmentation types
            diversity_bonus: Whether to give bonus for diverse predictions
            consistency_penalty: Whether to penalize inconsistent augmentations
        """
        self.augmentation_weights = augmentation_weights or {
            'original': 1.5,
            'basic': 1.0,
            'size': 0.8,
            'chain': 0.9,
            'repeat': 0.7
        }
        self.diversity_bonus = diversity_bonus
        self.consistency_penalty = consistency_penalty
        self.hasher = GridHasher()

    def vote_augmented_predictions(
        self,
        candidates: list[PredictionCandidate]
    ) -> VotingResult:
        """
        Vote on predictions from different augmentations.
        
        Args:
            candidates: List of prediction candidates from augmentations
            
        Returns:
            VotingResult with best prediction
        """
        if not candidates:
            return VotingResult(
                best_prediction=[[0]],
                confidence_score=0.0,
                vote_distribution={},
                agreement_ratio=0.0,
                total_candidates=0,
                voting_method="augmentation_voting",
                metadata={"error": "No candidates provided"}
            )

        # Group by augmentation type and prediction
        aug_groups = defaultdict(lambda: defaultdict(list))
        for candidate in candidates:
            aug_type = candidate.augmentation_info.get('augmentation_type', 'original')
            pred_hash = self.hasher.grid_to_hash(candidate.prediction)
            aug_groups[aug_type][pred_hash].append(candidate)

        # Calculate scores for each unique prediction
        prediction_scores = defaultdict(float)
        prediction_details = defaultdict(lambda: {
            'candidates': [],
            'augmentation_support': set(),
            'total_confidence': 0.0
        })

        for aug_type, pred_groups in aug_groups.items():
            aug_weight = self.augmentation_weights.get(aug_type, 1.0)

            for pred_hash, pred_candidates in pred_groups.items():
                # Calculate score for this prediction from this augmentation type
                avg_confidence = np.mean([c.confidence for c in pred_candidates])
                consistency_bonus = len(pred_candidates) / len(candidates)  # More consistent = higher bonus

                score = aug_weight * avg_confidence * (1 + consistency_bonus)
                prediction_scores[pred_hash] += score

                # Track details
                prediction_details[pred_hash]['candidates'].extend(pred_candidates)
                prediction_details[pred_hash]['augmentation_support'].add(aug_type)
                prediction_details[pred_hash]['total_confidence'] += avg_confidence

        # Apply diversity bonus
        if self.diversity_bonus:
            for pred_hash in prediction_scores:
                num_aug_types = len(prediction_details[pred_hash]['augmentation_support'])
                diversity_factor = 1 + (num_aug_types - 1) * 0.1  # 10% bonus per additional aug type
                prediction_scores[pred_hash] *= diversity_factor

        # Apply consistency penalty for low agreement
        if self.consistency_penalty:
            num_unique_predictions = len(prediction_scores)
            if num_unique_predictions > 3:  # High disagreement
                penalty_factor = 0.8
                for pred_hash in prediction_scores:
                    prediction_scores[pred_hash] *= penalty_factor

        # Find best prediction
        best_hash = max(prediction_scores.keys(), key=lambda h: prediction_scores[h])
        best_details = prediction_details[best_hash]
        best_prediction = best_details['candidates'][0].prediction

        # Calculate metrics
        total_score = sum(prediction_scores.values())
        agreement_ratio = prediction_scores[best_hash] / total_score if total_score > 0 else 0.0

        confidence_score = min(1.0, best_details['total_confidence'] / len(best_details['augmentation_support']))

        vote_distribution = {
            pred_hash: len(details['candidates'])
            for pred_hash, details in prediction_details.items()
        }

        return VotingResult(
            best_prediction=best_prediction,
            confidence_score=confidence_score,
            vote_distribution=vote_distribution,
            agreement_ratio=agreement_ratio,
            total_candidates=len(candidates),
            voting_method="augmentation_voting",
            metadata={
                "prediction_scores": dict(prediction_scores),
                "augmentation_support": {
                    pred_hash: list(details['augmentation_support'])
                    for pred_hash, details in prediction_details.items()
                },
                "diversity_bonus_applied": self.diversity_bonus,
                "consistency_penalty_applied": self.consistency_penalty
            }
        )


class HybridVoter:
    """
    Combines self-consistency and augmentation voting for optimal results.
    
    This is the main voting class that implements the full MIT TTT voting strategy.
    """

    def __init__(
        self,
        self_consistency_weight: float = 0.6,
        augmentation_weight: float = 0.4,
        min_confidence_threshold: float = 0.1
    ):
        """
        Initialize hybrid voter.
        
        Args:
            self_consistency_weight: Weight for self-consistency voting
            augmentation_weight: Weight for augmentation voting
            min_confidence_threshold: Minimum confidence to accept prediction
        """
        self.self_consistency_weight = self_consistency_weight
        self.augmentation_weight = augmentation_weight
        self.min_confidence_threshold = min_confidence_threshold

        self.self_consistency_voter = SelfConsistencyVoter()
        self.augmentation_voter = AugmentationVoter()
        self.hasher = GridHasher()

    def vote_all_predictions(
        self,
        candidates: list[PredictionCandidate],
        fallback_prediction: list[list[int]] | None = None
    ) -> VotingResult:
        """
        Vote on all predictions using hybrid approach.
        
        Args:
            candidates: All prediction candidates
            fallback_prediction: Fallback if voting fails
            
        Returns:
            Final voting result
        """
        if not candidates:
            fallback = fallback_prediction or [[0]]
            return VotingResult(
                best_prediction=fallback,
                confidence_score=0.0,
                vote_distribution={},
                agreement_ratio=0.0,
                total_candidates=0,
                voting_method="hybrid",
                metadata={"error": "No candidates", "used_fallback": True}
            )

        # Get results from both voting methods
        sc_result = self.self_consistency_voter.vote_predictions(candidates)
        aug_result = self.augmentation_voter.vote_augmented_predictions(candidates)

        # Check if both methods agree
        sc_hash = self.hasher.grid_to_hash(sc_result.best_prediction)
        aug_hash = self.hasher.grid_to_hash(aug_result.best_prediction)

        if sc_hash == aug_hash:
            # Both methods agree - high confidence
            combined_confidence = (
                sc_result.confidence_score * self.self_consistency_weight +
                aug_result.confidence_score * self.augmentation_weight
            )

            return VotingResult(
                best_prediction=sc_result.best_prediction,
                confidence_score=combined_confidence,
                vote_distribution=sc_result.vote_distribution,
                agreement_ratio=max(sc_result.agreement_ratio, aug_result.agreement_ratio),
                total_candidates=len(candidates),
                voting_method="hybrid_agreement",
                metadata={
                    "self_consistency_result": sc_result,
                    "augmentation_result": aug_result,
                    "methods_agree": True
                }
            )
        else:
            # Methods disagree - choose based on confidence and agreement
            sc_score = sc_result.confidence_score * sc_result.agreement_ratio * self.self_consistency_weight
            aug_score = aug_result.confidence_score * aug_result.agreement_ratio * self.augmentation_weight

            if sc_score >= aug_score:
                chosen_result = sc_result
                chosen_method = "hybrid_self_consistency"
            else:
                chosen_result = aug_result
                chosen_method = "hybrid_augmentation"

            # Penalize confidence due to disagreement
            disagreement_penalty = 0.8
            final_confidence = chosen_result.confidence_score * disagreement_penalty

            # Use fallback if confidence is too low
            if final_confidence < self.min_confidence_threshold and fallback_prediction:
                return VotingResult(
                    best_prediction=fallback_prediction,
                    confidence_score=self.min_confidence_threshold,
                    vote_distribution={},
                    agreement_ratio=0.0,
                    total_candidates=len(candidates),
                    voting_method="hybrid_fallback",
                    metadata={
                        "used_fallback": True,
                        "reason": "Low confidence due to disagreement",
                        "sc_score": sc_score,
                        "aug_score": aug_score
                    }
                )

            return VotingResult(
                best_prediction=chosen_result.best_prediction,
                confidence_score=final_confidence,
                vote_distribution=chosen_result.vote_distribution,
                agreement_ratio=chosen_result.agreement_ratio,
                total_candidates=len(candidates),
                voting_method=chosen_method,
                metadata={
                    "self_consistency_result": sc_result,
                    "augmentation_result": aug_result,
                    "methods_agree": False,
                    "sc_score": sc_score,
                    "aug_score": aug_score,
                    "disagreement_penalty": disagreement_penalty
                }
            )


def create_prediction_candidate(
    prediction: list[list[int]],
    confidence: float = 1.0,
    source_type: str = "original",
    augmentation_type: str = "original",
    original: bool = True,
    **kwargs
) -> PredictionCandidate:
    """
    Helper function to create prediction candidates.
    
    Args:
        prediction: Predicted grid
        confidence: Confidence score
        source_type: Source of prediction
        augmentation_type: Type of augmentation used
        original: Whether this is from original data
        **kwargs: Additional metadata
        
    Returns:
        PredictionCandidate instance
    """
    return PredictionCandidate(
        prediction=prediction,
        confidence=confidence,
        source_type=source_type,
        augmentation_info={
            "augmentation_type": augmentation_type,
            "original": original
        },
        generation_metadata=kwargs
    )
