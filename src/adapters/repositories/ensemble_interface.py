"""Ensemble interface for program cache integration."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from domain.dsl.base import DSLProgram

from .program_cache import ProgramCache, ProgramCacheEntry


@dataclass
class ProgramVote:
    """Represents a vote from a cached program."""

    program_id: str
    output_grid: np.ndarray
    confidence: float  # 0.0 to 1.0
    source_strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Result from ensemble voting."""

    final_output: np.ndarray
    confidence: float
    votes: list[ProgramVote]
    voting_method: str
    consensus_level: float  # 0.0 to 1.0, how much programs agree
    metadata: dict[str, Any] = field(default_factory=dict)


class EnsembleInterface:
    """Interface for ensemble voting using cached programs."""

    def __init__(
        self,
        program_cache: ProgramCache,
        config: dict[str, Any] | None = None
    ):
        """Initialize ensemble interface."""
        self.cache = program_cache
        self.config = config or {}

        # Voting configuration
        self.min_programs_for_vote = self.config.get('min_programs_for_vote', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.success_weight_multiplier = self.config.get('success_weight_multiplier', 2.0)
        self.voting_method = self.config.get('voting_method', 'weighted_majority')

        # Cache for ensemble results
        self.ensemble_cache = {}

    def get_programs_for_ensemble(
        self,
        task_id: str | None = None,
        similar_to: DSLProgram | None = None,
        min_accuracy: float = 0.7,
        limit: int = 20
    ) -> list[ProgramCacheEntry]:
        """Get programs suitable for ensemble voting."""
        programs = []

        if task_id:
            # Get programs for specific task
            task_programs = self.cache.get_programs_by_task(task_id)
            programs.extend(task_programs)

        if similar_to:
            # Get similar programs
            similar = self.cache.find_similar_programs(
                similar_to,
                max_results=limit
            )
            for prog_id, score, _sim_result in similar:
                if score >= self.confidence_threshold:
                    entry = self.cache.get_program(prog_id)
                    if entry and entry not in programs:
                        programs.append(entry)

        if not programs:
            # Get top successful programs
            programs = self.cache.get_successful_programs(
                min_accuracy=min_accuracy,
                limit=limit
            )

        # Filter by accuracy
        programs = [p for p in programs if p.accuracy_score >= min_accuracy]

        # Sort by accuracy and success
        programs.sort(
            key=lambda p: (p.success, p.accuracy_score),
            reverse=True
        )

        return programs[:limit]

    def ensemble_vote(
        self,
        input_grid: np.ndarray,
        candidate_programs: list[ProgramCacheEntry],
        execution_func: Any = None  # Function to execute programs
    ) -> EnsembleResult:
        """Perform ensemble voting on candidate programs."""
        if len(candidate_programs) < self.min_programs_for_vote:
            # Not enough programs for ensemble
            return self._single_best_program(
                input_grid, candidate_programs, execution_func
            )

        # Collect votes from each program
        votes = []
        for entry in candidate_programs:
            vote = self._get_program_vote(
                input_grid, entry, execution_func
            )
            if vote:
                votes.append(vote)

        if not votes:
            # No valid votes
            return EnsembleResult(
                final_output=input_grid,
                confidence=0.0,
                votes=[],
                voting_method="none",
                consensus_level=0.0
            )

        # Perform voting based on method
        if self.voting_method == 'weighted_majority':
            result = self._weighted_majority_vote(votes)
        elif self.voting_method == 'confidence_weighted':
            result = self._confidence_weighted_vote(votes)
        elif self.voting_method == 'consensus':
            result = self._consensus_vote(votes)
        else:
            # Default to simple majority
            result = self._simple_majority_vote(votes)

        return result

    def _get_program_vote(
        self,
        input_grid: np.ndarray,
        entry: ProgramCacheEntry,
        execution_func: Any
    ) -> ProgramVote | None:
        """Get vote from a single program."""
        try:
            # Execute program
            if execution_func:
                output = execution_func(entry.program, input_grid)
            else:
                # Dummy execution for testing
                output = input_grid.copy()

            # Calculate confidence based on program history
            confidence = self._calculate_confidence(entry)

            return ProgramVote(
                program_id=entry.program_id,
                output_grid=output,
                confidence=confidence,
                source_strategy=entry.metadata.get('strategy'),
                metadata={
                    'accuracy_score': entry.accuracy_score,
                    'execution_time': entry.execution_time_ms,
                    'generation': entry.generation
                }
            )

        except Exception as e:
            print(f"Error executing program {entry.program_id}: {e}")
            return None

    def _calculate_confidence(self, entry: ProgramCacheEntry) -> float:
        """Calculate confidence score for a program."""
        # Base confidence from accuracy
        confidence = entry.accuracy_score

        # Boost for successful programs
        if entry.success:
            confidence *= self.success_weight_multiplier

        # Adjust based on access frequency (popular programs)
        if entry.access_count > 10:
            confidence *= 1.1
        elif entry.access_count > 5:
            confidence *= 1.05

        # Fitness score bonus (for evolution programs)
        if entry.fitness_score is not None:
            confidence *= (0.5 + 0.5 * entry.fitness_score)

        # Normalize to [0, 1]
        return min(confidence, 1.0)

    def _simple_majority_vote(self, votes: list[ProgramVote]) -> EnsembleResult:
        """Simple majority voting."""
        # Count occurrences of each output
        output_counts = Counter()
        output_map = {}

        for vote in votes:
            # Convert array to hashable tuple
            output_key = tuple(vote.output_grid.flatten())
            output_counts[output_key] += 1
            output_map[output_key] = vote.output_grid

        # Get most common output
        most_common = output_counts.most_common(1)[0]
        winner_key, winner_count = most_common

        # Calculate consensus
        consensus = winner_count / len(votes)

        return EnsembleResult(
            final_output=output_map[winner_key],
            confidence=consensus,
            votes=votes,
            voting_method="simple_majority",
            consensus_level=consensus
        )

    def _weighted_majority_vote(self, votes: list[ProgramVote]) -> EnsembleResult:
        """Weighted majority voting using confidence scores."""
        # Weight votes by confidence
        weighted_counts = defaultdict(float)
        output_map = {}
        total_weight = 0

        for vote in votes:
            output_key = tuple(vote.output_grid.flatten())
            weighted_counts[output_key] += vote.confidence
            output_map[output_key] = vote.output_grid
            total_weight += vote.confidence

        # Get highest weighted output
        winner_key = max(weighted_counts, key=weighted_counts.get)
        winner_weight = weighted_counts[winner_key]

        # Calculate weighted consensus
        consensus = winner_weight / total_weight if total_weight > 0 else 0

        return EnsembleResult(
            final_output=output_map[winner_key],
            confidence=consensus,
            votes=votes,
            voting_method="weighted_majority",
            consensus_level=consensus,
            metadata={'total_weight': total_weight}
        )

    def _confidence_weighted_vote(self, votes: list[ProgramVote]) -> EnsembleResult:
        """Confidence-weighted averaging (for numeric grids)."""
        if not votes:
            return None

        # Initialize weighted sum
        shape = votes[0].output_grid.shape
        weighted_sum = np.zeros(shape)
        weight_sum = 0

        # Compute weighted average
        for vote in votes:
            if vote.output_grid.shape == shape:
                weighted_sum += vote.confidence * vote.output_grid
                weight_sum += vote.confidence

        if weight_sum > 0:
            # Compute average and round to integers
            avg_output = weighted_sum / weight_sum
            final_output = np.round(avg_output).astype(int)

            # Calculate consensus based on variance
            variances = []
            for vote in votes:
                if vote.output_grid.shape == shape:
                    diff = np.mean(np.abs(vote.output_grid - final_output))
                    variances.append(diff)

            avg_variance = np.mean(variances) if variances else 0
            consensus = 1.0 - min(avg_variance / 10, 1.0)  # Normalize

            return EnsembleResult(
                final_output=final_output,
                confidence=weight_sum / len(votes),
                votes=votes,
                voting_method="confidence_weighted",
                consensus_level=consensus
            )

        # Fallback to first vote
        return EnsembleResult(
            final_output=votes[0].output_grid,
            confidence=votes[0].confidence,
            votes=votes,
            voting_method="confidence_weighted_fallback",
            consensus_level=0.0
        )

    def _consensus_vote(self, votes: list[ProgramVote]) -> EnsembleResult:
        """Consensus voting - only accept if high agreement."""
        # First try simple majority
        majority_result = self._simple_majority_vote(votes)

        # Check consensus level
        if majority_result.consensus_level >= self.confidence_threshold:
            return majority_result

        # Try weighted majority
        weighted_result = self._weighted_majority_vote(votes)

        if weighted_result.consensus_level >= self.confidence_threshold:
            return weighted_result

        # No consensus - return best individual program
        best_vote = max(votes, key=lambda v: v.confidence)

        return EnsembleResult(
            final_output=best_vote.output_grid,
            confidence=best_vote.confidence * 0.5,  # Reduce confidence
            votes=votes,
            voting_method="consensus_failed",
            consensus_level=0.0,
            metadata={'reason': 'No consensus reached'}
        )

    def _single_best_program(
        self,
        input_grid: np.ndarray,
        programs: list[ProgramCacheEntry],
        execution_func: Any
    ) -> EnsembleResult:
        """Use single best program when ensemble not possible."""
        if not programs:
            return EnsembleResult(
                final_output=input_grid,
                confidence=0.0,
                votes=[],
                voting_method="no_programs",
                consensus_level=0.0
            )

        # Get best program
        best_program = max(programs, key=lambda p: (p.success, p.accuracy_score))

        # Get vote
        vote = self._get_program_vote(input_grid, best_program, execution_func)

        if vote:
            return EnsembleResult(
                final_output=vote.output_grid,
                confidence=vote.confidence,
                votes=[vote],
                voting_method="single_best",
                consensus_level=1.0
            )

        return EnsembleResult(
            final_output=input_grid,
            confidence=0.0,
            votes=[],
            voting_method="execution_failed",
            consensus_level=0.0
        )

    def cache_ensemble_result(
        self,
        task_id: str,
        result: EnsembleResult
    ) -> None:
        """Cache ensemble result for reuse."""
        self.ensemble_cache[task_id] = result

    def get_cached_ensemble_result(
        self,
        task_id: str
    ) -> EnsembleResult | None:
        """Get cached ensemble result."""
        return self.ensemble_cache.get(task_id)

    def analyze_ensemble_performance(
        self,
        results: list[EnsembleResult]
    ) -> dict[str, Any]:
        """Analyze performance of ensemble voting."""
        if not results:
            return {}

        # Calculate statistics
        confidences = [r.confidence for r in results]
        consensus_levels = [r.consensus_level for r in results]
        vote_counts = [len(r.votes) for r in results]

        # Method distribution
        method_counts = Counter(r.voting_method for r in results)

        return {
            'total_ensembles': len(results),
            'avg_confidence': np.mean(confidences),
            'avg_consensus': np.mean(consensus_levels),
            'avg_vote_count': np.mean(vote_counts),
            'confidence_std': np.std(confidences),
            'consensus_std': np.std(consensus_levels),
            'voting_methods': dict(method_counts),
            'high_confidence_rate': sum(1 for c in confidences if c >= self.confidence_threshold) / len(confidences),
            'high_consensus_rate': sum(1 for c in consensus_levels if c >= self.confidence_threshold) / len(consensus_levels)
        }
