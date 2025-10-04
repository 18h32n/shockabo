"""
Population merging for distributed evolution.

Implements hash-based deduplication and fitness-based conflict resolution
for merging populations from multiple platforms.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MergeResult:
    """Result of population merge operation."""
    merged_population: list[dict[str, Any]]
    total_individuals: int
    duplicates_removed: int
    platforms_merged: int
    avg_fitness: float
    best_fitness: float


class PopulationMerger:
    """Merges populations from multiple platforms without duplication."""

    def __init__(self):
        """Initialize population merger."""
        self._merge_history: list[dict[str, Any]] = []

    def calculate_program_hash(self, program: str) -> str:
        """
        Calculate SHA-256 hash of program.

        Args:
            program: Program string to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(program.encode('utf-8')).hexdigest()

    def _calculate_hash(self, program: str) -> str:
        """Private alias for calculate_program_hash."""
        return self.calculate_program_hash(program)

    def merge_populations(
        self,
        populations: dict[str, list[dict[str, Any]]],
        max_size: int | None = None,
        preserve_diversity: bool = True
    ) -> MergeResult:
        """
        Merge multiple populations with deduplication.

        Args:
            populations: Dict mapping platform_id to population list
            max_size: Optional maximum size of merged population
            preserve_diversity: Whether to preserve diverse solutions

        Returns:
            MergeResult with merged population and statistics
        """
        if not populations:
            return MergeResult(
                merged_population=[],
                total_individuals=0,
                duplicates_removed=0,
                platforms_merged=0,
                avg_fitness=0.0,
                best_fitness=0.0
            )

        # Track seen hashes for deduplication
        seen_hashes: dict[str, dict[str, Any]] = {}
        duplicates_removed = 0
        total_individuals = 0

        # Process all populations
        for platform_id, population in populations.items():
            for individual in population:
                total_individuals += 1

                # Get or calculate hash
                program_hash = individual.get('hash')
                if not program_hash:
                    program = individual.get('program', '')
                    program_hash = self.calculate_program_hash(program)
                    individual['hash'] = program_hash

                # Check for duplicate
                if program_hash in seen_hashes:
                    duplicates_removed += 1

                    # Keep individual with higher fitness
                    existing = seen_hashes[program_hash]
                    if individual.get('fitness', 0.0) > existing.get('fitness', 0.0):
                        seen_hashes[program_hash] = individual
                        logger.debug(
                            "duplicate_replaced",
                            hash=program_hash[:8],
                            old_fitness=existing.get('fitness'),
                            new_fitness=individual.get('fitness')
                        )
                else:
                    seen_hashes[program_hash] = individual
                    individual['source_platform'] = platform_id

        # Convert to list
        merged_population = list(seen_hashes.values())

        # Calculate statistics
        if merged_population:
            fitnesses = [ind.get('fitness', 0.0) for ind in merged_population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = max(fitnesses)
        else:
            avg_fitness = 0.0
            best_fitness = 0.0

        # Apply size limit if specified
        if max_size and len(merged_population) > max_size:
            if preserve_diversity:
                merged_population = self._select_diverse_subset(
                    merged_population,
                    max_size
                )
            else:
                # Keep top performers
                merged_population = sorted(
                    merged_population,
                    key=lambda x: x.get('fitness', 0.0),
                    reverse=True
                )[:max_size]

        result = MergeResult(
            merged_population=merged_population,
            total_individuals=total_individuals,
            duplicates_removed=duplicates_removed,
            platforms_merged=len(populations),
            avg_fitness=avg_fitness,
            best_fitness=best_fitness
        )

        # Record merge history
        self._merge_history.append({
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'platforms': list(populations.keys()),
            'total_individuals': total_individuals,
            'duplicates_removed': duplicates_removed,
            'final_size': len(merged_population)
        })

        logger.info(
            "populations_merged",
            platforms=len(populations),
            total=total_individuals,
            unique=len(merged_population),
            duplicates=duplicates_removed,
            deduplication_rate=f"{(duplicates_removed/total_individuals*100):.1f}%" if total_individuals > 0 else "0%"
        )

        return result

    def _select_diverse_subset(
        self,
        population: list[dict[str, Any]],
        target_size: int
    ) -> list[dict[str, Any]]:
        """
        Select diverse subset of population.

        Uses fitness-based selection with diversity preservation.

        Args:
            population: Full population
            target_size: Target subset size

        Returns:
            Diverse subset of population
        """
        if len(population) <= target_size:
            return population

        # Sort by fitness
        sorted_pop = sorted(
            population,
            key=lambda x: x.get('fitness', 0.0),
            reverse=True
        )

        # Always keep top 20%
        elite_size = max(1, target_size // 5)
        selected = sorted_pop[:elite_size]
        remaining = sorted_pop[elite_size:]

        # Select remaining based on diversity
        remaining_slots = target_size - elite_size

        # Simple diversity: select every Nth individual
        if remaining:
            step = max(1, len(remaining) // remaining_slots)
            for i in range(0, min(len(remaining), remaining_slots * step), step):
                if len(selected) >= target_size:
                    break
                selected.append(remaining[i])

        return selected

    def validate_merge(
        self,
        populations: dict[str, list[dict[str, Any]]],
        merge_result: MergeResult
    ) -> bool:
        """
        Validate merge result.

        Args:
            populations: Original populations
            merge_result: Merge result to validate

        Returns:
            True if merge is valid
        """
        # Check all individuals have required fields
        for individual in merge_result.merged_population:
            if 'program' not in individual:
                logger.error("merge_validation_failed", reason="missing_program")
                return False
            if 'fitness' not in individual:
                logger.error("merge_validation_failed", reason="missing_fitness")
                return False
            if 'hash' not in individual:
                logger.error("merge_validation_failed", reason="missing_hash")
                return False

        # Check no duplicates in result
        hashes = [ind['hash'] for ind in merge_result.merged_population]
        if len(hashes) != len(set(hashes)):
            logger.error("merge_validation_failed", reason="duplicates_in_result")
            return False

        # Check best fitness is preserved
        all_fitnesses = []
        for population in populations.values():
            all_fitnesses.extend([ind.get('fitness', 0.0) for ind in population])

        if all_fitnesses:
            original_best = max(all_fitnesses)
            if merge_result.best_fitness < original_best - 1e-6:
                logger.error(
                    "merge_validation_failed",
                    reason="best_fitness_lost",
                    original_best=original_best,
                    merged_best=merge_result.best_fitness
                )
                return False

        logger.info("merge_validation_passed")
        return True

    def get_merge_statistics(self) -> dict[str, Any]:
        """Get statistics from merge history."""
        if not self._merge_history:
            return {
                'total_merges': 0,
                'avg_deduplication_rate': 0.0,
                'total_individuals_processed': 0,
                'total_duplicates_removed': 0
            }

        total_individuals = sum(m['total_individuals'] for m in self._merge_history)
        total_duplicates = sum(m['duplicates_removed'] for m in self._merge_history)

        return {
            'total_merges': len(self._merge_history),
            'avg_deduplication_rate': (
                (total_duplicates / total_individuals * 100) if total_individuals > 0 else 0.0
            ),
            'total_individuals_processed': total_individuals,
            'total_duplicates_removed': total_duplicates,
            'merge_history': self._merge_history[-10:]  # Last 10 merges
        }
