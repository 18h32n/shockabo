"""Pattern mining and analysis for DSL programs."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from domain.dsl.base import DSLProgram

if TYPE_CHECKING:
    pass


@dataclass
class PatternAnalysis:
    """Stores discovered patterns and their frequencies."""

    pattern_id: str
    pattern_type: str  # 'sequence', 'structure', 'parameter'
    pattern_description: str
    operation_sequence: list[str]  # Sequence of operations in pattern
    frequency: int  # Number of occurrences
    success_rate: float  # Success rate when pattern is present
    program_ids: list[str]  # Programs containing this pattern
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PatternInstance:
    """Represents a single instance of a pattern found in a program."""

    pattern_type: str  # 'sequence', 'structure', 'parameter'
    pattern_value: Any  # The actual pattern (e.g., operation sequence)
    program_id: str
    position: int  # Position in program where pattern starts
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternStats:
    """Statistics about a discovered pattern."""

    pattern_id: str
    pattern_type: str
    frequency: int
    success_count: int
    total_count: int
    success_rate: float
    avg_accuracy: float
    avg_execution_time: float
    program_ids: list[str]
    first_seen: datetime
    last_seen: datetime


class ProgramPatternAnalyzer:
    """Analyzes DSL programs to discover common patterns."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize pattern analyzer."""
        self.config = config or {}

        # Pattern mining parameters
        self.min_frequency = self.config.get('min_frequency', 5)
        self.min_success_rate = self.config.get('min_success_rate', 0.7)
        self.max_patterns = self.config.get('max_patterns', 500)

        # Pattern types to analyze
        self.pattern_types = self.config.get('pattern_types', [
            'sequence',
            'structure',
            'parameter'
        ])

        # Sequence patterns parameters
        self.min_sequence_length = self.config.get('min_sequence_length', 2)
        self.max_sequence_length = self.config.get('max_sequence_length', 5)

    def analyze_programs(
        self,
        programs: list[Any]  # List[ProgramCacheEntry]
    ) -> dict[str, PatternAnalysis]:
        """Analyze a collection of programs to find patterns."""
        patterns = {}

        if 'sequence' in self.pattern_types:
            seq_patterns = self._find_sequence_patterns(programs)
            patterns.update(seq_patterns)

        if 'structure' in self.pattern_types:
            struct_patterns = self._find_structure_patterns(programs)
            patterns.update(struct_patterns)

        if 'parameter' in self.pattern_types:
            param_patterns = self._find_parameter_patterns(programs)
            patterns.update(param_patterns)

        # Filter and rank patterns
        filtered_patterns = self._filter_patterns(patterns)
        ranked_patterns = self._rank_patterns(filtered_patterns)

        # Limit to max patterns
        if len(ranked_patterns) > self.max_patterns:
            ranked_patterns = dict(list(ranked_patterns.items())[:self.max_patterns])

        return ranked_patterns

    def _find_sequence_patterns(
        self,
        programs: list[Any]  # List[ProgramCacheEntry]
    ) -> dict[str, PatternAnalysis]:
        """Find common operation sequences."""
        sequence_counts = defaultdict(list)

        for entry in programs:
            ops = [op.get('type', '') for op in entry.program.operations]

            # Extract all subsequences
            for length in range(self.min_sequence_length,
                              min(len(ops) + 1, self.max_sequence_length + 1)):
                for i in range(len(ops) - length + 1):
                    subseq = tuple(ops[i:i + length])
                    sequence_counts[subseq].append(entry)

        # Convert to PatternAnalysis
        patterns = {}
        for seq, entries in sequence_counts.items():
            if len(entries) >= self.min_frequency:
                pattern_id = f"seq_{hash(seq) % 1000000}"

                # Calculate statistics
                success_count = sum(1 for e in entries if e.success)
                success_rate = success_count / len(entries) if entries else 0

                if success_rate >= self.min_success_rate:
                    patterns[pattern_id] = PatternAnalysis(
                        pattern_id=pattern_id,
                        pattern_type='sequence',
                        pattern_description=f"Operation sequence: {' -> '.join(seq)}",
                        operation_sequence=list(seq),
                        frequency=len(entries),
                        success_rate=success_rate,
                        program_ids=[e.program_id for e in entries],
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )

        return patterns

    def _find_structure_patterns(
        self,
        programs: list[Any]  # List[ProgramCacheEntry]
    ) -> dict[str, PatternAnalysis]:
        """Find common structural patterns."""
        structure_counts = defaultdict(list)

        for entry in programs:
            # Extract structural features
            structure = self._extract_structure_signature(entry.program)
            structure_counts[structure].append(entry)

        # Convert to PatternAnalysis
        patterns = {}
        for struct, entries in structure_counts.items():
            if len(entries) >= self.min_frequency:
                pattern_id = f"struct_{hash(struct) % 1000000}"

                # Calculate statistics
                success_count = sum(1 for e in entries if e.success)
                success_rate = success_count / len(entries) if entries else 0

                if success_rate >= self.min_success_rate:
                    patterns[pattern_id] = PatternAnalysis(
                        pattern_id=pattern_id,
                        pattern_type='structure',
                        pattern_description=f"Structure pattern: {self._describe_structure(struct)}",
                        operation_sequence=[],  # Not applicable for structure
                        frequency=len(entries),
                        success_rate=success_rate,
                        program_ids=[e.program_id for e in entries],
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )

        return patterns

    def _find_parameter_patterns(
        self,
        programs: list[Any]  # List[ProgramCacheEntry]
    ) -> dict[str, PatternAnalysis]:
        """Find common parameter patterns."""
        param_patterns = defaultdict(list)

        for entry in programs:
            # Extract parameter combinations
            param_combos = self._extract_parameter_patterns(entry.program)
            for combo in param_combos:
                param_patterns[combo].append(entry)

        # Convert to PatternAnalysis
        patterns = {}
        for param_combo, entries in param_patterns.items():
            if len(entries) >= self.min_frequency:
                pattern_id = f"param_{hash(param_combo) % 1000000}"

                # Calculate statistics
                success_count = sum(1 for e in entries if e.success)
                success_rate = success_count / len(entries) if entries else 0

                if success_rate >= self.min_success_rate:
                    patterns[pattern_id] = PatternAnalysis(
                        pattern_id=pattern_id,
                        pattern_type='parameter',
                        pattern_description=f"Parameter pattern: {self._describe_parameters(param_combo)}",
                        operation_sequence=[],  # Not applicable for parameters
                        frequency=len(entries),
                        success_rate=success_rate,
                        program_ids=[e.program_id for e in entries],
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )

        return patterns

    def _extract_structure_signature(self, program: DSLProgram) -> tuple:
        """Extract structural signature from program."""
        # Simple structure: operation types and their counts
        op_counts = Counter(op.get('type', '') for op in program.operations)

        # Convert to sorted tuple for hashing
        return tuple(sorted(op_counts.items()))

    def _describe_structure(self, structure: tuple) -> str:
        """Create human-readable description of structure."""
        parts = []
        for op_type, count in structure:
            if count > 1:
                parts.append(f"{op_type}×{count}")
            else:
                parts.append(op_type)
        return ", ".join(parts)

    def _extract_parameter_patterns(self, program: DSLProgram) -> list[tuple]:
        """Extract parameter patterns from program."""
        patterns = []

        # Look for common parameter combinations
        for op in program.operations:
            op_type = op.get('type', '')
            params = op.get('params', {})

            # Create patterns for different parameter aspects
            if params:
                # Pattern 1: Operation with specific parameter names
                param_names = tuple(sorted(params.keys()))
                patterns.append(('op_params', op_type, param_names))

                # Pattern 2: Numeric parameter values (bucketed)
                for param_name, param_value in params.items():
                    if isinstance(param_value, int | float):
                        bucket = self._bucket_numeric_value(param_value)
                        patterns.append(('numeric_param', op_type, param_name, bucket))

        return patterns

    def _bucket_numeric_value(self, value: float) -> str:
        """Bucket numeric values for pattern matching."""
        if value == 0:
            return "zero"
        elif value == 1:
            return "one"
        elif value < 0:
            return "negative"
        elif value < 10:
            return "small"
        elif value < 100:
            return "medium"
        else:
            return "large"

    def _describe_parameters(self, param_combo: tuple) -> str:
        """Create human-readable description of parameter pattern."""
        if not param_combo:
            return "Empty"

        pattern_type = param_combo[0]

        if pattern_type == 'op_params':
            _, op_type, param_names = param_combo
            return f"{op_type} with params: {', '.join(param_names)}"
        elif pattern_type == 'numeric_param':
            _, op_type, param_name, bucket = param_combo
            return f"{op_type}.{param_name} = {bucket}"
        else:
            return str(param_combo)

    def _filter_patterns(
        self,
        patterns: dict[str, PatternAnalysis]
    ) -> dict[str, PatternAnalysis]:
        """Filter patterns based on criteria."""
        filtered = {}

        for pattern_id, pattern in patterns.items():
            if (pattern.frequency >= self.min_frequency and
                pattern.success_rate >= self.min_success_rate):
                filtered[pattern_id] = pattern

        return filtered

    def _rank_patterns(
        self,
        patterns: dict[str, PatternAnalysis]
    ) -> dict[str, PatternAnalysis]:
        """Rank patterns by importance."""
        # Calculate scores for each pattern
        scored_patterns = []

        for pattern_id, pattern in patterns.items():
            # Score based on frequency, success rate, and recency
            frequency_score = min(pattern.frequency / 100, 1.0)
            success_score = pattern.success_rate

            # Combined score (can be adjusted)
            total_score = 0.6 * success_score + 0.4 * frequency_score

            scored_patterns.append((total_score, pattern_id, pattern))

        # Sort by score (descending)
        scored_patterns.sort(reverse=True)

        # Return as ordered dict
        return {pid: pattern for _, pid, pattern in scored_patterns}

    def find_patterns_in_program(
        self,
        program: DSLProgram,
        known_patterns: dict[str, PatternAnalysis]
    ) -> list[str]:
        """Find which known patterns exist in a program."""
        found_patterns = []

        # Check sequence patterns
        ops = [op.get('type', '') for op in program.operations]
        for pattern_id, pattern in known_patterns.items():
            if pattern.pattern_type == 'sequence':
                seq = pattern.operation_sequence
                # Check if sequence exists in program
                for i in range(len(ops) - len(seq) + 1):
                    if ops[i:i + len(seq)] == seq:
                        found_patterns.append(pattern_id)
                        break

        # Check structure patterns
        self._extract_structure_signature(program)
        for _pattern_id, pattern in known_patterns.items():
            if pattern.pattern_type == 'structure':
                # Would need to store structure in pattern for exact matching
                # For now, use description matching as approximation
                pass

        # Check parameter patterns
        self._extract_parameter_patterns(program)
        for _pattern_id, pattern in known_patterns.items():
            if pattern.pattern_type == 'parameter':
                # Would need to store parameter pattern for exact matching
                pass

        return found_patterns

    def get_pattern_statistics(
        self,
        pattern_id: str,
        programs: list[Any]  # List[ProgramCacheEntry]
    ) -> PatternStats:
        """Get detailed statistics for a specific pattern."""
        matching_programs = []

        # Find all programs containing this pattern
        # (This is simplified - in practice would use stored pattern data)
        for entry in programs:
            # Check if program contains pattern
            # ... pattern matching logic ...
            matching_programs.append(entry)

        if not matching_programs:
            return None

        # Calculate statistics
        success_count = sum(1 for e in matching_programs if e.success)
        total_count = len(matching_programs)
        success_rate = success_count / total_count if total_count > 0 else 0

        avg_accuracy = sum(e.accuracy_score for e in matching_programs) / total_count
        avg_exec_time = sum(e.execution_time_ms for e in matching_programs) / total_count

        return PatternStats(
            pattern_id=pattern_id,
            pattern_type='unknown',  # Would be stored with pattern
            frequency=total_count,
            success_count=success_count,
            total_count=total_count,
            success_rate=success_rate,
            avg_accuracy=avg_accuracy,
            avg_execution_time=avg_exec_time,
            program_ids=[e.program_id for e in matching_programs],
            first_seen=min(e.created_at for e in matching_programs),
            last_seen=max(e.created_at for e in matching_programs)
        )

    def suggest_patterns_for_task(
        self,
        task_features: dict[str, Any],
        known_patterns: dict[str, PatternAnalysis],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Suggest patterns that might work for a given task."""
        # This is a placeholder for more sophisticated pattern recommendation
        # In practice, would use task features to match against pattern success contexts

        suggestions = []

        # For now, return patterns with highest success rates
        for pattern_id, pattern in known_patterns.items():
            suggestions.append((pattern_id, pattern.success_rate))

        # Sort by score and return top k
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions[:top_k]
