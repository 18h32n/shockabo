"""Advanced similarity detection for DSL programs."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from domain.dsl.base import DSLProgram


@dataclass
class SimilarityResult:
    """Result of similarity comparison between two programs."""

    program_id_1: str
    program_id_2: str
    overall_score: float  # 0.0 to 1.0
    exact_match: bool
    semantic_score: float
    structural_score: float
    parameter_score: float
    operation_overlap: list[str]
    details: dict[str, Any]


class ProgramSimilarityDetector:
    """Advanced similarity detection for DSL programs."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize similarity detector with configuration."""
        self.config = config or {}

        # Thresholds for different similarity types
        self.exact_threshold = self.config.get('exact_threshold', 1.0)
        self.semantic_threshold = self.config.get('semantic_threshold', 0.95)
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 0.85)

        # Weights for combined score
        self.weights = self.config.get('weights', {
            'semantic': 0.4,
            'structural': 0.3,
            'parameter': 0.3
        })

    def compute_hash(self, program: DSLProgram) -> str:
        """Compute deterministic hash of program for exact matching."""
        # Create canonical representation
        canonical_ops = []
        for op in program.operations:
            # Sort dictionary keys for deterministic representation
            sorted_op = dict(sorted(op.items()))
            canonical_ops.append(sorted_op)

        # Convert to string and hash
        program_str = str(canonical_ops)
        return hashlib.sha256(program_str.encode()).hexdigest()

    def detect_exact_duplicate(self, prog1: DSLProgram, prog2: DSLProgram) -> bool:
        """Check if two programs are exact duplicates."""
        return self.compute_hash(prog1) == self.compute_hash(prog2)

    def compute_semantic_similarity(self, prog1: DSLProgram, prog2: DSLProgram) -> float:
        """Compute semantic similarity based on operation sequences."""
        ops1 = [op.get('type', '') for op in prog1.operations]
        ops2 = [op.get('type', '') for op in prog2.operations]

        if not ops1 or not ops2:
            return 0.0

        # 1. Longest Common Subsequence (LCS) similarity
        lcs_length = self._longest_common_subsequence(ops1, ops2)
        lcs_score = (2 * lcs_length) / (len(ops1) + len(ops2))

        # 2. Edit distance (normalized)
        edit_dist = self._edit_distance(ops1, ops2)
        max_len = max(len(ops1), len(ops2))
        edit_score = 1 - (edit_dist / max_len)

        # 3. N-gram similarity
        ngram_score = self._ngram_similarity(ops1, ops2, n=2)

        # Combine scores
        return 0.4 * lcs_score + 0.4 * edit_score + 0.2 * ngram_score

    def compute_structural_similarity(self, prog1: DSLProgram, prog2: DSLProgram) -> float:
        """Compute structural similarity based on operation patterns."""
        # Extract structural features
        struct1 = self._extract_structure(prog1)
        struct2 = self._extract_structure(prog2)

        # Compare operation counts
        count_sim = self._compare_operation_counts(struct1['op_counts'], struct2['op_counts'])

        # Compare operation order patterns
        order_sim = self._compare_order_patterns(struct1['order_pattern'], struct2['order_pattern'])

        # Compare nesting depth (if applicable)
        depth_sim = 1.0 - abs(struct1['max_depth'] - struct2['max_depth']) / max(struct1['max_depth'], struct2['max_depth'], 1)

        return 0.5 * count_sim + 0.3 * order_sim + 0.2 * depth_sim

    def compute_parameter_similarity(self, prog1: DSLProgram, prog2: DSLProgram) -> float:
        """Compute similarity based on parameter patterns."""
        params1 = self._extract_parameters(prog1)
        params2 = self._extract_parameters(prog2)

        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0

        # Compare parameter types and values
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0

        similarities = []
        for param in common_params:
            sim = self._compare_parameter_values(params1[param], params2[param])
            similarities.append(sim)

        # Include penalty for missing parameters
        all_params = set(params1.keys()) | set(params2.keys())
        coverage = len(common_params) / len(all_params)

        return coverage * (sum(similarities) / len(similarities) if similarities else 0)

    def compute_fuzzy_similarity(self, prog1: DSLProgram, prog2: DSLProgram) -> SimilarityResult:
        """Compute comprehensive fuzzy similarity between programs."""
        # Check exact match first
        exact_match = self.detect_exact_duplicate(prog1, prog2)
        if exact_match:
            return SimilarityResult(
                program_id_1="",
                program_id_2="",
                overall_score=1.0,
                exact_match=True,
                semantic_score=1.0,
                structural_score=1.0,
                parameter_score=1.0,
                operation_overlap=self._get_operation_overlap(prog1, prog2),
                details={'match_type': 'exact'}
            )

        # Compute individual similarity scores
        semantic_score = self.compute_semantic_similarity(prog1, prog2)
        structural_score = self.compute_structural_similarity(prog1, prog2)
        parameter_score = self.compute_parameter_similarity(prog1, prog2)

        # Weighted combination
        overall_score = (
            self.weights['semantic'] * semantic_score +
            self.weights['structural'] * structural_score +
            self.weights['parameter'] * parameter_score
        )

        return SimilarityResult(
            program_id_1="",
            program_id_2="",
            overall_score=overall_score,
            exact_match=False,
            semantic_score=semantic_score,
            structural_score=structural_score,
            parameter_score=parameter_score,
            operation_overlap=self._get_operation_overlap(prog1, prog2),
            details={
                'match_type': self._classify_match_type(overall_score),
                'lcs_operations': self._get_lcs_operations(prog1, prog2)
            }
        )

    def cluster_similar_programs(
        self,
        programs: list[tuple[str, DSLProgram]],
        threshold: float = 0.85
    ) -> list[list[str]]:
        """Cluster programs by similarity."""
        n = len(programs)
        if n == 0:
            return []

        # Build similarity matrix
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_fuzzy_similarity(programs[i][1], programs[j][1])
                similarity_matrix[i, j] = sim.overall_score
                similarity_matrix[j, i] = sim.overall_score
            similarity_matrix[i, i] = 1.0

        # Simple clustering using connected components
        clusters = []
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            # Start new cluster
            cluster = []
            queue = [i]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.append(programs[current][0])

                # Add similar programs to queue
                for j in range(n):
                    if j not in visited and similarity_matrix[current, j] >= threshold:
                        queue.append(j)

            clusters.append(cluster)

        return clusters

    # Helper methods

    def _longest_common_subsequence(self, seq1: list[str], seq2: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _edit_distance(self, seq1: list[str], seq2: list[str]) -> int:
        """Compute edit distance between sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def _ngram_similarity(self, seq1: list[str], seq2: list[str], n: int = 2) -> float:
        """Compute n-gram similarity between sequences."""
        if len(seq1) < n or len(seq2) < n:
            # Fall back to unigram comparison
            set1, set2 = set(seq1), set(seq2)
            if not set1 and not set2:
                return 1.0
            return len(set1 & set2) / len(set1 | set2)

        # Extract n-grams
        ngrams1 = [tuple(seq1[i:i+n]) for i in range(len(seq1) - n + 1)]
        ngrams2 = [tuple(seq2[i:i+n]) for i in range(len(seq2) - n + 1)]

        # Compute Jaccard similarity
        set1, set2 = set(ngrams1), set(ngrams2)
        if not set1 and not set2:
            return 1.0

        return len(set1 & set2) / len(set1 | set2)

    def _extract_structure(self, program: DSLProgram) -> dict[str, Any]:
        """Extract structural features from program."""
        ops = program.operations

        # Operation counts
        op_counts = Counter(op.get('type', '') for op in ops)

        # Order pattern (simplified)
        order_pattern = tuple(op.get('type', '') for op in ops)

        # Max nesting depth (simplified - assumes flat structure)
        max_depth = 1

        return {
            'op_counts': op_counts,
            'order_pattern': order_pattern,
            'max_depth': max_depth
        }

    def _compare_operation_counts(self, counts1: Counter, counts2: Counter) -> float:
        """Compare operation count distributions."""
        all_ops = set(counts1.keys()) | set(counts2.keys())
        if not all_ops:
            return 1.0

        differences = []
        for op in all_ops:
            c1 = counts1.get(op, 0)
            c2 = counts2.get(op, 0)
            max_count = max(c1, c2, 1)
            diff = abs(c1 - c2) / max_count
            differences.append(1 - diff)

        return sum(differences) / len(differences)

    def _compare_order_patterns(self, pattern1: tuple, pattern2: tuple) -> float:
        """Compare operation order patterns."""
        if not pattern1 and not pattern2:
            return 1.0
        if not pattern1 or not pattern2:
            return 0.0

        # Use LCS-based comparison
        lcs = self._longest_common_subsequence(list(pattern1), list(pattern2))
        return (2 * lcs) / (len(pattern1) + len(pattern2))

    def _extract_parameters(self, program: DSLProgram) -> dict[str, list[Any]]:
        """Extract parameters from all operations."""
        params = {}

        for op in program.operations:
            op_type = op.get('type', '')
            op_params = op.get('params', {})

            for param_name, param_value in op_params.items():
                key = f"{op_type}.{param_name}"
                if key not in params:
                    params[key] = []
                params[key].append(param_value)

        return params

    def _compare_parameter_values(self, values1: list[Any], values2: list[Any]) -> float:
        """Compare lists of parameter values."""
        if not values1 and not values2:
            return 1.0
        if not values1 or not values2:
            return 0.0

        # For numeric values, compute relative difference
        if all(isinstance(v, int | float) for v in values1 + values2):
            avg1 = sum(values1) / len(values1)
            avg2 = sum(values2) / len(values2)
            max_avg = max(abs(avg1), abs(avg2), 1)
            return 1 - abs(avg1 - avg2) / max_avg

        # For other types, use exact matching
        matches = sum(1 for v1 in values1 if v1 in values2)
        return matches / max(len(values1), len(values2))

    def _get_operation_overlap(self, prog1: DSLProgram, prog2: DSLProgram) -> list[str]:
        """Get list of common operations."""
        ops1 = {op.get('type', '') for op in prog1.operations}
        ops2 = {op.get('type', '') for op in prog2.operations}
        return list(ops1 & ops2)

    def _classify_match_type(self, score: float) -> str:
        """Classify match type based on score."""
        if score >= self.exact_threshold:
            return 'exact'
        elif score >= self.semantic_threshold:
            return 'semantic'
        elif score >= self.fuzzy_threshold:
            return 'fuzzy'
        else:
            return 'low'

    def _get_lcs_operations(self, prog1: DSLProgram, prog2: DSLProgram) -> list[str]:
        """Get the longest common subsequence of operations."""
        ops1 = [op.get('type', '') for op in prog1.operations]
        ops2 = [op.get('type', '') for op in prog2.operations]

        # Reconstruct LCS
        m, n = len(ops1), len(ops2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ops1[i-1] == ops2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Backtrack to find LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if ops1[i-1] == ops2[j-1]:
                lcs.append(ops1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return lcs[::-1]
