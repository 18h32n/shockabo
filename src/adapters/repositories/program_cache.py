"""Program cache repository with persistence, similarity detection, and analytics."""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import diskcache as dc
import msgpack

# Domain imports
from domain.dsl.base import DSLProgram

from .analytics_dashboard import AnalyticsDashboard
from .pattern_analyzer import PatternAnalysis, ProgramPatternAnalyzer

# Configuration import
from .program_cache_config import ProgramCacheConfig
from .program_exporter import ProgramExporter
from .similarity_detector import ProgramSimilarityDetector, SimilarityResult


@dataclass
class ProgramCacheEntry:
    """Cache entry for a DSL program with metadata and performance metrics."""

    # Program identification
    program_id: str  # Unique ID for this program
    program_hash: str  # Hash of program operations for deduplication
    program: DSLProgram  # The actual DSL program

    # Task association
    task_id: str
    task_source: str  # 'training', 'evaluation', 'test'

    # Performance metrics
    success: bool  # Whether program successfully produced output
    accuracy_score: float  # Accuracy on task (0.0 to 1.0)
    execution_time_ms: float  # Execution time in milliseconds

    # Evolution metadata (if from evolution engine)
    generation: int | None = None
    parents: list[str] = field(default_factory=list)  # Parent program IDs
    mutation_type: str | None = None
    fitness_score: float | None = None

    # Timestamps and counters
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        # Convert DSLProgram to dict
        data['program'] = {
            'operations': self.program.operations,
            'version': self.program.version,
            'metadata': self.program.metadata
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProgramCacheEntry':
        """Create from dictionary."""
        # Convert datetime strings back to objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        # Reconstruct DSLProgram
        program_data = data['program']
        data['program'] = DSLProgram(
            operations=program_data['operations'],
            version=program_data['version'],
            metadata=program_data['metadata']
        )
        return cls(**data)


@dataclass
class SimilarityMatch:
    """Represents a similarity match between programs."""

    source_program_id: str
    target_program_id: str
    similarity_score: float  # 0.0 to 1.0
    similarity_type: str  # 'exact', 'semantic', 'fuzzy'
    matching_operations: list[str]  # Common operations
    created_at: datetime = field(default_factory=datetime.now)




@dataclass
class CacheStatistics:
    """Statistics about the program cache."""

    total_programs: int
    successful_programs: int
    unique_programs: int  # After deduplication
    total_size_bytes: int
    cache_hit_rate: float
    average_access_frequency: float
    most_successful_patterns: list[str]
    task_type_distribution: dict[str, int]
    generation_distribution: dict[int, int]  # For evolution programs
    timestamp: datetime = field(default_factory=datetime.now)


class ProgramCache:
    """Persistent cache for DSL programs with analytics and similarity detection."""

    def __init__(
        self,
        config: ProgramCacheConfig | None = None,
        config_path: str | None = None
    ):
        """Initialize program cache with configuration."""
        # Load configuration
        if config is None:
            if config_path is None:
                config_path = str(Path(__file__).parent.parent.parent.parent / "configs" / "strategies" / "program_cache.yaml")
            config = ProgramCacheConfig.from_yaml(config_path)

        self.config = config

        # Setup cache directory
        if Path(config.storage.cache_dir).is_absolute():
            self.cache_dir = Path(config.storage.cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent.parent / config.storage.cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Convert GB to bytes
        size_limit_bytes = int(config.storage.size_limit_gb * 1024 * 1024 * 1024)

        # Initialize diskcache with LRU eviction
        self.cache = dc.Cache(
            str(self.cache_dir),
            size_limit=size_limit_bytes,
            eviction_policy=config.storage.eviction_policy
        )

        # Initialize indexes (stored in separate cache files)
        self.index_dir = self.cache_dir / "indexes"
        self.index_dir.mkdir(exist_ok=True)

        # Store commonly used config values
        self.similarity_threshold = config.similarity.similarity_threshold
        self.max_similarity_checks = config.similarity.max_similarity_checks

        # Initialize similarity detector
        self.similarity_detector = ProgramSimilarityDetector({
            'semantic_threshold': config.similarity.similarity_threshold,
            'fuzzy_threshold': config.similarity.similarity_threshold * 0.9
        })

        # Initialize pattern analyzer
        self.pattern_analyzer = ProgramPatternAnalyzer({
            'min_frequency': config.pattern_mining.min_frequency,
            'min_success_rate': config.pattern_mining.min_success_rate,
            'max_patterns': config.pattern_mining.max_patterns,
            'pattern_types': config.pattern_mining.pattern_types
        })

        # Pattern storage
        self.patterns_file = self.cache_dir / "patterns.json"
        self.discovered_patterns: dict[str, PatternAnalysis] = {}
        self._load_patterns()

        # Analytics dashboard
        self.analytics_dashboard = None
        if config.analytics.enable_analytics:
            analytics_dir = self.cache_dir / "analytics"
            self.analytics_dashboard = AnalyticsDashboard(
                cache=self,
                output_dir=analytics_dir
            )

        # Ensemble interface
        self.ensemble_interface = None
        if config.ensemble.enable_ensemble:
            from .ensemble_interface import EnsembleInterface
            self.ensemble_interface = EnsembleInterface(
                program_cache=self,
                config={
                    'min_programs_for_vote': config.ensemble.min_programs_for_vote,
                    'confidence_threshold': config.ensemble.confidence_threshold,
                    'success_weight_multiplier': config.ensemble.success_weight_multiplier,
                    'voting_method': 'weighted_majority'
                }
            )

        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0,
            "similarity_checks": 0,
            "duplicates_found": 0,
            "pattern_analyses": 0,
            "ensemble_votes": 0
        }

    def generate_program_hash(self, program: DSLProgram) -> str:
        """Generate hash for program operations for deduplication."""
        # Use similarity detector for consistent hashing
        return self.similarity_detector.compute_hash(program)

    def save_program(
        self,
        program: DSLProgram,
        task_id: str,
        task_source: str,
        success: bool,
        accuracy_score: float,
        execution_time_ms: float,
        generation: int | None = None,
        parents: list[str] | None = None,
        mutation_type: str | None = None,
        fitness_score: float | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Save a program to the cache with deduplication."""
        # Generate program hash for deduplication
        program_hash = self.generate_program_hash(program)

        # Check for exact duplicate
        existing_id = self._find_duplicate_by_hash(program_hash)
        if existing_id:
            self.stats["duplicates_found"] += 1
            # Just return existing ID without updating access
            # Access will be updated when program is retrieved
            return existing_id

        # Generate unique program ID
        program_id = f"{task_id}_{program_hash[:8]}_{int(time.time() * 1000)}"

        # Create cache entry
        entry = ProgramCacheEntry(
            program_id=program_id,
            program_hash=program_hash,
            program=program,
            task_id=task_id,
            task_source=task_source,
            success=success,
            accuracy_score=accuracy_score,
            execution_time_ms=execution_time_ms,
            generation=generation,
            parents=parents or [],
            mutation_type=mutation_type,
            fitness_score=fitness_score,
            metadata=metadata or {}
        )

        # Serialize and save
        try:
            serialized = msgpack.packb(entry.to_dict(), default=str)
            self.cache.set(program_id, serialized)
            self.stats["saves"] += 1

            # Update indexes
            self._update_hash_index(program_hash, program_id)
            self._update_task_index(task_id, program_id)

            return program_id

        except Exception as e:
            print(f"Error saving program to cache: {e}")
            return ""

    def get_program(self, program_id: str) -> ProgramCacheEntry | None:
        """Retrieve a program from cache."""
        try:
            serialized = self.cache.get(program_id)
            if serialized is None:
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            data = msgpack.unpackb(serialized, raw=False)

            # Update access metadata before creating entry
            data['last_accessed'] = datetime.now().isoformat()
            data['access_count'] = data.get('access_count', 0) + 1

            # Save updated data back to cache
            self.cache.set(program_id, msgpack.packb(data, default=str))

            # Create entry with updated access count
            entry = ProgramCacheEntry.from_dict(data)

            return entry

        except Exception as e:
            print(f"Error retrieving program {program_id}: {e}")
            self.stats["misses"] += 1
            return None

    def find_similar_programs(
        self,
        program: DSLProgram,
        max_results: int = 10,
        similarity_type: str = "fuzzy"
    ) -> list[tuple[str, float, SimilarityResult]]:
        """Find similar programs in cache with detailed similarity analysis."""
        self.stats["similarity_checks"] += 1

        # First check exact match by hash
        program_hash = self.generate_program_hash(program)
        exact_match = self._find_duplicate_by_hash(program_hash)
        if exact_match:
            sim_result = SimilarityResult(
                program_id_1="",
                program_id_2=exact_match,
                overall_score=1.0,
                exact_match=True,
                semantic_score=1.0,
                structural_score=1.0,
                parameter_score=1.0,
                operation_overlap=[],
                details={'match_type': 'exact'}
            )
            return [(exact_match, 1.0, sim_result)]

        # Advanced similarity check
        similar_programs = []

        # Sample programs for comparison
        sample_size = min(self.max_similarity_checks, len(self.cache))
        for program_id in self._sample_program_ids(sample_size):
            entry = self.get_program(program_id)
            if entry:
                # Use advanced similarity detector
                sim_result = self.similarity_detector.compute_fuzzy_similarity(
                    program, entry.program
                )

                if sim_result.overall_score >= self.similarity_threshold:
                    similar_programs.append((
                        program_id,
                        sim_result.overall_score,
                        sim_result
                    ))

        # Sort by similarity score
        similar_programs.sort(key=lambda x: x[1], reverse=True)

        return similar_programs[:max_results]

    def deduplicate_programs(
        self,
        threshold: float = None
    ) -> dict[str, list[str]]:
        """Find and group duplicate/similar programs for deduplication."""
        if threshold is None:
            threshold = self.similarity_threshold

        # Get all programs
        programs = []
        for program_id in self.cache.iterkeys():
            entry = self.get_program(program_id)
            if entry:
                programs.append((program_id, entry.program))

        # Cluster similar programs
        clusters = self.similarity_detector.cluster_similar_programs(
            programs, threshold
        )

        # Create mapping of representative to duplicates
        dedupe_map = {}
        for cluster in clusters:
            if len(cluster) > 1:
                # First program in cluster is representative
                representative = cluster[0]
                duplicates = cluster[1:]
                dedupe_map[representative] = duplicates

        return dedupe_map

    def get_programs_by_task(self, task_id: str) -> list[ProgramCacheEntry]:
        """Get all programs associated with a task."""
        program_ids = self._get_task_program_ids(task_id)
        programs = []

        for pid in program_ids:
            entry = self.get_program(pid)
            if entry:
                programs.append(entry)

        # Sort by accuracy score and execution time
        programs.sort(key=lambda x: (-x.accuracy_score, x.execution_time_ms))

        return programs

    def get_successful_programs(
        self,
        min_accuracy: float = 0.8,
        limit: int = 100
    ) -> list[ProgramCacheEntry]:
        """Get successful programs above accuracy threshold."""
        successful = []

        for program_id in self.cache.iterkeys():
            entry = self.get_program(program_id)
            if entry and entry.success and entry.accuracy_score >= min_accuracy:
                successful.append(entry)
                if len(successful) >= limit:
                    break

        return successful

    def get_statistics(self) -> CacheStatistics:
        """Get comprehensive cache statistics."""
        total_programs = len(self.cache)
        successful_programs = 0
        unique_programs = set()
        task_types = {}
        generations = {}

        # Sample for statistics to avoid full scan
        sample_size = min(1000, total_programs)
        for program_id in self._sample_program_ids(sample_size):
            entry = self.get_program(program_id)
            if entry:
                if entry.success:
                    successful_programs += 1
                unique_programs.add(entry.program_hash)

                # Track task types
                task_types[entry.task_source] = task_types.get(entry.task_source, 0) + 1

                # Track generations for evolution programs
                if entry.generation is not None:
                    generations[entry.generation] = generations.get(entry.generation, 0) + 1

        # Extrapolate from sample
        if sample_size < total_programs:
            scale_factor = total_programs / sample_size
            successful_programs = int(successful_programs * scale_factor)

        # Calculate metrics
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        # Get top patterns
        top_patterns = []
        if self.discovered_patterns:
            sorted_patterns = sorted(
                self.discovered_patterns.values(),
                key=lambda p: p.success_rate * p.frequency,
                reverse=True
            )
            top_patterns = [p.pattern_description for p in sorted_patterns[:5]]

        return CacheStatistics(
            total_programs=total_programs,
            successful_programs=successful_programs,
            unique_programs=len(unique_programs),
            total_size_bytes=self.cache.volume(),
            cache_hit_rate=hit_rate,
            average_access_frequency=total_requests / max(1, total_programs),
            most_successful_patterns=top_patterns,
            task_type_distribution=task_types,
            generation_distribution=generations
        )

    def cleanup_old_programs(self, days: int | None = None) -> int:
        """Remove programs not accessed in specified days."""
        if days is None:
            days = self.config.storage.retention_days

        removed = 0
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for program_id in list(self.cache.iterkeys()):
            entry = self.get_program(program_id)
            if entry and entry.last_accessed.timestamp() < cutoff_date:
                if self.cache.delete(program_id):
                    removed += 1

        return removed

    def export_programs(
        self,
        output_dir: str,
        format: str = "json",
        filter_successful: bool = False,
        limit: int | None = None,
        single_file: bool = False
    ) -> int:
        """Export programs to files using ProgramExporter."""
        # Get programs to export
        programs = []
        count = 0

        for program_id in self.cache.iterkeys():
            if limit and count >= limit:
                break

            entry = self.get_program(program_id)
            if entry:
                if filter_successful and not entry.success:
                    continue

                programs.append(entry)
                count += 1

        if not programs:
            return 0

        # Use exporter
        exporter = ProgramExporter(include_metadata=self.config.export.include_metadata)

        try:
            # Map format names
            export_format = format
            if format == "msgpack":
                # For msgpack, export as JSON and convert
                export_format = "json"
            elif format in exporter.SUPPORTED_FORMATS:
                export_format = format
            else:
                # Default to configured format
                export_format = self.config.export.default_format

            exporter.export_programs_batch(
                programs,
                format=export_format,
                output_dir=output_dir,
                single_file=single_file
            )

            return len(programs)

        except Exception as e:
            print(f"Error exporting programs: {e}")
            return 0

    def export_program_by_id(
        self,
        program_id: str,
        format: str = "dsl",
        output_file: str | None = None
    ) -> str | None:
        """Export a specific program by ID."""
        entry = self.get_program(program_id)
        if not entry:
            return None

        exporter = ProgramExporter(include_metadata=self.config.export.include_metadata)

        try:
            return exporter.export_program(entry, format=format, output_file=output_file)
        except Exception as e:
            print(f"Error exporting program {program_id}: {e}")
            return None

    def analyze_patterns(self, force_refresh: bool = False) -> dict[str, PatternAnalysis]:
        """Analyze all cached programs to discover patterns."""
        self.stats["pattern_analyses"] += 1

        # Get all successful programs
        programs = self.get_successful_programs(
            min_accuracy=self.config.pattern_mining.min_success_rate,
            limit=10000  # Analyze many programs
        )

        if not programs:
            return {}

        # Run pattern analysis
        patterns = self.pattern_analyzer.analyze_programs(programs)

        # Update stored patterns
        self.discovered_patterns = patterns
        self._save_patterns()

        return patterns

    def get_patterns(self) -> dict[str, PatternAnalysis]:
        """Get currently discovered patterns."""
        return self.discovered_patterns.copy()

    def find_patterns_in_program(self, program: DSLProgram) -> list[str]:
        """Find which patterns exist in a program."""
        if not self.discovered_patterns:
            # Run analysis if no patterns discovered yet
            self.analyze_patterns()

        return self.pattern_analyzer.find_patterns_in_program(
            program, self.discovered_patterns
        )

    def suggest_patterns_for_task(
        self,
        task_features: dict[str, Any],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Suggest patterns that might work for a task."""
        if not self.discovered_patterns:
            self.analyze_patterns()

        return self.pattern_analyzer.suggest_patterns_for_task(
            task_features, self.discovered_patterns, top_k
        )

    def _load_patterns(self) -> None:
        """Load patterns from file."""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file) as f:
                    data = json.load(f)
                    self.discovered_patterns = {
                        pid: PatternAnalysis(**pdata)
                        for pid, pdata in data.items()
                    }
            except Exception as e:
                print(f"Error loading patterns: {e}")
                self.discovered_patterns = {}

    def _save_patterns(self) -> None:
        """Save patterns to file."""
        try:
            data = {
                pid: {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type,
                    'pattern_description': p.pattern_description,
                    'operation_sequence': p.operation_sequence,
                    'frequency': p.frequency,
                    'success_rate': p.success_rate,
                    'program_ids': p.program_ids,
                    'created_at': p.created_at.isoformat(),
                    'last_updated': p.last_updated.isoformat()
                }
                for pid, p in self.discovered_patterns.items()
            }

            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")

    def generate_analytics_report(
        self,
        format: str = 'html',
        include_visuals: bool = True
    ) -> str | None:
        """Generate analytics report for the cache."""
        if not self.analytics_dashboard:
            print("Analytics dashboard not enabled")
            return None

        return self.analytics_dashboard.generate_analytics_report(
            format=format,
            include_visuals=include_visuals
        )

    def collect_analytics_metrics(self) -> dict[str, Any] | None:
        """Collect current analytics metrics."""
        if not self.analytics_dashboard:
            return None

        return self.analytics_dashboard.collect_metrics()

    def export_analytics_data(
        self,
        format: str = 'csv',
        output_file: str | None = None
    ) -> str | None:
        """Export analytics data."""
        if not self.analytics_dashboard:
            return None

        if format == 'csv':
            return self.analytics_dashboard.export_metrics_csv(output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_ensemble_programs(
        self,
        task_id: str | None = None,
        similar_to: DSLProgram | None = None,
        min_accuracy: float = 0.7,
        limit: int = 20
    ) -> list[ProgramCacheEntry]:
        """Get programs for ensemble voting."""
        if not self.ensemble_interface:
            return []

        return self.ensemble_interface.get_programs_for_ensemble(
            task_id=task_id,
            similar_to=similar_to,
            min_accuracy=min_accuracy,
            limit=limit
        )

    def ensemble_vote(
        self,
        input_grid: Any,  # np.ndarray
        candidate_programs: list[ProgramCacheEntry] | None = None,
        task_id: str | None = None,
        execution_func: Any = None
    ) -> Any:  # EnsembleResult
        """Perform ensemble voting."""
        if not self.ensemble_interface:
            return None

        # Get candidate programs if not provided
        if candidate_programs is None:
            candidate_programs = self.get_ensemble_programs(task_id=task_id)

        if not candidate_programs:
            return None

        self.stats["ensemble_votes"] += 1

        result = self.ensemble_interface.ensemble_vote(
            input_grid=input_grid,
            candidate_programs=candidate_programs,
            execution_func=execution_func
        )

        # Cache result if task_id provided
        if task_id and result:
            self.ensemble_interface.cache_ensemble_result(task_id, result)

        return result

    def get_ensemble_performance_stats(self) -> dict[str, Any]:
        """Get ensemble performance statistics."""
        if not self.ensemble_interface:
            return {}

        # Get all cached results
        results = list(self.ensemble_interface.ensemble_cache.values())

        return self.ensemble_interface.analyze_ensemble_performance(results)

    # Private helper methods

    def _find_duplicate_by_hash(self, program_hash: str) -> str | None:
        """Find program ID by hash."""
        hash_index_path = self.index_dir / "hash_index.json"
        if hash_index_path.exists():
            with open(hash_index_path) as f:
                hash_index = json.load(f)
                return hash_index.get(program_hash)
        return None

    def _update_hash_index(self, program_hash: str, program_id: str) -> None:
        """Update hash index."""
        hash_index_path = self.index_dir / "hash_index.json"
        hash_index = {}

        if hash_index_path.exists():
            with open(hash_index_path) as f:
                hash_index = json.load(f)

        hash_index[program_hash] = program_id

        with open(hash_index_path, 'w') as f:
            json.dump(hash_index, f)

    def _update_task_index(self, task_id: str, program_id: str) -> None:
        """Update task index."""
        task_index_path = self.index_dir / "task_index.json"
        task_index = {}

        if task_index_path.exists():
            with open(task_index_path) as f:
                task_index = json.load(f)

        if task_id not in task_index:
            task_index[task_id] = []

        task_index[task_id].append(program_id)

        with open(task_index_path, 'w') as f:
            json.dump(task_index, f)

    def _get_task_program_ids(self, task_id: str) -> list[str]:
        """Get program IDs for a task."""
        task_index_path = self.index_dir / "task_index.json"
        if task_index_path.exists():
            with open(task_index_path) as f:
                task_index = json.load(f)
                return task_index.get(task_id, [])
        return []


    def _sample_program_ids(self, sample_size: int) -> list[str]:
        """Get a sample of program IDs."""
        all_ids = list(self.cache.iterkeys())
        if len(all_ids) <= sample_size:
            return all_ids

        import random
        return random.sample(all_ids, sample_size)

    def _compute_similarity(self, prog1: DSLProgram, prog2: DSLProgram) -> float:
        """Compute similarity between two programs."""
        # Get operation types in sequence (preserving order for better comparison)
        ops1 = [op.get('type', '') for op in prog1.operations]
        ops2 = [op.get('type', '') for op in prog2.operations]

        if not ops1 or not ops2:
            return 0.0

        # Combine Jaccard similarity with sequence similarity

        # 1. Jaccard similarity (set-based)
        set1, set2 = set(ops1), set(ops2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0

        # 2. Sequence similarity (order matters)
        # Count matching operations at same positions
        matches = sum(1 for i in range(min(len(ops1), len(ops2))) if ops1[i] == ops2[i])
        max_len = max(len(ops1), len(ops2))
        sequence_sim = matches / max_len if max_len > 0 else 0.0

        # 3. Length similarity
        length_sim = min(len(ops1), len(ops2)) / max(len(ops1), len(ops2))

        # Weighted combination (emphasize sequence similarity)
        return 0.3 * jaccard + 0.5 * sequence_sim + 0.2 * length_sim

    def close(self) -> None:
        """Close cache connection."""
        try:
            self.cache.close()
        except Exception as e:
            print(f"Error closing program cache: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
