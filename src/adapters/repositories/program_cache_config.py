"""Configuration loader for program cache."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StorageConfig:
    """Storage configuration for program cache."""
    size_limit_gb: float = 1.0
    cache_dir: str = "data/program_cache"
    eviction_policy: str = "least-recently-used"
    retention_days: int = 30


@dataclass
class SimilarityConfig:
    """Similarity detection configuration."""
    similarity_threshold: float = 0.95
    max_similarity_checks: int = 1000
    check_types: list[str] = field(default_factory=lambda: ["exact", "semantic", "fuzzy"])


@dataclass
class PatternMiningConfig:
    """Pattern mining configuration."""
    min_frequency: int = 5
    min_success_rate: float = 0.7
    max_patterns: int = 500
    pattern_types: list[str] = field(default_factory=lambda: ["sequence", "structure", "parameter"])


@dataclass
class PerformanceConfig:
    """Performance settings."""
    similarity_timeout_ms: int = 100
    pattern_analysis_timeout_s: int = 30
    lookup_timeout_ms: int = 10
    export_timeout_s: int = 5


@dataclass
class AnalyticsConfig:
    """Analytics configuration."""
    enable_analytics: bool = True
    update_interval_s: int = 60
    metrics: list[str] = field(default_factory=lambda: [
        "cache_hit_rate",
        "program_success_rate",
        "pattern_frequency",
        "task_type_distribution",
        "generation_distribution",
        "storage_efficiency"
    ])


@dataclass
class ExportConfig:
    """Export configuration."""
    default_format: str = "json"
    formats: list[str] = field(default_factory=lambda: ["json", "msgpack", "dsl", "python"])
    include_metadata: bool = True
    batch_size: int = 100


@dataclass
class EnsembleConfig:
    """Ensemble integration configuration."""
    enable_ensemble: bool = True
    success_weight_multiplier: float = 2.0
    min_programs_for_vote: int = 3
    confidence_threshold: float = 0.7


@dataclass
class ProgramCacheConfig:
    """Complete configuration for program cache."""
    storage: StorageConfig = field(default_factory=StorageConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    pattern_mining: PatternMiningConfig = field(default_factory=PatternMiningConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProgramCacheConfig':
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            # Return default config if file doesn't exist
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            storage=StorageConfig(**data.get('storage', {})),
            similarity=SimilarityConfig(**data.get('similarity', {})),
            pattern_mining=PatternMiningConfig(**data.get('pattern_mining', {})),
            performance=PerformanceConfig(**data.get('performance', {})),
            analytics=AnalyticsConfig(**data.get('analytics', {})),
            export=ExportConfig(**data.get('export', {})),
            ensemble=EnsembleConfig(**data.get('ensemble', {}))
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'storage': self.storage.__dict__,
            'similarity': self.similarity.__dict__,
            'pattern_mining': self.pattern_mining.__dict__,
            'performance': self.performance.__dict__,
            'analytics': self.analytics.__dict__,
            'export': self.export.__dict__,
            'ensemble': self.ensemble.__dict__
        }
