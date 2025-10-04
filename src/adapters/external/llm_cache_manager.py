"""LLM cache manager for efficient response caching and similarity matching."""

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.models import LLMCache

logger = logging.getLogger(__name__)


class LLMCacheManager:
    """Manager for LLM response caching with similarity matching."""

    def __init__(
        self,
        cache_dir: Path,
        max_cache_size_gb: float = 10.0,
        similarity_threshold: float = 0.85,
        cache_ttl_days: int = 30
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = timedelta(days=cache_ttl_days)

        # In-memory index for faster lookups
        self._cache_index: dict[str, LLMCache] = {}
        self._feature_vectors: dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()

        # Load existing cache index
        self._load_cache_index()

    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                    for cache_id, cache_data in data.items():
                        # Convert back to LLMCache object
                        cache_data['created_at'] = datetime.fromisoformat(cache_data['created_at'])
                        self._cache_index[cache_id] = LLMCache(**cache_data)

                        # Reconstruct feature vector
                        if cache_data.get('task_features'):
                            self._feature_vectors[cache_id] = self._features_to_vector(
                                cache_data['task_features']
                            )

                logger.info(f"Loaded {len(self._cache_index)} cache entries")
            except Exception as e:
                logger.error(f"Error loading cache index: {e}")

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.json"
        try:
            # Convert to serializable format
            data = {}
            for cache_id, cache_entry in self._cache_index.items():
                cache_dict = asdict(cache_entry)
                cache_dict['created_at'] = cache_dict['created_at'].isoformat()
                data[cache_id] = cache_dict

            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def _generate_cache_key(
        self,
        prompt: str,
        model_name: str,
        temperature: float
    ) -> str:
        """Generate unique cache key."""
        key_data = f"{prompt}_{model_name}_{temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate hash of prompt for similarity matching."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _features_to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy vector."""
        # Ensure consistent ordering
        feature_names = [
            "grid_size_score",
            "pattern_complexity",
            "color_diversity",
            "transformation_hints",
            "example_consistency"
        ]
        return np.array([features.get(name, 0.0) for name in feature_names])

    def _calculate_similarity(
        self,
        features1: dict[str, float],
        features2: dict[str, float]
    ) -> float:
        """Calculate cosine similarity between feature vectors."""
        vec1 = self._features_to_vector(features1)
        vec2 = self._features_to_vector(features2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def get(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        task_features: dict[str, float] | None = None
    ) -> str | None:
        """Get cached response if available."""
        async with self._lock:
            # Try exact match first
            cache_key = self._generate_cache_key(prompt, model_name, temperature)

            if cache_key in self._cache_index:
                cache_entry = self._cache_index[cache_key]

                # Check if expired
                if datetime.now() - cache_entry.created_at > self.cache_ttl:
                    await self._evict_entry(cache_key)
                    return None

                # Update access count
                cache_entry.access_count += 1

                # Load response from file
                response = await self._load_response(cache_key)
                if response:
                    logger.info(f"Cache hit (exact match) for {model_name}")
                    return response

            # Try similarity matching if features provided
            if task_features and self.similarity_threshold < 1.0:
                similar_entry = await self._find_similar_entry(
                    model_name, temperature, task_features
                )
                if similar_entry:
                    logger.info(f"Cache hit (similarity match) for {model_name}")
                    return similar_entry

            return None

    async def _find_similar_entry(
        self,
        model_name: str,
        temperature: float,
        task_features: dict[str, float]
    ) -> str | None:
        """Find similar cached entry based on task features."""
        best_match = None
        best_similarity = 0.0

        for cache_id, cache_entry in self._cache_index.items():
            # Check if same model and temperature
            if (cache_entry.model_name != model_name or
                abs(cache_entry.temperature - temperature) > 0.1):
                continue

            # Check expiration
            if datetime.now() - cache_entry.created_at > self.cache_ttl:
                continue

            # Calculate similarity
            if cache_entry.task_features:
                similarity = self._calculate_similarity(
                    task_features, cache_entry.task_features
                )

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = cache_id

        if best_match:
            # Update access count
            self._cache_index[best_match].access_count += 1

            # Load response
            return await self._load_response(best_match)

        return None

    async def put(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        response: str,
        token_count: int,
        task_features: dict[str, float] | None = None
    ):
        """Store response in cache."""
        async with self._lock:
            cache_key = self._generate_cache_key(prompt, model_name, temperature)
            prompt_hash = self._generate_prompt_hash(prompt)

            # Create cache entry
            cache_entry = LLMCache(
                cache_id=cache_key,
                prompt_hash=prompt_hash,
                model_name=model_name,
                temperature=temperature,
                response_text=response,
                token_count=token_count,
                created_at=datetime.now(),
                access_count=1,
                task_features=task_features or {}
            )

            # Check cache size before adding
            await self._ensure_cache_size_limit()

            # Store in index
            self._cache_index[cache_key] = cache_entry

            # Store feature vector if provided
            if task_features:
                self._feature_vectors[cache_key] = self._features_to_vector(task_features)

            # Save response to file
            await self._save_response(cache_key, response)

            # Update index file
            self._save_cache_index()

            logger.info(f"Cached response for {model_name} (key: {cache_key[:8]}...)")

    async def _load_response(self, cache_key: str) -> str | None:
        """Load response from cache file."""
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            try:
                with open(cache_file, encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error loading cached response: {e}")

        return None

    async def _save_response(self, cache_key: str, response: str):
        """Save response to cache file."""
        cache_file = self.cache_dir / f"{cache_key}.txt"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(response)
        except Exception as e:
            logger.error(f"Error saving response to cache: {e}")

    async def _ensure_cache_size_limit(self):
        """Ensure cache doesn't exceed size limit using LRU eviction."""
        # Calculate current cache size
        total_size = sum(
            (self.cache_dir / f"{cache_id}.txt").stat().st_size
            for cache_id in self._cache_index
            if (self.cache_dir / f"{cache_id}.txt").exists()
        )

        if total_size < self.max_cache_size_bytes * 0.9:  # 90% threshold
            return

        # Sort by access count and age for LRU eviction
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: (x[1].access_count, x[1].created_at)
        )

        # Evict until we're below 80% of limit
        target_size = self.max_cache_size_bytes * 0.8

        for cache_id, _ in sorted_entries:
            if total_size <= target_size:
                break

            file_size = await self._evict_entry(cache_id)
            total_size -= file_size

    async def _evict_entry(self, cache_id: str) -> int:
        """Evict a cache entry and return freed size."""
        cache_file = self.cache_dir / f"{cache_id}.txt"
        file_size = 0

        if cache_file.exists():
            file_size = cache_file.stat().st_size
            cache_file.unlink()

        # Remove from indices
        self._cache_index.pop(cache_id, None)
        self._feature_vectors.pop(cache_id, None)

        return file_size

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            (self.cache_dir / f"{cache_id}.txt").stat().st_size
            for cache_id in self._cache_index
            if (self.cache_dir / f"{cache_id}.txt").exists()
        )

        model_stats = {}
        for cache_entry in self._cache_index.values():
            model_name = cache_entry.model_name
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "count": 0,
                    "total_accesses": 0,
                    "total_tokens": 0
                }

            model_stats[model_name]["count"] += 1
            model_stats[model_name]["total_accesses"] += cache_entry.access_count
            model_stats[model_name]["total_tokens"] += cache_entry.token_count

        return {
            "total_entries": len(self._cache_index),
            "total_size_mb": total_size / (1024 * 1024),
            "size_limit_mb": self.max_cache_size_bytes / (1024 * 1024),
            "utilization_percent": (total_size / self.max_cache_size_bytes) * 100,
            "model_statistics": model_stats,
            "oldest_entry": min(
                (e.created_at for e in self._cache_index.values()),
                default=None
            ),
            "most_accessed": max(
                self._cache_index.values(),
                key=lambda x: x.access_count,
                default=None
            )
        }

    async def clear_expired(self):
        """Clear expired cache entries."""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                cache_id
                for cache_id, entry in self._cache_index.items()
                if now - entry.created_at > self.cache_ttl
            ]

            for cache_id in expired_keys:
                await self._evict_entry(cache_id)

            if expired_keys:
                self._save_cache_index()
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")

    async def clear_all(self):
        """Clear all cache entries."""
        async with self._lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.txt"):
                cache_file.unlink()

            # Clear indices
            self._cache_index.clear()
            self._feature_vectors.clear()

            # Save empty index
            self._save_cache_index()

            logger.info("Cleared all cache entries")
