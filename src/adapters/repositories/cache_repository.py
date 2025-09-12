"""Cache repository implementation using diskcache with LRU eviction."""

import sys
import time
from pathlib import Path
from typing import Any

import diskcache as dc

# Add source root to path for absolute imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

try:
    from domain.models import ARCTask
except ImportError:
    # Fallback to relative import
    from ...domain.models import ARCTask


class CacheRepository:
    """Persistent cache repository with size limits and LRU eviction."""

    def __init__(
        self,
        cache_dir: str | None = None,
        size_limit: int = 2 * 1024 * 1024 * 1024,  # 2GB default
        eviction_policy: str = "least-recently-used"
    ):
        """Initialize cache repository."""
        if cache_dir is None:
            cache_dir_path = Path(__file__).parent.parent.parent.parent / "data" / "cache"
            cache_dir = str(cache_dir_path)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diskcache with LRU eviction
        self.cache = dc.Cache(
            str(self.cache_dir),
            size_limit=size_limit,
            eviction_policy=eviction_policy
        )

        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "last_reset": time.time()
        }

    def get(self, key: str) -> ARCTask | None:
        """Get task from cache."""
        try:
            value = self.cache.get(key)
            if value is not None:
                self.stats["hits"] += 1
                return value
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            print(f"Cache get error for key {key}: {e}")
            self.stats["misses"] += 1
            return None

    def set(self, key: str, value: ARCTask, expire: float | None = None) -> bool:
        """Set task in cache with optional expiration."""
        try:
            result = self.cache.set(key, value, expire=expire)
            if result:
                self.stats["sets"] += 1
            return bool(result)
        except Exception as e:
            print(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete task from cache."""
        try:
            return bool(self.cache.delete(key))
        except Exception as e:
            print(f"Cache delete error for key {key}: {e}")
            return False

    def clear(self) -> None:
        """Clear entire cache."""
        try:
            self.cache.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "evictions": 0,
                "last_reset": time.time()
            }
        except Exception as e:
            print(f"Cache clear error: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache

    def keys(self) -> list[str]:
        """Get all cache keys."""
        try:
            return list(self.cache.iterkeys())
        except Exception:
            return []

    def get_size_info(self) -> dict[str, Any]:
        """Get cache size information."""
        try:
            return {
                "volume": self.cache.volume(),
                "size_limit": self.cache.size_limit,
                "usage_percent": (self.cache.volume() / self.cache.size_limit) * 100,
                "count": len(self.cache)
            }
        except Exception as e:
            print(f"Error getting cache size info: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0.0

        size_info = self.get_size_info()

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "sets": self.stats["sets"],
            "evictions": self.stats["evictions"],
            "uptime_seconds": time.time() - self.stats["last_reset"],
            **size_info
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "last_reset": time.time()
        }

    def warm_cache(
        self,
        tasks: dict[str, ARCTask],
        task_source: str = "training",
        preprocessing_options: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        """Warm cache with frequently accessed tasks."""
        results = {}

        for task_id, task in tasks.items():
            cache_key = self.create_cache_key(task_id, task_source, preprocessing_options)
            success = self.set(cache_key, task)
            results[task_id] = success

        return results

    def create_cache_key(
        self,
        task_id: str,
        task_source: str = "training",
        preprocessing_options: dict[str, Any] | None = None
    ) -> str:
        """Create cache key based on task_id, source, and preprocessing options."""
        base_key = f"{task_source}:{task_id}"

        if preprocessing_options:
            # Create deterministic hash of preprocessing options
            import hashlib
            import json

            options_str = json.dumps(preprocessing_options, sort_keys=True)
            options_hash = hashlib.md5(options_str.encode()).hexdigest()[:8]
            base_key += f":{options_hash}"

        return base_key

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        invalidated = 0
        keys_to_delete = []

        for key in self.keys():
            if pattern in key:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if self.delete(key):
                invalidated += 1

        return invalidated

    def cleanup_expired(self) -> int:
        """Cleanup expired entries (handled automatically by diskcache)."""
        initial_count = len(self.cache)
        self.cache.expire()
        final_count = len(self.cache)
        return initial_count - final_count

    def get_cache_efficiency(self) -> dict[str, float]:
        """Get cache efficiency metrics."""
        stats = self.get_statistics()
        size_info = self.get_size_info()

        return {
            "hit_rate": stats["hit_rate"],
            "storage_efficiency": size_info.get("usage_percent", 0) / 100,
            "requests_per_item": stats["total_requests"] / max(1, size_info.get("count", 1)),
            "average_access_frequency": stats["hits"] / max(1, size_info.get("count", 1))
        }

    def force_cleanup(self) -> None:
        """Force cleanup of cache resources - useful for Windows file handle issues."""
        try:
            # Check if Python is shutting down
            import sys
            if sys.meta_path is None:
                return  # Python is shutting down, skip cleanup

            import gc
            import platform
            import time

            # Close cache
            self.close()

            # Windows-specific aggressive cleanup
            if platform.system().lower() == 'windows':
                # Multiple garbage collection passes to ensure cleanup
                for _ in range(3):
                    gc.collect()
                    time.sleep(0.05)  # Small delay between collections

                # Try to explicitly close any remaining file handles
                if hasattr(self.cache, '_conn') and self.cache._conn:
                    try:
                        self.cache._conn.close()
                    except Exception:
                        pass

        except (ImportError, AttributeError):
            # Handle cases where Python is shutting down
            pass
        except Exception as e:
            try:
                print(f"Error during force cleanup: {e}")
            except Exception:
                pass

    def __del__(self):
        """Destructor to ensure cache is closed when object is garbage collected."""
        try:
            self.close()
        except Exception:
            # Ignore errors during destruction
            pass

    def close(self) -> None:
        """Close cache connection and ensure all resources are released."""
        try:
            # Check if Python is shutting down to avoid import errors
            import sys
            if sys.meta_path is None:
                return  # Python is shutting down, skip cleanup

            # Close the diskcache connection
            if hasattr(self.cache, 'close'):
                self.cache.close()

            # Additional cleanup for Windows: force garbage collection
            # to ensure any remaining file handles are released
            import gc
            import platform
            import time

            gc.collect()

            # On Windows, add a small delay to ensure file handles are released
            # This addresses the common Windows issue where SQLite database files
            # remain locked briefly after closing connections
            if platform.system().lower() == 'windows':
                time.sleep(0.1)  # Brief delay to allow Windows to release file handles
                gc.collect()  # Second garbage collection for Windows

        except (ImportError, AttributeError):
            # Handle cases where Python is shutting down or modules are not available
            pass
        except Exception as e:
            # Only print if we can safely access print (not during shutdown)
            try:
                print(f"Error closing cache: {e}")
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()
        # Don't suppress exceptions unless we're handling cleanup errors
        return False
