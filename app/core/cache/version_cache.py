"""
Version Cache System
Provides caching for document version operations to improve performance
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class VersionCache:
    """
    Cache system for document version operations
    Supports multiple cache backends and automatic expiration
    """
    
    def __init__(self, backend: str = "memory", max_size: int = 1000, default_ttl: int = 3600):
        self.backend = backend
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Memory cache storage
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        
        logger.info(f"Initialized VersionCache with backend={backend}, max_size={max_size}")
    
    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """Generate a unique cache key for the operation and parameters."""
        
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_data = f"{operation}:{json.dumps(sorted_kwargs, sort_keys=True)}"
        
        # Use SHA256 hash for consistent, collision-resistant keys
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        
        if "expires_at" not in cache_entry:
            return False
        
        return time.time() > cache_entry["expires_at"]
    
    def _evict_lru(self):
        """Evict least recently used items when cache is full."""
        
        if len(self._memory_cache) < self.max_size:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        del self._memory_cache[lru_key]
        del self._access_times[lru_key]
        
        self.stats["evictions"] += 1
        self.stats["size"] -= 1
        
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    async def get(self, operation: str, **kwargs) -> Optional[Any]:
        """Get cached result for an operation."""
        
        cache_key = self._generate_cache_key(operation, **kwargs)
        
        if cache_key not in self._memory_cache:
            self.stats["misses"] += 1
            return None
        
        cache_entry = self._memory_cache[cache_key]
        
        # Check expiration
        if self._is_expired(cache_entry):
            del self._memory_cache[cache_key]
            del self._access_times[cache_key]
            self.stats["misses"] += 1
            self.stats["size"] -= 1
            return None
        
        # Update access time
        self._access_times[cache_key] = time.time()
        self.stats["hits"] += 1
        
        logger.debug(f"Cache hit for operation: {operation}")
        return cache_entry["data"]
    
    async def set(self, operation: str, data: Any, ttl: Optional[int] = None, **kwargs):
        """Cache result for an operation."""
        
        cache_key = self._generate_cache_key(operation, **kwargs)
        ttl = ttl or self.default_ttl
        
        # Evict if necessary
        self._evict_lru()
        
        # Store cache entry
        cache_entry = {
            "data": data,
            "created_at": time.time(),
            "expires_at": time.time() + ttl if ttl > 0 else None
        }
        
        self._memory_cache[cache_key] = cache_entry
        self._access_times[cache_key] = time.time()
        
        self.stats["size"] += 1
        
        logger.debug(f"Cached result for operation: {operation}, TTL: {ttl}s")
    
    async def invalidate(self, operation: str, **kwargs):
        """Invalidate cached result for an operation."""
        
        cache_key = self._generate_cache_key(operation, **kwargs)
        
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
            del self._access_times[cache_key]
            self.stats["size"] -= 1
            logger.debug(f"Invalidated cache for operation: {operation}")
    
    async def invalidate_pattern(self, operation_pattern: str):
        """Invalidate all cached results matching an operation pattern."""
        
        keys_to_remove = []
        
        for cache_key in self._memory_cache.keys():
            # Simple pattern matching - could be enhanced with regex
            if operation_pattern in cache_key:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self._memory_cache[key]
            del self._access_times[key]
            self.stats["size"] -= 1
        
        logger.debug(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {operation_pattern}")
    
    async def clear(self):
        """Clear all cached data."""
        
        self._memory_cache.clear()
        self._access_times.clear()
        self.stats["size"] = 0
        
        logger.info("Cleared all cache data")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_usage_mb": len(pickle.dumps(self._memory_cache)) / (1024 * 1024)
        }


class DiffCache:
    """Specialized cache for document diff operations."""
    
    def __init__(self, cache: VersionCache):
        self.cache = cache
    
    async def get_diff(self, doc1_id: str, doc2_id: str, diff_type: str = "full") -> Optional[Dict[str, Any]]:
        """Get cached diff result."""
        
        return await self.cache.get(
            "document_diff",
            doc1_id=doc1_id,
            doc2_id=doc2_id,
            diff_type=diff_type
        )
    
    async def cache_diff(self, doc1_id: str, doc2_id: str, diff_result: Dict[str, Any], 
                        diff_type: str = "full", ttl: int = 7200):
        """Cache diff result."""
        
        await self.cache.set(
            "document_diff",
            diff_result,
            ttl=ttl,
            doc1_id=doc1_id,
            doc2_id=doc2_id,
            diff_type=diff_type
        )
    
    async def invalidate_document_diffs(self, doc_id: str):
        """Invalidate all diffs involving a specific document."""
        
        await self.cache.invalidate_pattern(f"document_diff.*{doc_id}")


class SearchCache:
    """Specialized cache for search operations."""
    
    def __init__(self, cache: VersionCache):
        self.cache = cache
    
    async def get_search_results(self, query: str, search_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached search results."""
        
        return await self.cache.get(
            "search_results",
            query=query,
            **search_params
        )
    
    async def cache_search_results(self, query: str, search_params: Dict[str, Any], 
                                  results: Dict[str, Any], ttl: int = 1800):
        """Cache search results."""
        
        await self.cache.set(
            "search_results",
            results,
            ttl=ttl,
            query=query,
            **search_params
        )
    
    async def invalidate_search_cache(self):
        """Invalidate all search cache entries."""
        
        await self.cache.invalidate_pattern("search_results")


class VersionMetadataCache:
    """Specialized cache for version metadata operations."""
    
    def __init__(self, cache: VersionCache):
        self.cache = cache
    
    async def get_family_versions(self, family_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached family version list."""
        
        return await self.cache.get("family_versions", family_id=family_id)
    
    async def cache_family_versions(self, family_id: str, versions: List[Dict[str, Any]], ttl: int = 3600):
        """Cache family version list."""
        
        await self.cache.set("family_versions", versions, ttl=ttl, family_id=family_id)
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document metadata."""
        
        return await self.cache.get("document_metadata", doc_id=doc_id)
    
    async def cache_document_metadata(self, doc_id: str, metadata: Dict[str, Any], ttl: int = 3600):
        """Cache document metadata."""
        
        await self.cache.set("document_metadata", metadata, ttl=ttl, doc_id=doc_id)
    
    async def invalidate_family_cache(self, family_id: str):
        """Invalidate cache for a document family."""
        
        await self.cache.invalidate("family_versions", family_id=family_id)
    
    async def invalidate_document_cache(self, doc_id: str):
        """Invalidate cache for a specific document."""
        
        await self.cache.invalidate("document_metadata", doc_id=doc_id)


# Global cache instances
_version_cache = None
_diff_cache = None
_search_cache = None
_metadata_cache = None


def get_version_cache() -> VersionCache:
    """Get the global version cache instance."""
    
    global _version_cache
    if _version_cache is None:
        _version_cache = VersionCache(
            backend="memory",
            max_size=1000,
            default_ttl=3600
        )
    return _version_cache


def get_diff_cache() -> DiffCache:
    """Get the global diff cache instance."""
    
    global _diff_cache
    if _diff_cache is None:
        _diff_cache = DiffCache(get_version_cache())
    return _diff_cache


def get_search_cache() -> SearchCache:
    """Get the global search cache instance."""
    
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchCache(get_version_cache())
    return _search_cache


def get_metadata_cache() -> VersionMetadataCache:
    """Get the global metadata cache instance."""
    
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = VersionMetadataCache(get_version_cache())
    return _metadata_cache


async def warm_cache_for_family(family_id: str, db_session):
    """Pre-warm cache with commonly accessed data for a document family."""
    
    try:
        from sqlalchemy import text
        
        # Load family versions
        result = await db_session.execute(
            text("""
            SELECT d.id, d.filename, d.version_number, d.is_latest_version,
                   d.upload_date, d.processing_status, dv.version_tag, dv.author
            FROM documents d
            LEFT JOIN document_versions dv ON d.id = dv.document_id
            WHERE d.document_family_id = :family_id OR d.id = :family_id
            ORDER BY d.version_number DESC
            """),
            {"family_id": family_id}
        )
        
        versions = []
        for row in result.fetchall():
            versions.append({
                "id": str(row.id),
                "filename": row.filename,
                "version_number": row.version_number,
                "is_latest_version": row.is_latest_version,
                "upload_date": row.upload_date.isoformat(),
                "processing_status": row.processing_status,
                "version_tag": row.version_tag,
                "author": row.author
            })
        
        # Cache family versions
        metadata_cache = get_metadata_cache()
        await metadata_cache.cache_family_versions(family_id, versions)
        
        logger.info(f"Warmed cache for family {family_id} with {len(versions)} versions")
        
    except Exception as e:
        logger.error(f"Failed to warm cache for family {family_id}: {e}")


async def cleanup_expired_cache():
    """Clean up expired cache entries (background task)."""
    
    cache = get_version_cache()
    
    expired_keys = []
    for key, entry in cache._memory_cache.items():
        if cache._is_expired(entry):
            expired_keys.append(key)
    
    for key in expired_keys:
        del cache._memory_cache[key]
        del cache._access_times[key]
        cache.stats["size"] -= 1
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


# Cache decorator for functions
def cached(operation: str, ttl: int = 3600):
    """Decorator to cache function results."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_version_cache()
            
            # Generate cache key from function arguments
            cache_key_data = {
                "args": str(args),
                "kwargs": kwargs
            }
            
            # Try to get from cache
            cached_result = await cache.get(operation, **cache_key_data)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(operation, result, ttl=ttl, **cache_key_data)
            
            return result
        
        return wrapper
    return decorator
