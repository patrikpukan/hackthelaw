"""
Search result caching system to avoid duplicate queries and improve performance.
Implements intelligent caching with TTL, similarity-based retrieval, and cache statistics.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    query: str
    query_hash: str
    results: List[Dict[str, Any]]
    timestamp: float
    ttl_seconds: int
    hit_count: int = 0
    similarity_threshold: float = 0.85
    metadata: Dict[str, Any] = None


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    similar_hits: int = 0
    expired_entries: int = 0
    total_entries: int = 0
    hit_rate: float = 0.0
    avg_results_per_query: float = 0.0
    cache_size_mb: float = 0.0


class QueryNormalizer:
    """Normalizes queries for consistent caching."""
    
    def __init__(self):
        # Common words to remove for similarity comparison
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'about', 'what', 'when', 'where',
            'who', 'why', 'how', 'which', 'can', 'could', 'should', 'would'
        }
    
    def normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove punctuation except question marks and quotes
        import string
        translator = str.maketrans('', '', string.punctuation.replace('?', '').replace('"', '').replace("'", ''))
        normalized = normalized.translate(translator)
        
        return normalized
    
    def extract_keywords(self, query: str) -> set:
        """Extract keywords from query for similarity comparison."""
        
        normalized = self.normalize_query(query)
        words = set(normalized.split())
        
        # Remove stop words
        keywords = words - self.stop_words
        
        # Remove very short words (less than 3 characters)
        keywords = {word for word in keywords if len(word) >= 3}
        
        return keywords
    
    def calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        
        keywords1 = self.extract_keywords(query1)
        keywords2 = self.extract_keywords(query2)
        
        if not keywords1 and not keywords2:
            return 1.0
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0


class SearchResultsCache:
    """Intelligent cache for search results with TTL and similarity matching."""
    
    def __init__(self, max_entries: int = 1000, default_ttl: int = 3600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.query_normalizer = QueryNormalizer()
        self.stats = CacheStats()
        
        # Index for similarity searches
        self.query_index: Dict[str, List[str]] = defaultdict(list)  # keyword -> [query_hashes]
    
    def _generate_hash(self, query: str) -> str:
        """Generate hash for query."""
        
        normalized = self.query_normalizer.normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        
        return time.time() - entry.timestamp > entry.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > entry.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats.expired_entries += 1
    
    def _remove_entry(self, query_hash: str):
        """Remove entry and update indexes."""
        
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            
            # Remove from keyword index
            keywords = self.query_normalizer.extract_keywords(entry.query)
            for keyword in keywords:
                if query_hash in self.query_index[keyword]:
                    self.query_index[keyword].remove(query_hash)
                if not self.query_index[keyword]:
                    del self.query_index[keyword]
            
            del self.cache[query_hash]
    
    def _evict_oldest(self):
        """Evict oldest entries when cache is full."""
        
        if len(self.cache) >= self.max_entries:
            # Sort by timestamp and remove oldest
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].timestamp)
            to_remove = len(sorted_entries) - self.max_entries + 100  # Remove extra entries
            
            for query_hash, _ in sorted_entries[:to_remove]:
                self._remove_entry(query_hash)
    
    def _find_similar_queries(self, query: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar cached queries."""
        
        keywords = self.query_normalizer.extract_keywords(query)
        candidate_hashes = set()
        
        # Find candidates based on keyword overlap
        for keyword in keywords:
            if keyword in self.query_index:
                candidate_hashes.update(self.query_index[keyword])
        
        # Calculate similarity for candidates
        similar_queries = []
        for query_hash in candidate_hashes:
            if query_hash in self.cache:
                entry = self.cache[query_hash]
                if not self._is_expired(entry):
                    similarity = self.query_normalizer.calculate_similarity(query, entry.query)
                    if similarity >= threshold:
                        similar_queries.append((query_hash, similarity))
        
        # Sort by similarity (descending)
        similar_queries.sort(key=lambda x: x[1], reverse=True)
        
        return similar_queries
    
    def get(self, query: str, similarity_threshold: float = 0.85) -> Optional[List[Dict[str, Any]]]:
        """Get cached results for query."""
        
        self.stats.total_queries += 1
        
        # Try exact match first
        query_hash = self._generate_hash(query)
        
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            if not self._is_expired(entry):
                entry.hit_count += 1
                self.stats.cache_hits += 1
                logger.info(f"Cache hit (exact): {query[:50]}...")
                return entry.results.copy()
            else:
                # Remove expired entry
                self._remove_entry(query_hash)
                self.stats.expired_entries += 1
        
        # Try similarity match
        similar_queries = self._find_similar_queries(query, similarity_threshold)
        
        if similar_queries:
            best_match_hash, similarity = similar_queries[0]
            entry = self.cache[best_match_hash]
            entry.hit_count += 1
            self.stats.cache_hits += 1
            self.stats.similar_hits += 1
            logger.info(f"Cache hit (similar {similarity:.3f}): {query[:50]}... -> {entry.query[:50]}...")
            return entry.results.copy()
        
        # Cache miss
        self.stats.cache_misses += 1
        logger.info(f"Cache miss: {query[:50]}...")
        return None
    
    def put(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        ttl_seconds: Optional[int] = None
    ):
        """Store results in cache."""
        
        if not results:
            return  # Don't cache empty results
        
        ttl = ttl_seconds or self.default_ttl
        query_hash = self._generate_hash(query)
        
        # Cleanup and eviction
        self._cleanup_expired()
        self._evict_oldest()
        
        # Create cache entry
        entry = CacheEntry(
            query=query,
            query_hash=query_hash,
            results=results.copy(),
            timestamp=time.time(),
            ttl_seconds=ttl,
            metadata={
                'result_count': len(results),
                'avg_similarity': sum(r.get('similarity_score', 0) for r in results) / len(results)
            }
        )
        
        # Store entry
        self.cache[query_hash] = entry
        
        # Update keyword index
        keywords = self.query_normalizer.extract_keywords(query)
        for keyword in keywords:
            self.query_index[keyword].append(query_hash)
        
        logger.info(f"Cached query: {query[:50]}... ({len(results)} results)")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        
        to_remove = []
        for query_hash, entry in self.cache.items():
            if pattern.lower() in entry.query.lower():
                to_remove.append(query_hash)
        
        for query_hash in to_remove:
            self._remove_entry(query_hash)
        
        logger.info(f"Invalidated {len(to_remove)} entries matching pattern: {pattern}")
    
    def clear(self):
        """Clear all cache entries."""
        
        self.cache.clear()
        self.query_index.clear()
        self.stats = CacheStats()
        logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        
        self.stats.total_entries = len(self.cache)
        
        if self.stats.total_queries > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_queries
        
        if self.stats.cache_hits > 0:
            total_results = sum(len(entry.results) for entry in self.cache.values())
            self.stats.avg_results_per_query = total_results / self.stats.cache_hits
        
        # Estimate cache size
        cache_size_bytes = 0
        for entry in self.cache.values():
            cache_size_bytes += len(json.dumps(asdict(entry)).encode())
        
        self.stats.cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        return self.stats
    
    def get_top_queries(self, limit: int = 10) -> List[Tuple[str, int, float]]:
        """Get most frequently accessed queries."""
        
        queries_with_hits = [
            (entry.query, entry.hit_count, entry.timestamp)
            for entry in self.cache.values()
        ]
        
        # Sort by hit count (descending)
        queries_with_hits.sort(key=lambda x: x[1], reverse=True)
        
        return queries_with_hits[:limit]
    
    def optimize(self):
        """Optimize cache by removing low-value entries."""
        
        current_time = time.time()
        
        # Remove entries that haven't been accessed recently
        cutoff_time = current_time - (self.default_ttl * 2)
        
        to_remove = []
        for query_hash, entry in self.cache.items():
            # Remove if old and not frequently accessed
            if (entry.timestamp < cutoff_time and entry.hit_count < 2):
                to_remove.append(query_hash)
        
        for query_hash in to_remove:
            self._remove_entry(query_hash)
        
        logger.info(f"Cache optimization: removed {len(to_remove)} low-value entries")


# Global cache instance
_search_cache = None


def get_search_cache() -> SearchResultsCache:
    """Get global search cache instance."""
    
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchResultsCache()
    
    return _search_cache


def cached_search(
    search_func,
    query: str,
    cache_ttl: Optional[int] = None,
    similarity_threshold: float = 0.85,
    **search_kwargs
) -> List[Dict[str, Any]]:
    """Decorator function for caching search results."""
    
    cache = get_search_cache()
    
    # Try to get from cache
    cached_results = cache.get(query, similarity_threshold)
    if cached_results is not None:
        return cached_results
    
    # Execute search
    results = search_func(query, **search_kwargs)
    
    # Cache results
    if results:
        cache.put(query, results, cache_ttl)
    
    return results


def invalidate_cache_for_document(document_id: str):
    """Invalidate cache entries that might contain results from a specific document."""
    
    cache = get_search_cache()
    
    # This is a simple approach - in a more sophisticated system,
    # we could track which documents contributed to each cache entry
    to_remove = []
    for query_hash, entry in cache.cache.items():
        # Check if any result comes from the specified document
        for result in entry.results:
            if result.get('document', {}).get('id') == document_id:
                to_remove.append(query_hash)
                break
    
    for query_hash in to_remove:
        cache._remove_entry(query_hash)
    
    logger.info(f"Invalidated {len(to_remove)} cache entries for document {document_id}")


def warm_cache_from_common_queries(common_queries: List[str], search_func):
    """Pre-populate cache with results for common queries."""
    
    cache = get_search_cache()
    
    for query in common_queries:
        if not cache.get(query):
            try:
                results = search_func(query)
                if results:
                    cache.put(query, results)
                    logger.info(f"Pre-cached query: {query}")
            except Exception as e:
                logger.error(f"Error pre-caching query '{query}': {e}")


# Common legal queries for warming up the cache
COMMON_LEGAL_QUERIES = [
    "termination clause",
    "liability limitations",
    "intellectual property rights",
    "confidentiality agreement",
    "force majeure",
    "governing law",
    "dispute resolution",
    "payment terms",
    "delivery obligations",
    "warranty provisions"
] 