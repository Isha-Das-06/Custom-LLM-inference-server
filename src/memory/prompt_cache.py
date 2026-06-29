"""Prompt prefix caching for request reuse"""

import hashlib
import logging
from collections import OrderedDict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PromptCache:
    """
    Cache KV-states for common prompts.

    Enables multiple requests with the same prefix to reuse computation.
    Uses content-hashing (SHA256) to ensure correctness.
    """

    def __init__(self, max_cache_size_mb: int = 1000):
        self.max_cache_size_mb = max_cache_size_mb
        self.cache: OrderedDict = OrderedDict()  # hash -> (tokens, kv_pages)
        self.cache_size_mb = 0
        self.hit_count = 0
        self.miss_count = 0

    def get_prompt_hash(self, prompt_ids: list[int]) -> str:
        """
        Generate SHA256 hash of prompt tokens.

        Args:
            prompt_ids: List of token IDs

        Returns:
            Hex string hash
        """
        tokens_bytes = bytes(prompt_ids)
        return hashlib.sha256(tokens_bytes).hexdigest()

    def get_or_compute(
        self,
        prompt_ids: list[int],
    ) -> Tuple[Optional[list[int]], bool]:
        """
        Retrieve cached KV-pages for prompt, or return None if not cached.

        Args:
            prompt_ids: Prompt token IDs

        Returns:
            (kv_pages, cache_hit) tuple
        """
        prompt_hash = self.get_prompt_hash(prompt_ids)

        if prompt_hash in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(prompt_hash)
            self.hit_count += 1
            logger.debug(f"Prompt cache hit: {prompt_hash[:8]}... (hit rate: {self.hit_rate:.2%})")
            kv_pages = self.cache[prompt_hash]
            return kv_pages, True

        self.miss_count += 1
        logger.debug(f"Prompt cache miss: {prompt_hash[:8]}...")
        return None, False

    def put(self, prompt_ids: list[int], kv_pages: list[int]):
        """
        Store KV-pages for a prompt.

        Args:
            prompt_ids: Prompt token IDs
            kv_pages: Cached KV-page IDs
        """
        prompt_hash = self.get_prompt_hash(prompt_ids)

        # Estimate cache size (2 bytes per token per head)
        # Rough estimate: page_size * num_pages * num_heads * head_dim * 2 bytes
        estimated_size_mb = (len(kv_pages) * 16 * 32 * 128 * 2) / (1024 * 1024)

        # Evict if cache full
        while self.cache_size_mb + estimated_size_mb > self.max_cache_size_mb and self.cache:
            evicted_hash = next(iter(self.cache))
            evicted_pages = self.cache.pop(evicted_hash)
            # Rough size calculation for evicted entry
            evicted_size = (len(evicted_pages) * 16 * 32 * 128 * 2) / (1024 * 1024)
            self.cache_size_mb -= evicted_size
            logger.debug(f"Evicted cache entry: {evicted_hash[:8]}...")

        self.cache[prompt_hash] = kv_pages
        self.cache_size_mb += estimated_size_mb
        logger.debug(f"Cached prompt: {prompt_hash[:8]}... ({self.cache_size_mb:.1f}MB used)")

    @property
    def hit_rate(self) -> float:
        """Cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cached_prompts": len(self.cache),
            "cache_size_mb": self.cache_size_mb,
            "max_cache_size_mb": self.max_cache_size_mb,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
        }

    def clear(self):
        """Clear all cached prompts"""
        self.cache.clear()
        self.cache_size_mb = 0
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Prompt cache cleared")
