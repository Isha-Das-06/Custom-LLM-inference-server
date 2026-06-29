"""Tests for KV-cache manager"""

import pytest
import torch

from src.memory.kv_cache_manager import KVCacheManager


@pytest.fixture
def kv_cache():
    return KVCacheManager(
        num_pages=256,
        page_size=16,
        num_heads=32,
        head_dim=128,
        dtype=torch.float16,
        device="cpu",  # Use CPU for tests
    )


def test_kv_cache_allocate_pages(kv_cache):
    """Test page allocation"""
    pages = kv_cache.allocate_pages("seq_1", 4)
    assert len(pages) == 4
    assert len(kv_cache.free_pages) == 256 - 4


def test_kv_cache_write_read(kv_cache):
    """Test write and read KV"""
    # Allocate pages
    kv_cache.allocate_pages("seq_1", 2)

    # Write KV
    k_tokens = torch.randn(1, 16, 32, 128, dtype=torch.float16)
    v_tokens = torch.randn(1, 16, 32, 128, dtype=torch.float16)
    kv_cache.write_kv("seq_1", k_tokens, v_tokens, position=0)

    # Read KV
    k_read, v_read = kv_cache.read_kv("seq_1", length=16)
    assert k_read.shape == (16, 32, 128)
    assert v_read.shape == (16, 32, 128)


def test_kv_cache_free_pages(kv_cache):
    """Test page freeing"""
    kv_cache.allocate_pages("seq_1", 4)
    kv_cache.allocate_pages("seq_2", 3)

    initial_free = len(kv_cache.free_pages)
    kv_cache.free_pages("seq_1")
    final_free = len(kv_cache.free_pages)

    assert final_free == initial_free + 4


def test_kv_cache_stats(kv_cache):
    """Test cache statistics"""
    kv_cache.allocate_pages("seq_1", 4)
    kv_cache.allocate_pages("seq_2", 2)

    stats = kv_cache.get_cache_stats()
    assert stats["num_sequences"] == 2
    assert stats["used_pages"] == 6
    assert stats["free_pages"] == 256 - 6


def test_kv_cache_out_of_memory(kv_cache):
    """Test behavior when out of memory"""
    # Allocate all pages
    for i in range(256 // 10):
        kv_cache.allocate_pages(f"seq_{i}", 10)

    # Try to allocate more (should trigger eviction)
    # This shouldn't crash, just log a warning
    pages = kv_cache.allocate_pages("seq_new", 5)
    assert len(pages) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
