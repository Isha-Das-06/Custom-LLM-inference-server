"""Pytest configuration and fixtures"""

import pytest
import torch

from src.core.request_queue import RequestQueue
from src.memory.kv_cache_manager import KVCacheManager


@pytest.fixture
def request_queue():
    """Fixture: RequestQueue"""
    return RequestQueue()


@pytest.fixture
def kv_cache_manager():
    """Fixture: KVCacheManager"""
    return KVCacheManager(
        num_pages=256,
        page_size=16,
        num_heads=32,
        head_dim=128,
        dtype=torch.float16,
        device="cpu",
    )
