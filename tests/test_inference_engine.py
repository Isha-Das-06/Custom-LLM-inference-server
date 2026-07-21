"""Tests for inference engine"""

import pytest
import torch

from src.core.batch_scheduler import BatchScheduler
from src.core.inference_engine import InferenceEngine
from src.core.request_queue import Request, RequestQueue
from src.memory.kv_cache_manager import KVCacheManager


@pytest.fixture
def inference_components():
    """Fixture: Create inference engine components"""
    request_queue = RequestQueue()

    batch_scheduler = BatchScheduler(
        request_queue=request_queue,
        max_batch_size=4096,
        batch_timeout_ms=100,
    )

    kv_cache = KVCacheManager(
        num_pages=256,
        page_size=16,
        num_heads=32,
        head_dim=128,
        dtype=torch.float16,
        device="cpu",
    )

    return request_queue, batch_scheduler, kv_cache


def test_inference_engine_initialization(inference_components):
    """Test engine initialization"""
    request_queue, batch_scheduler, kv_cache = inference_components

    # Create a mock model executor
    class MockModelExecutor:
        def forward(self, batch):
            return [(0, False, None)] * len(batch)

        def clear_kv_cache(self, request_id):
            pass

    engine = InferenceEngine(
        model_executor=MockModelExecutor(),
        request_queue=request_queue,
        batch_scheduler=batch_scheduler,
        kv_cache_manager=kv_cache,
    )

    assert not engine.is_running
    assert engine.system_metrics is not None
    assert engine.prompt_cache is not None


@pytest.mark.asyncio
async def test_inference_engine_process_batch(inference_components):
    """Test batch processing"""
    request_queue, batch_scheduler, kv_cache = inference_components

    class MockModelExecutor:
        def forward(self, batch):
            return [(1, False, None) for _ in batch]  # Return token 1 for each request

        def clear_kv_cache(self, request_id):
            pass

    engine = InferenceEngine(
        model_executor=MockModelExecutor(),
        request_queue=request_queue,
        batch_scheduler=batch_scheduler,
        kv_cache_manager=kv_cache,
    )

    # Add requests
    req = Request(
        request_id="test_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=5,
    )
    await request_queue.add_request(req)

    # Process batch
    await engine._process_batch()

    # Check that batch was scheduled
    assert len(batch_scheduler.inflight_requests) > 0


@pytest.mark.asyncio
async def test_inference_engine_metrics(inference_components):
    """Test metrics generation"""
    request_queue, batch_scheduler, kv_cache = inference_components

    class MockModelExecutor:
        def forward(self, batch):
            return [(1, False, None) for _ in batch]

        def clear_kv_cache(self, request_id):
            pass

    engine = InferenceEngine(
        model_executor=MockModelExecutor(),
        request_queue=request_queue,
        batch_scheduler=batch_scheduler,
        kv_cache_manager=kv_cache,
    )

    metrics = engine.get_metrics()

    assert "total_requests" in metrics
    assert "total_tokens" in metrics
    assert "gpu_memory_used_gb" in metrics
    assert "active_requests" in metrics
    assert "prompt_cache" in metrics


@pytest.mark.asyncio
async def test_inference_engine_stop(inference_components):
    """Test stopping the engine"""
    request_queue, batch_scheduler, kv_cache = inference_components

    class MockModelExecutor:
        def forward(self, batch):
            return [(1, False, None) for _ in batch]

        def clear_kv_cache(self, request_id):
            pass

    engine = InferenceEngine(
        model_executor=MockModelExecutor(),
        request_queue=request_queue,
        batch_scheduler=batch_scheduler,
        kv_cache_manager=kv_cache,
    )

    engine.is_running = True
    engine.stop()

    assert not engine.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
