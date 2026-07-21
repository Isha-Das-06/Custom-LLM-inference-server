"""Tests for batch scheduler"""

import pytest

from src.core.batch_scheduler import BatchScheduler
from src.core.request_queue import Request, RequestQueue


@pytest.fixture
def request_queue():
    return RequestQueue()


@pytest.fixture
def scheduler(request_queue):
    return BatchScheduler(
        request_queue=request_queue,
        max_batch_size=4096,
        batch_timeout_ms=100,
    )


@pytest.mark.asyncio
async def test_batch_scheduler_empty(scheduler):
    """Test scheduler with no requests"""
    batch = await scheduler.schedule_batch()
    assert batch == []


@pytest.mark.asyncio
async def test_batch_scheduler_single_request(scheduler, request_queue):
    """Test scheduler with single request"""
    req = Request(
        request_id="req_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )
    await request_queue.add_request(req)

    batch = await scheduler.schedule_batch()
    assert len(batch) == 1
    assert batch[0].request_id == "req_1"


@pytest.mark.asyncio
async def test_batch_scheduler_respects_max_size(scheduler, request_queue):
    """Test scheduler respects max_batch_size"""
    # Add requests with large prompts
    for i in range(10):
        req = Request(
            request_id=f"req_{i}",
            prompt_ids=list(range(500)),  # 500 tokens each
            max_new_tokens=10,
        )
        await request_queue.add_request(req)

    batch = await scheduler.schedule_batch()
    # With max_batch_size=4096, should include 8 requests (8*500=4000)
    total_tokens = sum(len(r.prompt_ids) for r in batch)
    assert total_tokens <= 4096


@pytest.mark.asyncio
async def test_batch_scheduler_in_flight_continuity(scheduler, request_queue):
    """Test in-flight requests stay in batch"""
    # Add first request
    req1 = Request(
        request_id="req_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )
    await request_queue.add_request(req1)

    # Schedule batch (req1 should be inflight now)
    batch1 = await scheduler.schedule_batch()
    assert len(batch1) == 1

    # Add second request
    req2 = Request(
        request_id="req_2",
        prompt_ids=[4, 5, 6],
        max_new_tokens=10,
    )
    await request_queue.add_request(req2)

    # Schedule next batch - req1 should still be included
    batch2 = await scheduler.schedule_batch()
    assert any(r.request_id == "req_1" for r in batch2)


@pytest.mark.asyncio
async def test_batch_scheduler_fairness(scheduler, request_queue):
    """Test fairness timeout"""
    # Add request with large prompt
    req_large = Request(
        request_id="req_large",
        prompt_ids=list(range(4000)),
        max_new_tokens=10,
    )
    await request_queue.add_request(req_large)

    # Add small request that shouldn't fit in first batch
    req_small = Request(
        request_id="req_small",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )
    await request_queue.add_request(req_small)

    # First batch should only have req_large
    batch1 = await scheduler.schedule_batch()
    assert len(batch1) == 1
    assert batch1[0].request_id == "req_large"

    # Second batch should have req_small (or wait for timeout)
    batch2 = await scheduler.schedule_batch()
    # req_small should eventually be included or in inflight
    assert (
        any(r.request_id == "req_small" for r in batch2)
        or "req_small" in scheduler.inflight_requests
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
