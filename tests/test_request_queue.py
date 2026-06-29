"""Tests for request queue"""

import asyncio
import pytest

from src.core.request_queue import Request, RequestQueue, RequestState


@pytest.mark.asyncio
async def test_request_queue_add_and_get():
    """Test adding and getting requests"""
    queue = RequestQueue()

    req = Request(
        request_id="test_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )

    await queue.add_request(req)
    assert queue.new_requests.qsize() == 1

    retrieved = await queue.get_new_request()
    assert retrieved.request_id == "test_1"


@pytest.mark.asyncio
async def test_request_queue_mark_inflight():
    """Test marking request as inflight"""
    queue = RequestQueue()

    req = Request(
        request_id="test_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )

    queue.mark_inflight(req)
    assert len(queue.inflight_requests) == 1
    assert queue.get_inflight_request("test_1") == req


@pytest.mark.asyncio
async def test_request_queue_mark_finished():
    """Test marking request as finished"""
    queue = RequestQueue()

    req = Request(
        request_id="test_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )

    queue.mark_inflight(req)
    queue.mark_finished(req)

    assert len(queue.inflight_requests) == 0
    assert req.is_done is True
    assert "test_1" in queue.finished_requests


@pytest.mark.asyncio
async def test_request_state_progression():
    """Test request state progression"""
    req = Request(
        request_id="test_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=5,
    )

    assert req.state == RequestState.WAITING

    # Step to prefill
    req.step(10)
    assert req.state == RequestState.DECODE
    assert len(req.generated_ids) == 1

    # Step through decode
    req.step(11)
    req.step(12)
    req.step(13)
    req.step(14)

    assert len(req.generated_ids) == 5
    assert req.is_done is True


@pytest.mark.asyncio
async def test_request_early_finish():
    """Test early finish (EOS)"""
    req = Request(
        request_id="test_1",
        prompt_ids=[1, 2, 3],
        max_new_tokens=100,
    )

    req.step(10)  # First token
    req.step(11, is_eos=True)  # EOS token

    assert req.is_done is True
    assert len(req.generated_ids) == 2


@pytest.mark.asyncio
async def test_request_queue_get_stats():
    """Test queue statistics"""
    queue = RequestQueue()

    req1 = Request(request_id="req_1", prompt_ids=[1, 2], max_new_tokens=10)
    req2 = Request(request_id="req_2", prompt_ids=[3, 4], max_new_tokens=10)

    await queue.add_request(req1)
    queue.mark_inflight(req2)

    stats = queue.get_queue_stats()
    assert stats["new_requests"] == 1
    assert stats["inflight_requests"] == 1
    assert stats["finished_requests"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
