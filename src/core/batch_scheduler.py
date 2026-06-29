"""Continuous batch scheduling algorithm"""

import logging
import time

from src.core.request_queue import Request, RequestQueue

logger = logging.getLogger(__name__)


class BatchScheduler:
    """
    Continuous batching scheduler.

    Key algorithm:
    1. Maintain in-flight requests (already have KV-cache)
    2. Always include in-flight in next batch (avoid context switch)
    3. Add new requests until max_batch_size reached
    4. Flush batch if timeout exceeded (fairness)
    """

    def __init__(
        self,
        request_queue: RequestQueue,
        max_batch_size: int = 4096,
        batch_timeout_ms: int = 100,
    ):
        self.request_queue = request_queue
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms / 1000.0  # Convert to seconds
        self.inflight_requests: set[str] = set()
        self.batch_counter = 0

    async def schedule_batch(self) -> list[Request]:
        """
        Select requests for next GPU batch.

        Returns:
            List of requests to process in this iteration
        """
        self.batch_counter += 1
        batch: list[Request] = []
        current_tokens = 0

        # Step 1: Include all in-flight requests
        # These already have allocated KV-cache, so keep processing them
        for req_id in list(self.inflight_requests):
            req = self.request_queue.get_inflight_request(req_id)
            if req and not req.is_done:
                batch.append(req)
                # Decode phase: 1 token per sequence
                current_tokens += 1
            else:
                # Request finished, remove from inflight
                if req and req.is_done:
                    self.inflight_requests.discard(req_id)

        # Step 2: Add new requests
        # Prefill phase generates multiple tokens at once (sequence length)
        while True:
            if current_tokens >= self.max_batch_size:
                break

            new_req = await self.request_queue.get_new_request(timeout=0.001)
            if not new_req:
                break

            # Check if adding this request exceeds budget
            tokens_needed = len(new_req.prompt_ids)
            if current_tokens + tokens_needed > self.max_batch_size:
                # Put back and break (would exceed budget)
                await self.request_queue.new_requests.put(new_req)
                break

            batch.append(new_req)
            self.request_queue.mark_inflight(new_req)
            self.inflight_requests.add(new_req.request_id)
            current_tokens += tokens_needed

        # Step 3: Timeout check (fairness - prevent starvation)
        # If new requests are waiting too long, flush batch to process them
        if not await self._check_new_requests_timeout():
            # No timeout issue, batch is ready
            pass

        # Log batch formation
        logger.debug(
            f"Batch {self.batch_counter}: {len(batch)} requests, "
            f"{current_tokens} tokens, inflight: {len(self.inflight_requests)}"
        )

        return batch

    async def _check_new_requests_timeout(self) -> bool:
        """Check if oldest new request exceeded timeout"""
        if self.request_queue.new_requests.empty():
            return False

        # Peek at oldest request (without removing)
        queue_copy = list(self.request_queue.new_requests._queue)
        if not queue_copy:
            return False

        oldest_req = queue_copy[0]
        wait_time = time.time() - oldest_req.arrival_time

        if wait_time > self.batch_timeout_ms:
            logger.warning(
                f"Request {oldest_req.request_id} waiting {wait_time:.2f}s " "(exceeds timeout)"
            )
            return True

        return False

    def mark_request_finished(self, request_id: str):
        """Mark request as finished and remove from inflight"""
        self.inflight_requests.discard(request_id)
        req = self.request_queue.get_inflight_request(request_id)
        if req:
            self.request_queue.mark_finished(req)

    def get_scheduler_stats(self) -> dict:
        """Get scheduler statistics"""
        return {
            "batch_counter": self.batch_counter,
            "inflight_requests": len(self.inflight_requests),
            "new_requests_queued": self.request_queue.new_requests.qsize(),
        }
