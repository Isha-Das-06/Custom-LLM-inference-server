"""Request queue management"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RequestState(Enum):
    """Request lifecycle states"""

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    """A single inference request"""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 1.0
    presence_penalty: float = 1.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    cache_prompt: bool = False
    arrival_time: float = field(default_factory=time.time)

    # State tracking
    state: RequestState = RequestState.WAITING
    generated_ids: list[int] = field(default_factory=list)
    kv_cache_pages: list[int] = field(default_factory=list)
    position: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:8]

    @property
    def is_done(self) -> bool:
        """Check if request is finished"""
        return self.finished or self.state == RequestState.FINISHED

    @property
    def seq_len(self) -> int:
        """Total sequence length (prompt + generated)"""
        return len(self.prompt_ids) + len(self.generated_ids)

    @property
    def remaining_tokens(self) -> int:
        """Tokens still to generate"""
        return max(0, self.max_new_tokens - len(self.generated_ids))

    def step(self, token_id: int, is_eos: bool = False):
        """Advance request by one token"""
        if self.state == RequestState.WAITING:
            self.state = RequestState.PREFILL
        elif self.state == RequestState.PREFILL:
            self.state = RequestState.DECODE
            self.position = len(self.prompt_ids)

        self.generated_ids.append(token_id)
        self.position += 1

        if is_eos or len(self.generated_ids) >= self.max_new_tokens:
            self.finish()

    def finish(self, reason: str = "length"):
        """Mark request as finished"""
        self.finished = True
        self.state = RequestState.FINISHED
        self.finish_reason = reason


class RequestQueue:
    """Manages pending and in-flight requests"""

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.new_requests: asyncio.Queue[Request] = asyncio.Queue()
        self.inflight_requests: dict[str, Request] = {}
        self.finished_requests: dict[str, Request] = {}

    async def add_request(self, request: Request):
        """Add new request to queue"""
        if self.new_requests.qsize() >= self.max_queue_size:
            raise RuntimeError("Request queue full")
        await self.new_requests.put(request)
        logger.debug(f"Request {request.request_id} added to queue")

    async def get_new_request(self, timeout: float = 0.1) -> Optional[Request]:
        """Get next new request (non-blocking with timeout)"""
        try:
            return self.new_requests.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_inflight_request(self, request_id: str) -> Optional[Request]:
        """Get in-flight request by ID"""
        return self.inflight_requests.get(request_id)

    def mark_inflight(self, request: Request):
        """Mark request as in-flight"""
        self.inflight_requests[request.request_id] = request
        logger.debug(f"Request {request.request_id} marked as inflight")

    def mark_finished(self, request: Request):
        """Mark request as finished"""
        request.finish()
        self.inflight_requests.pop(request.request_id, None)
        self.finished_requests[request.request_id] = request
        logger.debug(f"Request {request.request_id} finished")

    def get_finished_request(self, request_id: str) -> Optional[Request]:
        """Get finished request"""
        return self.finished_requests.get(request_id)

    async def wait_for_request(self, request_id: str, timeout: float = 60.0) -> Optional[Request]:
        """Wait for request to complete"""
        start = time.time()
        while time.time() - start < timeout:
            if request_id in self.finished_requests:
                return self.finished_requests[request_id]
            await asyncio.sleep(0.01)
        return None

    def get_queue_stats(self) -> dict:
        """Get queue statistics"""
        return {
            "new_requests": self.new_requests.qsize(),
            "inflight_requests": len(self.inflight_requests),
            "finished_requests": len(self.finished_requests),
        }
