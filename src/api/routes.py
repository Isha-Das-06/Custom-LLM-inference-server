"""API routes"""

import asyncio
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

from src.api.models import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)
from src.core.inference_engine import InferenceEngine
from src.core.model_executor import ModelExecutor
from src.core.request_queue import Request, RequestQueue
from src.memory.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


class APIRoutes:
    """API route handlers"""

    def __init__(
        self,
        model_executor: ModelExecutor,
        request_queue: RequestQueue,
        batch_scheduler,
        kv_cache_manager: KVCacheManager,
        tokenizer: AutoTokenizer,
        inference_engine: InferenceEngine,
    ):
        self.model_executor = model_executor
        self.request_queue = request_queue
        self.kv_cache_manager = kv_cache_manager
        self.tokenizer = tokenizer
        self.inference_engine = inference_engine
        self.router = APIRouter()
        self.start_time = time.time()
        self.request_counter = 0

        self._setup_routes()

    def _setup_routes(self):
        """Setup all routes"""
        self.router.post("/generate")(self.generate)
        self.router.post("/batch-generate")(self.batch_generate)
        self.router.get("/health")(self.health)
        self.router.get("/metrics")(self.metrics)

    async def generate(self, req_data: GenerateRequest) -> GenerateResponse:
        """Handle single generation request"""
        start_time = time.time()

        try:
            # Tokenize prompt
            prompt_ids = self.tokenizer.encode(req_data.prompt)

            # Create request
            request_id = req_data.request_id or f"req_{self.request_counter:06d}"
            self.request_counter += 1

            request = Request(
                request_id=request_id,
                prompt_ids=prompt_ids,
                max_new_tokens=req_data.max_new_tokens,
                temperature=req_data.temperature,
                top_p=req_data.top_p,
                top_k=req_data.top_k,
                frequency_penalty=req_data.frequency_penalty,
                presence_penalty=req_data.presence_penalty,
                repetition_penalty=req_data.repetition_penalty,
                seed=req_data.seed,
                cache_prompt=req_data.cache_prompt,
            )

            # Queue request
            await self.request_queue.add_request(request)

            # Handle streaming or blocking wait
            if req_data.stream:
                return StreamingResponse(
                    self._stream_tokens(request_id, req_data.stream_chunk_size),
                    media_type="application/x-ndjson",
                )
            else:
                # Wait for completion (with timeout)
                finished_req = await self.request_queue.wait_for_request(request_id, timeout=300.0)
                if not finished_req:
                    raise RuntimeError("Request timeout")

                # Decode generated tokens
                generated_text = self.tokenizer.decode(finished_req.generated_ids)

                latency_ms = (time.time() - start_time) * 1000

                return GenerateResponse(
                    request_id=request_id,
                    prompt=req_data.prompt,
                    generated_text=generated_text,
                    finish_reason=finished_req.finish_reason or "length",
                    num_prompt_tokens=len(prompt_ids),
                    num_generated_tokens=len(finished_req.generated_ids),
                    total_tokens=finished_req.seq_len,
                    latency_ms=latency_ms,
                    tokens_per_sec=(
                        len(finished_req.generated_ids) * 1000 / latency_ms if latency_ms > 0 else 0
                    ),
                )

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_tokens(self, request_id: str, chunk_size: int) -> AsyncGenerator[str, None]:
        """Stream tokens for a request"""
        last_count = 0
        timeout = time.time() + 300

        while time.time() < timeout:
            req = self.request_queue.get_inflight_request(request_id)
            if req is None:
                # Check finished
                req = self.request_queue.get_finished_request(request_id)
                if req:
                    # Stream remaining tokens
                    if last_count < len(req.generated_ids):
                        chunk = req.generated_ids[last_count:]
                        text = self.tokenizer.decode(chunk)
                        yield f'data: {{"token": "{text}", "finish_reason": "{req.finish_reason}"}}\n'
                    break
            else:
                # Stream new tokens
                if len(req.generated_ids) >= last_count + chunk_size:
                    chunk = req.generated_ids[last_count : last_count + chunk_size]
                    text = self.tokenizer.decode(chunk)
                    yield f'data: {{"token": "{text}", "cumulative_tokens": {len(req.generated_ids)}}}\n'
                    last_count += chunk_size

            await asyncio.sleep(0.01)

    async def batch_generate(self, req_data: BatchGenerateRequest) -> BatchGenerateResponse:
        """Handle batch generation request"""
        results = []

        for req in req_data.requests:
            result = await self.generate(req)
            results.append(result)

        return BatchGenerateResponse(results=results)

    async def health(self) -> HealthResponse:
        """Health check endpoint"""
        return HealthResponse(
            status="healthy" if self.inference_engine.is_running else "degraded",
            uptime_seconds=time.time() - self.start_time,
        )

    async def metrics(self) -> MetricsResponse:
        """Metrics endpoint"""
        engine_metrics = self.inference_engine.get_metrics()

        return MetricsResponse(
            gpu_memory_used_gb=engine_metrics.get("gpu_memory_used_gb", 0.0),
            gpu_memory_total_gb=engine_metrics.get("gpu_memory_total_gb", 0.0),
            active_requests=engine_metrics.get("active_requests", 0),
            queued_requests=engine_metrics.get("queued_requests", 0),
            total_requests_processed=engine_metrics.get("total_requests", 0),
            avg_throughput_tokens_per_sec=engine_metrics.get("avg_throughput_tokens_per_sec", 0.0),
            avg_latency_ms=engine_metrics.get("avg_latency_ms", 0.0),
            cache_hit_rate=engine_metrics.get("hit_rate", 0.0),
            kv_cache_pages_used=engine_metrics.get("num_sequences", 0),
            kv_cache_pages_total=engine_metrics.get("total_pages", 0),
        )
