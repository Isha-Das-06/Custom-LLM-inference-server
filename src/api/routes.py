"""API routes"""

import logging
import time
from typing import AsyncGenerator

import torch
from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer

from src.api.models import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)
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
        kv_cache_manager: KVCacheManager,
        tokenizer: AutoTokenizer,
    ):
        self.model_executor = model_executor
        self.request_queue = request_queue
        self.kv_cache_manager = kv_cache_manager
        self.tokenizer = tokenizer
        self.router = APIRouter()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency_ms = 0

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
            request = Request(
                request_id=req_data.request_id or f"req_{self.total_requests:06d}",
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

            # Wait for completion (simplified for demo)
            finished_req = await self.request_queue.wait_for_request(request.request_id)
            if not finished_req:
                raise RuntimeError("Request timeout")

            # Decode generated tokens
            generated_text = self.tokenizer.decode(finished_req.generated_ids)

            latency_ms = (time.time() - start_time) * 1000
            self.total_requests += 1
            self.total_tokens += len(finished_req.generated_ids)
            self.total_latency_ms += latency_ms

            return GenerateResponse(
                request_id=request.request_id,
                prompt=req_data.prompt,
                generated_text=generated_text,
                finish_reason=finished_req.finish_reason or "length",
                num_prompt_tokens=len(prompt_ids),
                num_generated_tokens=len(finished_req.generated_ids),
                total_tokens=finished_req.seq_len,
                latency_ms=latency_ms,
                tokens_per_sec=len(finished_req.generated_ids) * 1000 / latency_ms if latency_ms > 0 else 0,
            )

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def batch_generate(
        self, req_data: BatchGenerateRequest
    ) -> BatchGenerateResponse:
        """Handle batch generation request"""
        results = []

        for req in req_data.requests:
            result = await self.generate(req)
            results.append(result)

        return BatchGenerateResponse(results=results)

    async def health(self) -> HealthResponse:
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            uptime_seconds=time.time() - self.start_time,
        )

    async def metrics(self) -> MetricsResponse:
        """Metrics endpoint"""
        queue_stats = self.request_queue.get_queue_stats()
        cache_stats = self.kv_cache_manager.get_cache_stats()

        try:
            # Try to get GPU memory (requires GPU)
            gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            gpu_memory_used_gb = 0.0
            gpu_memory_total_gb = 0.0

        avg_throughput = (
            self.total_tokens / self.total_latency_ms * 1000
            if self.total_latency_ms > 0
            else 0
        )
        avg_latency = (
            self.total_latency_ms / self.total_requests
            if self.total_requests > 0
            else 0
        )

        return MetricsResponse(
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            active_requests=len(self.request_queue.inflight_requests),
            queued_requests=queue_stats["new_requests"],
            total_requests_processed=self.total_requests,
            avg_throughput_tokens_per_sec=avg_throughput,
            avg_latency_ms=avg_latency,
            cache_hit_rate=0.0,  # TODO: integrate prompt cache
            kv_cache_pages_used=cache_stats["num_sequences"],
            kv_cache_pages_total=cache_stats["total_pages"],
        )
