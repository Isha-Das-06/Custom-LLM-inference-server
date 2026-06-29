"""Main inference engine - orchestrates batching, execution, and caching"""

import asyncio
import logging
import time
from typing import Optional

from src.core.batch_scheduler import BatchScheduler
from src.core.model_executor import ModelExecutor
from src.core.request_queue import Request, RequestQueue
from src.memory.kv_cache_manager import KVCacheManager
from src.memory.prompt_cache import PromptCache
from src.utils.metrics import RequestMetrics, SystemMetrics

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main inference engine that orchestrates:
    - Batch scheduling
    - Model execution
    - KV-cache management
    - Prompt caching
    - Request completion
    """

    def __init__(
        self,
        model_executor: ModelExecutor,
        request_queue: RequestQueue,
        batch_scheduler: BatchScheduler,
        kv_cache_manager: KVCacheManager,
        enable_prompt_cache: bool = True,
        prompt_cache_size_mb: int = 1000,
    ):
        self.model_executor = model_executor
        self.request_queue = request_queue
        self.batch_scheduler = batch_scheduler
        self.kv_cache_manager = kv_cache_manager

        # Prompt caching
        self.enable_prompt_cache = enable_prompt_cache
        self.prompt_cache = PromptCache(max_cache_size_mb=prompt_cache_size_mb)

        # Metrics
        self.system_metrics = SystemMetrics()
        self.is_running = False

    async def run(self, poll_interval_ms: float = 10):
        """
        Main inference loop.

        Continuously:
        1. Schedule batch of requests
        2. Execute model
        3. Update KV-cache
        4. Check for completion
        """
        self.is_running = True
        logger.info("Starting inference engine")

        try:
            while self.is_running:
                await self._process_batch()
                await asyncio.sleep(poll_interval_ms / 1000.0)
        except Exception as e:
            logger.error(f"Error in inference loop: {e}")
            self.is_running = False
            raise

    async def _process_batch(self):
        """Process one batch of requests"""
        # Schedule batch
        batch = await self.batch_scheduler.schedule_batch()
        if not batch:
            return

        start_time = time.time()

        try:
            # Forward pass
            results = self.model_executor.forward(batch)

            # Update requests and cache
            for i, (request, result) in enumerate(zip(batch, results)):
                if len(result) == 3:
                    token_id, is_eos, past_kv = result
                else:
                    # Backward compatibility
                    token_id, is_eos = result
                    past_kv = None

                # Write to KV-cache manager
                self._update_kv_cache(request, past_kv)

                # Check if finished
                if request.is_done:
                    self.batch_scheduler.mark_request_finished(request.request_id)
                    await self._complete_request(request)

            # Record metrics
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Batch {len(batch)} requests processed in {elapsed_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Mark requests as failed
            for request in batch:
                request.finish(reason="error")
                self.batch_scheduler.mark_request_finished(request.request_id)

    def _update_kv_cache(self, request: Request, past_key_values: Optional[tuple] = None):
        """
        Update KV-cache for a request after token generation.

        The KV cache is maintained in the ModelExecutor's buffer (past_key_values).
        This method can optionally write to the paged KV-cache manager for long-running
        sequences that need memory optimization.
        """
        if not request.generated_ids:
            return

        # For now, the primary KV cache is maintained in ModelExecutor's _kv_cache_buffer
        # This allows efficient reuse across decode steps without paging overhead.

        # In production with many long-running sequences, you would:
        # 1. Write new KV tokens to the paged cache manager periodically
        # 2. Evict old tokens from ModelExecutor buffer to free memory
        # 3. Reconstruct past_key_values from paged storage on page hits

        # Track that cache exists for metrics
        if past_key_values is not None:
            request.kv_cache_pages = [1]  # Mark as cached (simplified)

    async def _complete_request(self, request: Request):
        """Mark request as complete and store result"""
        elapsed_ms = (time.time() - request.arrival_time) * 1000

        # Record metrics
        metrics = RequestMetrics(
            request_id=request.request_id,
            prompt_tokens=len(request.prompt_ids),
            generated_tokens=len(request.generated_ids),
            total_latency_ms=elapsed_ms,
            cache_hit=False,  # TODO: track from prompt_cache
        )
        self.system_metrics.update_request(metrics)

        # Clean up KV cache
        self.model_executor.clear_kv_cache(request.request_id)
        if request.kv_cache_pages:
            self.kv_cache_manager.free_pages(request.request_id)

        # Store finished request
        self.request_queue.mark_finished(request)

        logger.info(
            f"Request {request.request_id} completed: "
            f"{len(request.generated_ids)} tokens in {elapsed_ms:.1f}ms"
        )

    def stop(self):
        """Stop the inference engine"""
        self.is_running = False
        logger.info("Inference engine stopped")

    def get_metrics(self) -> dict:
        """Get system metrics"""
        from src.utils.gpu_utils import GPUMemoryMonitor

        gpu_monitor = GPUMemoryMonitor()
        gpu_info = gpu_monitor.get_memory_info()

        cache_stats = self.kv_cache_manager.get_cache_stats()
        queue_stats = self.request_queue.get_queue_stats()
        system_stats = self.system_metrics.get_stats_dict()

        return {
            **system_stats,
            "gpu_memory_allocated_mb": gpu_info.get("allocated_mb", 0.0),
            "gpu_memory_total_mb": gpu_info.get("total_mb", 0.0),
            "gpu_memory_used_gb": gpu_info.get("allocated_mb", 0.0) / 1024,
            "gpu_memory_total_gb": gpu_info.get("total_mb", 0.0) / 1024,
            "gpu_percent_used": gpu_info.get("percent_used", 0.0),
            "active_requests": len(self.request_queue.inflight_requests),
            "queued_requests": queue_stats["new_requests"],
            **cache_stats,
            "prompt_cache": self.prompt_cache.get_stats(),
        }
