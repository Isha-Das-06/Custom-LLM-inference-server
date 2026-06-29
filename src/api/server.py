"""FastAPI server application"""

import asyncio
import logging

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer

from src.api.routes import APIRoutes
from src.config import ServerConfig
from src.core.batch_scheduler import BatchScheduler
from src.core.inference_engine import InferenceEngine
from src.core.model_executor import ModelExecutor
from src.core.request_queue import RequestQueue
from src.memory.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


def create_app(config: ServerConfig) -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="Custom LLM Inference Server",
        description="High-performance LLM inference with continuous batching and KV-cache optimization",
        version="0.1.0",
    )

    # Initialize components
    logger.info("Initializing server components...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model executor
    model_executor = ModelExecutor(
        model_path=config.model_path,
        dtype=config.dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Initialize KV-cache manager
    num_heads = model_executor.model.config.num_attention_heads
    head_dim = model_executor.model.config.hidden_size // num_heads

    kv_cache_manager = KVCacheManager(
        num_pages=4096,
        page_size=config.page_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=config.get_torch_dtype(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Initialize request queue and scheduler
    request_queue = RequestQueue()
    batch_scheduler = BatchScheduler(
        request_queue=request_queue,
        max_batch_size=config.max_batch_size,
        batch_timeout_ms=config.batch_timeout_ms,
    )

    # Initialize inference engine
    inference_engine = InferenceEngine(
        model_executor=model_executor,
        request_queue=request_queue,
        batch_scheduler=batch_scheduler,
        kv_cache_manager=kv_cache_manager,
        enable_prompt_cache=config.enable_prompt_cache,
        prompt_cache_size_mb=config.prompt_cache_size_mb,
    )

    # Initialize API routes
    api_routes = APIRoutes(
        model_executor=model_executor,
        request_queue=request_queue,
        batch_scheduler=batch_scheduler,
        kv_cache_manager=kv_cache_manager,
        tokenizer=tokenizer,
        inference_engine=inference_engine,
    )

    # Include routes
    app.include_router(api_routes.router)

    # Inference engine task
    inference_task = None

    @app.on_event("startup")
    async def startup():
        nonlocal inference_task
        logger.info("Server startup - starting inference engine")
        # Start inference engine as background task
        inference_task = asyncio.create_task(inference_engine.run(poll_interval_ms=10))

    @app.on_event("shutdown")
    async def shutdown():
        nonlocal inference_task
        logger.info("Server shutdown - stopping inference engine")
        inference_engine.stop()
        if inference_task:
            try:
                await asyncio.wait_for(inference_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Inference task did not stop gracefully")

    # Store components in app state for access in routes
    app.state.model_executor = model_executor
    app.state.request_queue = request_queue
    app.state.batch_scheduler = batch_scheduler
    app.state.kv_cache_manager = kv_cache_manager
    app.state.tokenizer = tokenizer
    app.state.inference_engine = inference_engine

    return app
