"""Pydantic request/response models"""

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request for single inference"""

    prompt: str = Field(..., description="Input text")
    max_new_tokens: int = Field(256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(50, ge=0, description="Top-K sampling")
    frequency_penalty: float = Field(1.0, ge=0.0, description="Frequency penalty")
    presence_penalty: float = Field(1.0, ge=0.0, description="Presence penalty")
    repetition_penalty: float = Field(1.0, ge=0.0, description="Repetition penalty")
    stream: bool = Field(False, description="Stream tokens")
    stream_chunk_size: int = Field(10, ge=1, description="Tokens per chunk (streaming)")
    cache_prompt: bool = Field(False, description="Enable prefix caching")
    seed: Optional[int] = Field(None, description="Random seed")
    request_id: Optional[str] = Field(None, description="Custom request ID")


class GenerateResponse(BaseModel):
    """Response from inference"""

    request_id: str = Field(..., description="Request ID")
    prompt: str = Field(..., description="Input prompt")
    generated_text: str = Field(..., description="Generated text")
    finish_reason: str = Field(..., description="Why generation stopped")
    num_prompt_tokens: int = Field(..., description="Tokens in prompt")
    num_generated_tokens: int = Field(..., description="Tokens generated")
    total_tokens: int = Field(..., description="Total tokens")
    latency_ms: float = Field(..., description="Total latency (ms)")
    tokens_per_sec: float = Field(..., description="Generation speed")


class BatchGenerateRequest(BaseModel):
    """Request for batch inference"""

    requests: list[GenerateRequest] = Field(..., description="List of requests")


class BatchGenerateResponse(BaseModel):
    """Response from batch inference"""

    results: list[GenerateResponse] = Field(..., description="Results for each request")


class HealthResponse(BaseModel):
    """Server health check response"""

    status: str = Field(..., description="Server status")
    uptime_seconds: float = Field(..., description="Server uptime")


class MetricsResponse(BaseModel):
    """Server metrics response"""

    status: str = Field("healthy", description="Server status")
    gpu_memory_used_gb: float = Field(..., description="GPU memory used")
    gpu_memory_total_gb: float = Field(..., description="Total GPU memory")
    active_requests: int = Field(..., description="Currently processing")
    queued_requests: int = Field(..., description="Waiting in queue")
    total_requests_processed: int = Field(..., description="Lifetime requests")
    avg_throughput_tokens_per_sec: float = Field(..., description="Average throughput")
    avg_latency_ms: float = Field(..., description="Average latency")
    cache_hit_rate: float = Field(..., description="Prompt cache hit rate")
    kv_cache_pages_used: int = Field(..., description="KV-cache pages in use")
    kv_cache_pages_total: int = Field(..., description="Total KV-cache pages")
