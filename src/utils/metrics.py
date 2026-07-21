"""Metrics and monitoring utilities"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""

    request_id: str
    prompt_tokens: int
    generated_tokens: int
    total_latency_ms: float
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    cache_hit: bool = False

    @property
    def tokens_per_sec(self) -> float:
        """Generate speed"""
        if self.total_latency_ms <= 0:
            return 0.0
        return self.generated_tokens * 1000 / self.total_latency_ms

    @property
    def total_tokens(self) -> int:
        """Total tokens processed"""
        return self.prompt_tokens + self.generated_tokens


@dataclass
class SystemMetrics:
    """System-wide metrics"""

    total_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(default_factory=time.time)
    request_times: list[float] = field(default_factory=list)

    @property
    def uptime_seconds(self) -> float:
        """Server uptime"""
        return time.time() - self.start_time

    @property
    def avg_throughput(self) -> float:
        """Average tokens per second"""
        if self.total_latency_ms <= 0:
            return 0.0
        return self.total_tokens * 1000 / self.total_latency_ms

    @property
    def avg_latency(self) -> float:
        """Average request latency"""
        if self.total_requests <= 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def cache_hit_rate(self) -> float:
        """Prompt cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total <= 0:
            return 0.0
        return self.cache_hits / total

    def update_request(self, metrics: RequestMetrics):
        """Update with request metrics"""
        self.total_requests += 1
        self.total_tokens += metrics.total_tokens
        self.total_latency_ms += metrics.total_latency_ms
        self.request_times.append(metrics.total_latency_ms)

        if metrics.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_stats_dict(self) -> dict:
        """Get metrics as dictionary"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "uptime_seconds": self.uptime_seconds,
            "avg_throughput_tokens_per_sec": self.avg_throughput,
            "avg_latency_ms": self.avg_latency,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
