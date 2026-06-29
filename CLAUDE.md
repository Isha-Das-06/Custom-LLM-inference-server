# Custom LLM Inference Server - Developer Guide

## Project Context

This is a **production-grade LLM inference engine** demonstrating mastery of modern serving techniques:
- Continuous batching (vs. static batching)
- Paged attention + KV-cache optimization
- Performance competitive with vLLM
- Built from first principles (not a wrapper around vLLM)

**Target audience:** Senior engineers, hiring managers who want to see deep understanding of ML infrastructure.

---

## Architecture Overview

### Request Flow

```
Client Request
    ↓
[Request Validator] → Validate prompt, params, auth
    ↓
[Request Scheduler] → Add to queue, assign to batch
    ↓
[Batch Coordinator] → Decide: continue batch or flush?
    ↓
[Model Executor] → Forward pass (prefill or decode)
    ↓
[Token Sampler] → Sample next token (temperature, top-p, top-k)
    ↓
[KV-Cache Manager] → Store/update cached K, V tensors
    ↓
[Response Handler] → Stream or collect tokens
    ↓
Client Response
```

### Key Components

| Component | Responsibility | Complexity |
|---|---|---|
| `RequestQueue` | FIFO queue, request validation | Low |
| `BatchScheduler` | Select requests for next iteration | Medium |
| `KVCacheManager` | Paged memory allocation, prefix caching | High |
| `ModelExecutor` | Forward pass, attention computation | Medium |
| `TokenSampler` | Temperature, top-p, top-k sampling | Low |
| `PromptCache` | Hash-based prompt caching for prefix sharing | High |

---

## Core Algorithms

### 1. Continuous Batching Scheduler

**Problem:** Static batching requires all sequences same length.

**Solution:** Dynamic micro-batching that changes size each iteration.

**Implementation:**
- Maintain two queues: `new_requests` (not yet scheduled) and `inflight_requests` (in progress)
- Each iteration:
  1. Keep all inflight requests (save context switch)
  2. Add new requests until batch is full
  3. If wait time exceeds threshold, flush batch (fairness)

**Pseudo-code:**
```python
def schedule_batch(max_batch_tokens, timeout_ms):
    batch = []
    
    # Include all in-flight requests
    batch.extend(inflight_requests)
    current_tokens = sum(r.tokens_needed for r in batch)
    
    # Add new requests
    while (new_requests and 
           current_tokens < max_batch_tokens):
        req = new_requests.pop_oldest()
        if req.tokens_needed <= (max_batch_tokens - current_tokens):
            batch.append(req)
            current_tokens += req.tokens_needed
            inflight_requests.add(req)
    
    # Timeout check (prevent starvation)
    if new_requests and (time.now() - new_requests[0].arrival_time) > timeout_ms:
        batch.append(new_requests.pop_oldest())
    
    return batch
```

**Tradeoffs:**
- ✅ High throughput (no padding waste)
- ✅ Fair (timeout ensures no starvation)
- ✅ Simple implementation
- ❌ Variable batch size (less predictable GPU utilization)

### 2. Paged KV-Cache Attention

**Problem:** KV-cache is O(seq_len) memory per sequence; difficult to share across requests.

**Solution:** Allocate cache in fixed-size pages; sequences reference page lists.

**Design:**
```
KV Pages (fixed-size, e.g., 16 tokens per page):
┌─────────────┐
│ Page 0 (16) │
├─────────────┤
│ Page 1 (16) │
├─────────────┤
│ Page 2 (16) │
├─────────────┤
│ Page 3 (16) │
└─────────────┘

Sequence references:
Seq A: [page 0, page 1, page 2] (48 tokens cached)
Seq B: [page 3, page 4]         (32 tokens cached)
Seq C: [page 0, page 1, page 5] (48 tokens cached) ← reuses pages with Seq A
```

**Advantages:**
- Pages can be shared (common prompt prefix)
- Fine-grained eviction (per-page, not per-sequence)
- Memory pooling reduces fragmentation
- CPU swapping support (page 0-2 → CPU, page 3-5 → GPU)

**Implementation:**
```python
class PagedAttention:
    def __init__(self, page_size=16, num_pages=4096):
        self.pages_free = BitSet(num_pages)  # Free pages
        self.pages_in_use = defaultdict(list)  # seq_id → page indices
        self.kv_storage = torch.zeros(
            (num_pages, page_size, heads, head_dim),
            dtype=torch.float16
        )
    
    def allocate_pages(self, num_pages_needed, seq_id):
        pages = self.pages_free.find_free(num_pages_needed)
        self.pages_in_use[seq_id] = pages
        return pages
    
    def forward(self, q, k_new, v_new, seq_id):
        # Retrieve cached KV from pages
        page_indices = self.pages_in_use[seq_id]
        k_cached = self.kv_storage[page_indices]  # (num_pages, page_size, heads, dim)
        k_full = k_cached.reshape(-1, self.heads, self.head_dim)  # Flatten to (seq_len, heads, dim)
        
        # Append new tokens
        k_full = torch.cat([k_full, k_new], dim=0)
        v_full = torch.cat([v_cached, v_new], dim=0)
        
        # Attention (normal computation)
        attn_output = self.attention(q, k_full, v_full)
        
        # Update cache
        next_page = find_next_free_page()
        self.kv_storage[next_page] = torch.stack([k_new, v_new])
        
        return attn_output
```

### 3. Prompt Caching

**Idea:** Many requests share identical prefixes (e.g., system prompts). Cache the KV-states.

**Example:**
```
Request 1: "[SYSTEM: You are helpful] User: What is 2+2?"
Request 2: "[SYSTEM: You are helpful] User: What is 3+3?"
Request 3: "[SYSTEM: You are helpful] User: What is 4+4?"

Shared prefix: "[SYSTEM: You are helpful]" (20 tokens)
Unique: "User: What is X+X?" (variable)

Without caching: Process 20 tokens 3 times = 60 ops
With caching:    Process 20 tokens 1 time, then append unique tokens = 20 + unique
Savings: ~60% for this example
```

**Implementation:**
```python
class PromptCache:
    def get_or_compute(self, prompt_ids, model):
        prompt_hash = hash(tuple(prompt_ids))
        
        if prompt_hash in self.cache:
            return self.cache[prompt_hash], cache_hit=True
        
        # Compute KV for this prompt
        kv_pages = self._forward_prefix(prompt_ids, model)
        
        # LRU eviction if cache full
        if self.memory_used + len(kv_pages) > self.max_memory:
            self._evict_oldest()
        
        self.cache[prompt_hash] = kv_pages
        return kv_pages, cache_hit=False
```

**Tradeoffs:**
- ✅ Huge savings for repeated prompts (60-80% reduction)
- ✅ Applicable to RAG, chat with system prompts, etc.
- ❌ Cache invalidation complexity
- ❌ Hash collisions (use SHA256, not Python hash)

---

## Implementation Checklist

### Core (MVP)
- [ ] Request queue + validation
- [ ] Single-GPU model executor
- [ ] Continuous batch scheduler
- [ ] Basic KV-cache management
- [ ] Token sampling (temperature, top-p, top-k)
- [ ] HTTP API (FastAPI)

### Optimization (v1)
- [ ] Paged attention (KV-cache pages)
- [ ] Prompt caching
- [ ] Memory pooling
- [ ] LRU eviction

### Advanced (v1.5)
- [ ] Multi-GPU (tensor parallelism)
- [ ] Dynamic batching with SLOs (latency targets)
- [ ] Speculative decoding
- [ ] Prefix caching with content-based hashing

### Production (v2)
- [ ] Distributed serving (multiple instances)
- [ ] OpenAI-compatible API
- [ ] Canary deployments
- [ ] Monitoring (Prometheus + Grafana)

---

## Testing Strategy

### Unit Tests
```python
# tests/test_batch_scheduler.py
def test_continuous_batching_no_starvation():
    """New requests should not wait > timeout_ms."""
    
def test_batch_respects_max_tokens():
    """Batch should not exceed max_batch_tokens."""
    
# tests/test_kv_cache.py
def test_paged_attention_same_as_standard():
    """Verify paged attention matches standard attention."""
    
def test_prompt_cache_hit():
    """Verify cached prompts are reused."""
```

### Integration Tests
```python
# tests/integration/test_e2e_inference.py
def test_full_request_flow():
    """End-to-end: request → batch → model → response."""
    
def test_concurrent_requests():
    """Multiple concurrent clients."""
```

### Benchmarks
```python
# benchmark/throughput.py
def benchmark_vs_vllm():
    """Compare tokens/sec, latency, memory."""
```

---

## Performance Targets

| Metric | Target | Tolerance |
|---|---|---|
| Throughput | 8000+ tokens/sec | ±10% |
| p50 latency | <300ms | ±20% |
| p99 latency | <2s | ±30% |
| Memory efficiency | ≥ vLLM | -5% acceptable |
| Cache hit rate | >60% (with prompt cache) | >50% |

---

## Common Pitfalls

### 1. Forgetting to Update Sequence Positions
If you don't track `seq_position` (how many tokens generated), the KV-cache will grow indefinitely.

```python
# ❌ Wrong
def decode(req):
    kv = cache.get(req.id)  # Always same key
    output = model(input, kv=kv)
    cache.update(req.id, new_kv)

# ✅ Right
def decode(req):
    seq_pos = req.prompt_len + req.generated_len
    kv = cache.get_slice(req.id, start=seq_pos)
    output = model(input, kv=kv)
    new_kv = torch.cat([kv, new_tokens])
    cache.set_slice(req.id, start=seq_pos, kv=new_kv)
```

### 2. Not Handling Batch Size 0
If scheduler returns empty batch (no requests ready), don't crash.

```python
# ❌ Wrong
batch = scheduler.get_batch()
output = model(batch)  # Fails if batch is empty

# ✅ Right
batch = scheduler.get_batch()
if batch:
    output = model(batch)
else:
    time.sleep(0.001)  # Wait for new requests
```

### 3. Cache Invalidation
If you use prompt hashing, be careful with:
- Tokenizer changes (same text → different tokens)
- Different models (same tokens → different KV)

Use `(model_id, tokenizer_version, prompt_tokens)` as cache key, not just prompt text.

### 4. Memory Leaks
Release KV-cache pages after requests finish.

```python
# ✅ Cleanup
def finish_request(req_id):
    pages = self.kv_cache.get_pages(req_id)
    self.kv_cache.free_pages(pages)  # IMPORTANT
    self.requests.pop(req_id)
```

---

## Debugging Tips

### Batch Size is 1 (No Batching)
- Check scheduler: are new requests reaching the scheduler?
- Check batch timeout: is it too aggressive?
- Add logging:
```python
logger.info(f"Batch size: {len(batch)}, Inflight: {len(inflight)}, New: {len(new_queue)}")
```

### High Latency Spikes
- Monitor `queued_requests` metric
- If queue grows, increase `max_batch_size` or `max_num_seqs`
- Profile GPU: if utilization < 50%, you're I/O bound

### Cache Hit Rate is 0%
- Verify prefix caching is enabled
- Check that requests actually share prompts
- Verify cache key is computed correctly (not using Python `hash()`)

---

## Resources

### Papers
- [Paged Attention (vLLM)](https://arxiv.org/abs/2309.06180) - Core inspiration
- [Efficient Transformers Survey](https://arxiv.org/abs/2202.02557) - Background on attention optimization
- [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) - Memory optimization

### Code References
- [vLLM source](https://github.com/lm-sys/vllm/blob/main/vllm/core_scheduler.py) - Real-world scheduler
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Production serving
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Fused kernels

### Tools
- `nvidia-smi` — Monitor GPU memory and utilization
- `python -m torch.utils.bottleneck` — Profile CPU/GPU time
- `py-spy` — Flame graphs for Python
- `nvprof` — NVIDIA's profiler

---

## Git Workflow

```bash
# Feature branch
git checkout -b feature/continuous-batching

# Commit with clear messages
git commit -m "impl: continuous batch scheduler with fairness timeout"

# Before merge: benchmark
python benchmark/run_throughput_test.py --baseline

# Merge with PR review
```

---

## Deployment Checklist

- [ ] Tested on target GPU (RTX 4090 / A100)
- [ ] Benchmarks pass (throughput ≥ baseline)
- [ ] No memory leaks (profile for 1 hour)
- [ ] Error handling for edge cases (empty batch, OOM, etc.)
- [ ] Logging is production-ready (no spam)
- [ ] README is updated
- [ ] CHANGELOG reflects changes

