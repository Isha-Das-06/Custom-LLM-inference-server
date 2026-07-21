# KV-Cache Integration Fixes

## Problem Statement

The inference engine had critical architectural gaps preventing correct token generation:

1. **Missing Context in Decode Phase**: During token generation beyond the first token, the model executor only fed the single most recent token without access to cached key-value (KV) states from the prompt and previous tokens. This would cause:
   - Complete loss of context after token 1
   - Incoherent or nonsensical output from token 2 onwards
   - Model essentially generating tokens blindly without seeing the full context

2. **Stub KV-Cache Implementation**: The `_update_kv_cache()` method in `InferenceEngine` was an empty pass statement, meaning cached KV states were never persisted between iterations.

3. **Disconnected Components**: The `KVCacheManager` class existed with full paging implementation but was never wired into the actual generation loop.

## Root Cause

In `model_executor.py` lines 82-87 (old code):
```python
else:
    # Decode: use last generated token
    if req.generated_ids:
        input_ids = [req.generated_ids[-1]]  # ← Only the last token!
    else:
        input_ids = [req.prompt_ids[-1]]
    # No past_key_values passed to model()
```

Combined with the model forward pass at line 114:
```python
outputs = self.model(
    input_ids=input_tensor,
    attention_mask=attention_tensor,
    # ← No past_key_values argument
)
```

## Fixes Applied

### 1. Model Executor - KV-Cache Buffering (model_executor.py)

**Added KV cache buffer to ModelExecutor init (line 54):**
```python
self._kv_cache_buffer = {}  # request_id -> past_key_values
```

**Modified forward() to use cached KV states (lines 85-97):**
- Prefill phase (position == 0): Process full prompt, no cached KV
- Decode phase: Only feed last generated token BUT use cached KV from previous iterations
- Retrieve cached KV: `past_key_values = self._kv_cache_buffer.get(req.request_id, None)`

**Updated model call (lines 135-140):**
```python
outputs = self.model(
    input_ids=input_tensor_single,
    past_key_values=past_kv,  # ← Pass cached KV
    use_cache=True,           # ← Request new KV output
)
```

**Cache new KV for next iteration (lines 142-143, 165-167):**
```python
new_past_kv = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
...
if new_past_kv is not None:
    self._kv_cache_buffer[req.request_id] = new_past_kv
```

**Added cleanup method (line 172-175):**
```python
def clear_kv_cache(self, request_id: str):
    """Clear cached KV states for a request"""
    self._kv_cache_buffer.pop(request_id, None)
```

### 2. Inference Engine - Cache Integration (inference_engine.py)

**Updated batch processing to handle new return format (lines 89-98):**
```python
for i, (request, result) in enumerate(zip(batch, results)):
    if len(result) == 3:
        token_id, is_eos, past_kv = result
    else:
        token_id, is_eos = result
        past_kv = None
    
    self._update_kv_cache(request, past_kv)
```

**Implemented _update_kv_cache (lines 118-139):**
- Accepts past_key_values from model executor
- Tracks cache state for metrics
- Includes comments on production optimizations (spilling to paged cache)

**Added cleanup in _complete_request (lines 155-158):**
```python
self.model_executor.clear_kv_cache(request.request_id)
if request.kv_cache_pages:
    self.kv_cache_manager.free_pages(request.request_id)
```

## How It Works Now

### Generation Flow (Fixed)
```
Request 1 with prompt "Hello world"
  │
  ├─ Iteration 1 (Prefill): 
  │   Input: [Hello, world] 
  │   Output: Token "A", past_kv stored
  │
  ├─ Iteration 2 (Decode):
  │   Input: [A] + past_kv (contains prompt context)
  │   Output: Token "B", past_kv updated
  │
  ├─ Iteration 3 (Decode):
  │   Input: [B] + past_kv (contains prompt + A)
  │   Output: Token "C", past_kv updated
  │
  └─ Continue until EOS or max_tokens
```

### Memory Management
1. **During Generation**: KV cached in ModelExecutor._kv_cache_buffer (fast, on-GPU)
2. **Request Completion**: Cache cleaned up via clear_kv_cache()
3. **Future Enhancement**: Can spill long sequences to paged KVCacheManager

## Testing Recommendations

1. **Verify Context Preservation**: Generate text and check that tokens depend on full context
   ```python
   prompt = "The capital of France is"
   # Should consistently produce "Paris" or related token
   # NOT random tokens
   ```

2. **Check Memory Cleanup**: Monitor that cache buffer doesn't grow unbounded
   ```python
   executor._kv_cache_buffer should be empty after request completes
   ```

3. **Batch Consistency**: Multi-request batches should each maintain independent contexts
   ```python
   batch = [req1 (prompt A), req2 (prompt B)]
   # Both should generate coherently from their own prompts
   ```

## Performance Impact

- **Positive**: Enables correct multi-token generation (was broken before)
- **Minimal overhead**: Per-request dict lookup, one tensor copy per generation step
- **Scalability**: Can integrate paged cache for memory-constrained deployments

## Next Steps (Not Implemented)

1. Integrate output KV into KVCacheManager for production deployments
2. Add memory budgets to spill long-running sequences to CPU/disk
3. Implement prompt caching to avoid recomputing shared prefixes
4. Add tests verifying KV-cache correctness vs. full context attention
