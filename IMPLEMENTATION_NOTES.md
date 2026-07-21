# KV-Cache Integration: Technical Implementation Notes

## Executive Summary

Fixed a critical bug preventing correct token generation beyond the first token. The model now properly maintains context across the entire generation sequence using cached KV (key-value) states.

## The Bug in Detail

### What Was Broken
During token generation (decode phase), the model executor fed only the single most recently generated token to the model **without any cached context**:

```python
# OLD CODE - Context Loss
for iteration in range(max_tokens):
    if iteration == 0:
        input_ids = full_prompt  # [token0, token1, ..., tokenN]
    else:
        input_ids = [last_token]  # [tokenN+iteration] ← ONLY THIS!
    
    logits = model(input_ids)  # Model can't see prompt or prior tokens
    next_token = sample(logits)
```

This is fundamentally broken for transformers, which are **entirely attention-based**. Without access to the prompt and previous outputs, the model generates tokens with zero context—essentially random generation that happens to be tokenized.

### Example Failure
```
Prompt: "The capital of France is"

Token 1: "Paris" ✓ (model seen full prompt)
Token 2: "???" (model only sees [Paris], not prompt)
Token 3: "!#$" (random)
Token 4: "%^&" (random)
```

### Root Cause
The `model.forward()` call was missing the `past_key_values` parameter:
```python
outputs = self.model(
    input_ids=input_tensor,
    attention_mask=attention_tensor,
    # ← past_key_values NOT PASSED
)
```

## The Fix

### 1. KV Cache Buffer (ModelExecutor)

Added in-memory cache to store KV states between inference steps:

```python
class ModelExecutor:
    def __init__(self, ...):
        self._kv_cache_buffer = {}  # request_id -> past_key_values
```

**Why in-memory?** 
- Fast GPU-GPU transfer (no memory movement)
- Simple single-request isolation
- Sufficient for most workloads
- Future: Can spill to paged cache for long sequences

### 2. Forward Pass With KV Cache

Two distinct code paths depending on phase:

```python
def forward(self, requests, eos_token_id=None):
    for req in requests:
        if req.position == 0:  # PREFILL
            input_ids = req.prompt_ids
            past_kv = None
        else:  # DECODE
            input_ids = [req.generated_ids[-1]]  # Just last token
            past_kv = self._kv_cache_buffer[req.request_id]  # Load cached KV
        
        # Forward pass WITH cached KV
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,  # ← CRITICAL
            use_cache=True,  # ← Get new KV
        )
        
        # Cache the new KV for next iteration
        new_kv = outputs.past_key_values
        self._kv_cache_buffer[req.request_id] = new_kv
```

### 3. Memory Management

Cleanup on request completion:

```python
def clear_kv_cache(self, request_id: str):
    self._kv_cache_buffer.pop(request_id, None)
```

Called by InferenceEngine after request finishes:
```python
self.model_executor.clear_kv_cache(request.request_id)
```

## How KV-Cache Works (Primer)

Transformers compute attention using keys (K) and values (V):

```
Q(new) @ K(all).T → Attention weights
Attention weights @ V(all) → Output
```

**KV-Cache stores** the K and V matrices from all previous tokens so you don't recompute:

```
Iteration 1 (Prefill):
  Input: [tok0, tok1, tok2]  (prompt)
  Output: tok3 (next token)
  Cached: K=[k0,k1,k2], V=[v0,v1,v2]

Iteration 2 (Decode):
  Input: [tok3]  (just new token)
  K_cached: [k0,k1,k2,k3_new]  (append new K)
  V_cached: [v0,v1,v2,v3_new]  (append new V)
  Attention: Q(tok3) @ K_cached → uses full context!
  Output: tok4
```

**Memory savings**: Without cache, iteration N must recompute N tokens. With cache, compute only 1 new token + lookup.

## Architecture After Fix

```
Request "Hello world" (2 tokens)
│
├─ Prefill Step (req.position == 0)
│  ├─ Input: "Hello world" (2 tokens)
│  ├─ Model forward: Q @ [K_hello, K_world]
│  ├─ Output: next token = "!"
│  └─ Cache: K=[k_hello, k_world], V=[v_hello, v_world]
│
├─ Decode Step 1 (req.position == 2)
│  ├─ Input: "!" (1 token) + past_kv
│  ├─ Model forward: Q(!) @ [K_hello, K_world, K_!]
│  ├─ Output: next token = "How"
│  └─ Cache: K=[..., k_!], V=[..., v_!]
│
├─ Decode Step 2 (req.position == 3)
│  ├─ Input: "How" (1 token) + past_kv
│  ├─ Model forward: Q(How) @ [K_hello, K_world, K_!, K_How]
│  ├─ Output: next token = "are"
│  └─ Cache: K=[..., k_How], V=[..., v_How]
│
└─ Continue until EOS or max_tokens
```

## Return Value Changes

Model executor now returns KV state:

```python
# Before
results = [(token_id, is_eos), ...]

# After
results = [(token_id, is_eos, past_key_values), ...]
```

InferenceEngine handles both formats for backward compatibility:
```python
if len(result) == 3:
    token_id, is_eos, past_kv = result
else:
    token_id, is_eos = result
    past_kv = None
```

## Test Updates

All mock ModelExecutor instances updated to return 3-tuple:

```python
# Before
def forward(self, batch):
    return [(1, False) for _ in batch]

# After  
def forward(self, batch):
    return [(1, False, None) for _ in batch]  # None = no KV cache
```

## Verification Steps

1. **Check context is preserved**:
   ```python
   prompt = "Q: What is 2+2? A:"
   # Should consistently generate related math content
   # NOT random tokens
   ```

2. **Verify cache cleanup**:
   ```python
   assert len(model_executor._kv_cache_buffer) == 0  # After request finishes
   ```

3. **Memory profile**:
   ```python
   import psutil
   # Should not grow unbounded over multiple requests
   ```

4. **Batch consistency**:
   ```python
   batch = [Request(prompt_A), Request(prompt_B)]
   # Each should maintain independent context
   ```

## Performance Characteristics

| Metric | Impact | Notes |
|--------|--------|-------|
| Correctness | **Critical Fix** | Was generating nonsense, now works |
| Throughput | No change | Still 1 forward pass per token |
| Memory | +Small | One dict entry per active request |
| Latency | No change | Same compute + dict lookup |

## Future Optimizations Not Implemented

1. **Paged KV-Cache**: Write to KVCacheManager for multi-GPU or long sequences
2. **Memory Pooling**: Reuse pages across sequences sharing common prefix
3. **Prompt Caching**: Hash identical prefixes to avoid recomputing
4. **CPU Offloading**: Spill old KV tokens to CPU memory
5. **Attention Sparsity**: Skip computing attention for very old tokens

Each would require:
- Extracting K, V from model outputs (currently we just store past_key_values tuple)
- Custom attention kernels to handle paged layout
- Cache key management and eviction policies

## Code Locations

| File | Change | Lines |
|------|--------|-------|
| `src/core/model_executor.py` | Add KV buffer | 54, 85-97, 135-170 |
| `src/core/inference_engine.py` | Handle KV returns, cleanup | 89-103, 118-139, 155-158 |
| `tests/test_inference_engine.py` | Update mocks for 3-tuple | All MockModelExecutor |

## Safety and Edge Cases

✅ **Handled**:
- Empty batch (no requests to process)
- First token (prefill with no cached KV)
- Request completion (cache cleanup)
- Backward compatibility (accept 2 or 3-tuple)

⚠️ **Assumptions**:
- Model supports `use_cache=True` (standard for HF models)
- `past_key_values` can be pickled/stored (true for torch tensors)
- Same tokenizer throughout request lifetime
- No model parameter changes mid-request (true for inference)

## Debugging Tips

If output still seems incoherent:

1. **Check if KV cache is actually cached**:
   ```python
   # Add logging
   logger.info(f"Cache hit: {request.request_id in model_executor._kv_cache_buffer}")
   ```

2. **Verify model supports use_cache**:
   ```python
   model.config.use_cache = True
   assert hasattr(model.forward, 'use_cache')
   ```

3. **Inspect cache size**:
   ```python
   for past in outputs.past_key_values:
       print(past.shape)  # Should grow by 1 token per iteration
   ```

4. **Test without cache** (sanity check):
   ```python
   # Remove KV passing, verify it breaks (expected)
   outputs = model(input_ids, past_key_values=None)
   ```
