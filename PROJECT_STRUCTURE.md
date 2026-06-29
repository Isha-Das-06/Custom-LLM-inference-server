# Project Structure

```
custom-llm-inference-server/
├── README.md                          # Main documentation (comprehensive)
├── CLAUDE.md                          # Developer guide (architecture, algorithms)
├── PROJECT_STRUCTURE.md               # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
├── .pre-commit-config.yaml            # Pre-commit hooks (linting, formatting)
│
├── server.py                          # Main entry point (server initialization)
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration management (Pydantic)
│   ├── logging_config.py              # Logging setup
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py                  # FastAPI app definition
│   │   ├── routes.py                  # API endpoints (/generate, /batch-generate, /health)
│   │   ├── models.py                  # Pydantic request/response schemas
│   │   └── middleware.py              # Auth, rate limiting, monitoring
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── request_queue.py           # Request buffering and validation
│   │   ├── batch_scheduler.py         # Continuous batch scheduling algorithm
│   │   ├── model_executor.py          # Model forward pass (prefill/decode)
│   │   ├── token_sampler.py           # Sampling (temperature, top-p, top-k)
│   │   └── response_handler.py        # Response streaming and collection
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── kv_cache_manager.py        # KV-cache paging and allocation
│   │   ├── paged_attention.py         # Paged attention mechanism
│   │   ├── prompt_cache.py            # Prompt prefix caching
│   │   └── memory_pool.py             # Memory pooling and defragmentation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_loader.py            # Load models from HF / local paths
│   │
│   └── utils/
│       ├── __init__.py
│       ├── torch_utils.py             # Tensor operations, device management
│       ├── tokenizer_utils.py         # Tokenization helpers
│       ├── metrics.py                 # Throughput, latency tracking
│       └── timing.py                  # Performance profiling decorators
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures (model, tokenizer, etc.)
│   │
│   ├── unit/
│   │   ├── test_batch_scheduler.py    # Scheduler correctness + fairness
│   │   ├── test_kv_cache.py           # Paged attention, cache correctness
│   │   ├── test_prompt_cache.py       # Prefix caching + eviction
│   │   ├── test_token_sampler.py      # Sampling correctness
│   │   └── test_request_queue.py      # Request buffering
│   │
│   ├── integration/
│   │   ├── test_e2e_inference.py      # Full request flow
│   │   ├── test_concurrent_requests.py # Multiple clients
│   │   └── test_error_handling.py     # OOM, model errors, etc.
│   │
│   └── fixtures/
│       ├── sample_prompts.py          # Test prompts
│       └── model_fixtures.py          # Mock models for testing
│
├── benchmark/
│   ├── __init__.py
│   ├── run_throughput_test.py         # Tokens/sec benchmark
│   ├── run_latency_test.py            # Latency percentile benchmark
│   ├── profile_memory.py              # Memory usage profiling
│   ├── compare_with_vllm.py           # Head-to-head with vLLM
│   ├── utils.py                       # Benchmark utilities
│   └── results/
│       ├── throughput_baseline.json   # Baseline results
│       ├── latency_baseline.json
│       └── memory_baseline.json
│
├── scripts/
│   ├── download_model.sh              # Model download script
│   ├── setup_venv.sh                  # Virtual environment setup
│   ├── run_server.sh                  # Server startup script
│   └── profile_inference.py           # PyTorch profiler script
│
├── examples/
│   ├── single_request.py              # Example: single request
│   ├── batch_requests.py              # Example: batch requests
│   ├── streaming.py                   # Example: streaming response
│   ├── prompt_caching.py              # Example: prefix caching
│   └── benchmark_comparison.py        # Example: compare with baseline
│
├── docs/
│   ├── ARCHITECTURE.md                # System architecture details
│   ├── API_REFERENCE.md               # Detailed API docs
│   ├── PERFORMANCE_TUNING.md          # Optimization guide
│   ├── DEPLOYMENT.md                  # Production deployment
│   └── TROUBLESHOOTING.md             # Common issues + solutions
│
└── .github/
    ├── workflows/
    │   ├── tests.yml                  # Run tests on PR
    │   ├── benchmarks.yml             # Run benchmarks
    │   └── style.yml                  # Linting + formatting
    │
    └── CONTRIBUTING.md                # Contribution guidelines
```

## Key Files Explained

### `server.py`
Entry point that initializes the server:
```python
from src.api.server import create_app
from src.config import ServerConfig

config = ServerConfig.from_env()
app = create_app(config)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
```

### `src/core/batch_scheduler.py`
Core algorithm that decides which requests to process next:
- Maintains `inflight_requests` and `new_request_queue`
- Implements timeout-based fairness
- Respects `max_batch_tokens` limit

### `src/memory/paged_attention.py`
KV-cache implementation with paging:
- Page allocation/deallocation
- Sequence-to-pages mapping
- Shared prefix handling

### `src/api/routes.py`
FastAPI routes:
- `POST /generate` — Single request
- `POST /batch-generate` — Multiple requests
- `GET /health` — Server health
- `GET /metrics` — Performance metrics

### `tests/unit/test_batch_scheduler.py`
Tests for continuous batching:
- No starvation (fairness timeout works)
- Respects max_batch_tokens
- Handles request lifecycle (prefill → decode → finish)

### `benchmark/run_throughput_test.py`
Performance benchmark:
- Sends N concurrent requests
- Measures tokens/sec, requests/sec
- Compares against baseline

## Data Flow

```
Client
  ↓
/generate endpoint (routes.py)
  ↓
Request validator (models.py)
  ↓
Request queue (request_queue.py)
  ↓
Batch scheduler loop (batch_scheduler.py)
  ├─→ Select requests for next batch
  ├─→ Model executor (model_executor.py)
  │   ├─→ Retrieve KV-cache (kv_cache_manager.py)
  │   ├─→ Forward pass (model_loader.py)
  │   ├─→ Sample token (token_sampler.py)
  │   └─→ Update cache (paged_attention.py)
  ├─→ Response handler (response_handler.py)
  │   └─→ Stream or collect tokens
  └─→ Repeat until request finishes
  ↓
Response back to client
```

## Testing Pyramid

```
           Benchmarks
          (performance)
        /            \
   Integration       (head-to-head with vLLM)
   (full flow)
    /      \
  Unit     (components)
(functions)
```

- **Unit Tests**: Fast, run on CPU, test individual components
- **Integration Tests**: Run on GPU, test full request flow
- **Benchmarks**: Compare against baseline/vLLM

## Dependency Graph

```
API Layer (FastAPI)
    ↓
Core Layer (Scheduling, Execution, Sampling)
    ↓
Memory Layer (KV-Cache, Paging, Prompt Cache)
    ↓
Model Layer (Model Loading, Forward Pass)
    ↓
Utilities (Tokenization, Metrics, Profiling)
```

## Adding New Features

### Example: Add Speculative Decoding

1. Create `src/core/speculative_decoder.py`
2. Add tests in `tests/unit/test_speculative_decoder.py`
3. Integrate into `batch_scheduler.py` (call draft model in parallel)
4. Update benchmark to measure latency improvement
5. Update README with new feature

### Example: Add Multi-GPU Support

1. Create `src/core/tensor_parallelism.py`
2. Modify `model_executor.py` to split model across GPUs
3. Add distributed communication layer
4. Update configuration (`src/config.py`)
5. Benchmark and update README

---

## File Naming Conventions

- `*_test.py` — Test files
- `*_utils.py` — Utility functions
- `*_config.py` — Configuration
- `*_manager.py` — Component that manages state
- `*_executor.py` — Component that executes operations

## Documentation Standards

- **README.md** — User-facing, comprehensive
- **CLAUDE.md** — Developer-facing, algorithms and architecture
- **docs/*.md** — Detailed topic guides (tuning, deployment, etc.)
- **Code comments** — Only for "why", not "what"

