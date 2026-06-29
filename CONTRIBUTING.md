# Contributing to Custom LLM Inference Server

Thank you for your interest in contributing! This document provides guidelines and best practices for contributing to this project.

## Getting Started

### 1. Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/custom-llm-inference-server.git
cd custom-llm-inference-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (with dev tools)
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Understand the Codebase

Before jumping into coding:
1. Read [README.md](README.md) for project overview
2. Review [CLAUDE.md](CLAUDE.md) for architecture and algorithms
3. Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for file organization
4. Explore `tests/` to understand testing patterns

### 3. Run Tests Locally

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration tests (requires GPU)
pytest tests/integration/ -v --gpu

# Run specific test
pytest tests/unit/test_batch_scheduler.py::test_continuous_batching_no_starvation -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Types of Contributions

### 🐛 Bug Fixes

**Reporting a bug:**
1. Check [existing issues](https://github.com/yourusername/custom-llm-inference-server/issues)
2. Open a new issue with:
   - Clear title
   - Minimal reproduction script
   - Expected vs. actual behavior
   - Environment (GPU, CUDA version, PyTorch version)

**Fixing a bug:**
1. Create branch: `git checkout -b fix/issue-description`
2. Write a test that reproduces the bug
3. Fix the bug
4. Verify the test passes: `pytest tests/unit/test_*.py -v`
5. Run pre-commit checks: `pre-commit run --all-files`
6. Open PR with reference to issue: "Fixes #123"

### ✨ Features

**Proposing a feature:**
1. Open discussion in [GitHub Discussions](https://github.com/yourusername/custom-llm-inference-server/discussions)
2. Describe use case and how it benefits users
3. Wait for feedback before implementing

**Implementing a feature:**
1. Create branch: `git checkout -b feature/feature-name`
2. Follow the checklist below
3. Open PR with detailed description

### 📊 Benchmarks & Performance

**Adding benchmark:**
1. Create benchmark script in `benchmark/`
2. Include setup, execution, and teardown
3. Compare against baseline (if applicable)
4. Document results in PR description

### 📚 Documentation

**Fixing or improving docs:**
1. Create branch: `git checkout -b docs/description`
2. Update relevant markdown files
3. Check formatting: `md --check`
4. Open PR with rationale

---

## Development Workflow

### Branch Naming

```
fix/                          Bug fixes
feature/                      New features
perf/                         Performance improvements
docs/                         Documentation
refactor/                     Code refactoring
test/                         Test improvements
```

Examples:
- `fix/kv-cache-memory-leak`
- `feature/prompt-caching`
- `perf/batch-scheduler-optimization`
- `docs/api-reference`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type: subject (under 50 chars)

Body (wrapped at 72 chars, optional)
- Explain WHY, not WHAT
- Reference issues: Fixes #123

Footer (optional)
Co-Authored-By: Name <email>
```

Examples:
```
fix: prevent memory leak in kv_cache_manager

KV-cache pages were not being freed after sequence completion.
Ensure all pages are returned to free_pages pool.

Fixes #45
```

```
feat: implement continuous batching scheduler

Use FIFO fairness with timeout-based batch flushing.
Achieves 2-3x throughput improvement over static batching.

- Maintain inflight_requests and new_request_queue
- Implement timeout fairness mechanism
- Add tests for starvation prevention
```

### Code Style

**Python Formatting:**
```bash
# Format code
black src/ tests/ benchmark/

# Import sorting
isort src/ tests/ benchmark/

# Linting
flake8 src/ tests/ --max-line-length=100

# Type checking
mypy src/
```

**Automated via pre-commit:**
```bash
# Run all checks before commit
pre-commit run --all-files

# Checks include: black, isort, flake8, mypy, trailing-whitespace, etc.
```

**Style Guidelines:**
- Max line length: 100 characters
- Type hints on all functions
- Docstrings for public functions (one-line or Google style)
- No magic numbers (use constants)
- Meaningful variable names

```python
# ❌ Bad
def foo(x, y):
    z = x * 2 + y
    return z

# ✅ Good
def compute_scaling_factor(base_value: float, offset: float) -> float:
    """Scale base_value by 2 and add offset."""
    scaled = base_value * SCALE_FACTOR + offset
    return scaled
```

---

## Testing Requirements

### Unit Tests

Every new feature/fix should have unit tests:

```python
# tests/unit/test_my_feature.py
import pytest
from src.core.my_feature import MyComponent

@pytest.fixture
def component():
    return MyComponent()

def test_happy_path(component):
    """Test normal operation."""
    result = component.process(valid_input)
    assert result == expected_output

def test_edge_case(component):
    """Test boundary conditions."""
    result = component.process(edge_case_input)
    assert result == expected_output

def test_error_handling(component):
    """Test error cases."""
    with pytest.raises(ValueError):
        component.process(invalid_input)
```

**Coverage Requirements:**
- Aim for ≥90% line coverage
- ≥95% for critical paths (scheduler, cache manager)
- Use `pytest --cov` to check

### Integration Tests

For features that span multiple components:

```python
# tests/integration/test_feature_flow.py
@pytest.mark.gpu
def test_end_to_end_inference():
    """Test full request → response flow."""
    server = start_test_server()
    
    response = server.generate(
        prompt="Hello",
        max_new_tokens=10
    )
    
    assert response.finish_reason == "length"
    assert len(response.generated_tokens) == 10
```

### Performance Tests

For performance-critical components:

```bash
# Run benchmark with baseline comparison
python benchmark/run_throughput_test.py \
    --baseline benchmark/results/throughput_baseline.json \
    --threshold 5  # Fail if throughput drops >5%
```

---

## Pull Request Process

### Before Opening PR

1. **Code Quality**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   pre-commit run --all-files
   ```

2. **Tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

3. **Benchmarks** (if performance-related)
   ```bash
   python benchmark/run_throughput_test.py --baseline
   ```

4. **Documentation**
   - Update README if adding features
   - Update CLAUDE.md if changing architecture
   - Add docstrings to new functions

### PR Description Template

```markdown
## Description
Brief explanation of what this PR does.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Changed X from A to B because...
- Added new component Y with features Z
- Removed deprecated code

## Testing
- [ ] Unit tests pass (`pytest tests/unit/ -v`)
- [ ] Integration tests pass (`pytest tests/integration/ -v --gpu`)
- [ ] Benchmarks show no regression (link results)

## Benchmarks (if applicable)
```
Throughput: 8234 → 8542 tokens/sec (+3.8%)
p50 latency: 301ms → 287ms (-4.6%)
Memory: 18.2 → 17.1 GB (-6.0%)
```

## Related Issues
Closes #123

## Checklist
- [ ] Code follows style guidelines
- [ ] No new warnings in lint/type-check
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Performance impact understood
```

### Review Process

1. **Automated checks** (GitHub Actions)
   - Tests pass
   - Coverage meets threshold
   - Linting/type-checking pass

2. **Peer review**
   - At least 1 approval from maintainer
   - Address reviewer feedback
   - Update PR with changes

3. **Merge**
   - Squash commits if multiple
   - Use PR description as commit message

---

## Common Tasks

### Adding a New Feature

**Example: Add Speculative Decoding**

1. **Design (discussion phase)**
   - Open GitHub Discussion
   - Outline algorithm and API
   - Wait for feedback

2. **Implementation**
   ```bash
   git checkout -b feature/speculative-decoding
   ```
   - Create `src/core/speculative_decoder.py`
   - Integrate into `batch_scheduler.py`
   - Add configuration options to `src/config.py`

3. **Testing**
   ```bash
   # Create tests
   touch tests/unit/test_speculative_decoder.py
   
   # Write tests for:
   # - Correctness (draft model predictions)
   # - Efficiency (latency improvement)
   # - Edge cases (short sequences, OOM)
   pytest tests/unit/test_speculative_decoder.py -v
   ```

4. **Benchmarking**
   ```bash
   python benchmark/run_latency_test.py --with-speculative
   ```

5. **Documentation**
   - Add section to README.md
   - Update PROJECT_STRUCTURE.md
   - Document configuration in CLAUDE.md

6. **Open PR**
   - Reference design discussion
   - Include benchmark results
   - Wait for review

### Optimizing Performance

1. **Profile first**
   ```python
   from torch.profiler import profile
   
   with profile(...) as prof:
       output = model.forward(batch)
   
   print(prof.key_averages().table(...))
   ```

2. **Identify bottleneck**
   - GPU time vs. CPU time?
   - Memory bandwidth vs. compute?

3. **Implement optimization**
   - Kernel fusion? Memory pooling? Caching?

4. **Measure impact**
   ```bash
   python benchmark/run_throughput_test.py --baseline
   ```

5. **Document trade-offs**
   - Did we trade memory for speed?
   - Does optimization work for all model sizes?

### Fixing a Bug

1. **Reproduce**
   ```python
   # Create minimal reproduction in tests/
   def test_bug_reproduction():
       # This should fail with current code
       assert False
   ```

2. **Fix**
   - Understand root cause
   - Implement minimal fix
   - Avoid unnecessary refactoring

3. **Verify**
   ```bash
   pytest tests/unit/test_*.py -v  # Fix should make test pass
   ```

4. **Test edge cases**
   - Does fix work with empty batch?
   - Long sequences? Short sequences?
   - Different models?

---

## Performance Guidelines

### Acceptable Performance Regressions

- Throughput: ±5% tolerance
- Latency (p50): ±10% tolerance
- Latency (p99): ±15% tolerance
- Memory: ±5% tolerance

PRs exceeding these thresholds require discussion/approval.

### Optimization Opportunities

If you spot inefficiencies:

- **Reduce memory allocations**: Use memory pools, reuse tensors
- **Reduce data copies**: Avoid `clone()`, `detach()`, `.cpu()`
- **Kernel fusion**: Combine multiple operations
- **Batch size tuning**: Profile across different batch sizes
- **Cache optimization**: Prefix caching, token sharing

---

## Getting Help

### Questions?

- Check [GitHub Discussions](https://github.com/yourusername/custom-llm-inference-server/discussions)
- Read [CLAUDE.md](CLAUDE.md) for architecture details
- Ask in code review

### Stuck on a Bug?

1. Add debug logging
2. Profile with `torch.profiler`
3. Check existing issues/PRs
4. Open discussion with minimal repro

### Need Feedback on Design?

1. Open GitHub Discussion
2. Share architecture diagram
3. Ask specific questions
4. Iterate based on feedback

---

## Code of Conduct

- Be respectful and inclusive
- Assume good intent
- Welcome diverse perspectives
- Report violations to maintainers

---

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Thanked in README acknowledgments

Thank you for making this project better! 🚀

