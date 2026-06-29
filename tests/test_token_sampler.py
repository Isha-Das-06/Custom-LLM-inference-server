"""Tests for token sampler"""

import pytest
import torch

from src.core.token_sampler import TokenSampler


def test_token_sampler_greedy():
    """Test greedy sampling"""
    logits = torch.tensor(
        [[1.0, 5.0, 2.0, 1.0]], dtype=torch.float32
    )
    token = TokenSampler.sample(logits, temperature=0.0)
    # Should pick highest logit (index 1)
    assert token.item() == 1


def test_token_sampler_temperature():
    """Test temperature scaling"""
    logits = torch.tensor(
        [[1.0, 2.0, 1.0]], dtype=torch.float32
    )

    # Low temperature = more peaked
    token_low = TokenSampler.sample(logits, temperature=0.1)
    # High temperature = flatter
    token_high = TokenSampler.sample(logits, temperature=10.0)

    # Both should work (may pick any token)
    assert 0 <= token_low.item() < 3
    assert 0 <= token_high.item() < 3


def test_token_sampler_top_k():
    """Test top-k filtering"""
    logits = torch.tensor(
        [[10.0, 9.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    token = TokenSampler.sample(logits, temperature=0.7, top_k=2)
    # Should only sample from top 2 (indices 0, 1)
    assert token.item() in [0, 1]


def test_token_sampler_top_p():
    """Test nucleus (top-p) sampling"""
    logits = torch.tensor(
        [[100.0, 10.0, 1.0, 1.0]], dtype=torch.float32
    )
    token = TokenSampler.sample(logits, temperature=0.7, top_p=0.9)
    # Should sample from high probability tokens
    assert token.item() in [0, 1]


def test_apply_penalties():
    """Test repetition penalties"""
    logits = torch.tensor(
        [[1.0, 1.0, 1.0]], dtype=torch.float32
    )
    generated = [0, 0, 1]

    penalized = TokenSampler.apply_penalties(
        logits,
        generated,
        frequency_penalty=1.0,
        presence_penalty=1.0,
    )

    # Token 0 should have lower logits (appears twice)
    # Token 1 should have lower logits (appears once)
    # Token 2 should keep original logit
    assert penalized[0, 0] < logits[0, 2]  # More penalized
    assert penalized[0, 1] < logits[0, 2]  # Penalized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
