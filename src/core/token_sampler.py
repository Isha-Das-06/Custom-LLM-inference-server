"""Token sampling strategies"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TokenSampler:
    """Sampling from model logits"""

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = None,
    ) -> torch.Tensor:
        """
        Sample tokens from logits using temperature, top-p, and top-k.

        Args:
            logits: Model output logits (batch, vocab_size)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold (0-1)
            top_k: Keep top K logits
            seed: Random seed for reproducibility

        Returns:
            Sampled token IDs (batch,)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy: argmax
            return torch.argmax(logits, dim=-1)

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # Keep at least one token
            logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return token_ids

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        generated_ids: list[int],
        frequency_penalty: float = 1.0,
        presence_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply penalties to discourage repetition.

        Args:
            logits: Model output logits
            generated_ids: Previously generated token IDs
            frequency_penalty: Penalty for token frequency
            presence_penalty: Penalty for token presence
            repetition_penalty: Penalty for repetition

        Returns:
            Modified logits
        """
        if not generated_ids:
            return logits

        logits = logits.clone()

        for token_id in set(generated_ids):
            if token_id < logits.shape[-1]:
                # Frequency penalty: penalize based on count
                count = generated_ids.count(token_id)
                logits[:, token_id] -= frequency_penalty * count

                # Presence penalty: penalize if token appears at all
                logits[:, token_id] -= presence_penalty

        # Repetition penalty (GPTQ style)
        for token_id in set(generated_ids):
            if token_id < logits.shape[-1]:
                if logits[:, token_id] < 0:
                    logits[:, token_id] *= repetition_penalty
                else:
                    logits[:, token_id] /= repetition_penalty

        return logits
