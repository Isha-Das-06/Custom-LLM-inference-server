"""Model execution on GPU"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.request_queue import Request
from src.core.token_sampler import TokenSampler
from src.memory.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


class ModelExecutor:
    """Executes model inference on GPU"""

    def __init__(
        self,
        model_path: str,
        dtype: str = "float16",
        device: str = "cuda",
        kv_cache_manager: Optional[KVCacheManager] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.kv_cache_manager = kv_cache_manager

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float16)

        logger.info(f"Loading model from {model_path} with dtype {dtype}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set model to eval mode
        self.model.eval()
        logger.info(f"Model loaded successfully")

    @torch.no_grad()
    def forward(
        self,
        requests: list[Request],
        eos_token_id: Optional[int] = None,
    ) -> list[tuple[int, bool]]:
        """
        Forward pass for a batch of requests.

        Args:
            requests: List of Request objects
            eos_token_id: End-of-sequence token ID

        Returns:
            List of (token_id, is_eos) tuples
        """
        if not requests:
            return []

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        results = []

        # Prepare input: for each request, use last token (prefill or decode)
        input_ids_list = []
        for req in requests:
            if req.position == 0:
                # Prefill: use full prompt
                input_ids = req.prompt_ids
            else:
                # Decode: use last generated token
                if req.generated_ids:
                    input_ids = [req.generated_ids[-1]]
                else:
                    input_ids = [req.prompt_ids[-1]]

            input_ids_list.append(input_ids)

        # Pad to same length for batch processing
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        attention_masks = []

        for input_ids in input_ids_list:
            pad_len = max_len - len(input_ids)
            padded = input_ids + [self.tokenizer.pad_token_id] * pad_len
            mask = [1] * len(input_ids) + [0] * pad_len

            padded_input_ids.append(padded)
            attention_masks.append(mask)

        # Convert to tensors
        input_tensor = torch.tensor(
            padded_input_ids, dtype=torch.long, device=self.device
        )
        attention_tensor = torch.tensor(
            attention_masks, dtype=torch.long, device=self.device
        )

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
            )

        logits = outputs.logits[:, -1, :]  # Get last token logits

        # Sample tokens
        for i, req in enumerate(requests):
            logit = logits[i]

            # Apply penalties
            logit = TokenSampler.apply_penalties(
                logit.unsqueeze(0),
                req.generated_ids,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                repetition_penalty=req.repetition_penalty,
            )

            # Sample
            token_id = TokenSampler.sample(
                logit,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
            ).item()

            # Check if EOS
            is_eos = token_id == eos_token_id

            results.append((token_id, is_eos))
            req.step(token_id, is_eos=is_eos)

        return results

    def get_vocab_size(self) -> int:
        """Get model vocabulary size"""
        return self.model.config.vocab_size

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "vocab_size": self.model.config.vocab_size,
            "num_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "num_attention_heads": self.model.config.num_attention_heads,
            "dtype": str(self.dtype),
            "device": self.device,
        }
