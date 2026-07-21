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

        # Set model to eval mode and enable output of past_key_values
        self.model.eval()
        self.model.config.use_cache = True
        logger.info("Model loaded successfully")

        # Cache for storing KV states between batches
        self._kv_cache_buffer = {}  # request_id -> past_key_values

    @torch.no_grad()
    def forward(
        self,
        requests: list[Request],
        eos_token_id: Optional[int] = None,
    ) -> list[tuple[int, bool, Optional[tuple]]]:
        """
        Forward pass for a batch of requests with KV-cache support.

        Args:
            requests: List of Request objects
            eos_token_id: End-of-sequence token ID

        Returns:
            List of (token_id, is_eos, past_key_values) tuples
        """
        if not requests:
            return []

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        results = []

        # Prepare input: for each request, use appropriate tokens and cached KV
        input_ids_list = []
        attention_masks_list = []
        past_key_values_list = []

        for req in requests:
            if req.position == 0:
                # Prefill: use full prompt, no cached KV
                input_ids = req.prompt_ids
                past_key_values = None
            else:
                # Decode: use only last generated token + cached KV
                if req.generated_ids:
                    input_ids = [req.generated_ids[-1]]
                else:
                    input_ids = [req.prompt_ids[-1]]
                # Retrieve cached KV from buffer
                past_key_values = self._kv_cache_buffer.get(req.request_id, None)

            input_ids_list.append(input_ids)
            past_key_values_list.append(past_key_values)
            # For simplicity, always use full attention mask
            attention_masks_list.append([1] * len(input_ids))

        # For batches with mixed prefill/decode, we need to handle carefully
        # For now, process each request individually to properly use their cached KV
        for i, (req, input_ids, past_kv) in enumerate(
            zip(requests, input_ids_list, past_key_values_list)
        ):
            input_tensor_single = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            # Forward pass with optional cached KV
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor_single,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]  # Get last token logits
            new_past_kv = outputs.past_key_values if hasattr(outputs, "past_key_values") else None

            # Apply penalties
            logit = TokenSampler.apply_penalties(
                logits[0].unsqueeze(0),
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

            # Cache the new KV states for next iteration
            if new_past_kv is not None:
                self._kv_cache_buffer[req.request_id] = new_past_kv

            results.append((token_id, is_eos, new_past_kv))
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

    def clear_kv_cache(self, request_id: str):
        """Clear cached KV states for a request"""
        self._kv_cache_buffer.pop(request_id, None)
        logger.debug(f"Cleared KV cache for request {request_id}")
