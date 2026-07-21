"""Server configuration"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ServerConfig:
    """Server configuration"""

    model_path: str
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    gpu_memory_fraction: float = 0.9
    max_num_seqs: int = 256
    max_seq_len: int = 2048
    max_batch_size: int = 4096
    batch_timeout_ms: int = 100
    enable_kv_cache_paging: bool = True
    page_size: int = 16
    enable_prompt_cache: bool = True
    prompt_cache_size_mb: int = 1000
    port: int = 8000
    host: str = "0.0.0.0"
    log_level: str = "info"

    def get_torch_dtype(self):
        """Get PyTorch dtype from config"""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[self.dtype]
