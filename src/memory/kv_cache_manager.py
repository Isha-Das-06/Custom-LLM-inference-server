"""KV-cache memory management with paging"""

import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    Manages KV-cache with paging strategy.

    Pages: Fixed-size chunks of K and V tensors
    Each sequence references a list of page indices
    Pages can be shared (prompt caching) or reused (memory pooling)
    """

    def __init__(
        self,
        num_pages: int = 4096,
        page_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate KV storage
        self.kv_storage = torch.zeros(
            (num_pages, page_size, 2, num_heads, head_dim),  # 2 for K and V
            dtype=dtype,
            device=device,
        )

        # Track free and used pages
        self.free_pages = set(range(num_pages))
        self.used_pages = defaultdict(list)  # seq_id -> [page_ids]
        self.page_contents = {}  # page_id -> (seq_id, start_pos)

        logger.info(
            f"Initialized KVCacheManager with {num_pages} pages, " f"{page_size} tokens per page"
        )

    def allocate_pages(self, seq_id: str, num_pages_needed: int) -> list[int]:
        """
        Allocate pages for a sequence.

        Args:
            seq_id: Sequence ID
            num_pages_needed: Number of pages needed

        Returns:
            List of allocated page IDs
        """
        if num_pages_needed > len(self.free_pages):
            logger.warning(
                f"Not enough free pages: need {num_pages_needed}, " f"have {len(self.free_pages)}"
            )
            # Trigger LRU eviction if needed
            self._evict_lru_pages(num_pages_needed)

        # Allocate pages
        allocated = []
        for _ in range(num_pages_needed):
            page_id = self.free_pages.pop()
            allocated.append(page_id)
            self.page_contents[page_id] = (seq_id, len(allocated) - 1)

        self.used_pages[seq_id].extend(allocated)
        logger.debug(f"Allocated {num_pages_needed} pages for seq {seq_id}")

        return allocated

    def free_pages(self, seq_id: str):
        """Free all pages for a sequence"""
        pages = self.used_pages.pop(seq_id, [])
        for page_id in pages:
            self.free_pages.add(page_id)
            self.page_contents.pop(page_id, None)

        logger.debug(f"Freed {len(pages)} pages for seq {seq_id}")

    def write_kv(
        self,
        seq_id: str,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        position: int,
    ):
        """
        Write K and V tokens to cache.

        Args:
            seq_id: Sequence ID
            k_tokens: Key tokens (batch, seq_len, heads, head_dim)
            v_tokens: Value tokens (batch, seq_len, heads, head_dim)
            position: Current position in sequence
        """
        if seq_id not in self.used_pages:
            num_pages = (position + k_tokens.shape[1] + self.page_size - 1) // self.page_size
            self.allocate_pages(seq_id, num_pages)

        pages = self.used_pages[seq_id]
        page_idx = position // self.page_size
        offset = position % self.page_size

        if page_idx < len(pages):
            page_id = pages[page_idx]
            # Write to storage
            remaining_space = self.page_size - offset
            write_len = min(k_tokens.shape[1], remaining_space)

            self.kv_storage[page_id, offset : offset + write_len, 0] = k_tokens[0, :write_len]
            self.kv_storage[page_id, offset : offset + write_len, 1] = v_tokens[0, :write_len]

    def read_kv(self, seq_id: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read K and V for a sequence.

        Args:
            seq_id: Sequence ID
            length: Number of tokens to read

        Returns:
            (K_tensor, V_tensor) of shape (length, heads, head_dim)
        """
        if seq_id not in self.used_pages:
            return (
                torch.zeros(
                    (0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device
                ),
                torch.zeros(
                    (0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device
                ),
            )

        pages = self.used_pages[seq_id]
        k_list = []
        v_list = []

        for i in range(min(length, len(pages) * self.page_size)):
            page_idx = i // self.page_size
            offset = i % self.page_size

            if page_idx < len(pages):
                page_id = pages[page_idx]
                k_list.append(self.kv_storage[page_id, offset, 0])
                v_list.append(self.kv_storage[page_id, offset, 1])

        if k_list:
            k = torch.stack(k_list)
            v = torch.stack(v_list)
        else:
            k = torch.zeros(
                (0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device
            )
            v = torch.zeros(
                (0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device
            )

        return k, v

    def _evict_lru_pages(self, num_pages_to_free: int):
        """Evict least recently used pages"""
        # Simple LRU: evict pages from sequences with most allocated
        evicted = 0
        for seq_id in sorted(
            self.used_pages.keys(), key=lambda s: len(self.used_pages[s]), reverse=True
        ):
            if evicted >= num_pages_to_free:
                break
            # Remove half of this sequence's pages
            pages = self.used_pages[seq_id]
            to_free = min(len(pages) // 2, num_pages_to_free - evicted)
            for page_id in pages[:to_free]:
                self.free_pages.add(page_id)
                self.page_contents.pop(page_id, None)
                evicted += 1
            self.used_pages[seq_id] = pages[to_free:]

        logger.warning(f"Evicted {evicted} pages due to memory pressure")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        total_cached_pages = sum(len(pages) for pages in self.used_pages.values())
        return {
            "total_pages": self.num_pages,
            "free_pages": len(self.free_pages),
            "used_pages": total_cached_pages,
            "num_sequences": len(self.used_pages),
            "memory_used_mb": (
                total_cached_pages * self.page_size * 2 * self.num_heads * self.head_dim * 2
            )
            / (1024 * 1024),
        }
