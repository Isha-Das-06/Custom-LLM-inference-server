"""GPU memory management and monitoring utilities"""

import logging

import torch

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitor and track GPU memory usage"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_gpu = torch.cuda.is_available() and device == "cuda"

    def get_memory_info(self) -> dict:
        """Get current GPU memory information"""
        if not self.is_gpu:
            return {
                "used_mb": 0.0,
                "total_mb": 0.0,
                "free_mb": 0.0,
                "percent_used": 0.0,
            }

        try:
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            free = total - allocated

            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "total_mb": total,
                "free_mb": free,
                "percent_used": (allocated / total * 100) if total > 0 else 0.0,
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
                "free_mb": 0.0,
                "percent_used": 0.0,
            }

    def get_device_info(self) -> dict:
        """Get GPU device information"""
        if not self.is_gpu:
            return {"device_name": "CPU", "compute_capability": "N/A"}

        try:
            props = torch.cuda.get_device_properties(0)
            return {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024**3),
                "max_threads_per_block": props.max_threads_per_block,
            }
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return {}

    def clear_cache(self):
        """Clear GPU cache"""
        if self.is_gpu:
            try:
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.error(f"Error clearing GPU cache: {e}")

    def reset_peak_memory(self):
        """Reset peak memory tracking"""
        if self.is_gpu:
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage"""
        if not self.is_gpu:
            return 0.0

        try:
            return torch.cuda.max_memory_allocated() / (1024**2)
        except:
            return 0.0
