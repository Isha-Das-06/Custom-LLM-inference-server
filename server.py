#!/usr/bin/env python3
"""
Custom LLM Inference Server - Main Entry Point
"""

import argparse
import asyncio
import logging
from pathlib import Path

from src.api.server import create_app
from src.config import ServerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Custom LLM Inference Server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Model data type",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.9,
        help="GPU memory fraction to use (0-1)",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=4096, help="Max tokens per batch"
    )
    parser.add_argument(
        "--batch-timeout-ms",
        type=int,
        default=100,
        help="Batch formation timeout (ms)",
    )
    parser.add_argument(
        "--enable-kv-cache-paging",
        type=bool,
        default=True,
        help="Enable paged KV-cache",
    )
    parser.add_argument(
        "--page-size", type=int, default=16, help="Tokens per KV-cache page"
    )
    parser.add_argument(
        "--enable-prompt-cache",
        type=bool,
        default=True,
        help="Enable prompt prefix caching",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Create config
    config = ServerConfig(
        model_path=args.model_path,
        dtype=args.dtype,
        gpu_memory_fraction=args.gpu_memory_fraction,
        max_num_seqs=args.max_num_seqs,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        enable_kv_cache_paging=args.enable_kv_cache_paging,
        page_size=args.page_size,
        enable_prompt_cache=args.enable_prompt_cache,
        port=args.port,
        host=args.host,
    )

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info(f"Starting LLM Inference Server with config: {config}")

    # Create and run app
    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
