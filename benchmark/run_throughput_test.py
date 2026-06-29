"""Benchmark: Throughput test"""

import argparse
import json
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model name or path"
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Number of requests"
    )
    parser.add_argument(
        "--prompt-length", type=int, default=128, help="Prompt length"
    )
    parser.add_argument(
        "--completion-length", type=int, default=256, help="Completion length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output file"
    )

    args = parser.parse_args()

    logger.info(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info(f"Running benchmark on {device}")
    logger.info(
        f"Generating {args.num_requests} prompts with length {args.prompt_length}"
    )

    # Generate dummy prompts
    prompts = [
        " ".join(["hello"] * (args.prompt_length // 6))
        for _ in range(args.num_requests)
    ]

    # Warmup
    logger.info("Warmup...")
    with torch.no_grad():
        tokens = tokenizer.encode(prompts[0])[:10]
        input_ids = torch.tensor([tokens], device=device)
        model(input_ids)

    # Benchmark
    logger.info("Benchmarking...")
    start_time = time.time()
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, args.num_requests, args.batch_size):
            batch = prompts[i : i + args.batch_size]
            inputs = tokenizer(batch, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.completion_length,
                do_sample=False,
            )

            total_tokens += (
                outputs.shape[0]
                * outputs.shape[1]
                * args.completion_length
                // 100
            )

    elapsed = time.time() - start_time

    throughput = total_tokens / elapsed
    logger.info(f"Throughput: {throughput:.1f} tokens/sec")
    logger.info(f"Latency: {elapsed / args.num_requests * 1000:.1f} ms/request")

    # Save results
    results = {
        "model": args.model,
        "num_requests": args.num_requests,
        "prompt_length": args.prompt_length,
        "completion_length": args.completion_length,
        "batch_size": args.batch_size,
        "device": device,
        "throughput_tokens_per_sec": throughput,
        "latency_ms_per_request": elapsed / args.num_requests * 1000,
        "total_time_sec": elapsed,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
