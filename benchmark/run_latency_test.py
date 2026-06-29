"""Benchmark: Latency test"""

import argparse
import json
import logging
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def percentile(data, p):
    """Calculate percentile"""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark")
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
        "--output", type=str, default="latency_results.json", help="Output file"
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

    logger.info(f"Running latency benchmark on {device}")

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
    logger.info("Benchmarking latency...")
    latencies = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.completion_length,
                do_sample=False,
            )
            elapsed = (time.time() - start) * 1000  # Convert to ms

            latencies.append(elapsed)

    # Calculate statistics
    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p99 = percentile(latencies, 99)
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0

    logger.info(f"P50 latency: {p50:.1f} ms")
    logger.info(f"P90 latency: {p90:.1f} ms")
    logger.info(f"P99 latency: {p99:.1f} ms")
    logger.info(f"Mean latency: {mean:.1f} ms")
    logger.info(f"Stdev: {stdev:.1f} ms")

    # Save results
    results = {
        "model": args.model,
        "num_requests": args.num_requests,
        "prompt_length": args.prompt_length,
        "completion_length": args.completion_length,
        "device": device,
        "latency_ms": {
            "p50": p50,
            "p90": p90,
            "p99": p99,
            "mean": mean,
            "median": median,
            "stdev": stdev,
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
