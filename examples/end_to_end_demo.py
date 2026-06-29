"""End-to-end demo of the inference server"""

import asyncio
import httpx
import json
import time


async def single_request_demo():
    """Demo: Single request inference"""
    print("\n=== Single Request Demo ===")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "What is artificial intelligence?",
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        result = response.json()
        print(f"Request ID: {result['request_id']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"Tokens: {result['num_generated_tokens']}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
        print(f"Throughput: {result['tokens_per_sec']:.1f} tokens/sec")


async def batch_requests_demo():
    """Demo: Batch inference"""
    print("\n=== Batch Requests Demo ===")

    requests = [
        {"prompt": "What is machine learning?", "max_new_tokens": 40},
        {"prompt": "Explain deep learning", "max_new_tokens": 40},
        {"prompt": "Define neural networks", "max_new_tokens": 40},
        {"prompt": "What is NLP?", "max_new_tokens": 40},
        {"prompt": "Tell me about GPT models", "max_new_tokens": 40},
    ]

    async with httpx.AsyncClient(timeout=120) as client:
        start = time.time()
        response = await client.post(
            "http://localhost:8000/batch-generate",
            json={"requests": requests},
        )
        elapsed = time.time() - start

        results = response.json()["results"]

        print(f"Processed {len(results)} requests in {elapsed:.2f}s")
        print(f"Average latency: {sum(r['latency_ms'] for r in results) / len(results):.1f}ms")
        print(f"Total tokens generated: {sum(r['num_generated_tokens'] for r in results)}")

        for i, result in enumerate(results):
            print(f"\nRequest {i + 1}: {result['num_generated_tokens']} tokens, {result['latency_ms']:.1f}ms")


async def streaming_demo():
    """Demo: Streaming responses"""
    print("\n=== Streaming Demo ===")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "Write a short story about AI",
                "max_new_tokens": 100,
                "stream": True,
                "stream_chunk_size": 10,
            },
        )

        print("Streaming tokens:")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                print(data.get("token", ""), end="", flush=True)
        print()


async def health_and_metrics():
    """Demo: Health check and metrics"""
    print("\n=== Health Check & Metrics ===")

    async with httpx.AsyncClient() as client:
        # Health
        response = await client.get("http://localhost:8000/health")
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Uptime: {health['uptime_seconds']:.1f}s")

        # Metrics
        response = await client.get("http://localhost:8000/metrics")
        metrics = response.json()

        print(f"\nMetrics:")
        print(f"  Requests processed: {metrics['total_requests_processed']}")
        print(f"  Active requests: {metrics['active_requests']}")
        print(f"  Queued requests: {metrics['queued_requests']}")
        print(f"  Avg throughput: {metrics['avg_throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"  GPU memory: {metrics['gpu_memory_used_gb']:.2f}GB / {metrics['gpu_memory_total_gb']:.2f}GB")
        print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        print(f"  KV-cache pages: {metrics['kv_cache_pages_used']}/{metrics['kv_cache_pages_total']}")


async def concurrent_requests_demo():
    """Demo: Multiple concurrent requests"""
    print("\n=== Concurrent Requests Demo ===")

    prompts = [
        "What is AI?",
        "Explain ML",
        "Define deep learning",
        "What is NLP?",
        "Tell me about transformers",
    ]

    async def send_request(prompt):
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 30,
                },
            )
            return response.json()

    start = time.time()
    results = await asyncio.gather(*[send_request(p) for p in prompts])
    elapsed = time.time() - start

    print(f"Sent {len(results)} concurrent requests in {elapsed:.2f}s")
    total_tokens = sum(r['num_generated_tokens'] for r in results)
    print(f"Total tokens generated: {total_tokens}")
    print(f"Effective throughput: {total_tokens / elapsed:.1f} tokens/sec")


async def main():
    """Run all demos"""
    print("=" * 50)
    print("Custom LLM Inference Server - End-to-End Demo")
    print("=" * 50)

    try:
        await health_and_metrics()
        await single_request_demo()
        await batch_requests_demo()
        await concurrent_requests_demo()
        # Streaming demo commented out for now (requires different client handling)
        # await streaming_demo()
        await health_and_metrics()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the server is running: python server.py --model-path gpt2")


if __name__ == "__main__":
    asyncio.run(main())
