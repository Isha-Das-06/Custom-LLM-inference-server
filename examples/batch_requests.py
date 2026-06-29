"""Example: Batch inference requests"""

import asyncio
import httpx


async def main():
    """Run batch inference"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/batch-generate",
            json={
                "requests": [
                    {
                        "prompt": "What is machine learning?",
                        "max_new_tokens": 50,
                    },
                    {
                        "prompt": "Tell me about deep learning",
                        "max_new_tokens": 75,
                    },
                    {
                        "prompt": "Explain neural networks",
                        "max_new_tokens": 60,
                    },
                ]
            },
        )

        results = response.json()
        for i, result in enumerate(results["results"]):
            print(f"\nRequest {i + 1}:")
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated_text']}")
            print(f"Tokens: {result['num_generated_tokens']}")
            print(f"Latency: {result['latency_ms']:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
