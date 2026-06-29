"""Example: Single request inference"""

import asyncio
import httpx


async def main():
    """Run single inference request"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "What is machine learning?",
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        print("Response:")
        print(response.json())


if __name__ == "__main__":
    asyncio.run(main())
