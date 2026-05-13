"""
a2a_test_client.py  —  End-to-end test client for RAPHI A2A Server

Discovers the RAPHI agent, then sends test queries covering each skill.

Run:
    1. Start the A2A server:  .venv/bin/python -m backend.a2a_server
    2. Run this client:       .venv/bin/python -m backend.a2a_test_client
"""

import asyncio
import json
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)

BASE_URL = "http://localhost:9999"

TEST_QUERIES = [
    ("Market Intelligence", "What's the current market overview?"),
    ("Stock Detail", "Get me stock details and fundamentals for NVDA"),
    ("SEC Research", "Search SEC filings for Microsoft"),
    ("ML Signal", "Generate an ML trading signal for AAPL"),
    ("Portfolio", "Show my portfolio snapshot with risk metrics"),
    ("Investment Memo", "Give me a brief investment analysis for TSLA"),
]


async def main():
    async with httpx.AsyncClient() as httpx_client:
        # Discover the agent
        print(f"Connecting to RAPHI A2A Server at {BASE_URL}...")
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=BASE_URL,
        )
        card = await resolver.get_agent_card()

        print(f"\nAgent: {card.name}")
        print(f"Description: {card.description}")
        print(f"Version: {card.version}")
        print(f"Skills: {[s.name for s in card.skills]}")
        print(f"Streaming: {card.capabilities.streaming}")
        print("=" * 70)

        # Initialize client
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=card,
        )

        # Send test queries
        for skill_name, query in TEST_QUERIES:
            print(f"\n{'─' * 70}")
            print(f"Skill: {skill_name}")
            print(f"Query: {query}")
            print(f"{'─' * 70}")

            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        "role": "user",
                        "parts": [{"kind": "text", "text": query}],
                        "messageId": uuid4().hex,
                    },
                ),
            )

            try:
                response = await client.send_message(request)
                result = response.model_dump(mode="json", exclude_none=True)

                # Extract response text
                if "result" in result:
                    r = result["result"]
                    # Handle different response shapes
                    if isinstance(r, dict):
                        # Look for message parts in the result
                        parts = None
                        if "status" in r and isinstance(r["status"], dict):
                            msg = r["status"].get("message")
                            if msg and isinstance(msg, dict):
                                parts = msg.get("parts", [])
                        if not parts and "artifacts" in r:
                            for artifact in r.get("artifacts", []):
                                parts = artifact.get("parts", [])
                                if parts:
                                    break

                        if parts:
                            for part in parts:
                                if isinstance(part, dict) and part.get("kind") == "text":
                                    text = part["text"]
                                    # Truncate for display
                                    if len(text) > 500:
                                        print(f"Response: {text[:500]}...")
                                    else:
                                        print(f"Response: {text}")
                        else:
                            print(f"Raw result: {json.dumps(r, indent=2)[:500]}")
                    else:
                        print(f"Result: {r}")
                else:
                    print(f"Full response: {json.dumps(result, indent=2)[:500]}")

            except Exception as e:
                print(f"Error: {e}")

        print(f"\n{'=' * 70}")
        print("Test complete.")


if __name__ == "__main__":
    asyncio.run(main())
