import asyncio
import aiohttp
import json

AIRA_HUB = "http://localhost:8015"
AGENT_URL = "http://localhost:8094/"
AGENT_NAME = "MemoryAgent"
CAPABILITIES = "mcp,a2a"

init_payload = {
    "url": AGENT_URL,
    "description": "Memory + Semantic Recall Agent",
    "skills": [
        {
            "id": "semantic-recall",
            "name": "Semantic Recall",
            "description": "Retrieves and reformulates memory based on latent intent",
            "tags": ["memory", "recall"]
        }
    ],
    "shared_resources": [
        {
            "uri": f"{AGENT_URL}/tools/recall",
            "description": "Recall semantic memory",
            "type": "mcp_tool",
            "version": "1.0.0",
            "metadata": {"example": "memory retrieval"}
        }
    ],
    "tags": ["streamed", "cognition"],
    "category": "memory"
}


async def connect_and_send_init():
    stream_url = f"{AIRA_HUB}/connect/stream?agent_url={AGENT_URL}&name={AGENT_NAME}&aira_capabilities={CAPABILITIES}"

    async with aiohttp.ClientSession() as session:
        # Open SSE stream
        async with session.get(stream_url) as resp:
            if resp.status != 200:
                print(f" Failed to connect to SSE stream: {resp.status}")
                return

            print(" Connected to AIRA via SSE. Sending init payload...")
            await asyncio.sleep(1)  # wait a bit to ensure SSE is open

            # Send init info via POST
            init_url = f"{AIRA_HUB}/connect/stream/init"
            async with session.post(init_url, json=init_payload) as post_resp:
                if post_resp.status == 200:
                    print(" Init payload sent!")
                else:
                    print(f"Failed to send init: {post_resp.status}")
                    return

            # Handle SSE messages (heartbeat)
            async for line in resp.content:
                if not line:
                    continue

                decoded = line.decode().strip()

                if not decoded or not decoded.startswith("data: "):
                    continue  # Ignore keep-alive or blank lines

                try:
                    payload = json.loads(decoded[6:])
                    print(f"[{payload['agent']}] Status: {payload['status']} @ {payload['ts']}")
                except json.JSONDecodeError:
                    print(f"️ Invalid JSON received: {decoded}")


if __name__ == "__main__":
    asyncio.run(connect_and_send_init())
