import asyncio
import aiohttp
import json
import os


class SimpleA2AAgent:
    def __init__(self, hub_url, agent_url, agent_name):
        self.hub_url = hub_url
        self.agent_url = agent_url
        self.agent_name = agent_name
        self.session = None
        self.tools = {}

    async def __aenter__(self):
        """Async context manager entry point"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point - properly close session"""
        if self.session:
            await self.session.close()

    def add_tool(self, name, function):
        """Add a tool that can be invoked by other agents"""
        self.tools[name] = function

    async def register_with_hub(self):
        """Register the agent with the AIRA Hub"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")

        payload = {
            "url": self.agent_url,
            "name": self.agent_name,
            "skills": [
                {
                    "id": name,
                    "name": name,
                    "description": f"Tool: {name}",
                    "tags": ["tool"]
                } for name in self.tools.keys()
            ],
            "aira_capabilities": ["a2a"]
        }

        async with self.session.post(f"{self.hub_url}/register", json=payload) as resp:
            if resp.status == 201:
                print(f"✅ Registered {self.agent_name} with hub")
                result = await resp.json()
                return result
            else:
                error_text = await resp.text()
                print(f"❌ Registration failed: {error_text}")
                return None

    async def handle_a2a_request(self, request):
        """Handle incoming A2A protocol requests"""
        method = request.get("method")
        if method == "tasks/send":
            return await self._handle_tool_invocation(request)
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32601, "message": "Method not supported"}
        }

    async def _handle_tool_invocation(self, request):
        """Process tool invocation requests"""
        params = request.get("params", {})
        message = params.get("message", {})
        text = next((p.get("text", "") for p in message.get("parts", [])
                     if p.get("type") == "text"), "")

        # Extract tool name and parameters
        tool_name = None
        for name in self.tools.keys():
            if name.lower() in text.lower():
                tool_name = name
                break

        if not tool_name:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": "Tool not found"}
            }

        # Extract parameters (simplified)
        try:
            tool_function = self.tools[tool_name]
            result = tool_function({"text": text})

            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "artifacts": [{
                        "parts": [{
                            "type": "text",
                            "text": json.dumps(result)
                        }]
                    }]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": str(e)}
            }


async def main():
    # Configuration from environment or defaults
    hub_url = os.environ.get("AIRA_HUB_URL", "https://aira-fl8f.onrender.com")
    agent_url = os.environ.get("AGENT_URL", "http://localhost:8088")

    # Use async context manager to ensure proper session handling
    async with SimpleA2AAgent(hub_url, agent_url, "EchoAgent") as agent:
        # Add a simple tool
        def echo_tool(params):
            return {"echo": params.get("text", "")}

        agent.add_tool("echo", echo_tool)

        # Register with hub
        await agent.register_with_hub()

        # Keep the script running (in a real implementation,
        # you'd set up a web server or long-running process)
        print("Agent running. Press Ctrl+C to exit.")
        await asyncio.get_event_loop().create_future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped.")
