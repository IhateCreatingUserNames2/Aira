import asyncio
import aiohttp
import json


class SimpleA2AAgent:
    def __init__(self, hub_url, agent_url, agent_name):
        self.hub_url = hub_url
        self.agent_url = agent_url
        self.agent_name = agent_name
        self.session = aiohttp.ClientSession()
        self.tools = {}

    def add_tool(self, name, function):
        """Add a tool that can be invoked by other agents"""
        self.tools[name] = function

    async def register_with_hub(self):
        """Register the agent with the AIRA Hub"""
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
            else:
                print(f"❌ Registration failed: {await resp.text()}")

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
    # Example usage
    hub_url = "https://aira-fl8f.onrender.com"
    agent_url = "http://localhost:8088"

    agent = SimpleA2AAgent(hub_url, agent_url, "EchoAgent")

    # Add a simple tool
    def echo_tool(params):
        return {"echo": params.get("text", "")}

    agent.add_tool("echo", echo_tool)

    # Register with hub
    await agent.register_with_hub()


if __name__ == "__main__":
    asyncio.run(main())