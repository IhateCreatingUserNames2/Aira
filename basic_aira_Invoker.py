import asyncio
import aiohttp
import json
import os
import logging


class DynamicA2AAgent:
    def __init__(self, hub_url, agent_url, agent_name):
        self.hub_url = hub_url
        self.agent_url = agent_url
        self.agent_name = agent_name
        self.session = None
        self.discovered_tools = {}

    async def __aenter__(self):
        """Async context manager entry point"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point"""
        if self.session:
            await self.session.close()

    async def discover_tools(self):
        """Discover tools from the AIRA Hub"""
        try:
            # Try agents endpoint
            async with self.session.get(f"{self.hub_url}/agents") as resp:
                if resp.status == 200:
                    agents = await resp.json()
                    print(f"Agents found: {len(agents)}")

                    # Collect tools from agents
                    tools = []
                    for agent in agents:
                        agent_skills = agent.get('skills', [])
                        for skill in agent_skills:
                            # Add more robust skill parsing
                            tool = {
                                'resource': {
                                    'name': skill.get('name'),
                                    'description': skill.get('description'),
                                    'id': skill.get('id'),
                                    'type': 'a2a_skill'
                                },
                                'agent': {
                                    'name': agent.get('name'),
                                    'url': agent.get('url')
                                }
                            }
                            tools.append(tool)

                    # Store discovered tools
                    for tool in tools:
                        tool_name = tool['resource'].get('name')
                        self.discovered_tools[tool_name] = tool

                    print(f"🔍 Discovered {len(self.discovered_tools)} tools:")
                    for name, tool in self.discovered_tools.items():
                        print(f"- {name}: {tool['resource'].get('description', 'No description')}")

                    return self.discovered_tools

        except Exception as e:
            print(f"❌ Tool discovery error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    async def invoke_tool(self, tool_name, parameters=None):
        """Invoke a discovered tool"""
        if tool_name not in self.discovered_tools:
            print(f"❌ Tool '{tool_name}' not found")
            return None

        tool = self.discovered_tools[tool_name]
        agent_url = tool.get('agent', {}).get('url')

        if not agent_url:
            print(f"❌ No agent URL found for tool '{tool_name}'")
            return None

        # Construct A2A protocol request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "id": f"task-{tool_name}",
                "message": {
                    "role": "user",
                    "parts": [{
                        "type": "text",
                        "text": f"Use the {tool_name} tool with parameters: {json.dumps(parameters or {})}"
                    }]
                }
            }
        }

        try:
            # Append /a2a to the agent URL if not already present
            if not agent_url.endswith('/a2a'):
                agent_url = f"{agent_url.rstrip('/')}/a2a"

            async with self.session.post(agent_url, json=request) as resp:
                if resp.status == 200:
                    response = await resp.json()

                    # Extract result from A2A response
                    if 'result' in response and 'artifacts' in response['result']:
                        artifacts = response['result']['artifacts']
                        if artifacts and 'parts' in artifacts[0]:
                            text_part = next((p for p in artifacts[0]['parts'] if p.get('type') == 'text'), None)
                            if text_part:
                                try:
                                    return json.loads(text_part['text'])
                                except json.JSONDecodeError:
                                    return text_part['text']

                    return response
                else:
                    print(f"❌ Tool invocation failed: {await resp.text()}")
                    return None
        except Exception as e:
            print(f"❌ Error invoking tool: {e}")
            return None
    async def check_hub_connectivity(self):
        """Check connectivity to the AIRA Hub"""
        try:
            async with self.session.get(f"{self.hub_url}/status") as resp:
                if resp.status == 200:
                    status = await resp.json()
                    print("Hub Status:")
                    print(json.dumps(status, indent=2))
                    return True
                else:
                    print(f"❌ Hub status check failed: {await resp.text()}")
                    return False
        except Exception as e:
            print(f"❌ Connectivity error: {e}")
            return False


async def main():
    hub_url = os.environ.get("AIRA_HUB_URL", "https://aira-fl8f.onrender.com")
    agent_url = os.environ.get("AGENT_URL", "http://localhost:8088")

    print(f"Connecting to Hub: {hub_url}")

    async with DynamicA2AAgent(hub_url, agent_url, "DynamicToolConsumer") as agent:
        # Check connectivity first
        if not await agent.check_hub_connectivity():
            print("❌ Unable to connect to AIRA Hub")
            return

        # Discover tools
        tools = await agent.discover_tools()

        # Attempt to invoke a tool if any are discovered
        if tools:
            # You might want to prompt user or choose a tool dynamically
            first_tool_name = list(tools.keys())[0]
            print(f"\n🚀 Attempting to invoke first discovered tool: {first_tool_name}")

            result = await agent.invoke_tool(first_tool_name, {"city": "London"})
            print(f"🔧 Tool Invocation Result: {result}")
        else:
            print("❌ No tools discovered")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped.")