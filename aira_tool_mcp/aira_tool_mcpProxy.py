import asyncio
import aiohttp
import json
import argparse
import sys
import os
from typing import Dict, List, Any, Optional


class MCPtoA2AAdapter:
    """Adapter that bridges between MCP servers and A2A protocol"""

    def __init__(self, hub_url: str, adapter_url: str, server_url: str):
        """Initialize the adapter with hub, adapter URL and target MCP server URL"""
        self.hub_url = hub_url.rstrip('/')
        self.adapter_url = adapter_url.rstrip('/')
        self.server_url = server_url.rstrip('/')
        self.session = None
        self.server_name = None
        self.tools = []

    async def __aenter__(self):
        """Set up async context"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context"""
        if self.session:
            await self.session.close()

    async def initialize(self):
        """Initialize the adapter by discovering the MCP server and registering with AIRA Hub"""
        # Connect to MCP server and discover tools
        await self.discover_mcp_server()

        # Register with AIRA Hub
        await self.register_with_hub()

    async def discover_mcp_server(self):
        """Connect to MCP server and discover its tools"""
        try:
            # Try to get server info
            async with self.session.post(
                    f"{self.server_url}/initialize",
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
            ) as resp:
                if resp.status == 200:
                    server_info = await resp.json()
                    self.server_name = server_info.get("result", {}).get("server_name", "Unknown MCP Server")
                    print(f"✅ Connected to MCP server: {self.server_name}")
                else:
                    self.server_name = os.path.basename(self.server_url)
                    print(f"⚠️ Server didn't respond to initialization: {resp.status}")

            # Try to list tools
            async with self.session.post(
                    f"{self.server_url}/list_tools",
                    json={"jsonrpc": "2.0", "id": 2, "method": "list_tools"}
            ) as resp:
                if resp.status == 200:
                    tools_response = await resp.json()
                    self.tools = tools_response.get("result", [])
                    print(f"✅ Found {len(self.tools)} tools on server")
                    for tool in self.tools:
                        print(f"  - {tool.get('name')}: {tool.get('description')}")
                else:
                    print(f"⚠️ Failed to list tools: {resp.status}")

        except Exception as e:
            print(f"❌ Error discovering MCP server: {str(e)}")
            raise RuntimeError(f"Failed to connect to MCP server: {str(e)}")

    async def register_with_hub(self):
        """Register this adapter as an agent with AIRA Hub"""
        try:
            # Convert MCP tools to A2A skills
            skills = []
            for tool in self.tools:
                skill_id = f"tool-{tool.get('name', 'unknown')}"
                skill_name = tool.get("name", "Unknown Tool")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})

                skills.append({
                    "id": skill_id,
                    "name": skill_name,
                    "description": description,
                    "tags": ["mcp", "tool"],
                    "parameters": parameters
                })

            # Create registration payload
            payload = {
                "url": self.adapter_url,
                "name": f"{self.server_name} Adapter",
                "description": f"MCP-to-A2A adapter for {self.server_name}",
                "skills": skills,
                "shared_resources": [],
                "aira_capabilities": ["mcp", "a2a"],
                "auth": {},
                "tags": ["mcp", "adapter"]
            }

            # Register with hub
            async with self.session.post(f"{self.hub_url}/register", json=payload) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    print(f"✅ Registered with AIRA Hub: {result}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"❌ Registration failed: {error_text}")
                    return False

        except Exception as e:
            print(f"❌ Error registering with hub: {str(e)}")
            return False

    async def handle_a2a_request(self, a2a_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle A2A protocol request by translating to MCP and forwarding to MCP server"""
        try:
            method = a2a_request.get("method")

            if method == "tasks/send":
                # Handle the A2A task/send request
                params = a2a_request.get("params", {})
                task_id = params.get("id")
                message = params.get("message", {})

                # Extract text from the user message
                text = ""
                for part in message.get("parts", []):
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        break

                # Parse the message to identify tool and parameters
                tool_name = None
                tool_params = {}

                # Try to extract tool name and parameters from the message
                for tool in self.tools:
                    current_tool_name = tool.get("name")
                    if current_tool_name.lower() in text.lower():
                        tool_name = current_tool_name

                        # Try to parse parameters from JSON if available
                        try:
                            # Look for parameters as JSON
                            json_start = text.find('{')
                            if json_start != -1:
                                json_part = text[json_start:]
                                # Parse the parameters
                                tool_params = json.loads(json_part)
                                print(f"📦 Extracted parameters: {tool_params}")
                        except json.JSONDecodeError:
                            # Basic parameter extraction
                            props = tool.get("parameters", {}).get("properties", {})
                            for param_name in props:
                                param_match = text.lower().find(param_name.lower())
                                if param_match != -1:
                                    # Try to find value after param name
                                    value_start = param_match + len(param_name)
                                    value_end = text.find(" ", value_start)
                                    if value_end == -1:
                                        value_end = len(text)

                                    param_value = text[value_start:value_end].strip(": ")
                                    if param_value:
                                        tool_params[param_name] = param_value

                        break

                if not tool_name:
                    # Return help message
                    return {
                        "jsonrpc": "2.0",
                        "id": a2a_request.get("id"),
                        "result": {
                            "id": task_id,
                            "sessionId": params.get("sessionId", "session-" + str(task_id)),
                            "status": {"state": "completed"},
                            "artifacts": [{
                                "parts": [{
                                    "type": "text",
                                    "text": f"I'm not sure which tool you want to use. Available tools: {', '.join(t.get('name') for t in self.tools)}"
                                }]
                            }]
                        }
                    }

                # Forward to MCP server
                mcp_request = {
                    "jsonrpc": "2.0",
                    "id": a2a_request.get("id"),
                    "method": "call_tool",
                    "params": {
                        "name": tool_name,
                        "arguments": tool_params
                    }
                }

                print(f"🔄 Forwarding to MCP server: {json.dumps(mcp_request)}")

                async with self.session.post(f"{self.server_url}/call_tool", json=mcp_request) as resp:
                    if resp.status == 200:
                        mcp_result = await resp.json()

                        # Convert MCP result to A2A response
                        result_content = mcp_result.get("result", [])
                        if not isinstance(result_content, list):
                            result_content = [result_content]

                        # Extract text content
                        result_text = ""
                        for content in result_content:
                            if content.get("type") == "text":
                                result_text += content.get("text", "")

                        return {
                            "jsonrpc": "2.0",
                            "id": a2a_request.get("id"),
                            "result": {
                                "id": task_id,
                                "sessionId": params.get("sessionId", "session-" + str(task_id)),
                                "status": {"state": "completed"},
                                "artifacts": [{
                                    "parts": [{
                                        "type": "text",
                                        "text": result_text
                                    }]
                                }]
                            }
                        }
                    else:
                        error_text = await resp.text()
                        print(f"❌ MCP server error: {error_text}")

                        return {
                            "jsonrpc": "2.0",
                            "id": a2a_request.get("id"),
                            "error": {
                                "code": -32000,
                                "message": f"Error from MCP server: {error_text}"
                            }
                        }

            elif method == "tasks/get":
                # Handle task/get - not implementing full history
                task_id = a2a_request.get("params", {}).get("id")

                return {
                    "jsonrpc": "2.0",
                    "id": a2a_request.get("id"),
                    "result": {
                        "id": task_id,
                        "status": {"state": "completed"},
                        "artifacts": [],
                        "history": []
                    }
                }

            else:
                # Method not supported
                return {
                    "jsonrpc": "2.0",
                    "id": a2a_request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method {method} not supported"
                    }
                }

        except Exception as e:
            print(f"❌ Error handling A2A request: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": a2a_request.get("id"),
                "error": {
                    "code": -32000,
                    "message": f"Error: {str(e)}"
                }
            }


async def run_adapter_server(hub_url: str, adapter_host: str, adapter_port: int, mcp_server_url: str):
    """Run the adapter server with web endpoints"""
    from aiohttp import web

    # Create and initialize the adapter
    adapter = await MCPtoA2AAdapter(hub_url, f"http://{adapter_host}:{adapter_port}", mcp_server_url).__aenter__()
    await adapter.initialize()

    # Create web app
    app = web.Application()

    # Define routes
    async def a2a_handler(request):
        """Handle A2A protocol requests"""
        try:
            body = await request.text()
            a2a_request = json.loads(body)
            response = await adapter.handle_a2a_request(a2a_request)
            return web.json_response(response)
        except Exception as e:
            return web.json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }, status=500)

    async def index_handler(request):
        """Handle root endpoint - show info"""
        return web.json_response({
            "name": f"{adapter.server_name} Adapter",
            "description": f"MCP-to-A2A adapter for {adapter.server_url}",
            "tools": [t.get("name") for t in adapter.tools]
        })

    async def agent_card_handler(request):
        """Handle /.well-known/agent.json for A2A discovery"""
        skills = []
        for tool in adapter.tools:
            skill_id = f"tool-{tool.get('name', 'unknown')}"
            skill_name = tool.get("name", "Unknown Tool")
            description = tool.get("description", "")

            skills.append({
                "id": skill_id,
                "name": skill_name,
                "description": description,
                "tags": ["mcp", "tool"]
            })

        return web.json_response({
            "name": f"{adapter.server_name} Adapter",
            "description": f"MCP-to-A2A adapter for {adapter.server_name}",
            "url": adapter.adapter_url,
            "skills": skills
        })

    # Add routes
    app.router.add_post('/a2a', a2a_handler)
    app.router.add_get('/', index_handler)
    app.router.add_get('/.well-known/agent.json', agent_card_handler)

    # Run the server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, adapter_host, adapter_port)

    print(f"🚀 Starting adapter server at http://{adapter_host}:{adapter_port}")
    await site.start()

    try:
        while True:
            await asyncio.sleep(3600)  # Keep alive
    finally:
        # Cleanup
        await adapter.__aexit__(None, None, None)
        await runner.cleanup()


def main():
    """Command line interface for MCP-to-A2A adapter"""
    parser = argparse.ArgumentParser(description="MCP-to-A2A adapter for AIRA Hub")
    parser.add_argument("--hub", default="https://aira-fl8f.onrender.com", help="AIRA Hub URL")
    parser.add_argument("--host", default="localhost", help="Adapter host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Adapter port to bind to")
    parser.add_argument("--server", required=True, help="MCP Server URL to proxy")

    args = parser.parse_args()

    try:
        asyncio.run(run_adapter_server(args.hub, args.host, args.port, args.server))
    except KeyboardInterrupt:
        print("Server stopped")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()