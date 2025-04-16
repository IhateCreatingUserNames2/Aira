import asyncio
import aiohttp
import json
import argparse
import sys
import os
from typing import Dict, List, Any, Optional


class MCPServerRegistration:
    """Tool to register remote MCP servers with AIRA Hub"""

    def __init__(self, hub_url: str):
        """Initialize with AIRA Hub URL"""
        self.hub_url = hub_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        """Set up async context"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context"""
        if self.session:
            await self.session.close()

    async def discover_mcp_server(self, mcp_server_url: str) -> Dict[str, Any]:
        """Discover tools and capabilities from an MCP server"""
        try:
            # First try to get the server info
            server_url = mcp_server_url.rstrip('/')

            # Try to get initialization data
            async with self.session.post(
                    f"{server_url}/initialize",
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
            ) as resp:
                if resp.status == 200:
                    server_info = await resp.json()
                    print(f"✅ Connected to MCP server: {server_url}")
                else:
                    print(f"⚠️ Server didn't respond to initialization: {resp.status}")
                    server_info = {"name": os.path.basename(server_url)}

            # Try to list tools
            async with self.session.post(
                    f"{server_url}/list_tools",
                    json={"jsonrpc": "2.0", "id": 2, "method": "list_tools"}
            ) as resp:
                if resp.status == 200:
                    tools_response = await resp.json()
                    tools = tools_response.get("result", [])
                    print(f"✅ Found {len(tools)} tools on server")
                else:
                    tools = []
                    print(f"⚠️ Failed to list tools: {resp.status}")

            # Extract server name from info
            server_name = server_info.get("result", {}).get("server_name", "Unknown MCP Server")

            return {
                "url": server_url,
                "name": server_name,
                "tools": tools
            }

        except Exception as e:
            print(f"❌ Error discovering MCP server: {str(e)}")
            return {"url": mcp_server_url, "name": "Unknown MCP Server", "tools": []}

    async def register_with_aira_hub(self, server_data: Dict[str, Any]) -> bool:
        """Register MCP server with AIRA Hub as an agent"""
        try:
            server_url = server_data["url"]

            # Convert MCP tools to A2A skills
            skills = []
            for tool in server_data.get("tools", []):
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
                "url": server_url,
                "name": f"{server_data.get('name', 'MCP Server')}",
                "description": f"MCP Server exposed through AIRA Hub",
                "skills": skills,
                "shared_resources": [],
                "aira_capabilities": ["mcp", "a2a"],
                "auth": {},
                "tags": ["mcp", "remote-server"]
            }

            # Register with hub
            async with self.session.post(f"{self.hub_url}/register", json=payload) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    print(f"✅ Registered MCP server with AIRA Hub: {result}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"❌ Registration failed: {error_text}")
                    return False

        except Exception as e:
            print(f"❌ Error registering with hub: {str(e)}")
            return False


async def register_mcp_server(hub_url: str, mcp_server_url: str) -> bool:
    """Register an MCP server with AIRA Hub"""
    async with MCPServerRegistration(hub_url) as registrar:
        server_data = await registrar.discover_mcp_server(mcp_server_url)
        if server_data:
            return await registrar.register_with_aira_hub(server_data)
    return False


def main():
    """Command line interface for MCP server registration"""
    parser = argparse.ArgumentParser(description="Register remote MCP servers with AIRA Hub")
    parser.add_argument("--hub", default="https://aira-fl8f.onrender.com", help="AIRA Hub URL")
    parser.add_argument("--server", required=True, help="MCP Server URL to register")

    args = parser.parse_args()

    try:
        result = asyncio.run(register_mcp_server(args.hub, args.server))
        if result:
            print(f"✅ Successfully registered MCP server: {args.server}")
            sys.exit(0)
        else:
            print(f"❌ Failed to register MCP server")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()