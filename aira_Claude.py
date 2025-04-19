#!/usr/bin/env python3
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aira-connector")

# Initialize FastMCP server
mcp = FastMCP("aira-connector")

# AIRA Hub configuration
AIRA_HUB_URL = os.environ.get("AIRA_HUB_URL", "http://localhost:8015")
USERNAME = os.environ.get("AIRA_USERNAME", "admin")
PASSWORD = os.environ.get("AIRA_PASSWORD", "password123")

# Store the access token
access_token = None


async def get_access_token() -> str:
    """Get an access token from AIRA Hub"""
    global access_token
    if access_token:
        return access_token

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{AIRA_HUB_URL}/token",
                data={"username": USERNAME, "password": PASSWORD},
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if resp.status_code == 200:
                token_data = resp.json()
                access_token = token_data.get("access_token")
                return access_token
            else:
                logger.error(f"Failed to get token: {resp.status_code} {resp.text}")
                return ""
        except Exception as e:
            logger.error(f"Error getting token: {e}")
            return ""


async def get_all_tools() -> List[Dict[str, Any]]:
    """Get all available tools from AIRA Hub"""
    token = await get_access_token()
    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{AIRA_HUB_URL}/mcp/tools", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("tools", [])
            else:
                logger.error(f"Failed to get tools: {resp.status_code} {resp.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []


@mcp.tool()
async def list_aira_tools() -> str:
    """List all available tools from AIRA Hub"""
    tools = await get_all_tools()

    if not tools:
        return "No tools available from AIRA Hub. Make sure the hub is running and your credentials are correct."

    result = "Available tools from AIRA Hub:\n\n"
    for tool in tools:
        agent_name = tool.get("agent_name", "Unknown")
        tool_info = tool.get("tool", {})
        tool_name = tool_info.get("name", "Unknown")
        description = tool_info.get("description", "No description available")

        result += f"Tool: {tool_name}\n"
        result += f"Agent: {agent_name}\n"
        result += f"Description: {description}\n"
        result += "---\n"

    return result


@mcp.tool()
async def invoke_aira_tool(tool_name: str, agent_url: str, arguments: str) -> str:
    """Invoke a tool from AIRA Hub

    Args:
        tool_name: Name of the tool to invoke
        agent_url: URL of the agent that provides the tool
        arguments: Arguments for the tool as a JSON string
    """
    token = await get_access_token()
    if not token:
        return "Failed to authenticate with AIRA Hub"

    headers = {"Authorization": f"Bearer {token}"}

    try:
        # Parse arguments from string
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return "Invalid arguments JSON format"

    request_data = {
        "agent_url": agent_url,
        "tool_name": tool_name,
        "arguments": args
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{AIRA_HUB_URL}/mcp/invoke-tool",
                json=request_data,
                headers=headers
            )

            if resp.status_code == 200:
                result = resp.json()
                if result.get("success"):
                    content_list = result.get("content", [])
                    response_text = ""
                    for item in content_list:
                        if item.get("type") == "text":
                            response_text += item.get("text", "") + "\n"
                    return response_text
                else:
                    return f"Tool invocation failed: {result.get('message', 'Unknown error')}"
            else:
                return f"Request failed: {resp.status_code} {resp.text}"
        except Exception as e:
            return f"Error invoking tool: {e}"


@mcp.tool()
async def search_tools(query: str) -> str:
    """Search for tools in AIRA Hub that match a query

    Args:
        query: Search query to find tools
    """
    tools = await get_all_tools()

    if not tools:
        return "No tools available from AIRA Hub"

    query = query.lower()
    matching_tools = []

    for tool in tools:
        tool_info = tool.get("tool", {})
        tool_name = tool_info.get("name", "").lower()
        description = tool_info.get("description", "").lower()
        agent_name = tool.get("agent_name", "").lower()

        if query in tool_name or query in description or query in agent_name:
            matching_tools.append(tool)

    if not matching_tools:
        return f"No tools found matching '{query}'"

    result = f"Found {len(matching_tools)} tools matching '{query}':\n\n"
    for tool in matching_tools:
        agent_name = tool.get("agent_name", "Unknown")
        agent_url = tool.get("agent_url", "Unknown")
        tool_info = tool.get("tool", {})
        tool_name = tool_info.get("name", "Unknown")
        description = tool_info.get("description", "No description available")

        result += f"Tool: {tool_name}\n"
        result += f"Agent: {agent_name}\n"
        result += f"Agent URL: {agent_url}\n"
        result += f"Description: {description}\n"
        result += "---\n"

    return result


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
