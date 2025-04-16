#!/usr/bin/env python3
"""
AIRA MCP Server Manager - Command Line Tool

This tool helps manage remote MCP servers with AIRA Hub, allowing you to:
- Register remote MCP servers with AIRA Hub
- List registered MCP servers
- Start an adapter server for a specific MCP server
- Run a simple web UI for managing servers
"""

import argparse
import asyncio
import json
import os
import requests
import sys
import subprocess
from typing import Dict, List, Any, Optional

from aira_tool_mcpProxy import run_adapter_server
from aira_tool_mcpSA import register_mcp_server


# Import implementation from previous parts
# MCPServerRegistration class, MCPtoA2AAdapter class, etc.

def register_command(args):
    """Register a remote MCP server with AIRA Hub"""
    try:
        # Run the registration process
        result = asyncio.run(register_mcp_server(args.hub, args.server))
        if not result:
            print("❌ Registration failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def list_command(args):
    """List registered servers in AIRA Hub"""
    try:
        # Query AIRA Hub for agents
        response = requests.get(f"{args.hub}/agents")

        if response.status_code != 200:
            print(f"❌ Failed to retrieve agents: {response.text}")
            sys.exit(1)

        agents = response.json()

        # Filter for MCP servers
        mcp_servers = []
        for agent in agents:
            tags = agent.get("tags", [])
            if "mcp" in tags:
                mcp_servers.append(agent)

        if not mcp_servers:
            print("No MCP servers registered with this hub")
            sys.exit(0)

        print(f"Found {len(mcp_servers)} MCP servers:")
        for i, server in enumerate(mcp_servers, 1):
            print(f"\n{i}. {server.get('name', 'Unknown Server')}")
            print(f"   URL: {server.get('url', 'No URL')}")
            print(f"   Tags: {', '.join(server.get('tags', []))}")

            skills = server.get("skills", [])
            if skills:
                print(f"   Tools ({len(skills)}):")
                for skill in skills[:5]:  # Show just the first 5
                    print(f"     - {skill.get('name')}: {skill.get('description', '')}")

                if len(skills) > 5:
                    print(f"     ... and {len(skills) - 5} more")
            else:
                print("   No tools available")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def adapter_command(args):
    """Start an adapter server for a specific MCP server"""
    try:
        # Run the adapter server
        asyncio.run(run_adapter_server(args.hub, args.host, args.port, args.server))
    except KeyboardInterrupt:
        print("\nAdapter server stopped")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def webui_command(args):
    """Start the web UI for managing remote MCP servers"""
    try:
        # Set environment variables
        os.environ['FLASK_APP'] = 'app.py'
        os.environ['FLASK_ENV'] = 'development'
        os.environ['PORT'] = str(args.port)

        # Run the Flask app
        subprocess.run(['python', '-m', 'flask', 'run', '--host', args.host, '--port', str(args.port)])
    except KeyboardInterrupt:
        print("\nWeb UI stopped")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the command-line tool"""
    parser = argparse.ArgumentParser(
        description="AIRA MCP Server Manager - Manage remote MCP servers with AIRA Hub"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a remote MCP server with AIRA Hub")
    register_parser.add_argument("--hub", default="https://aira-fl8f.onrender.com", help="AIRA Hub URL")
    register_parser.add_argument("--server", required=True, help="MCP server URL to register")
    register_parser.set_defaults(func=register_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List registered MCP servers")
    list_parser.add_argument("--hub", default="https://aira-fl8f.onrender.com", help="AIRA Hub URL")
    list_parser.set_defaults(func=list_command)

    # Adapter command
    adapter_parser = subparsers.add_parser("adapter", help="Start an adapter server for a specific MCP server")
    adapter_parser.add_argument("--hub", default="https://aira-fl8f.onrender.com", help="AIRA Hub URL")
    adapter_parser.add_argument("--host", default="localhost", help="Host to bind the adapter server to")
    adapter_parser.add_argument("--port", type=int, default=8080, help="Port to bind the adapter server to")
    adapter_parser.add_argument("--server", required=True, help="MCP server URL to adapt")
    adapter_parser.set_defaults(func=adapter_command)

    # Web UI command
    webui_parser = subparsers.add_parser("webui", help="Start the web UI for managing remote MCP servers")
    webui_parser.add_argument("--host", default="localhost", help="Host to bind the web UI to")
    webui_parser.add_argument("--port", type=int, default=5000, help="Port to bind the web UI to")
    webui_parser.set_defaults(func=webui_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()