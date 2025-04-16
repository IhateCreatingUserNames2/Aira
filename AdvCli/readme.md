# Enhanced MCP-A2A Bridge

![MCP-A2A Bridge](https://via.placeholder.com/800x200?text=MCP-A2A+Bridge)

A robust bidirectional bridge between the Model Context Protocol (MCP) and the Agent-to-Agent (A2A) protocol, enabling seamless interoperability between agents using different protocols.

## Key Features

1. **MCP to A2A Integration**: MCP servers are automatically exposed as A2A agents
2. **A2A to MCP Integration**: A2A tools can be accessed as MCP resources
3. **Streamlined Authentication**: Comprehensive authentication and permission management
4. **Streaming Responses**: Proper handling of streaming responses across protocols
5. **Built-in Web Server**: Easy deployment with integrated HTTP endpoints
6. **Client Adapters**: Convenient client adapters for both protocols

## Overview

The Enhanced MCP-A2A Bridge creates a seamless connection between two powerful agent protocols:

- **Model Context Protocol (MCP)**: A protocol designed to provide LLMs with access to tools, resources, and context
- **Agent-to-Agent Protocol (A2A)**: A protocol designed to facilitate communication between different AI agents

This bridge allows agents built with either protocol to communicate with and leverage capabilities from the other ecosystem, creating a unified agent network.

## Installation

```bash
# Basic installation
pip install mcp-a2a-bridge

# With web server support
pip install "mcp-a2a-bridge[web]"
```

## Basic Usage

```python
import asyncio
from mcp_a2a_bridge import create_bridge, McpTool

async def main():
    # Create and start the bridge
    bridge = create_bridge(
        name="My Bridge",
        description="Bridge between MCP and A2A",
        url="http://localhost:8000"
    )
    await bridge.start()
    
    try:
        # Define an MCP tool
        async def calculator(params):
            operation = params.get("operation", "add")
            a = params.get("a", 0)
            b = params.get("b", 0)
            
            if operation == "add":
                return {"result": a + b}
            elif operation == "subtract":
                return {"result": a - b}
            else:
                return {"error": f"Unknown operation: {operation}"}
        
        # Create MCP tool definition
        calculator_tool = McpTool(
            name="calculator",
            description="A simple calculator",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract"],
                        "description": "The operation to perform"
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }
        )
        
        # Expose the MCP tool as an A2A skill
        bridge.expose_mcp_tool_as_a2a_skill(calculator_tool, calculator)
        
        # Create a web server
        server = bridge.create_web_server()
        
        print("Bridge running at http://localhost:8000")
        print("Press Ctrl+C to stop")
        
        # Run the server
        await server.start_async()
        
    finally:
        await bridge.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Bridge Components

### EnhancedMcpA2ABridge

The core bridge class that manages both MCP and A2A protocol communication:

```python
from mcp_a2a_bridge import create_bridge

# Create a bridge
bridge = create_bridge(
    name="My Bridge",
    description="Bridge between MCP and A2A",
    url="http://localhost:8000"
)
```

### MCP Tools as A2A Skills

Expose MCP tools to A2A agents:

```python
from mcp_a2a_bridge import McpTool

# Define MCP tool
calculator_tool = McpTool(
    name="calculator",
    description="A simple calculator",
    parameters={...}
)

# Expose as A2A skill
bridge.expose_mcp_tool_as_a2a_skill(calculator_tool, my_calculator_function)
```

### A2A Skills as MCP Tools

Expose A2A skills to MCP clients:

```python
# Expose A2A skill as MCP tool
bridge.expose_a2a_skill_as_mcp_tool(
    skill_id="weather",
    name="get_weather",
    description="Get weather information",
    parameters={...},
    implementation=my_weather_function
)
```

### MCP Resources

Expose MCP resources and access them via A2A:

```python
from mcp_a2a_bridge import McpResource

# Create resource
docs_resource = McpResource(
    uri="docs://myapp/readme",
    description="Application documentation",
    mime_type="text/markdown"
)

# Expose resource
bridge.expose_mcp_resource_as_a2a_skill(docs_resource, get_docs_function)
```

### Authentication and Permission Management

Create and verify tokens with specific permission scopes:

```python
# Permission scopes
from mcp_a2a_bridge import PermissionScope

# Create a token with read permissions
read_token = bridge.mcp_server.auth_manager.create_token(
    scopes=[PermissionScope.READ],
    expires_in_days=1
)

# Create an admin token
admin_token = bridge.mcp_server.auth_manager.create_token(
    scopes=[PermissionScope.ADMIN],
    expires_in_days=7
)

# Validate token
is_valid = bridge.mcp_server.auth_manager.validate_token(token)

# Check permission
can_execute = bridge.mcp_server.auth_manager.check_permission(
    token, PermissionScope.EXECUTE
)
```

### Web Server

Create a web server to handle both MCP and A2A requests:

```python
# Create a web server
server = bridge.create_web_server(host="localhost", port=8000)

# Run the server (blocking)
server.run()

# Or run asynchronously
await server.start_async()
```

### Client Adapters

Connect to the bridge as either an MCP or A2A client:

```python
from mcp_a2a_bridge import connect_to_bridge_as_mcp, connect_to_bridge_as_a2a

# Connect as MCP client
mcp_client = await connect_to_bridge_as_mcp("http://localhost:8000")

# List tools
tools = await mcp_client.list_tools()

# Call a tool
result = await mcp_client.call_tool("calculator", {"operation": "add", "a": 5, "b": 3})

# Connect as A2A client
a2a_client = await connect_to_bridge_as_a2a("http://localhost:8000")

# Get agent card
agent_card = await a2a_client.get_agent_card()

# Send a task
response = await a2a_client.send_task("task-123", message)
```

## Streaming Support

Handle streaming responses from either protocol:

```python
# Streaming text generator
class StreamingTextGenerator:
    async def stream(self, params):
        prompt = params.get("prompt", "")
        length = params.get("length", 5)
        
        for i in range(length):
            yield f"Paragraph {i+1} about {prompt}..."
            await asyncio.sleep(0.5)  # Simulate generation delay

# Use with A2A client
async for event in a2a_client.send_streaming_task(task_id, message):
    # Process streaming events
    print(f"Received event: {event}")
```

## Server Endpoints

When running the web server, the following endpoints are available:

- **A2A Protocol**: `/a2a` - Handle A2A protocol requests
- **A2A Streaming**: `/a2a/stream/{task_id}` - Handle A2A streaming requests
- **MCP List Tools**: `/mcp/list_tools` - List available MCP tools
- **MCP Call Tool**: `/mcp/call_tool` - Call an MCP tool
- **MCP List Resources**: `/mcp/list_resources` - List available MCP resources
- **MCP Read Resource**: `/mcp/read_resource?uri={uri}` - Read an MCP resource
- **Agent Card**: `/.well-known/agent.json` - Get the A2A agent card

## Use Cases

### AI Orchestration Hub

Create a central hub where AI agents from different frameworks can discover and utilize each other's capabilities:

```python
# Create a bridge
bridge = create_bridge(name="AI Orchestration Hub", ...)

# Add tools and skills from various sources
bridge.expose_mcp_tool_as_a2a_skill(tool1, implementation1)
bridge.expose_a2a_skill_as_mcp_tool(skill_id="skill1", ...)

# Run the server for all agents to connect to
server = bridge.create_web_server()
server.run()
```

### Bridge Between Frameworks

Connect agents built with frameworks using different protocols:

```python
# Bridge Google ADK (using A2A) with MCP-based agents
from google.adk.tools.function_tool import FunctionTool

# Create ADK tools
adk_tool = FunctionTool(my_adk_function)

# Expose to MCP
bridge.expose_a2a_skill_as_mcp_tool(
    skill_id="adk-tool",
    name=adk_tool.name,
    description=adk_tool.description,
    parameters={...},
    implementation=adk_tool.run_async
)
```

### Enterprise API Gateway

Create a secure gateway that manages authentication and permissions for agent-to-agent communication:

```python
# Create tokens for different teams/services
team1_token = bridge.mcp_server.auth_manager.create_token(
    scopes=[PermissionScope.EXECUTE],
    expires_in_days=30
)

# Expose tools with permission checking
async def secure_tool(params, token=None):
    if not bridge.mcp_server.auth_manager.check_permission(token, PermissionScope.EXECUTE):
        return {"error": "Permission denied"}
    # Proceed with tool execution
    return {"result": "Success"}
```

## Complete Example

A complete example demonstrating all the capabilities of the bridge is available in the `examples` directory.

## API Reference

For detailed API documentation, see the [API reference](https://example.com/api-reference).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
