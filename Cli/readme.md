# AIRA Universal Client Documentation

## Overview

The AIRA Universal Client is a Python library designed to seamlessly connect AI agents built with different frameworks to the AIRA network. This allows agents to discover, communicate with, and utilize tools from other agents regardless of their underlying frameworks.

## Features

- **Framework Agnostic**: Works with Google ADK, LangChain, MCP, and custom agents
- **Tool Discovery**: Easily find and use tools across the AIRA network
- **Tool Exposure**: Expose your agent's tools to other agents on the network
- **A2A Protocol Support**: Complies with the Agent-to-Agent (A2A) protocol
- **MCP Integration**: Bridge between Model Context Protocol (MCP) and A2A protocol
- **Web Server Support**: Built-in web server for handling A2A requests

## Installation

```bash
pip install aira-connect
```

## Quickstart

```python
import asyncio
from aira_connect import setup_aira_client

async def main():
    # Create an AIRA client
    client = setup_aira_client(
        hub_url="https://aira-fl8f.onrender.com",
        agent_url="http://localhost:8000",
        agent_name="MyAgent"
    )
    
    # Start the client (registers with the hub)
    await client.start()
    
    try:
        # Define a simple tool
        def greeting(params):
            name = params.get("name", "World")
            return {"message": f"Hello, {name}!"}
        
        # Add the tool to the client
        client.add_tool(
            greeting,
            description="Generate a greeting message",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to greet"
                    }
                }
            }
        )
        
        # Discover weather agents
        agents = await client.discover_agents(tags=["weather"])
        
        if agents:
            # Select the first weather agent
            weather_agent = agents[0]
            
            # Discover its tools
            tools = await client.discover_agent_tools(weather_agent.get("url"))
            
            if tools:
                # Find a tool that provides weather information
                weather_tool = next((t for t in tools if "weather" in t.get("name", "").lower()), None)
                
                if weather_tool:
                    # Call the weather tool
                    result = await client.call_tool(
                        agent_url=weather_agent.get("url"),
                        tool_name=weather_tool.get("name"),
                        parameters={"city": "London"}
                    )
                    
                    print(f"Weather result: {result}")
    
    finally:
        # Stop the client
        await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Concepts

### AiraClient

The main client class that provides high-level access to the AIRA network:

```python
from aira_connect import AiraClient

client = AiraClient(
    hub_url="https://aira-fl8f.onrender.com",
    agent_url="http://localhost:8000",
    agent_name="MyAgent"
)
```

### Adding Tools

Expose your agent's tools to the AIRA network:

```python
def calculator(params):
    """A simple calculator."""
    operation = params.get("operation")
    a = params.get("a", 0)
    b = params.get("b", 0)
    
    if operation == "add":
        return {"result": a + b}
    elif operation == "subtract":
        return {"result": a - b}
    else:
        return {"error": f"Unknown operation: {operation}"}

client.add_tool(
    calculator,
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    }
)
```

### Discovering Agents and Tools

Find other agents and their tools on the AIRA network:

```python
# Find agents by tags
agents = await client.discover_agents(tags=["weather"])

# Find agents by category
finance_agents = await client.discover_agents(category="finance")

# Discover tools from a specific agent
tools = await client.discover_agent_tools("https://weather-agent.example.com")
```

### Calling Tools

Call tools on other agents:

```python
result = await client.call_tool(
    agent_url="https://weather-agent.example.com",
    tool_name="get_weather",
    parameters={"city": "London"}
)
```

### Web Server Setup

Set up a web server to handle incoming A2A requests:

```python
# Create a web server
server = client.create_web_server(host="localhost", port=8000)

# Run the server (blocking)
server.run()

# Or run it asynchronously
await server.start_async()
```

## Framework-Specific Adapters

### Google ADK Adapter

```python
from aira_connect import GoogleAdkAdapter

# Create the adapter
adk_adapter = GoogleAdkAdapter(client)

# Expose an ADK tool to AIRA
adk_adapter.expose_adk_tool(my_adk_tool)

# Create an ADK tool from an AIRA tool
adk_tool = adk_adapter.create_adk_tool_from_aira(
    agent_url="https://weather-agent.example.com",
    tool_name="get_weather"
)
```

### LangChain Adapter

```python
from aira_connect import LangchainAdapter

# Create the adapter
lc_adapter = LangchainAdapter(client)

# Expose a LangChain tool to AIRA
lc_adapter.expose_langchain_tool(my_langchain_tool)

# Create a LangChain tool from an AIRA tool
lc_tool = lc_adapter.create_langchain_tool_from_aira(
    agent_url="https://weather-agent.example.com",
    tool_name="get_weather"
)
```

### MCP Adapter

```python
from aira_connect import McpAdapter

# Create the adapter
mcp_adapter = McpAdapter(client)

# Expose an MCP tool to AIRA
mcp_adapter.expose_mcp_tool(
    tool_name="translate",
    description="Translate text to another language",
    parameters={...},
    implementation=my_translate_function
)

# Create MCP tools from AIRA tools
mcp_tools = await mcp_adapter.create_mcp_tools_from_aira(
    agent_url="https://weather-agent.example.com"
)

# Register tools with an MCP server
await mcp_adapter.register_mcp_tools_with_server(
    agent_url="https://weather-agent.example.com",
    mcp_server=my_mcp_server
)
```

## Advanced Usage

### Asynchronous Context Manager

You can use the client as an asynchronous context manager:

```python
async with AiraClient(...) as client:
    # Use the client
    agents = await client.discover_agents()
    # ...
```

### Custom Tool Parameters

Define complex parameter schemas for your tools:

```python
client.add_tool(
    complex_tool,
    description="A tool with complex parameters",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Input text"
            },
            "options": {
                "type": "object",
                "properties": {
                    "length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text", "html"]
                    }
                }
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["text"]
    }
)
```

### Searching for Tools

Search for tools across all agents based on a query:

```python
results = await client.search_tools("weather")

for tool, agent in results:
    print(f"Found tool '{tool.get('name')}' on agent '{agent.get('name')}'")
```

### Adding Agent Metadata

Add additional metadata to your agent:

```python
client.set_agent_metadata({
    "category": "utility",
    "tags": ["helper", "tools"],
    "provider": {
        "organization": "My Company",
        "url": "https://example.com"
    }
})
```

## Example Applications

### Weather Assistant

```python
async def weather_assistant():
    client = setup_aira_client(agent_name="WeatherAssistant")
    await client.start()
    
    try:
        # Find weather agents
        agents = await client.discover_agents(tags=["weather"])
        
        if agents:
            weather_agent = agents[0]
            tools = await client.discover_agent_tools(weather_agent.get("url"))
            
            # Create a simple command-line interface
            while True:
                city = input("Enter city name (or 'quit' to exit): ")
                if city.lower() == "quit":
                    break
                
                # Find the weather tool
                weather_tool = next((t for t in tools if "weather" in t.get("name", "").lower()), None)
                
                if weather_tool:
                    result = await client.call_tool(
                        agent_url=weather_agent.get("url"),
                        tool_name=weather_tool.get("name"),
                        parameters={"city": city}
                    )
                    
                    print(f"Weather: {result}")
                else:
                    print("No weather tool found.")
    
    finally:
        await client.stop()
```

### Tool Aggregator

```python
async def tool_aggregator():
    client = setup_aira_client(agent_name="ToolAggregator")
    await client.start()
    
    try:
        # Discover all agents
        agents = await client.discover_agents()
        
        # Collect tools from all agents
        all_tools = []
        for agent in agents:
            agent_url = agent.get("url")
            tools = await client.discover_agent_tools(agent_url)
            
            for tool in tools:
                all_tools.append({
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "agent": agent.get("name"),
                    "agent_url": agent_url
                })
        
        # Expose a meta-tool that can call any tool
        async def meta_tool(params):
            tool_name = params.get("tool")
            agent_name = params.get("agent")
            tool_params = params.get("parameters", {})
            
            # Find the agent URL
            agent_url = None
            for agent in agents:
                if agent.get("name") == agent_name:
                    agent_url = agent.get("url")
                    break
            
            if not agent_url:
                return {"error": f"Agent '{agent_name}' not found"}
            
            # Call the tool
            try:
                result = await client.call_tool(
                    agent_url=agent_url,
                    tool_name=tool_name,
                    parameters=tool_params
                )
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}
        
        client.add_tool(
            meta_tool,
            name="call_any_tool",
            description="Call any tool on any agent",
            parameters={
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "Name of the tool to call"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Name of the agent hosting the tool"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters to pass to the tool"
                    }
                },
                "required": ["tool", "agent"]
            }
        )
        
        # Create a web server to handle requests
        server = client.create_web_server()
        
        # Run the server
        print(f"Tool Aggregator running at {client.config.agent_url}")
        print(f"Available tools: {len(all_tools)}")
        server.run()
    
    finally:
        await client.stop()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
