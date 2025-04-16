"""
Enhanced MCP-A2A Bridge Example
==============================

This example demonstrates how to use the Enhanced MCP-A2A Bridge to:
1. Create a bridge between MCP and A2A protocols
2. Expose MCP tools as A2A skills
3. Access A2A tools as MCP resources
4. Handle streaming responses
5. Manage authentication and permissions
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any, List, Optional

# Import the bridge components
from mcp_a2a_bridge import (
    create_bridge,
    connect_to_bridge_as_mcp,
    connect_to_bridge_as_a2a,
    McpTool,
    McpResource
)

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bridge_example")


# --- Example Tool Implementations ---

async def calculate(params: Dict[str, Any]) -> Dict[str, Any]:
    """A calculator tool that supports basic operations."""
    operation = params.get("operation", "add")
    a = params.get("a", 0)
    b = params.get("b", 0)

    result = None
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"error": "Cannot divide by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation: {operation}"}

    return {"result": result}


async def translate(params: Dict[str, Any]) -> Dict[str, Any]:
    """A translation tool that supports a few languages."""
    text = params.get("text", "")
    target_language = params.get("target_language", "").lower()

    # Mock translations
    translations = {
        "hello": {
            "spanish": "hola",
            "french": "bonjour",
            "german": "hallo"
        },
        "goodbye": {
            "spanish": "adiós",
            "french": "au revoir",
            "german": "auf wiedersehen"
        },
        "thank you": {
            "spanish": "gracias",
            "french": "merci",
            "german": "danke"
        }
    }

    text_lower = text.lower()

    if text_lower in translations and target_language in translations[text_lower]:
        translated = translations[text_lower][target_language]
        return {
            "original": text,
            "translated": translated,
            "language": target_language
        }
    else:
        return {
            "error": f"Cannot translate '{text}' to {target_language}.",
            "supported_texts": list(translations.keys()),
            "supported_languages": ["spanish", "french", "german"]
        }


async def get_weather(params: Dict[str, Any]) -> Dict[str, Any]:
    """A weather tool that provides weather information for cities."""
    city = params.get("city", "").lower()

    # Mock weather data
    weather_data = {
        "london": {"temperature": 15, "condition": "cloudy", "humidity": 75},
        "new york": {"temperature": 22, "condition": "sunny", "humidity": 60},
        "tokyo": {"temperature": 28, "condition": "partly cloudy", "humidity": 65},
        "paris": {"temperature": 20, "condition": "light rain", "humidity": 80},
        "sydney": {"temperature": 25, "condition": "clear", "humidity": 55}
    }

    if city in weather_data:
        return {
            "city": city,
            "weather": weather_data[city],
            "unit": "celsius"
        }
    else:
        return {
            "error": f"No weather data available for {city}",
            "available_cities": list(weather_data.keys())
        }


class StreamingTextGenerator:
    """A streaming text generator for demonstrating streaming responses."""

    def __init__(self):
        """Initialize StreamingTextGenerator."""
        pass

    async def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text based on input parameters."""
        prompt = params.get("prompt", "")
        length = min(params.get("length", 5), 10)  # Limit to 10 paragraphs max

        paragraphs = []
        for i in range(length):
            paragraphs.append(f"Paragraph {i + 1} about {prompt}. This is an example of streaming content generation.")

        return {"text": "\n\n".join(paragraphs)}

    async def stream(self, params: Dict[str, Any]):
        """Stream text generation results."""
        prompt = params.get("prompt", "")
        length = min(params.get("length", 5), 10)  # Limit to 10 paragraphs max

        for i in range(length):
            paragraph = f"Paragraph {i + 1} about {prompt}. This is an example of streaming content generation."
            yield paragraph
            await asyncio.sleep(0.5)  # Simulate generation delay


# --- Example Resource Implementation ---

async def get_documentation(params: Dict[str, Any]) -> str:
    """Return documentation content."""
    return """
    # Enhanced MCP-A2A Bridge Documentation

    This bridge enables seamless interoperability between MCP and A2A protocols.

    ## Features

    - Expose MCP tools as A2A skills
    - Access A2A tools as MCP resources
    - Handle streaming responses
    - Manage authentication and permissions

    ## Usage

    See the example script for usage details.
    """


# --- Main Example App ---

async def run_complete_example():
    """Run a complete example demonstrating various bridge features."""
    print("\n=== Enhanced MCP-A2A Bridge Example ===\n")

    # Create a bridge
    bridge = create_bridge(
        name="Example Bridge",
        description="Demonstration of the Enhanced MCP-A2A Bridge",
        url="http://localhost:8000"
    )

    # Start the bridge
    await bridge.start()

    try:
        # --- 1. Expose MCP Tools as A2A Skills ---
        print("\n1. Exposing MCP Tools as A2A Skills")

        # Create MCP tools
        calculator_tool = McpTool(
            name="calculator",
            description="A calculator tool that supports basic operations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        )

        translator_tool = McpTool(
            name="translator",
            description="A translation tool that supports a few languages",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to translate"
                    },
                    "target_language": {
                        "type": "string",
                        "enum": ["spanish", "french", "german"],
                        "description": "The target language"
                    }
                },
                "required": ["text", "target_language"]
            }
        )

        # Expose MCP tools as A2A skills
        bridge.expose_mcp_tool_as_a2a_skill(calculator_tool, calculate)
        bridge.expose_mcp_tool_as_a2a_skill(translator_tool, translate)

        print("  ✅ Exposed calculator and translator tools as A2A skills")

        # --- 2. Expose A2A Skills as MCP Tools ---
        print("\n2. Exposing A2A Skills as MCP Tools")

        # Expose A2A skills as MCP tools
        bridge.expose_a2a_skill_as_mcp_tool(
            skill_id="weather",
            name="get_weather",
            description="Get weather information for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            },
            implementation=get_weather
        )

        # Create a streaming text generator
        text_generator = StreamingTextGenerator()

        # Expose streaming text generator as an A2A skill
        bridge.expose_a2a_skill_as_mcp_tool(
            skill_id="text-generator",
            name="generate_text",
            description="Generate text based on a prompt",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The text prompt"
                    },
                    "length": {
                        "type": "integer",
                        "description": "The number of paragraphs to generate"
                    }
                },
                "required": ["prompt"]
            },
            implementation=text_generator.generate
        )

        print("  ✅ Exposed weather and text generator skills as MCP tools")

        # --- 3. Expose MCP Resources ---
        print("\n3. Exposing MCP Resources")

        # Create an MCP resource
        docs_resource = McpResource(
            uri="docs://bridge/readme",
            description="Bridge documentation",
            mime_type="text/markdown"
        )

        # Expose MCP resource as A2A skill
        bridge.expose_mcp_resource_as_a2a_skill(docs_resource, get_documentation)

        print("  ✅ Exposed documentation resource")

        # --- 4. Create Authentication Tokens ---
        print("\n4. Creating Authentication Tokens")

        # Create tokens with different permission scopes
        read_token = bridge.mcp_server.auth_manager.create_token(
            scopes=[bridge.mcp_server.auth_manager.PermissionScope.READ],
            expires_in_days=1
        )

        execute_token = bridge.mcp_server.auth_manager.create_token(
            scopes=[bridge.mcp_server.auth_manager.PermissionScope.EXECUTE],
            expires_in_days=1
        )

        admin_token = bridge.mcp_server.auth_manager.create_token(
            scopes=[bridge.mcp_server.auth_manager.PermissionScope.ADMIN],
            expires_in_days=7
        )

        print(f"  ✅ Created tokens with different permission scopes:")
        print(f"    - Read token: {read_token[:8]}...")
        print(f"    - Execute token: {execute_token[:8]}...")
        print(f"    - Admin token: {admin_token[:8]}...")

        # --- 5. Create a Web Server ---
        print("\n5. Creating Web Server")

        server = bridge.create_web_server(host="localhost", port=8000)

        print(f"  ✅ Created web server at http://localhost:8000")
        print(f"    - A2A endpoint: http://localhost:8000/a2a")
        print(f"    - MCP endpoints: http://localhost:8000/mcp/*")
        print(f"    - Agent card: http://localhost:8000/.well-known/agent.json")

        # --- 6. Connect as Clients ---
        print("\n6. Connecting as Clients")

        # Connect as MCP client
        mcp_client = await connect_to_bridge_as_mcp("http://localhost:8000")

        # List tools from the bridge
        mcp_tools = await mcp_client.list_tools()

        print(f"  ✅ Connected as MCP client and found {len(mcp_tools)} tools:")
        for tool in mcp_tools:
            print(f"    - {tool['name']}: {tool['description']}")

        # Connect as A2A client
        a2a_client = await connect_to_bridge_as_a2a("http://localhost:8000")

        # Get the agent card
        agent_card = await a2a_client.get_agent_card()

        print(f"  ✅ Connected as A2A client and got agent card:")
        print(f"    - Name: {agent_card.get('name')}")
        print(f"    - Description: {agent_card.get('description')}")
        print(f"    - Skills: {len(agent_card.get('skills', []))}")

        # --- 7. Call Tools and Skills ---
        print("\n7. Calling Tools and Skills")

        # Call the calculator tool as an MCP client
        calculator_result = await mcp_client.call_tool(
            "calculator",
            {"operation": "multiply", "a": 6, "b": 7}
        )

        print(f"  ✅ Called calculator tool as MCP client:")
        print(f"    - Result: {calculator_result}")

        # Call the weather skill as an A2A client
        weather_task_id = f"task-{int(asyncio.get_event_loop().time())}"
        weather_message = {
            "role": "user",
            "parts": [{
                "type": "text",
                "text": "Use the get_weather tool with parameters: {\"city\": \"london\"}"
            }]
        }

        weather_response = await a2a_client.send_task(weather_task_id, weather_message)

        if "result" in weather_response:
            result = weather_response["result"]
            if "artifacts" in result and result["artifacts"]:
                artifact = result["artifacts"][0]
                if "parts" in artifact and artifact["parts"]:
                    part = artifact["parts"][0]
                    if "text" in part:
                        try:
                            weather_data = json.loads(part["text"])
                            print(f"  ✅ Called weather skill as A2A client:")
                            print(f"    - City: {weather_data.get('city', 'unknown')}")
                            if "weather" in weather_data:
                                print(f"    - Temperature: {weather_data['weather'].get('temperature')}°C")
                                print(f"    - Condition: {weather_data['weather'].get('condition')}")
                        except json.JSONDecodeError:
                            print(f"  ✅ Called weather skill as A2A client:")
                            print(f"    - Response: {part['text']}")

        # --- 8. Streaming Responses ---
        print("\n8. Streaming Responses")

        # Call the text generator with streaming
        text_gen_task_id = f"task-{int(asyncio.get_event_loop().time())}"
        text_gen_message = {
            "role": "user",
            "parts": [{
                "type": "text",
                "text": "Use the generate_text tool with parameters: {\"prompt\": \"AI and interoperability\", \"length\": 3}"
            }]
        }

        print(f"  ✅ Calling text generator with streaming:")

        async for event in a2a_client.send_streaming_task(text_gen_task_id, text_gen_message):
            if "result" in event:
                result = event["result"]

                if "artifact" in result:
                    artifact = result["artifact"]
                    if "parts" in artifact and artifact["parts"]:
                        part = artifact["parts"][0]
                        if "text" in part:
                            print(f"    - Chunk: {part['text']}")

                if "status" in result and result.get("final", False):
                    state = result["status"].get("state")
                    print(f"    - Final state: {state}")

        # --- 9. Read Resource ---
        print("\n9. Reading Resource")

        # Read the documentation resource
        docs_content, mime_type = await mcp_client.read_resource("docs://bridge/readme")

        # Convert bytes to string if needed
        if isinstance(docs_content, bytes):
            docs_content = docs_content.decode("utf-8")

        print(f"  ✅ Read documentation resource:")
        print(f"    - MIME type: {mime_type}")
        print(f"    - Content preview: {docs_content[:100]}...")

        print("\n=== Example Complete ===")
        print("\nThe web server is ready to accept requests.")
        print("Press Ctrl+C to stop.\n")

        # Run the server
        await asyncio.sleep(0.1)  # Give a moment for any pending operations

        # In a real application, you would run the server here:
        # await server.start_async()

        # For the example, we'll just wait for user input
        print("Press Enter to exit...")
        await asyncio.get_event_loop().run_in_executor(None, input)

    finally:
        # Stop clients and bridge
        if 'mcp_client' in locals():
            await mcp_client.stop()

        if 'a2a_client' in locals():
            await a2a_client.stop()

        await bridge.stop()


# --- Run the example ---
if __name__ == "__main__":
    try:
        asyncio.run(run_complete_example())
    except KeyboardInterrupt:
        print("\nExample stopped by user.")
    except Exception as e:
        print(f"Error running example: {str(e)}")