"""
Basic Weather Agent with ADK and A2A via AIRA
---------------------------------------------
This file demonstrates how to create a simple weather agent using Google's Agent
Development Kit (ADK) and expose it to other agents via Agent-to-Agent (A2A)
protocol using the AIRA client library.
"""

import asyncio
import os
import json
from typing import Dict, Any, Optional

# --- AIRA Client Imports (Assuming aira_client.py is in the same directory) ---
# Modified to fix registration issue
class AiraNode:
    """Simple AIRA Node implementation with registration fix."""

    def __init__(self, hub_url: str, node_url: str, node_name: str):
        """Initialize the AIRA Node."""
        self.hub_url = hub_url.rstrip('/')
        self.node_url = node_url
        self.node_name = node_name
        self.registered = False
        self._heartbeat_task = None
        self.mcp_adapter = None

        # Import aiohttp here for cleaner error handling if it's not installed
        import aiohttp
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Clean up resources."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.session.close()

    def set_mcp_adapter(self, adapter):
        """Set the MCP adapter."""
        self.mcp_adapter = adapter

    async def register_with_hub(self):
        """Register this node with the AIRA hub."""
        # Create agent card if MCP adapter exists
        agent_card = None
        if self.mcp_adapter:
            agent_card = self.mcp_adapter.generate_agent_card()
        else:
            # Create a basic agent card
            agent_card = {
                "name": self.node_name,
                "description": f"AIRA node for {self.node_name}",
                "url": self.node_url,
                "skills": []
            }

        # Convert agent card to registration payload
        if hasattr(agent_card, "skills"):
            skills = agent_card.skills
        else:
            skills = agent_card.get("skills", [])

        # Prepare payload
        payload = {
            "url": self.node_url,
            "name": self.node_name,
            "description": "Weather service providing current conditions and forecasts",
            "skills": skills,
            "shared_resources": [],
            "aira_capabilities": ["a2a", "mcp"],
            "auth": {}
        }

        try:
            # Send registration request
            async with self.session.post(f"{self.hub_url}/register", json=payload) as resp:
                if resp.status == 201:  # Success status for registration
                    result = await resp.json()
                    print(f"✅ Successfully registered with hub: {result}")
                    self.registered = True
                    self._start_heartbeat()
                    return result
                else:
                    error_text = await resp.text()
                    raise ValueError(f"Registration failed with status {resp.status}: {error_text}")
        except Exception as e:
            print(f"❌ Error registering with hub {self.hub_url}: {str(e)}")
            raise ValueError(f"Failed to register with hub: {str(e)}")

    def _start_heartbeat(self):
        """Start the heartbeat background task."""
        if not self._heartbeat_task:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to the hub."""
        while True:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                if not self.registered:
                    continue

                # URL encode properly to avoid 404 errors
                import urllib.parse
                encoded_url = urllib.parse.quote(self.node_url, safe='')

                async with self.session.post(f"{self.hub_url}/heartbeat/{encoded_url}") as resp:
                    if resp.status != 200:
                        print(f"⚠️ Heartbeat failed: {await resp.text()}")
                        # If heartbeat failed, try to re-register
                        self.registered = False
                        await self.register_with_hub()
                    else:
                        print("💓 Heartbeat sent successfully")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def handle_a2a_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming A2A protocol request."""
        if not self.mcp_adapter:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32000,
                    "message": "No MCP adapter configured to handle requests"
                }
            }

        return await self.mcp_adapter.handle_a2a_request(request)


class McpServerAdapter:
    """Adapter for MCP servers to expose their tools via A2A protocol."""

    def __init__(self, server_name: str, server_description: str, base_url: str):
        self.server_name = server_name
        self.server_description = server_description
        self.base_url = base_url
        self.tools = []
        self.tool_implementations = {}

    def add_tool(self, tool, implementation):
        """Add an MCP tool with its implementation."""
        self.tools.append(tool)
        self.tool_implementations[tool.name] = implementation

    def generate_agent_card(self):
        """Generate an A2A Agent Card from registered tools."""
        skills = []

        # Convert tools to skills
        for tool in self.tools:
            skills.append({
                "id": f"tool-{tool.name}",
                "name": tool.name,
                "description": tool.description,
                "tags": ["mcp", "tool"],
                "parameters": tool.parameters
            })

        return {
            "name": self.server_name,
            "description": self.server_description,
            "url": self.base_url,
            "skills": skills
        }

    async def handle_a2a_request(self, req):
        """Handle an A2A protocol request."""
        method = req.get("method")

        if method == "tasks/send":
            return await self._handle_tasks_send(req.get("params", {}), req.get("id", 1))
        elif method == "tasks/get":
            return await self._handle_tasks_get(req.get("params", {}), req.get("id", 1))
        else:
            return {
                "jsonrpc": "2.0",
                "id": req.get("id", 1),
                "error": {
                    "code": -32601,
                    "message": f"Method {method} not supported"
                }
            }

    async def _handle_tasks_send(self, params, req_id):
        """Handle A2A tasks/send method."""
        task_id = params.get("id")
        message = params.get("message", {})

        if message.get("role") != "user" or not message.get("parts"):
            return self._create_error_response("Invalid message format", req_id)

        # Extract the message text
        text_part = next((p for p in message.get("parts", []) if p.get("type") == "text"), None)
        if not text_part or not text_part.get("text"):
            return self._create_error_response("No text content found", req_id)

        # Parse the message to identify tool request
        text = text_part.get("text")
        tool_name = None
        tool_params = {}

        # Simple parsing - in a real implementation, you'd use an LLM or parser
        for tool in self.tools:
            if tool.name.lower() in text.lower():
                tool_name = tool.name
                # Extract parameters - this is simplified
                if "city" in text.lower():
                    import re
                    city_match = re.search(r'(?:in|for|of|at)\s+([A-Za-z\s]+)', text)
                    if city_match:
                        tool_params["city"] = city_match.group(1).strip()
                    else:
                        tool_params["city"] = "New York"  # Default

                if "forecast" in text.lower() and "days" in text.lower():
                    days_match = re.search(r'(\d+)\s+days?', text)
                    if days_match:
                        tool_params["days"] = int(days_match.group(1))
                break

        if not tool_name or tool_name not in self.tool_implementations:
            # Return help message
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "sessionId": params.get("sessionId", "session-" + str(task_id)),
                    "status": {"state": "completed"},
                    "artifacts": [{
                        "parts": [{
                            "type": "text",
                            "text": f"I'm not sure which tool you want to use. Available tools: {', '.join(t.name for t in self.tools)}"
                        }]
                    }]
                }
            }

        # Execute the tool
        try:
            tool_func = self.tool_implementations[tool_name]

            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(tool_params)
            else:
                result = tool_func(tool_params)

            # Format the result as an A2A task response
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "sessionId": params.get("sessionId", "session-" + str(task_id)),
                    "status": {"state": "completed"},
                    "artifacts": [{
                        "parts": [{
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }]
                    }]
                }
            }
        except Exception as e:
            print(f"❌ Error executing tool {tool_name}: {str(e)}")
            return self._create_error_response(f"Error executing tool: {str(e)}", req_id)

    async def _handle_tasks_get(self, params, req_id):
        """Handle A2A tasks/get method."""
        task_id = params.get("id")

        # Return a simple response
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "id": task_id,
                "sessionId": "session-" + str(task_id),
                "status": {"state": "completed"},
                "artifacts": [],
                "history": []
            }
        }

    def _create_error_response(self, message, req_id):
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32000,
                "message": message
            }
        }


class McpTool:
    """Definition of an MCP tool."""

    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters


# --- Google ADK Imports ---
from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# --- A2A-ADK Adapter ---
class AdkToAiraAdapter:
    """Adapter that bridges between Google ADK tools and the AIRA network."""

    def __init__(self, hub_url, agent_url, agent_name, agent_description):
        """Initialize the adapter."""
        self.aira_node = AiraNode(
            hub_url=hub_url,
            node_url=agent_url,
            node_name=agent_name
        )

        self.mcp_adapter = McpServerAdapter(
            server_name=agent_name,
            server_description=agent_description,
            base_url=agent_url
        )

        self.aira_node.set_mcp_adapter(self.mcp_adapter)
        self.adk_tools = {}
        self.agent = None
        self.runner = None
        self.session_service = None

    async def start(self):
        """Start the adapter and register with the AIRA hub."""
        await self.aira_node.register_with_hub()
        print(f"🚀 Agent '{self.aira_node.node_name}' registered with hub at {self.aira_node.hub_url}")

    async def stop(self):
        """Stop the adapter and clean up resources."""
        await self.aira_node.close()
        print(f"🛑 Agent '{self.aira_node.node_name}' disconnected from hub")

    def add_adk_tool(self, adk_tool):
        """Add an ADK tool to be exposed as an MCP tool through A2A."""
        # Extract tool metadata from ADK tool
        tool_name = adk_tool.name
        tool_description = getattr(adk_tool, "description", f"ADK Tool: {tool_name}")

        # Extract parameters from the function signature (simplified)
        parameters = self._extract_parameters_from_adk_tool(adk_tool)

        # Create MCP tool
        mcp_tool = McpTool(
            name=tool_name,
            description=tool_description,
            parameters=parameters
        )

        # Create implementation that delegates to the ADK tool
        async def tool_implementation(params):
            print(f"🔧 Executing ADK tool '{tool_name}' with params: {params}")

            # In a real implementation, you'd use ToolContext properly
            # For simplicity, we're passing empty params for missing args
            if hasattr(adk_tool, 'run_async'):
                return await adk_tool.run_async(args=params, tool_context=None)
            else:
                return adk_tool.run(args=params, tool_context=None)

        # Add to the MCP adapter
        self.mcp_adapter.add_tool(mcp_tool, tool_implementation)
        self.adk_tools[tool_name] = adk_tool
        print(f"➕ Added ADK tool '{tool_name}' to A2A exposure")

    def _extract_parameters_from_adk_tool(self, adk_tool):
        """Extract parameter schema from an ADK tool."""
        # Create simple schema
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days for forecast (if applicable)"
                }
            },
            "required": ["city"]
        }

    def setup_adk_agent(self, model, agent_name, instruction):
        """Set up the ADK agent with the registered tools."""
        tools = list(self.adk_tools.values())

        self.agent = Agent(
            name=agent_name,
            model=model,
            instruction=instruction,
            tools=tools
        )

        self.session_service = InMemorySessionService()

        self.runner = Runner(
            agent=self.agent,
            app_name="adk_aira_agent",
            session_service=self.session_service
        )

        print(f"🤖 ADK Agent '{agent_name}' initialized with {len(tools)} tools")

    async def handle_a2a_request(self, request_body):
        """Handle an incoming A2A request."""
        try:
            request = json.loads(request_body)
            response = await self.aira_node.handle_a2a_request(request)
            return json.dumps(response)
        except Exception as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32000,
                    "message": f"Error handling request: {str(e)}"
                }
            })


# --- Define Weather Tool ---
def get_weather(city: str) -> dict:
    """
    Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo")

    Returns:
        dict: A dictionary containing the weather information
    """
    print(f"🌦️ Weather lookup for city: {city}")

    # Mock weather data for demonstration
    mock_weather_db = {
        "new york": {"temperature": 22, "condition": "sunny", "humidity": 60},
        "london": {"temperature": 15, "condition": "cloudy", "humidity": 75},
        "tokyo": {"temperature": 28, "condition": "partly cloudy", "humidity": 65},
        "paris": {"temperature": 20, "condition": "light rain", "humidity": 80},
        "sydney": {"temperature": 25, "condition": "clear", "humidity": 55},
    }

    city_normalized = city.lower().replace(" ", "")

    if city_normalized in mock_weather_db:
        return {
            "city": city,
            "weather": mock_weather_db[city_normalized],
            "unit": "celsius"
        }
    else:
        return {
            "city": city,
            "error": "City not found in weather database",
            "available_cities": list(mock_weather_db.keys())
        }


# --- Define forecast tool ---
def get_forecast(city: str, days: int = 3) -> dict:
    """
    Retrieves a weather forecast for the specified city and number of days.

    Args:
        city (str): The name of the city
        days (int): Number of days for the forecast (default: 3, max: 7)

    Returns:
        dict: A dictionary containing the forecast information
    """
    print(f"🔮 Forecast lookup for city: {city}, days: {days}")

    # Limit days to valid range
    days = min(max(1, days), 7)

    # Mock forecast data
    city_normalized = city.lower().replace(" ", "")
    forecast = []

    base_temps = {
        "new york": 22,
        "london": 15,
        "tokyo": 28,
        "paris": 20,
        "sydney": 25
    }

    conditions = ["sunny", "partly cloudy", "cloudy", "light rain", "rain", "clear"]

    if city_normalized in base_temps:
        base_temp = base_temps[city_normalized]
        import random

        for i in range(days):
            # Random variation in temperature
            temp_variation = random.uniform(-3, 3)
            # Random condition
            condition = random.choice(conditions)

            forecast.append({
                "day": i + 1,
                "temperature": round(base_temp + temp_variation, 1),
                "condition": condition
            })

        return {
            "city": city,
            "days": days,
            "unit": "celsius",
            "forecast": forecast
        }
    else:
        return {
            "city": city,
            "error": "City not found in forecast database",
            "available_cities": list(base_temps.keys())
        }


# --- Main application ---
async def main():
    # Configuration
    HUB_URL = os.environ.get("AIRA_HUB_URL", "http://localhost:8000")
    AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8001")
    AGENT_NAME = "WeatherAgent"
    AGENT_DESCRIPTION = "Provides current weather data and forecasts for cities worldwide"

    # Create ADK tools
    weather_tool = FunctionTool(get_weather)
    forecast_tool = FunctionTool(get_forecast)

    try:
        # Initialize the adapter
        adapter = AdkToAiraAdapter(
            hub_url=HUB_URL,
            agent_url=AGENT_URL,
            agent_name=AGENT_NAME,
            agent_description=AGENT_DESCRIPTION
        )

        # Add ADK tools
        adapter.add_adk_tool(weather_tool)
        adapter.add_adk_tool(forecast_tool)

        # Setup ADK agent (optional - for local usage)
        adapter.setup_adk_agent(
            model="gemini-1.0-pro",  # Update with your preferred model
            agent_name=AGENT_NAME,
            instruction=(
                "You are a weather assistant that provides accurate weather "
                "information and forecasts. Use the get_weather tool for current "
                "conditions and get_forecast for multi-day predictions."
            )
        )

        # Start the adapter and register with the hub
        await adapter.start()

        # ------------------------------------
        # Simple Web Server for handling A2A requests
        # ------------------------------------
        from aiohttp import web

        app = web.Application()

        async def a2a_handler(request):
            request_body = await request.text()
            response_body = await adapter.handle_a2a_request(request_body)
            return web.Response(text=response_body, content_type='application/json')

        async def agent_info_handler(request):
            return web.json_response({
                "name": AGENT_NAME,
                "description": AGENT_DESCRIPTION,
                "tools": ["get_weather", "get_forecast"]
            })

        async def well_known_agent_handler(request):
            """Handle the /.well-known/agent.json endpoint for A2A discovery."""
            agent_card = adapter.mcp_adapter.generate_agent_card()
            return web.json_response(agent_card)

        # Add routes
        app.router.add_post('/a2a', a2a_handler)
        app.router.add_get('/', agent_info_handler)
        app.router.add_get('/.well-known/agent.json', well_known_agent_handler)

        # Runner
        runner = web.AppRunner(app)
        await runner.setup()

        # Extract host and port from AGENT_URL
        import urllib.parse
        parsed_url = urllib.parse.urlparse(AGENT_URL)
        host = parsed_url.hostname or 'localhost'
        port = parsed_url.port or 8001

        # Start site
        site = web.TCPSite(runner, host, port)
        await site.start()

        print(f"📡 Agent server running at {AGENT_URL}")
        print("⏳ Press Ctrl+C to exit")

        # Keep the application running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n⛔ Shutting down...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Cleanup
        if 'adapter' in locals():
            await adapter.stop()
        if 'runner' in locals():
            await runner.cleanup()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    # Set your Google API key for ADK
    os.environ.setdefault("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    # Run the main application
    asyncio.run(main())