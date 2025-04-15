"""
AIRA Consumer Agent (Improved)
------------------------------
This agent discovers and invokes tools from the WeatherAgent through the AIRA network,
with improved handling of city names and error conditions.
"""

import asyncio
import os
import json
import sys
import datetime
from typing import Dict, Any, Optional, List

# --- AIRA Client ---
class AiraNode:
    """Simple AIRA Node implementation for consumer agent."""

    def __init__(self, hub_url: str, node_url: str, node_name: str):
        """Initialize the AIRA Node."""
        self.hub_url = hub_url.rstrip('/')
        self.node_url = node_url
        self.node_name = node_name
        self.registered = False
        self._heartbeat_task = None

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

    async def register_with_hub(self):
        """Register this node with the AIRA hub."""
        # Prepare payload
        payload = {
            "url": self.node_url,
            "name": self.node_name,
            "description": "Consumer agent that discovers and calls other agents' tools",
            "skills": [],
            "shared_resources": [],
            "aira_capabilities": ["a2a"],
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

    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents from the hub."""
        try:
            async with self.session.get(f"{self.hub_url}/agents") as resp:
                if resp.status == 200:
                    agents = await resp.json()
                    return agents
                else:
                    error_text = await resp.text()
                    print(f"⚠️ Failed to discover agents: {error_text}")
                    return []
        except Exception as e:
            print(f"❌ Error discovering agents: {str(e)}")
            return []

    async def discover_agent_capabilities(self, agent_url: str) -> Dict[str, Any]:
        """Discover the capabilities of a specific agent."""
        try:
            # First try the A2A agent card endpoint
            normalized_url = agent_url
            if normalized_url.endswith('/'):
                normalized_url = normalized_url[:-1]

            async with self.session.get(f"{normalized_url}/.well-known/agent.json") as resp:
                if resp.status == 200:
                    agent_card = await resp.json()
                    return agent_card
                else:
                    error_text = await resp.text()
                    print(f"⚠️ Failed to get agent card: {error_text}")
                    return {}
        except Exception as e:
            print(f"❌ Error discovering agent capabilities: {str(e)}")
            return {}

    async def invoke_agent_tool(self, agent_url: str, tool_name: str, params: Dict[str, Any]) -> Any:
        """Invoke a tool on another agent."""
        try:
            # Create a tasks/send request
            task_id = f"task-{int(datetime.datetime.now().timestamp())}"

            # Format the request for A2A protocol
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tasks/send",
                "params": {
                    "id": task_id,
                    "message": {
                        "role": "user",
                        "parts": [{
                            "type": "text",
                            "text": f"Use the {tool_name} tool with parameters: {json.dumps(params)}"
                        }]
                    }
                }
            }

            # Ensure the URL ends with /a2a
            if not agent_url.endswith('/a2a'):
                if agent_url.endswith('/'):
                    agent_url = agent_url + 'a2a'
                else:
                    agent_url = agent_url + '/a2a'

            print(f"🔄 Invoking tool '{tool_name}' on agent at {agent_url}")
            print(f"📤 Request: {json.dumps(request, indent=2)}")

            # Send the request
            async with self.session.post(agent_url, json=request) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"📥 Response: {json.dumps(result, indent=2)}")

                    # Extract the result from the artifacts
                    if "result" in result:
                        task_result = result["result"]
                        artifacts = task_result.get("artifacts", [])

                        if artifacts:
                            # Get the text part from the first artifact
                            parts = artifacts[0].get("parts", [])
                            text_part = next((p for p in parts if p.get("type") == "text"), None)

                            if text_part and "text" in text_part:
                                try:
                                    # Try to parse as JSON
                                    return json.loads(text_part["text"])
                                except:
                                    # Return as plain text if not JSON
                                    return text_part["text"]

                    return result
                else:
                    error_text = await resp.text()
                    print(f"⚠️ Failed to invoke tool: {error_text}")
                    return {"error": f"Failed to invoke tool: {error_text}"}
        except Exception as e:
            print(f"❌ Error invoking agent tool: {str(e)}")
            return {"error": f"Error invoking tool: {str(e)}"}


# --- Consumer Agent ---
class ImprovedConsumerAgent:
    """Agent that discovers and consumes tools from other agents with improved handling."""

    def __init__(self, hub_url: str, agent_url: str, agent_name: str):
        """Initialize the Consumer Agent."""
        self.aira_node = AiraNode(
            hub_url=hub_url,
            node_url=agent_url,
            node_name=agent_name
        )
        self.discovered_agents = {}
        self.discovered_tools = {}
        self.available_cities = set()  # Track available cities

    async def start(self):
        """Start the agent and register with the AIRA hub."""
        await self.aira_node.register_with_hub()
        print(f"🚀 Agent '{self.aira_node.node_name}' registered with hub at {self.aira_node.hub_url}")

    async def stop(self):
        """Stop the agent and clean up resources."""
        await self.aira_node.close()
        print(f"🛑 Agent '{self.aira_node.node_name}' disconnected from hub")

    async def discover_all_agents(self):
        """Discover all agents from the hub."""
        agents = await self.aira_node.discover_agents()
        print(f"🔍 Discovered {len(agents)} agents")

        for agent in agents:
            agent_url = agent.get("url")
            agent_name = agent.get("name")

            if agent_url != self.aira_node.node_url:  # Skip self
                self.discovered_agents[agent_url] = agent
                print(f"  📌 {agent_name} at {agent_url}")

        return self.discovered_agents

    async def discover_agent_tools(self, agent_url: str):
        """Discover tools provided by a specific agent."""
        agent_card = await self.aira_node.discover_agent_capabilities(agent_url)

        if not agent_card:
            print(f"⚠️ No agent card found for {agent_url}")
            return []

        agent_name = agent_card.get("name", "Unknown Agent")
        skills = agent_card.get("skills", [])

        tools = []
        for skill in skills:
            if "tool" in skill.get("tags", []):
                tool_name = skill.get("name")
                tool_description = skill.get("description", "")
                tool_parameters = skill.get("parameters", {})

                tools.append({
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_parameters
                })

                print(f"  🔧 Found tool: {tool_name} - {tool_description}")

        self.discovered_tools[agent_url] = tools
        return tools

    async def discover_available_cities(self, agent_url: str):
        """Use the weather tool to discover available cities."""
        # Make a generic request to discover available cities
        result = await self.invoke_weather_tool(agent_url, "test")

        if result and "available_cities" in result:
            self.available_cities = set(result["available_cities"])
            print(f"📍 Available cities: {', '.join(self.available_cities)}")

        return self.available_cities

    def normalize_city_name(self, city: str) -> str:
        """Find the best match for a city name in available cities."""
        if not self.available_cities:
            return city

        # Normalize input
        city_normalized = city.lower().strip()

        # Exact match
        if city_normalized in self.available_cities:
            return city_normalized

        # Try without spaces
        city_nospace = city_normalized.replace(" ", "")
        for available in self.available_cities:
            if available.replace(" ", "") == city_nospace:
                return available

        # Check if city is a case-insensitive match
        for available in self.available_cities:
            if available.lower() == city_normalized:
                return available

        # No match found, return original
        return city_normalized

    async def invoke_weather_tool(self, agent_url: str, city: str):
        """Invoke the weather tool on the weather agent."""
        # Find the weather tool
        tools = self.discovered_tools.get(agent_url, [])
        weather_tool = next((t for t in tools if "weather" in t["name"].lower() and "forecast" not in t["name"].lower()), None)

        if not weather_tool:
            print(f"⚠️ No weather tool found for {agent_url}")
            return None

        # Normalize city name
        normalized_city = self.normalize_city_name(city)
        print(f"🌦️ Invoking weather tool for {city} (normalized to {normalized_city})")

        result = await self.aira_node.invoke_agent_tool(
            agent_url=agent_url,
            tool_name=weather_tool["name"],
            params={"city": normalized_city}
        )

        # Update available cities if provided
        if result and "available_cities" in result:
            self.available_cities = set(result["available_cities"])

        return result

    async def invoke_forecast_tool(self, agent_url: str, city: str, days: int = 3):
        """Invoke the forecast tool on the weather agent."""
        # Find the forecast tool
        tools = self.discovered_tools.get(agent_url, [])
        forecast_tool = next((t for t in tools if "forecast" in t["name"].lower()), None)

        if not forecast_tool:
            print(f"⚠️ No forecast tool found for {agent_url}")
            return None

        # Normalize city name
        normalized_city = self.normalize_city_name(city)
        print(f"🔮 Invoking forecast tool for {city} (normalized to {normalized_city}) ({days} days)")

        result = await self.aira_node.invoke_agent_tool(
            agent_url=agent_url,
            tool_name=forecast_tool["name"],
            params={"city": normalized_city, "days": days}
        )

        # Update available cities if provided
        if result and "available_cities" in result:
            self.available_cities = set(result["available_cities"])

        return result


async def interact_with_user(consumer_agent, weather_agent_url=None):
    """Interactive CLI for testing the consumer agent."""

    # If no weather agent URL, discover weather agent
    if not weather_agent_url:
        print("\n🔍 Discovering agents...")
        await consumer_agent.discover_all_agents()

        # Find the weather agent
        weather_agents = []
        for url, agent in consumer_agent.discovered_agents.items():
            if "weather" in agent.get("name", "").lower():
                weather_agents.append((url, agent.get("name")))

        if not weather_agents:
            print("❌ No weather agents found.")
            return

        if len(weather_agents) == 1:
            weather_agent_url = weather_agents[0][0]
            print(f"✅ Found weather agent: {weather_agents[0][1]} at {weather_agent_url}")
        else:
            print("Multiple weather agents found:")
            for i, (url, name) in enumerate(weather_agents):
                print(f"{i+1}. {name} ({url})")

            choice = input("Select agent number: ")
            try:
                idx = int(choice) - 1
                weather_agent_url = weather_agents[idx][0]
            except:
                print("❌ Invalid choice.")
                return

    # Discover tools
    print(f"\n🔧 Discovering tools for {weather_agent_url}...")
    tools = await consumer_agent.discover_agent_tools(weather_agent_url)

    if not tools:
        print("❌ No tools found for this agent.")
        return

    # Discover available cities
    print("\n📍 Discovering available cities...")
    cities = await consumer_agent.discover_available_cities(weather_agent_url)

    print("\n✅ Ready to use weather tools!")

    # Interactive loop
    while True:
        print("\n--- AIRA Weather Consumer (Improved) ---")
        print("1. Get current weather")
        print("2. Get weather forecast")
        print("3. List available cities")
        print("4. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            print(f"\nAvailable cities: {', '.join(consumer_agent.available_cities)}")
            city = input("Enter city name: ")
            print(f"\n🔄 Getting weather for {city}...")
            result = await consumer_agent.invoke_weather_tool(weather_agent_url, city)

            if result and "error" in result:
                print(f"\n⚠️ Error: {result['error']}")
            else:
                print(f"\n🌦️ Weather Result:\n{json.dumps(result, indent=2)}")

        elif choice == "2":
            print(f"\nAvailable cities: {', '.join(consumer_agent.available_cities)}")
            city = input("Enter city name: ")
            days_input = input("Enter number of days (default 3): ")
            days = 3
            try:
                if days_input.strip():
                    days = int(days_input)
            except:
                print("Using default 3 days...")

            print(f"\n🔄 Getting {days}-day forecast for {city}...")
            result = await consumer_agent.invoke_forecast_tool(weather_agent_url, city, days)

            if result and "error" in result:
                print(f"\n⚠️ Error: {result['error']}")
            else:
                print(f"\n🔮 Forecast Result:\n{json.dumps(result, indent=2)}")

        elif choice == "3":
            print(f"\n📍 Available cities: {', '.join(consumer_agent.available_cities)}")

        elif choice == "4":
            print("👋 Goodbye!")
            break

        else:
            print("⚠️ Invalid option. Please try again.")


# --- Main function ---
async def main():
    # Configuration
    HUB_URL = os.environ.get("AIRA_HUB_URL", "http://localhost:8000")
    AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8002") # Different port than weather agent
    AGENT_NAME = "ImprovedWeatherConsumerAgent"
    WEATHER_AGENT_URL = os.environ.get("WEATHER_AGENT_URL", "http://localhost:8001")

    # Command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto":
            # Automated mode (non-interactive)
            auto_mode = True
            if len(sys.argv) > 2:
                city = sys.argv[2]
            else:
                city = "london"  # Note lowercase
        else:
            # Take first arg as WEATHER_AGENT_URL
            WEATHER_AGENT_URL = sys.argv[1]
            auto_mode = False
    else:
        auto_mode = False

    # Create and start the agent
    try:
        consumer_agent = ImprovedConsumerAgent(
            hub_url=HUB_URL,
            agent_url=AGENT_URL,
            agent_name=AGENT_NAME
        )

        await consumer_agent.start()

        if auto_mode:
            # Automated mode - run a predefined sequence
            print(f"\n🤖 Running in automated mode for city: {city}")

            # Discover tools
            print(f"\n🔧 Discovering tools for {WEATHER_AGENT_URL}...")
            tools = await consumer_agent.discover_agent_tools(WEATHER_AGENT_URL)

            if not tools:
                print("❌ No tools found for weather agent.")
                return

            # Discover available cities first
            print("\n📍 Discovering available cities...")
            await consumer_agent.discover_available_cities(WEATHER_AGENT_URL)

            # Get current weather
            print(f"\n🔄 Getting weather for {city}...")
            weather = await consumer_agent.invoke_weather_tool(WEATHER_AGENT_URL, city)
            if weather and "error" in weather:
                print(f"\n⚠️ Error: {weather['error']}")
            else:
                print(f"\n🌦️ Weather Result:\n{json.dumps(weather, indent=2)}")

            # Get forecast
            print(f"\n🔄 Getting 5-day forecast for {city}...")
            forecast = await consumer_agent.invoke_forecast_tool(WEATHER_AGENT_URL, city, 5)
            if forecast and "error" in forecast:
                print(f"\n⚠️ Error: {forecast['error']}")
            else:
                print(f"\n🔮 Forecast Result:\n{json.dumps(forecast, indent=2)}")

        else:
            # Interactive mode
            await interact_with_user(consumer_agent, WEATHER_AGENT_URL)

    except KeyboardInterrupt:
        print("\n⛔ Shutting down...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Cleanup
        if 'consumer_agent' in locals():
            await consumer_agent.stop()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("📦 Please install aiohttp: pip install aiohttp")
        sys.exit(1)

    # Run the main function
    asyncio.run(main())