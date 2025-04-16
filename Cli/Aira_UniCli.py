"""
AIRA Universal Client
=====================

A universal adapter library for connecting AI agents to the AIRA network,
regardless of the underlying framework (ADK, LangChain, etc.).

This library simplifies the process of:
1. Discovering and using tools exposed by other agents on the AIRA network
2. Exposing your own agent's tools to the network
3. Handling authentication and communication via the A2A protocol
4. Bridging between MCP and A2A protocols

Basic usage:
```python
from aira_connect import AiraClient

# Create a client
client = AiraClient(hub_url="https://aira-hub.example.com")

# Discover agents
agents = await client.discover_agents(tags=["weather"])

# Call a tool on another agent
result = await client.call_tool(
    agent_url="https://weather-agent.example.com",
    tool_name="get_weather",
    parameters={"city": "London"}
)
```

Advanced usage with framework-specific adapters shown in adapter modules.
"""

import asyncio
import aiohttp
import json
import logging
import os
import time
import uuid
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aira_connect")

# Type aliases
ToolFunction = Callable[[Dict[str, Any]], Any]
AsyncToolFunction = Callable[[Dict[str, Any]], Awaitable[Any]]
AgentInfo = Dict[str, Any]
ToolInfo = Dict[str, Any]


class AiraClientConfig:
    """Configuration for the AIRA client."""

    def __init__(
            self,
            hub_url: str = "https://aira-fl8f.onrender.com",
            agent_url: Optional[str] = None,
            agent_name: Optional[str] = None,
            agent_description: str = "AIRA Universal Client Agent",
            auto_register: bool = True,
            heartbeat_interval: int = 30,
            max_retries: int = 3,
            timeout: int = 30,
            log_level: int = logging.INFO
    ):
        """Initialize AiraClientConfig.

        Args:
            hub_url: URL of the AIRA hub
            agent_url: URL where this agent is accessible (if None, a local URL is generated)
            agent_name: Name of this agent (if None, a name is generated)
            agent_description: Description of this agent
            auto_register: Whether to automatically register with the hub on startup
            heartbeat_interval: Interval in seconds for sending heartbeats to the hub
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for requests
            log_level: Logging level
        """
        self.hub_url = hub_url.rstrip('/')
        self.agent_url = agent_url
        self.agent_name = agent_name or f"AiraAgent-{uuid.uuid4().hex[:8]}"
        self.agent_description = agent_description
        self.auto_register = auto_register
        self.heartbeat_interval = heartbeat_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.log_level = log_level


class AiraAgent:
    """Base class representing an AIRA agent with core functionality."""

    def __init__(self, config: AiraClientConfig):
        """Initialize AiraAgent.

        Args:
            config: Configuration for the AIRA client
        """
        self.config = config
        self.session = None
        self.registered = False
        self._heartbeat_task = None
        self.tools: Dict[str, Tuple[ToolFunction, Dict[str, Any]]] = {}
        self.agent_metadata: Dict[str, Any] = {}

        # Set logging level
        logger.setLevel(config.log_level)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the agent and register with the hub if auto_register is True."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout))

        if self.config.auto_register:
            await self.register_with_hub()

    async def stop(self):
        """Stop the agent and clean up resources."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self.session is not None:
            await self.session.close()
            self.session = None

    async def register_with_hub(self) -> Dict[str, Any]:
        """Register this agent with the AIRA hub."""
        if self.session is None:
            await self.start()

        # Create agent skills payload from registered tools
        skills = []
        for tool_name, (_, tool_metadata) in self.tools.items():
            skill = {
                "id": f"tool-{tool_name}",
                "name": tool_name,
                "description": tool_metadata.get("description", f"Tool: {tool_name}"),
                "tags": tool_metadata.get("tags", ["tool"]),
                "parameters": tool_metadata.get("parameters", {})
            }
            skills.append(skill)

        # Prepare payload
        payload = {
            "url": self.config.agent_url,
            "name": self.config.agent_name,
            "description": self.config.agent_description,
            "skills": skills,
            "shared_resources": [],
            "aira_capabilities": ["a2a"],
            "auth": {},
            **self.agent_metadata
        }

        try:
            # Send registration request
            async with self.session.post(f"{self.config.hub_url}/register", json=payload) as resp:
                if resp.status == 201:  # Success status for registration
                    result = await resp.json()
                    logger.info(f"Successfully registered with hub: {result}")
                    self.registered = True
                    self._start_heartbeat()
                    return result
                else:
                    error_text = await resp.text()
                    logger.error(f"Registration failed with status {resp.status}: {error_text}")
                    return {"error": f"Registration failed with status {resp.status}", "message": error_text}
        except Exception as e:
            logger.error(f"Error registering with hub {self.config.hub_url}: {str(e)}")
            return {"error": f"Error registering with hub", "message": str(e)}

    def _start_heartbeat(self):
        """Start the heartbeat background task."""
        if not self._heartbeat_task:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to the hub."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if not self.registered:
                    continue

                # URL encode the agent URL
                encoded_url = urllib.parse.quote(self.config.agent_url, safe='')

                # Send heartbeat
                async with self.session.post(f"{self.config.hub_url}/heartbeat/{encoded_url}") as resp:
                    if resp.status != 200:
                        logger.warning(f"Heartbeat failed: {await resp.text()}")
                        # If heartbeat failed, try to re-register
                        self.registered = False
                        await self.register_with_hub()
                    else:
                        logger.debug("Heartbeat sent successfully")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def discover_agents(self, tags: Optional[List[str]] = None, category: Optional[str] = None) -> List[
        AgentInfo]:
        """Discover agents on the AIRA network.

        Args:
            tags: List of tags to filter agents by
            category: Category to filter agents by

        Returns:
            List of agent information dictionaries
        """
        if self.session is None:
            await self.start()

        try:
            # Prepare query parameters
            query_params = {}
            if tags:
                query_params["tags"] = ",".join(tags)
            if category:
                query_params["category"] = category

            # Send discovery request
            async with self.session.get(f"{self.config.hub_url}/agents", params=query_params) as resp:
                if resp.status == 200:
                    agents = await resp.json()
                    logger.info(f"Discovered {len(agents)} agents")
                    return agents
                else:
                    error_text = await resp.text()
                    logger.error(f"Discovery failed with status {resp.status}: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error discovering agents: {str(e)}")
            return []

    async def discover_agent_tools(self, agent_url: str) -> List[ToolInfo]:
        """Discover tools provided by a specific agent.

        Args:
            agent_url: URL of the agent to discover tools from

        Returns:
            List of tool information dictionaries
        """
        if self.session is None:
            await self.start()

        try:
            # First try the A2A agent card endpoint
            normalized_url = agent_url.rstrip('/')

            # Try the well-known agent card endpoint
            async with self.session.get(f"{normalized_url}/.well-known/agent.json") as resp:
                if resp.status == 200:
                    agent_card = await resp.json()
                    # Extract tools/skills from agent card
                    skills = agent_card.get("skills", [])
                    tools = []

                    for skill in skills:
                        if "tool" in skill.get("tags", []):
                            tool_info = {
                                "name": skill.get("name"),
                                "description": skill.get("description", ""),
                                "parameters": skill.get("parameters", {}),
                                "agent_url": agent_url
                            }
                            tools.append(tool_info)

                    logger.info(f"Discovered {len(tools)} tools from agent {agent_url}")
                    return tools
                else:
                    # Fallback to hub agent endpoint
                    async with self.session.get(
                            f"{self.config.hub_url}/agents/{urllib.parse.quote(agent_url, safe='')}") as agent_resp:
                        if agent_resp.status == 200:
                            agent_info = await agent_resp.json()
                            skills = agent_info.get("skills", [])
                            tools = []

                            for skill in skills:
                                if "tool" in skill.get("tags", []):
                                    tool_info = {
                                        "name": skill.get("name"),
                                        "description": skill.get("description", ""),
                                        "parameters": skill.get("parameters", {}),
                                        "agent_url": agent_url
                                    }
                                    tools.append(tool_info)

                            logger.info(f"Discovered {len(tools)} tools from agent {agent_url} via hub")
                            return tools
                        else:
                            logger.error(f"Failed to get agent info: {await agent_resp.text()}")
                            return []
        except Exception as e:
            logger.error(f"Error discovering agent tools: {str(e)}")
            return []

    async def call_tool(self, agent_url: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool on another agent.

        Args:
            agent_url: URL of the agent that hosts the tool
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool

        Returns:
            The result of the tool call
        """
        if self.session is None:
            await self.start()

        # Ensure the agent URL is normalized
        normalized_url = agent_url.rstrip('/')

        # Ensure the URL ends with /a2a for the A2A protocol
        if not normalized_url.endswith('/a2a'):
            a2a_url = f"{normalized_url}/a2a"
        else:
            a2a_url = normalized_url

        # Create a task ID
        task_id = f"task-{int(time.time())}"

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
                        "text": f"Use the {tool_name} tool with parameters: {json.dumps(parameters)}"
                    }]
                }
            }
        }

        logger.info(f"Calling tool '{tool_name}' on agent at {a2a_url}")
        logger.debug(f"Request: {json.dumps(request, indent=2)}")

        try:
            # Send the request
            retries = 0
            response = None

            while retries <= self.config.max_retries:
                try:
                    async with self.session.post(a2a_url, json=request) as resp:
                        if resp.status == 200:
                            response = await resp.json()
                            break
                        else:
                            error_text = await resp.text()
                            logger.warning(f"Tool call failed with status {resp.status}: {error_text}")
                            retries += 1
                            if retries <= self.config.max_retries:
                                await asyncio.sleep(1)  # Wait before retrying
                except Exception as e:
                    logger.warning(f"Error during tool call: {str(e)}")
                    retries += 1
                    if retries <= self.config.max_retries:
                        await asyncio.sleep(1)  # Wait before retrying

            if response is None:
                return {"error": "Tool call failed after retries"}

            logger.debug(f"Response: {json.dumps(response, indent=2)}")

            # Extract the result from the artifacts
            if "result" in response:
                task_result = response["result"]
                artifacts = task_result.get("artifacts", [])

                if artifacts:
                    # Get the text part from the first artifact
                    parts = artifacts[0].get("parts", [])
                    text_part = next((p for p in parts if p.get("type") == "text"), None)

                    if text_part and "text" in text_part:
                        try:
                            # Try to parse as JSON
                            return json.loads(text_part["text"])
                        except json.JSONDecodeError:
                            # Return as plain text if not JSON
                            return text_part["text"]

            # Return the whole response if we couldn't extract a better result
            return response
        except Exception as e:
            logger.error(f"Error calling tool: {str(e)}")
            return {"error": f"Error calling tool: {str(e)}"}

    def add_tool(self, func: Union[ToolFunction, AsyncToolFunction], name: Optional[str] = None,
                 description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None) -> str:
        """Add a tool to this agent.

        Args:
            func: Function that implements the tool
            name: Name of the tool (if None, function name is used)
            description: Description of the tool
            parameters: JSON Schema for the tool parameters
            tags: Tags for the tool

        Returns:
            The name of the tool
        """
        # Get tool name from function if not provided
        tool_name = name or func.__name__

        # Get description from function docstring if not provided
        if description is None and func.__doc__:
            description = func.__doc__.strip()
        else:
            description = description or f"Tool: {tool_name}"

        # Create tool metadata
        tool_metadata = {
            "description": description,
            "parameters": parameters or {},
            "tags": tags or ["tool"]
        }

        # Store the tool
        self.tools[tool_name] = (func, tool_metadata)

        logger.info(f"Added tool '{tool_name}' to agent")

        # If already registered, re-register to update tools
        if self.registered:
            asyncio.create_task(self.register_with_hub())

        return tool_name

    async def handle_a2a_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming A2A protocol request.

        Args:
            request: A2A protocol request

        Returns:
            A2A protocol response
        """
        method = request.get("method")
        req_id = request.get("id", 1)

        if method == "tasks/send":
            return await self._handle_tasks_send(request.get("params", {}), req_id)
        elif method == "tasks/get":
            return await self._handle_tasks_get(request.get("params", {}), req_id)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method {method} not supported"
                }
            }

    async def _handle_tasks_send(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        """Handle an A2A tasks/send request.

        Args:
            params: Request parameters
            req_id: Request ID

        Returns:
            A2A protocol response
        """
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

        # Parse parameters from JSON if available
        if "parameters:" in text.lower() or "with parameters:" in text.lower():
            try:
                # Extract the JSON part
                json_start = text.find('{')
                if json_start != -1:
                    json_part = text[json_start:]
                    # Parse the parameters
                    parsed_params = json.loads(json_part)
                    tool_params = parsed_params
                    logger.debug(f"Extracted parameters: {tool_params}")

                    # Also extract tool name from text if not already set
                    for name in self.tools.keys():
                        if name.lower() in text.lower():
                            tool_name = name
                            break
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing parameters JSON: {str(e)}")

        # Simple parsing fallback - look for tool name in text
        if not tool_name:
            for name in self.tools.keys():
                if name.lower() in text.lower():
                    tool_name = name
                    break

        if not tool_name or tool_name not in self.tools:
            # Return help message
            available_tools = list(self.tools.keys())
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "sessionId": params.get("sessionId", f"session-{task_id}"),
                    "status": {"state": "completed"},
                    "artifacts": [{
                        "parts": [{
                            "type": "text",
                            "text": f"I'm not sure which tool you want to use. Available tools: {', '.join(available_tools)}"
                        }]
                    }]
                }
            }

        # Execute the tool
        try:
            tool_func, _ = self.tools[tool_name]

            # Execute the tool function (async or sync)
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(tool_params)
            else:
                result = tool_func(tool_params)

            # Convert result to string if it's not a dict
            if not isinstance(result, dict) and not isinstance(result, str):
                result = str(result)

            # Format the result as an A2A task response
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "sessionId": params.get("sessionId", f"session-{task_id}"),
                    "status": {"state": "completed"},
                    "artifacts": [{
                        "parts": [{
                            "type": "text",
                            "text": json.dumps(result) if isinstance(result, dict) else result
                        }]
                    }]
                }
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return self._create_error_response(f"Error executing tool: {str(e)}", req_id)

    async def _handle_tasks_get(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        """Handle an A2A tasks/get request.

        Args:
            params: Request parameters
            req_id: Request ID

        Returns:
            A2A protocol response
        """
        task_id = params.get("id")

        # Simple implementation - just return an empty task
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "id": task_id,
                "sessionId": f"session-{task_id}",
                "status": {"state": "completed"},
                "artifacts": [],
                "history": []
            }
        }

    def _create_error_response(self, message: str, req_id: Any) -> Dict[str, Any]:
        """Create an A2A protocol error response.

        Args:
            message: Error message
            req_id: Request ID

        Returns:
            A2A protocol error response
        """
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32000,
                "message": message
            }
        }


class AiraClient:
    """High-level client for the AIRA network."""

    def __init__(self,
                 hub_url: str = "https://aira-fl8f.onrender.com",
                 agent_url: Optional[str] = None,
                 agent_name: Optional[str] = None,
                 agent_description: str = "AIRA Universal Client Agent",
                 auto_register: bool = True,
                 heartbeat_interval: int = 30,
                 max_retries: int = 3,
                 timeout: int = 30,
                 log_level: int = logging.INFO):
        """Initialize AiraClient.

        Args:
            hub_url: URL of the AIRA hub
            agent_url: URL where this agent is accessible (if None, a local URL is generated)
            agent_name: Name of this agent (if None, a name is generated)
            agent_description: Description of this agent
            auto_register: Whether to automatically register with the hub on startup
            heartbeat_interval: Interval in seconds for sending heartbeats to the hub
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for requests
            log_level: Logging level
        """
        # Create configuration
        self.config = AiraClientConfig(
            hub_url=hub_url,
            agent_url=agent_url,
            agent_name=agent_name,
            agent_description=agent_description,
            auto_register=auto_register,
            heartbeat_interval=heartbeat_interval,
            max_retries=max_retries,
            timeout=timeout,
            log_level=log_level
        )

        # Create agent
        self.agent = AiraAgent(self.config)

        # Initialize caches
        self.discovered_agents: Dict[str, AgentInfo] = {}
        self.discovered_tools: Dict[str, Dict[str, ToolInfo]] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the client."""
        await self.agent.start()

    async def stop(self):
        """Stop the client."""
        await self.agent.stop()

    async def discover_agents(self, tags: Optional[List[str]] = None, category: Optional[str] = None,
                              refresh: bool = False) -> List[AgentInfo]:
        """Discover agents on the AIRA network.

        Args:
            tags: List of tags to filter agents by
            category: Category to filter agents by
            refresh: Whether to force a refresh from the hub

        Returns:
            List of agent information dictionaries
        """
        if refresh or not self.discovered_agents:
            agents = await self.agent.discover_agents(tags, category)

            # Update cache
            for agent in agents:
                if "url" in agent:
                    self.discovered_agents[agent["url"]] = agent

            return agents
        else:
            # Filter cached agents by tags and category
            filtered_agents = list(self.discovered_agents.values())

            if tags:
                filtered_agents = [a for a in filtered_agents if set(tags).issubset(set(a.get("tags", [])))]

            if category:
                filtered_agents = [a for a in filtered_agents if a.get("category") == category]

            return filtered_agents

    async def discover_agent_tools(self, agent_url: str, refresh: bool = False) -> List[ToolInfo]:
        """Discover tools provided by a specific agent.

        Args:
            agent_url: URL of the agent to discover tools from
            refresh: Whether to force a refresh from the agent

        Returns:
            List of tool information dictionaries
        """
        if refresh or agent_url not in self.discovered_tools:
            tools = await self.agent.discover_agent_tools(agent_url)

            # Update cache
            self.discovered_tools[agent_url] = {tool["name"]: tool for tool in tools}

            return tools
        else:
            return list(self.discovered_tools[agent_url].values())

    async def call_tool(self, agent_url: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool on another agent.

        Args:
            agent_url: URL of the agent that hosts the tool
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool

        Returns:
            The result of the tool call
        """
        return await self.agent.call_tool(agent_url, tool_name, parameters)

    def add_tool(self, func: Union[ToolFunction, AsyncToolFunction], name: Optional[str] = None,
                 description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None) -> str:
        """Add a tool to this client's agent.

        Args:
            func: Function that implements the tool
            name: Name of the tool (if None, function name is used)
            description: Description of the tool
            parameters: JSON Schema for the tool parameters
            tags: Tags for the tool

        Returns:
            The name of the tool
        """
        return self.agent.add_tool(func, name, description, parameters, tags)

    def set_agent_metadata(self, metadata: Dict[str, Any]):
        """Set additional metadata for this client's agent.

        Args:
            metadata: Metadata dictionary to set
        """
        self.agent.agent_metadata.update(metadata)

        # Re-register if already registered
        if self.agent.registered:
            asyncio.create_task(self.agent.register_with_hub())

    async def handle_a2a_request(self, request_body: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Handle an incoming A2A protocol request.

        Args:
            request_body: A2A protocol request as JSON string or dict

        Returns:
            A2A protocol response as dict
        """
        if isinstance(request_body, str):
            try:
                request = json.loads(request_body)
            except json.JSONDecodeError:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
        else:
            request = request_body

        return await self.agent.handle_a2a_request(request)

    async def search_tools(self, query: str) -> List[Tuple[ToolInfo, AgentInfo]]:
        """Search for tools across all agents that match a query.

        Args:
            query: Search query string

        Returns:
            List of (tool_info, agent_info) tuples
        """
        # First discover all agents
        agents = await self.discover_agents()

        results = []

        # For each agent, discover tools and filter by query
        for agent in agents:
            agent_url = agent.get("url")
            if not agent_url:
                continue

            tools = await self.discover_agent_tools(agent_url)

            for tool in tools:
                # Check if the query matches the tool name or description
                if (query.lower() in tool.get("name", "").lower() or
                        query.lower() in tool.get("description", "").lower()):
                    results.append((tool, agent))

        return results

    # --- Framework-specific adapters ---

    async def get_agent_urls_by_name(self, name_pattern: str) -> List[str]:
        """Find agent URLs based on a name pattern.

        Args:
            name_pattern: Pattern to match against agent names

        Returns:
            List of matching agent URLs
        """
        agents = await self.discover_agents()

        matching_urls = []
        for agent in agents:
            if name_pattern.lower() in agent.get("name", "").lower():
                if "url" in agent:
                    matching_urls.append(agent["url"])

        return matching_urls

    def create_web_server(self, host: str = "localhost", port: int = 8000):
        """Create a simple web server to handle incoming A2A requests.

        Args:
            host: Host to bind to
            port: Port to bind to

        Returns:
            Web server application (must be run with app.run() or similar)
        """
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            import uvicorn

            app = FastAPI(title="AIRA Agent")

            @app.post("/a2a")
            async def a2a_handler(request: Request):
                request_body = await request.body()
                response = await self.handle_a2a_request(request_body)
                return JSONResponse(content=response)

            @app.get("/")
            async def root():
                return {
                    "name": self.config.agent_name,
                    "description": self.config.agent_description,
                    "tools": list(self.agent.tools.keys())
                }

            @app.get("/.well-known/agent.json")
            async def agent_card():
                skills = []
                for tool_name, (_, tool_metadata) in self.agent.tools.items():
                    skill = {
                        "id": f"tool-{tool_name}",
                        "name": tool_name,
                        "description": tool_metadata.get("description", f"Tool: {tool_name}"),
                        "tags": tool_metadata.get("tags", ["tool"]),
                        "parameters": tool_metadata.get("parameters", {})
                    }
                    skills.append(skill)

                return {
                    "name": self.config.agent_name,
                    "description": self.config.agent_description,
                    "url": self.config.agent_url,
                    "skills": skills
                }

            class Server:
                def __init__(self, app, host, port):
                    self.app = app
                    self.host = host
                    self.port = port

                def run(self):
                    """Run the server (blocking)."""
                    uvicorn.run(self.app, host=self.host, port=self.port)

                async def start_async(self):
                    """Start the server asynchronously."""
                    config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
                    server = uvicorn.Server(config)
                    await server.serve()

            return Server(app, host, port)
        except ImportError:
            logger.error("FastAPI and uvicorn are required for web server. Install with: pip install fastapi uvicorn")
            raise


# --- Framework-specific adapters ---

class GoogleAdkAdapter:
    """Adapter for Google's Agent Development Kit (ADK)."""

    def __init__(self, aira_client: AiraClient):
        """Initialize GoogleAdkAdapter.

        Args:
            aira_client: AiraClient instance
        """
        self.aira_client = aira_client

    def expose_adk_tool(self, adk_tool):
        """Expose an ADK tool to the AIRA network.

        Args:
            adk_tool: ADK tool (e.g., FunctionTool instance)

        Returns:
            The name of the exposed tool
        """
        import inspect

        # Extract tool information
        tool_name = getattr(adk_tool, "name", None)
        tool_description = getattr(adk_tool, "description", None)

        # Extract parameters from the tool
        parameters = {}
        if hasattr(adk_tool, "run") and inspect.ismethod(adk_tool.run):
            sig = inspect.signature(adk_tool.run)
            for param_name, param in sig.parameters.items():
                if param_name != "self" and param_name != "tool_context":
                    param_type = param.annotation
                    if param_type is inspect.Parameter.empty:
                        param_type = "string"

                    parameters[param_name] = {
                        "type": self._convert_type(param_type),
                        "description": f"Parameter: {param_name}"
                    }

        # Define the AIRA tool function
        async def aira_tool_function(params):
            try:
                # In a real implementation, you'd need proper ToolContext handling
                if hasattr(adk_tool, "run_async"):
                    return await adk_tool.run_async(args=params, tool_context=None)
                else:
                    return adk_tool.run(args=params, tool_context=None)
            except Exception as e:
                logger.error(f"Error executing ADK tool: {str(e)}")
                return {"error": str(e)}

        # Add the tool to the AIRA client
        return self.aira_client.add_tool(
            aira_tool_function,
            name=tool_name,
            description=tool_description,
            parameters={"type": "object", "properties": parameters},
            tags=["adk", "tool"]
        )

    def create_adk_tool_from_aira(self, agent_url: str, tool_name: str):
        """Create an ADK tool that calls an AIRA tool.

        Args:
            agent_url: URL of the agent hosting the tool
            tool_name: Name of the tool

        Returns:
            ADK tool instance
        """
        try:
            from google.adk.tools.function_tool import FunctionTool

            # Define the function that will be wrapped as an ADK tool
            async def tool_function(args, tool_context=None):
                return await self.aira_client.call_tool(agent_url, tool_name, args)

            # Discover tool parameters (if possible)
            tool_info = None
            try:
                tools =  self.aira_client.discover_agent_tools(agent_url)
                tool_info = next((t for t in tools if t["name"] == tool_name), None)
            except Exception as e:
                logger.warning(f"Could not discover tool parameters: {str(e)}")

            # Create the ADK tool
            tool_function.__name__ = tool_name
            if tool_info:
                tool_function.__doc__ = tool_info.get("description", f"AIRA tool: {tool_name}")

            return FunctionTool(tool_function)
        except ImportError:
            logger.error("Google ADK is required for this adapter. Install with: pip install google-adk")
            raise

    def _convert_type(self, param_type):
        """Convert Python type to JSON Schema type.

        Args:
            param_type: Python type

        Returns:
            JSON Schema type string
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        return type_map.get(param_type, "string")


class LangchainAdapter:
    """Adapter for LangChain framework."""

    def __init__(self, aira_client: AiraClient):
        """Initialize LangchainAdapter.

        Args:
            aira_client: AiraClient instance
        """
        self.aira_client = aira_client

    def expose_langchain_tool(self, lc_tool):
        """Expose a LangChain tool to the AIRA network.

        Args:
            lc_tool: LangChain tool instance

        Returns:
            The name of the exposed tool
        """
        # Extract tool information
        tool_name = getattr(lc_tool, "name", None)
        tool_description = getattr(lc_tool, "description", None)

        # Define the AIRA tool function
        async def aira_tool_function(params):
            try:
                # Check if the tool takes a single string or dict input
                if hasattr(lc_tool, "_run") and callable(lc_tool._run):
                    if len(params) == 1 and "input" in params:
                        result = lc_tool._run(params["input"])
                    else:
                        result = lc_tool._run(json.dumps(params))
                elif hasattr(lc_tool, "func") and callable(lc_tool.func):
                    if len(params) == 1 and "input" in params:
                        result = lc_tool.func(params["input"])
                    else:
                        result = lc_tool.func(**params)
                else:
                    return {"error": "Unsupported LangChain tool type"}

                return result
            except Exception as e:
                logger.error(f"Error executing LangChain tool: {str(e)}")
                return {"error": str(e)}

        # Add the tool to the AIRA client
        return self.aira_client.add_tool(
            aira_tool_function,
            name=tool_name,
            description=tool_description,
            parameters={"type": "object", "properties": {"input": {"type": "string"}}},
            tags=["langchain", "tool"]
        )

    def create_langchain_tool_from_aira(self, agent_url: str, tool_name: str):
        """Create a LangChain tool that calls an AIRA tool.

        Args:
            agent_url: URL of the agent hosting the tool
            tool_name: Name of the tool

        Returns:
            LangChain tool instance
        """
        try:
            from langchain.tools import BaseTool

            # Discover tool info
            tool_info = None
            try:
                tools = self.aira_client.discover_agent_tools(agent_url)
                tool_info = next((t for t in tools if t["name"] == tool_name), None)
            except Exception as e:
                logger.warning(f"Could not discover tool parameters: {str(e)}")

            class AiraTool(BaseTool):
                name = tool_name
                description = tool_info.get("description",
                                            f"AIRA tool: {tool_name}") if tool_info else f"AIRA tool: {tool_name}"

                def _run(self, input_str):
                    """Run the tool with a string input."""
                    try:
                        # Try to parse as JSON first
                        try:
                            params = json.loads(input_str)
                            if not isinstance(params, dict):
                                params = {"input": input_str}
                        except json.JSONDecodeError:
                            params = {"input": input_str}

                        loop = asyncio.get_event_loop()
                        return loop.run_until_complete(
                            self.aira_client.call_tool(agent_url, tool_name, params)
                        )
                    except Exception as e:
                        logger.error(f"Error calling AIRA tool: {str(e)}")
                        return {"error": str(e)}

                async def _arun(self, input_str):
                    """Run the tool asynchronously with a string input."""
                    try:
                        # Try to parse as JSON first
                        try:
                            params = json.loads(input_str)
                            if not isinstance(params, dict):
                                params = {"input": input_str}
                        except json.JSONDecodeError:
                            params = {"input": input_str}

                        return await self.aira_client.call_tool(agent_url, tool_name, params)
                    except Exception as e:
                        logger.error(f"Error calling AIRA tool: {str(e)}")
                        return {"error": str(e)}

            # Add the AIRA client to the tool instance
            tool = AiraTool()
            tool.aira_client = self.aira_client

            return tool
        except ImportError:
            logger.error("LangChain is required for this adapter. Install with: pip install langchain")
            raise


class McpAdapter:
    """Adapter for Model Context Protocol (MCP)."""

    def __init__(self, aira_client: AiraClient):
        """Initialize McpAdapter.

        Args:
            aira_client: AiraClient instance
        """
        self.aira_client = aira_client

    def expose_mcp_tool(self, tool_name: str, description: str, parameters: Dict[str, Any],
                        implementation: AsyncToolFunction):
        """Expose an MCP tool to the AIRA network.

        Args:
            tool_name: Name of the tool
            description: Description of the tool
            parameters: JSON Schema for the tool parameters
            implementation: Function that implements the tool

        Returns:
            The name of the exposed tool
        """
        # Add the tool to the AIRA client
        return self.aira_client.add_tool(
            implementation,
            name=tool_name,
            description=description,
            parameters=parameters,
            tags=["mcp", "tool"]
        )

    async def create_mcp_tools_from_aira(self, agent_url: str):
        """Create MCP tools from tools provided by an AIRA agent.

        Args:
            agent_url: URL of the agent to get tools from

        Returns:
            Dictionary mapping tool names to MCP tool definitions and implementations
        """
        try:
            # Import MCP types
            from mcp import types as mcp_types

            # Discover tools
            tools = await self.aira_client.discover_agent_tools(agent_url)

            mcp_tools = {}

            for tool_info in tools:
                tool_name = tool_info.get("name")
                description = tool_info.get("description", f"AIRA tool: {tool_name}")
                parameters = tool_info.get("parameters", {})

                # Create MCP tool definition
                tool_def = mcp_types.Tool(
                    name=tool_name,
                    description=description,
                    parameters=parameters
                )

                # Create implementation function
                async def tool_implementation(tool_args):
                    try:
                        return await self.aira_client.call_tool(agent_url, tool_name, tool_args)
                    except Exception as e:
                        logger.error(f"Error calling AIRA tool: {str(e)}")
                        return {"error": str(e)}

                # Add to dictionary
                mcp_tools[tool_name] = (tool_def, tool_implementation)

            return mcp_tools
        except ImportError:
            logger.error("MCP is required for this adapter. Install with: pip install mcp")
            raise

    async def register_mcp_tools_with_server(self, agent_url: str, mcp_server):
        """Register AIRA tools with an MCP server.

        Args:
            agent_url: URL of the agent to get tools from
            mcp_server: MCP server instance

        Returns:
            List of registered tool names
        """
        mcp_tools = await self.create_mcp_tools_from_aira(agent_url)

        registered_tools = []

        for tool_name, (tool_def, tool_impl) in mcp_tools.items():
            if hasattr(mcp_server, "add_tool"):
                mcp_server.add_tool(tool_def, tool_impl)
                registered_tools.append(tool_name)
            else:
                logger.warning(f"MCP server does not have add_tool method. Cannot register {tool_name}")

        return registered_tools


# --- Main setup function ---

def setup_aira_client(hub_url: Optional[str] = None, agent_url: Optional[str] = None,
                      agent_name: Optional[str] = None, log_level: int = logging.INFO,
                      auto_register: bool = True) -> AiraClient:
    """Set up an AIRA client with common defaults.

    Args:
        hub_url: URL of the AIRA hub (defaults to environment variable AIRA_HUB_URL or predefined default)
        agent_url: URL where this agent is accessible (defaults to environment variable AIRA_AGENT_URL)
        agent_name: Name of this agent (defaults to environment variable AIRA_AGENT_NAME or a generated name)
        log_level: Logging level
        auto_register: Whether to automatically register with the hub on startup

    Returns:
        Configured AiraClient instance
    """
    # Get hub URL from environment or use default
    if hub_url is None:
        hub_url = os.environ.get("AIRA_HUB_URL", "https://aira-fl8f.onrender.com")

    # Get agent URL from environment or use default
    if agent_url is None:
        agent_url = os.environ.get("AIRA_AGENT_URL", None)

    # Get agent name from environment or use default
    if agent_name is None:
        agent_name = os.environ.get("AIRA_AGENT_NAME", None)

    # Create client
    return AiraClient(
        hub_url=hub_url,
        agent_url=agent_url,
        agent_name=agent_name,
        log_level=log_level,
        auto_register=auto_register
    )