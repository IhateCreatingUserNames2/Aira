"""
Enhanced MCP-A2A Bridge
=======================

A robust bidirectional bridge between the Model Context Protocol (MCP) and
the Agent-to-Agent (A2A) protocol, enabling seamless interoperability between
agents using different protocols.

Key features:
1. MCP servers are automatically exposed as A2A agents
2. A2A tools can be accessed as MCP resources
3. Streamlined authentication and permission management
4. Proper handling of streaming responses

This bridge is designed to work with both protocols natively, with minimal
configuration required from the user.
"""

import asyncio
import json
import logging
import os
import time
import uuid
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set, Tuple, Type, TypeVar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_a2a_bridge")

# Type aliases
ToolFunction = Callable[[Dict[str, Any]], Any]
AsyncToolFunction = Callable[[Dict[str, Any]], Awaitable[Any]]


# --- Authentication and Permission Management ---

class PermissionScope:
    """Permission scope for authentication and authorization."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class AuthToken:
    """Authentication token with scopes and expiration."""

    def __init__(self, token: str, scopes: List[str], expires_at: Optional[float] = None):
        """Initialize AuthToken.

        Args:
            token: Token string
            scopes: Permission scopes
            expires_at: Expiration timestamp (None for no expiration)
        """
        self.token = token
        self.scopes = scopes
        self.expires_at = expires_at
        self.created_at = time.time()

    def is_valid(self) -> bool:
        """Check if the token is valid (not expired)."""
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Check if the token has a specific scope."""
        return scope in self.scopes or PermissionScope.ADMIN in self.scopes


class AuthManager:
    """Manager for authentication and authorization."""

    def __init__(self):
        """Initialize AuthManager."""
        self.tokens: Dict[str, AuthToken] = {}

    def create_token(self, scopes: List[str], expires_in_days: Optional[int] = None) -> str:
        """Create a new authentication token.

        Args:
            scopes: Permission scopes
            expires_in_days: Token expiration in days (None for no expiration)

        Returns:
            Token string
        """
        token = uuid.uuid4().hex

        expires_at = None
        if expires_in_days is not None:
            expires_at = time.time() + (expires_in_days * 86400)

        self.tokens[token] = AuthToken(token, scopes, expires_at)

        return token

    def validate_token(self, token: str) -> bool:
        """Validate a token.

        Args:
            token: Token string

        Returns:
            True if token is valid, False otherwise
        """
        if token not in self.tokens:
            return False

        if not self.tokens[token].is_valid():
            del self.tokens[token]
            return False

        return True

    def check_permission(self, token: str, scope: str) -> bool:
        """Check if a token has a specific permission scope.

        Args:
            token: Token string
            scope: Permission scope

        Returns:
            True if token has permission, False otherwise
        """
        if not self.validate_token(token):
            return False

        return self.tokens[token].has_scope(scope)

    def revoke_token(self, token: str) -> bool:
        """Revoke a token.

        Args:
            token: Token string

        Returns:
            True if token was revoked, False if token was not found
        """
        if token in self.tokens:
            del self.tokens[token]
            return True
        return False


# --- MCP Components ---

class McpTool:
    """MCP tool definition."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """Initialize McpTool.

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters


class McpResource:
    """MCP resource definition."""

    def __init__(self, uri: str, description: str, mime_type: str = "text/plain"):
        """Initialize McpResource.

        Args:
            uri: Resource URI
            description: Resource description
            mime_type: Resource MIME type
        """
        self.uri = uri
        self.description = description
        self.mime_type = mime_type


class McpServer:
    """Base MCP server implementation."""

    def __init__(self, name: str, description: str):
        """Initialize McpServer.

        Args:
            name: Server name
            description: Server description
        """
        self.name = name
        self.description = description
        self.tools: Dict[str, Tuple[McpTool, AsyncToolFunction]] = {}
        self.resources: Dict[str, Tuple[McpResource, AsyncToolFunction]] = {}
        self.auth_manager = AuthManager()

    def add_tool(self, tool: McpTool, implementation: AsyncToolFunction):
        """Add a tool to the server.

        Args:
            tool: Tool definition
            implementation: Tool implementation function
        """
        self.tools[tool.name] = (tool, implementation)

    def add_resource(self, resource: McpResource, implementation: AsyncToolFunction):
        """Add a resource to the server.

        Args:
            resource: Resource definition
            implementation: Resource implementation function
        """
        self.resources[resource.uri] = (resource, implementation)

    async def call_tool(self, name: str, arguments: Dict[str, Any], token: Optional[str] = None) -> Any:
        """Call a tool.

        Args:
            name: Tool name
            arguments: Tool arguments
            token: Authentication token

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found or permission denied
        """
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")

        # Check permissions if token provided
        if token is not None and not self.auth_manager.check_permission(token, PermissionScope.EXECUTE):
            raise ValueError(f"Permission denied for tool {name}")

        _, implementation = self.tools[name]

        return await implementation(arguments)

    async def read_resource(self, uri: str, token: Optional[str] = None) -> Tuple[Any, str]:
        """Read a resource.

        Args:
            uri: Resource URI
            token: Authentication token

        Returns:
            Tuple of (resource content, mime type)

        Raises:
            ValueError: If resource not found or permission denied
        """
        if uri not in self.resources:
            raise ValueError(f"Resource {uri} not found")

        # Check permissions if token provided
        if token is not None and not self.auth_manager.check_permission(token, PermissionScope.READ):
            raise ValueError(f"Permission denied for resource {uri}")

        resource, implementation = self.resources[uri]

        return await implementation({}), resource.mime_type

    async def list_tools(self, token: Optional[str] = None) -> List[McpTool]:
        """List available tools.

        Args:
            token: Authentication token

        Returns:
            List of tools

        Raises:
            ValueError: If permission denied
        """
        # Check permissions if token provided
        if token is not None and not self.auth_manager.check_permission(token, PermissionScope.READ):
            raise ValueError("Permission denied for listing tools")

        return [tool for tool, _ in self.tools.values()]

    async def list_resources(self, token: Optional[str] = None) -> List[McpResource]:
        """List available resources.

        Args:
            token: Authentication token

        Returns:
            List of resources

        Raises:
            ValueError: If permission denied
        """
        # Check permissions if token provided
        if token is not None and not self.auth_manager.check_permission(token, PermissionScope.READ):
            raise ValueError("Permission denied for listing resources")

        return [resource for resource, _ in self.resources.values()]


# --- A2A Components ---

class A2AAgent:
    """Base A2A agent implementation."""

    def __init__(self, name: str, description: str, url: str):
        """Initialize A2AAgent.

        Args:
            name: Agent name
            description: Agent description
            url: Agent URL
        """
        self.name = name
        self.description = description
        self.url = url
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.auth_manager = AuthManager()

    def add_skill(self, skill_id: str, name: str, description: str, parameters: Dict[str, Any],
                  implementation: AsyncToolFunction, tags: List[str] = None):
        """Add a skill to the agent.

        Args:
            skill_id: Skill ID
            name: Skill name
            description: Skill description
            parameters: Skill parameters (JSON Schema)
            implementation: Skill implementation function
            tags: Skill tags
        """
        self.skills[skill_id] = {
            "id": skill_id,
            "name": name,
            "description": description,
            "parameters": parameters,
            "implementation": implementation,
            "tags": tags or ["tool"]
        }

    def generate_agent_card(self) -> Dict[str, Any]:
        """Generate an A2A agent card.

        Returns:
            Agent card dictionary
        """
        skills = []

        for skill_id, skill in self.skills.items():
            skill_data = {
                "id": skill_id,
                "name": skill["name"],
                "description": skill["description"],
                "tags": skill["tags"],
                "parameters": skill["parameters"]
            }
            skills.append(skill_data)

        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "skills": skills
        }

    async def handle_a2a_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an A2A protocol request.

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
        elif method == "tasks/cancel":
            return await self._handle_tasks_cancel(request.get("params", {}), req_id)
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

        # Parse the message to identify skill
        text = text_part.get("text")
        skill_id = None
        skill_params = {}

        # Parse parameters from JSON if available
        if "parameters:" in text.lower() or "with parameters:" in text.lower():
            try:
                # Extract the JSON part
                json_start = text.find('{')
                if json_start != -1:
                    json_part = text[json_start:]
                    # Parse the parameters
                    parsed_params = json.loads(json_part)
                    skill_params = parsed_params
                    logger.debug(f"Extracted parameters: {skill_params}")

                    # Extract skill ID from text
                    for sid in self.skills.keys():
                        if sid.lower() in text.lower() or self.skills[sid]["name"].lower() in text.lower():
                            skill_id = sid
                            break
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing parameters JSON: {str(e)}")

        # Simple parsing fallback - look for skill ID in text
        if not skill_id:
            for sid, skill in self.skills.items():
                if sid.lower() in text.lower() or skill["name"].lower() in text.lower():
                    skill_id = sid
                    break

        if not skill_id or skill_id not in self.skills:
            # Return help message
            available_skills = [f"{skill['name']} ({sid})" for sid, skill in self.skills.items()]
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
                            "text": f"I'm not sure which skill you want to use. Available skills: {', '.join(available_skills)}"
                        }]
                    }]
                }
            }

        # Execute the skill
        try:
            skill = self.skills[skill_id]
            implementation = skill["implementation"]

            # Execute the skill function (async)
            result = await implementation(skill_params)

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
            logger.error(f"Error executing skill {skill_id}: {str(e)}")
            return self._create_error_response(f"Error executing skill: {str(e)}", req_id)

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

    async def _handle_tasks_cancel(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        """Handle an A2A tasks/cancel request.

        Args:
            params: Request parameters
            req_id: Request ID

        Returns:
            A2A protocol response
        """
        task_id = params.get("id")

        # Simple implementation - just acknowledge the cancellation
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "id": task_id,
                "sessionId": f"session-{task_id}",
                "status": {"state": "canceled"}
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


# --- Streaming Response Handling ---

class StreamingResponse:
    """Streaming response for SSE (Server-Sent Events)."""

    def __init__(self, task_id: str):
        """Initialize StreamingResponse.

        Args:
            task_id: Task ID
        """
        self.task_id = task_id
        self.buffer = []
        self.is_complete = False

    def add_chunk(self, chunk: str, is_final: bool = False):
        """Add a chunk to the response.

        Args:
            chunk: Response chunk
            is_final: Whether this is the final chunk
        """
        self.buffer.append(chunk)

        if is_final:
            self.is_complete = True

    def get_chunks(self) -> List[str]:
        """Get all buffered chunks.

        Returns:
            List of response chunks
        """
        chunks = self.buffer.copy()
        self.buffer = []
        return chunks

    def to_sse_event(self, chunk: str) -> str:
        """Convert a chunk to an SSE event.

        Args:
            chunk: Response chunk

        Returns:
            SSE event string
        """
        event_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "id": self.task_id,
                "artifact": {
                    "parts": [{"type": "text", "text": chunk}],
                    "append": len(self.buffer) > 0,
                    "lastChunk": self.is_complete
                }
            }
        }

        return f"data: {json.dumps(event_data)}\n\n"


# --- Enhanced MCP-A2A Bridge ---

class EnhancedMcpA2ABridge:
    """Enhanced bidirectional bridge between MCP and A2A protocols."""

    def __init__(self, name: str, description: str, url: str):
        """Initialize EnhancedMcpA2ABridge.

        Args:
            name: Bridge name
            description: Bridge description
            url: Bridge URL
        """
        self.name = name
        self.description = description
        self.url = url
        self.mcp_server = McpServer(name, description)
        self.a2a_agent = A2AAgent(name, description, url)
        self.streaming_responses: Dict[str, StreamingResponse] = {}
        self.http_session = None

    async def start(self):
        """Start the bridge."""
        import aiohttp

        if self.http_session is None:
            self.http_session = aiohttp.ClientSession()

    async def stop(self):
        """Stop the bridge."""
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None

    def expose_mcp_tool_as_a2a_skill(self, tool: McpTool, implementation: AsyncToolFunction):
        """Expose an MCP tool as an A2A skill.

        Args:
            tool: MCP tool
            implementation: Tool implementation
        """
        # Add to MCP server
        self.mcp_server.add_tool(tool, implementation)

        # Add to A2A agent
        self.a2a_agent.add_skill(
            skill_id=f"tool-{tool.name}",
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            implementation=implementation,
            tags=["mcp", "tool"]
        )

    def expose_a2a_skill_as_mcp_tool(self, skill_id: str, name: str, description: str,
                                     parameters: Dict[str, Any], implementation: AsyncToolFunction):
        """Expose an A2A skill as an MCP tool.

        Args:
            skill_id: Skill ID
            name: Skill name
            description: Skill description
            parameters: Skill parameters
            implementation: Skill implementation
        """
        # Add to A2A agent
        self.a2a_agent.add_skill(
            skill_id=skill_id,
            name=name,
            description=description,
            parameters=parameters,
            implementation=implementation,
            tags=["a2a", "tool"]
        )

        # Add to MCP server
        self.mcp_server.add_tool(
            McpTool(
                name=name,
                description=description,
                parameters=parameters
            ),
            implementation
        )

    def expose_mcp_resource_as_a2a_skill(self, resource: McpResource, implementation: AsyncToolFunction):
        """Expose an MCP resource as an A2A skill.

        Args:
            resource: MCP resource
            implementation: Resource implementation
        """
        # Add to MCP server
        self.mcp_server.add_resource(resource, implementation)

        # Create a wrapper function that returns the resource content
        async def resource_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            content, mime_type = await self.mcp_server.read_resource(resource.uri)
            return {
                "content": content,
                "mime_type": mime_type,
                "uri": resource.uri
            }

        # Add to A2A agent
        self.a2a_agent.add_skill(
            skill_id=f"resource-{resource.uri.replace('://', '-').replace('/', '-')}",
            name=f"Get {resource.uri}",
            description=resource.description,
            parameters={},
            implementation=resource_wrapper,
            tags=["mcp", "resource"]
        )

    def create_streaming_response(self, task_id: str) -> StreamingResponse:
        """Create a streaming response.

        Args:
            task_id: Task ID

        Returns:
            Streaming response
        """
        response = StreamingResponse(task_id)
        self.streaming_responses[task_id] = response
        return response

    async def handle_a2a_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an A2A protocol request.

        Args:
            request: A2A protocol request

        Returns:
            A2A protocol response
        """
        method = request.get("method")

        # Handle streaming requests
        if method == "tasks/sendSubscribe":
            return await self._handle_streaming_request(request)

        # Pass all other requests to the A2A agent
        return await self.a2a_agent.handle_a2a_request(request)

    async def _handle_streaming_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an A2A streaming request.

        Args:
            request: A2A protocol request

        Returns:
            A2A protocol response
        """
        req_id = request.get("id", 1)
        params = request.get("params", {})
        task_id = params.get("id")

        # Create a streaming response
        streaming_response = self.create_streaming_response(task_id)

        try:
            # Process the request asynchronously
            asyncio.create_task(self._process_streaming_task(task_id, params, streaming_response))

            # Return initial response
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "status": {"state": "working"},
                    "final": False
                }
            }
        except Exception as e:
            logger.error(f"Error handling streaming request: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": f"Error handling streaming request: {str(e)}"
                }
            }

    async def _process_streaming_task(self, task_id: str, params: Dict[str, Any],
                                      streaming_response: StreamingResponse):
        """Process a streaming task.

        Args:
            task_id: Task ID
            params: Request parameters
            streaming_response: Streaming response
        """
        message = params.get("message", {})

        if message.get("role") != "user" or not message.get("parts"):
            streaming_response.add_chunk("Invalid message format", is_final=True)
            return

        # Extract the message text
        text_part = next((p for p in message.get("parts", []) if p.get("type") == "text"), None)
        if not text_part or not text_part.get("text"):
            streaming_response.add_chunk("No text content found", is_final=True)
            return

        # Parse the message to identify skill
        text = text_part.get("text")
        skill_id = None
        skill_params = {}

        # Parse parameters from JSON if available
        if "parameters:" in text.lower() or "with parameters:" in text.lower():
            try:
                # Extract the JSON part
                json_start = text.find('{')
                if json_start != -1:
                    json_part = text[json_start:]
                    # Parse the parameters
                    parsed_params = json.loads(json_part)
                    skill_params = parsed_params

                    # Extract skill ID from text
                    for sid in self.a2a_agent.skills.keys():
                        if sid.lower() in text.lower() or self.a2a_agent.skills[sid]["name"].lower() in text.lower():
                            skill_id = sid
                            break
            except json.JSONDecodeError:
                pass

        # Simple parsing fallback - look for skill ID in text
        if not skill_id:
            for sid, skill in self.a2a_agent.skills.items():
                if sid.lower() in text.lower() or skill["name"].lower() in text.lower():
                    skill_id = sid
                    break

        if not skill_id or skill_id not in self.a2a_agent.skills:
            # Return help message
            available_skills = [f"{skill['name']} ({sid})" for sid, skill in self.a2a_agent.skills.items()]
            streaming_response.add_chunk(
                f"I'm not sure which skill you want to use. Available skills: {', '.join(available_skills)}",
                is_final=True
            )
            return

        # Execute the skill
        try:
            skill = self.a2a_agent.skills[skill_id]
            implementation = skill["implementation"]

            # Execute the skill function with streaming
            if hasattr(implementation, "__self__") and hasattr(implementation.__self__, "stream"):
                # Streaming implementation
                async for chunk in implementation.__self__.stream(skill_params):
                    streaming_response.add_chunk(chunk, is_final=False)

                streaming_response.add_chunk("", is_final=True)
            else:
                # Non-streaming implementation
                result = await implementation(skill_params)

                # Convert result to string if it's not a dict
                if isinstance(result, dict):
                    result = json.dumps(result)
                elif not isinstance(result, str):
                    result = str(result)

                streaming_response.add_chunk(result, is_final=True)
        except Exception as e:
            logger.error(f"Error executing streaming skill {skill_id}: {str(e)}")
            streaming_response.add_chunk(f"Error executing skill: {str(e)}", is_final=True)

    def get_streaming_chunks(self, task_id: str) -> List[str]:
        """Get streaming chunks for a task.

        Args:
            task_id: Task ID

        Returns:
            List of response chunks
        """
        if task_id in self.streaming_responses:
            return self.streaming_responses[task_id].get_chunks()
        return []

    def is_streaming_complete(self, task_id: str) -> bool:
        """Check if streaming is complete for a task.

        Args:
            task_id: Task ID

        Returns:
            True if streaming is complete, False otherwise
        """
        if task_id in self.streaming_responses:
            return self.streaming_responses[task_id].is_complete
        return True

    async def handle_mcp_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle an MCP protocol request.

        Args:
            method: MCP method
            params: Method parameters

        Returns:
            MCP response
        """
        if method == "list_tools":
            return await self.mcp_server.list_tools()
        elif method == "call_tool":
            return await self.mcp_server.call_tool(params["name"], params["arguments"])
        elif method == "list_resources":
            return await self.mcp_server.list_resources()
        elif method == "read_resource":
            return await self.mcp_server.read_resource(params["uri"])
        else:
            raise ValueError(f"Unsupported MCP method: {method}")

    def create_web_server(self, host: str = "localhost", port: int = 8000):
        """Create a web server for handling A2A and MCP requests.

        Args:
            host: Host to bind to
            port: Port to bind to

        Returns:
            Web server application
        """
        try:
            from fastapi import FastAPI, Request, Response, HTTPException
            from fastapi.responses import JSONResponse, StreamingResponse
            import uvicorn

            app = FastAPI(title=f"{self.name} - MCP-A2A Bridge")

            @app.post("/a2a")
            async def a2a_handler(request: Request):
                """Handle A2A protocol requests."""
                try:
                    request_body = await request.json()
                    response = await self.handle_a2a_request(request_body)
                    return JSONResponse(content=response)
                except Exception as e:
                    logger.error(f"Error handling A2A request: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32000,
                                "message": f"Error handling request: {str(e)}"
                            }
                        }
                    )

            @app.get("/a2a/stream/{task_id}")
            async def a2a_stream_handler(task_id: str):
                """Handle A2A streaming requests."""

                async def event_generator():
                    while not self.is_streaming_complete(task_id):
                        chunks = self.get_streaming_chunks(task_id)
                        for chunk in chunks:
                            if self.streaming_responses[task_id].is_complete and chunk == chunks[-1]:
                                yield self.streaming_responses[task_id].to_sse_event(chunk)

                                # Send final status event
                                final_event = {
                                    "jsonrpc": "2.0",
                                    "id": 1,
                                    "result": {
                                        "id": task_id,
                                        "status": {"state": "completed"},
                                        "final": True
                                    }
                                }
                                yield f"data: {json.dumps(final_event)}\n\n"
                                return
                            else:
                                yield self.streaming_responses[task_id].to_sse_event(chunk)

                        await asyncio.sleep(0.1)

                return StreamingResponse(event_generator(), media_type="text/event-stream")

            @app.post("/mcp/list_tools")
            async def mcp_list_tools(request: Request):
                """Handle MCP list_tools requests."""
                try:
                    tools = await self.mcp_server.list_tools()
                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                        for tool in tools
                    ]
                except Exception as e:
                    logger.error(f"Error handling MCP list_tools request: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/mcp/call_tool")
            async def mcp_call_tool(request: Request):
                """Handle MCP call_tool requests."""
                try:
                    params = await request.json()
                    result = await self.mcp_server.call_tool(params["name"], params["arguments"])
                    return result
                except Exception as e:
                    logger.error(f"Error handling MCP call_tool request: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/mcp/list_resources")
            async def mcp_list_resources(request: Request):
                """Handle MCP list_resources requests."""
                try:
                    resources = await self.mcp_server.list_resources()
                    return [
                        {
                            "uri": resource.uri,
                            "description": resource.description,
                            "mime_type": resource.mime_type
                        }
                        for resource in resources
                    ]
                except Exception as e:
                    logger.error(f"Error handling MCP list_resources request: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/mcp/read_resource")
            async def mcp_read_resource(uri: str):
                """Handle MCP read_resource requests."""
                try:
                    content, mime_type = await self.mcp_server.read_resource(uri)
                    return Response(content=content, media_type=mime_type)
                except Exception as e:
                    logger.error(f"Error handling MCP read_resource request: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/.well-known/agent.json")
            async def agent_card():
                """Return the A2A agent card."""
                return self.a2a_agent.generate_agent_card()

            @app.get("/")
            async def root():
                """Return basic information about the bridge."""
                return {
                    "name": self.name,
                    "description": self.description,
                    "type": "MCP-A2A Bridge",
                    "endpoints": {
                        "a2a": "/a2a",
                        "a2a_stream": "/a2a/stream/{task_id}",
                        "mcp_list_tools": "/mcp/list_tools",
                        "mcp_call_tool": "/mcp/call_tool",
                        "mcp_list_resources": "/mcp/list_resources",
                        "mcp_read_resource": "/mcp/read_resource?uri={uri}",
                        "agent_card": "/.well-known/agent.json"
                    }
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


# --- MCP Client Adapter ---

class McpClientAdapter:
    """Adapter for MCP clients to connect to the bridge."""

    def __init__(self, bridge_url: str):
        """Initialize McpClientAdapter.

        Args:
            bridge_url: URL of the MCP-A2A Bridge
        """
        self.bridge_url = bridge_url.rstrip("/")
        self.http_session = None

    async def start(self):
        """Start the adapter."""
        import aiohttp

        if self.http_session is None:
            self.http_session = aiohttp.ClientSession()

    async def stop(self):
        """Stop the adapter."""
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List tools from the bridge.

        Returns:
            List of tool definitions
        """
        if self.http_session is None:
            await self.start()

        async with self.http_session.post(f"{self.bridge_url}/mcp/list_tools") as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error listing tools: {error_text}")
                return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the bridge.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if self.http_session is None:
            await self.start()

        async with self.http_session.post(
                f"{self.bridge_url}/mcp/call_tool",
                json={"name": name, "arguments": arguments}
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error calling tool: {error_text}")
                raise ValueError(f"Error calling tool: {error_text}")

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List resources from the bridge.

        Returns:
            List of resource definitions
        """
        if self.http_session is None:
            await self.start()

        async with self.http_session.post(f"{self.bridge_url}/mcp/list_resources") as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error listing resources: {error_text}")
                return []

    async def read_resource(self, uri: str) -> Tuple[str, str]:
        """Read a resource from the bridge.

        Args:
            uri: Resource URI

        Returns:
            Tuple of (content, mime_type)
        """
        if self.http_session is None:
            await self.start()

        async with self.http_session.get(
                f"{self.bridge_url}/mcp/read_resource",
                params={"uri": uri}
        ) as resp:
            if resp.status == 200:
                content = await resp.read()
                mime_type = resp.headers.get("Content-Type", "application/octet-stream")
                return content, mime_type
            else:
                error_text = await resp.text()
                logger.error(f"Error reading resource: {error_text}")
                raise ValueError(f"Error reading resource: {error_text}")


# --- A2A Client Adapter ---

class A2AClientAdapter:
    """Adapter for A2A clients to connect to the bridge."""

    def __init__(self, bridge_url: str):
        """Initialize A2AClientAdapter.

        Args:
            bridge_url: URL of the MCP-A2A Bridge
        """
        self.bridge_url = bridge_url.rstrip("/")
        self.http_session = None

    async def start(self):
        """Start the adapter."""
        import aiohttp

        if self.http_session is None:
            self.http_session = aiohttp.ClientSession()

    async def stop(self):
        """Stop the adapter."""
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None

    async def get_agent_card(self) -> Dict[str, Any]:
        """Get the agent card from the bridge.

        Returns:
            Agent card
        """
        if self.http_session is None:
            await self.start()

        async with self.http_session.get(f"{self.bridge_url}/.well-known/agent.json") as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error getting agent card: {error_text}")
                return {}

    async def send_task(self, task_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a task to the bridge.

        Args:
            task_id: Task ID
            message: Message

        Returns:
            Task result
        """
        if self.http_session is None:
            await self.start()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": message
            }
        }

        async with self.http_session.post(
                f"{self.bridge_url}/a2a",
                json=request
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error sending task: {error_text}")
                raise ValueError(f"Error sending task: {error_text}")

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get a task from the bridge.

        Args:
            task_id: Task ID

        Returns:
            Task result
        """
        if self.http_session is None:
            await self.start()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/get",
            "params": {
                "id": task_id
            }
        }

        async with self.http_session.post(
                f"{self.bridge_url}/a2a",
                json=request
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error getting task: {error_text}")
                raise ValueError(f"Error getting task: {error_text}")

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task on the bridge.

        Args:
            task_id: Task ID

        Returns:
            Task result
        """
        if self.http_session is None:
            await self.start()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/cancel",
            "params": {
                "id": task_id
            }
        }

        async with self.http_session.post(
                f"{self.bridge_url}/a2a",
                json=request
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                logger.error(f"Error canceling task: {error_text}")
                raise ValueError(f"Error canceling task: {error_text}")

    async def send_streaming_task(self, task_id: str, message: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a streaming task to the bridge.

        Args:
            task_id: Task ID
            message: Message

        Returns:
            Async generator of task events
        """
        if self.http_session is None:
            await self.start()

        # First send the task
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/sendSubscribe",
            "params": {
                "id": task_id,
                "message": message
            }
        }

        async with self.http_session.post(
                f"{self.bridge_url}/a2a",
                json=request
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Error sending streaming task: {error_text}")
                raise ValueError(f"Error sending streaming task: {error_text}")

            # Get the initial response
            initial_response = await resp.json()
            yield initial_response

        # Then connect to the streaming endpoint
        import aiohttp

        async with self.http_session.get(
                f"{self.bridge_url}/a2a/stream/{task_id}",
                headers={"Accept": "text/event-stream"}
        ) as stream_resp:
            if stream_resp.status != 200:
                error_text = await stream_resp.text()
                logger.error(f"Error connecting to stream: {error_text}")
                raise ValueError(f"Error connecting to stream: {error_text}")

            # Parse SSE events
            buffer = ""

            async for line in stream_resp.content:
                line = line.decode("utf-8")
                buffer += line

                if buffer.endswith("\n\n"):
                    # Process complete event
                    event = buffer.strip()
                    buffer = ""

                    if event.startswith("data: "):
                        data = event[6:]
                        try:
                            event_data = json.loads(data)
                            yield event_data

                            # Check if this is the final event
                            if "result" in event_data and event_data["result"].get("final", False):
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing event data: {data}")


# --- Convenience Functions ---

def create_bridge(name: str, description: str, url: str) -> EnhancedMcpA2ABridge:
    """Create an Enhanced MCP-A2A Bridge.

    Args:
        name: Bridge name
        description: Bridge description
        url: Bridge URL

    Returns:
        EnhancedMcpA2ABridge instance
    """
    return EnhancedMcpA2ABridge(name, description, url)


async def connect_to_bridge_as_mcp(bridge_url: str) -> McpClientAdapter:
    """Connect to an Enhanced MCP-A2A Bridge as an MCP client.

    Args:
        bridge_url: URL of the bridge

    Returns:
        McpClientAdapter instance
    """
    adapter = McpClientAdapter(bridge_url)
    await adapter.start()
    return adapter


async def connect_to_bridge_as_a2a(bridge_url: str) -> A2AClientAdapter:
    """Connect to an Enhanced MCP-A2A Bridge as an A2A client.

    Args:
        bridge_url: URL of the bridge

    Returns:
        A2AClientAdapter instance
    """
    adapter = A2AClientAdapter(bridge_url)
    await adapter.start()
    return adapter