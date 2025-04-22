"""
AIRA Hub: A central registry for AI agents to discover and communicate with each other.

This implementation provides:
- Agent registration and discovery
- Resource sharing
- Authentication and security
- Health monitoring
- Metrics and analytics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, constr
from typing import Dict, List, Optional, Set, Any, Union
import asyncio
import time
import json
import uuid
import logging
import hashlib
import os
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aira_hub.log")
    ]
)
logger = logging.getLogger("aira_hub")

# Constants
DEFAULT_HEARTBEAT_TIMEOUT = 300  # 5 minutes
MAX_RESOURCES_PER_AGENT = 100
DEFAULT_API_RATE_LIMIT = 100  # requests per minute
DB_PERSIST_INTERVAL = 60  # seconds

# Storage options
class StorageBackend(Enum):
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    MONGODB = "mongodb"

# ==================== Data Models ====================

class AgentStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"

class ResourceType(str, Enum):
    MCP_TOOL = "mcp_tool"
    MCP_RESOURCE = "mcp_resource"
    A2A_SKILL = "a2a_skill"
    API_ENDPOINT = "api_endpoint"
    DATASET = "dataset"
    OTHER = "other"

class ApiKey(BaseModel):
    key: str
    expires_at: Optional[float] = None
    scopes: List[str] = []
    created_at: float = Field(default_factory=time.time)

class AgentAuth(BaseModel):
    type: str = "none"  # none, apiKey, oauth2, etc.
    scheme: Optional[str] = None  # bearer, basic, etc.
    parameter_name: Optional[str] = None  # header name, query param, etc.
    location: Optional[str] = None  # header, query, cookie, etc.

class Resource(BaseModel):
    uri: str
    description: str
    type: Union[ResourceType, str]
    version: str = "1.0.0"
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = {}

    @validator('uri')
    def uri_must_be_valid(cls, v):
        if not v or len(v) < 3:
            raise ValueError('URI must be at least 3 characters long')
        return v

    @validator('type')
    def type_must_be_valid(cls, v):
        if isinstance(v, str) and v not in [t.value for t in ResourceType]:
            return v  # Allow custom types as strings
        return v

class Skill(BaseModel):
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = []
    parameters: Dict[str, Any] = {}
    examples: List[str] = []
    metadata: Dict[str, Any] = {}

class AgentMetrics(BaseModel):
    request_count: int = 0
    error_count: int = 0
    last_response_time: Optional[float] = None
    avg_response_time: Optional[float] = None
    uptime: float = 0

class AgentRegistration(BaseModel):
    url: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    skills: List[Dict[str, Any]] = []
    shared_resources: List[Resource] = []
    aira_capabilities: List[str] = []
    auth: Dict[str, Any] = {}
    status: AgentStatus = AgentStatus.ONLINE
    last_seen: float = Field(default_factory=time.time)
    created_at: float = Field(default_factory=time.time)
    metrics: Optional[AgentMetrics] = None
    tags: List[str] = []
    category: Optional[str] = None
    provider: Optional[Dict[str, str]] = None

    @validator('url')
    def url_must_be_valid(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

    @validator('shared_resources')
    def max_resources(cls, v):
        if len(v) > MAX_RESOURCES_PER_AGENT:
            raise ValueError(f'Maximum of {MAX_RESOURCES_PER_AGENT} resources allowed')
        return v

    class Config:
        schema_extra = {
            "example": {
                "url": "https://myagent.example.com",
                "name": "Weather Agent",
                "description": "Provides weather forecasts and historical data",
                "version": "1.0.0",
                "skills": [
                    {
                        "id": "get-forecast",
                        "name": "Get Weather Forecast",
                        "description": "Get weather forecast for a location",
                        "tags": ["weather", "forecast"]
                    }
                ],
                "shared_resources": [
                    {
                        "uri": "mcp://tool/get-weather",
                        "description": "Weather forecast tool",
                        "type": "mcp_tool",
                        "version": "1.0.0"
                    }
                ],
                "aira_capabilities": ["mcp", "a2a"],
                "auth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
                "tags": ["weather", "forecast"],
                "category": "utilities"
            }
        }

class HubHealth(BaseModel):
    status: str = "healthy"
    uptime: float = 0
    registered_agents: int = 0
    active_agents: int = 0
    total_resources: int = 0
    total_skills: int = 0
    version: str = "1.0.0"

class DiscoverQuery(BaseModel):
    skill_id: Optional[str] = None
    skill_tags: Optional[List[str]] = None
    resource_type: Optional[Union[ResourceType, str]] = None
    agent_tags: Optional[List[str]] = None
    category: Optional[str] = None
    status: Optional[AgentStatus] = None
    offset: int = 0
    limit: int = 100

# ==================== Storage Backends ====================

class BaseStorage:
    """Base class for storage backends"""

    async def init(self):
        """Initialize the storage backend"""
        pass

    async def close(self):
        """Close the storage backend"""
        pass

    async def save_agent(self, agent: AgentRegistration):
        """Save an agent to storage"""
        raise NotImplementedError()

    async def get_agent(self, url: str) -> Optional[AgentRegistration]:
        """Get an agent by URL"""
        raise NotImplementedError()

    async def list_agents(self) -> List[AgentRegistration]:
        """List all agents"""
        raise NotImplementedError()

    async def delete_agent(self, url: str):
        """Delete an agent by URL"""
        raise NotImplementedError()

    async def update_agent_heartbeat(self, url: str, timestamp: float):
        """Update agent heartbeat timestamp"""
        raise NotImplementedError()

class MemoryStorage(BaseStorage):
    """In-memory storage backend"""

    def __init__(self):
        self.agents: Dict[str, AgentRegistration] = {}
        self.metrics = {
            "start_time": time.time(),
            "agent_requests": {},
            "total_requests": 0
        }

    async def save_agent(self, agent: AgentRegistration):
        self.agents[agent.url] = agent

    async def get_agent(self, url: str) -> Optional[AgentRegistration]:
        return self.agents.get(url)

    async def list_agents(self) -> List[AgentRegistration]:
        return list(self.agents.values())

    async def delete_agent(self, url: str):
        if url in self.agents:
            del self.agents[url]

    async def update_agent_heartbeat(self, url: str, timestamp: float):
        if url in self.agents:
            self.agents[url].last_seen = timestamp

class FileStorage(BaseStorage):
    """File-based storage backend"""

    def __init__(self, file_path: str = "aira_db.json"):
        self.file_path = file_path
        self.agents: Dict[str, AgentRegistration] = {}
        self.metrics = {
            "start_time": time.time(),
            "agent_requests": {},
            "total_requests": 0
        }
        self.last_save = 0

    async def init(self):
        await self._load_from_file()

    async def close(self):
        await self._save_to_file()

    async def _load_from_file(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.agents = {url: AgentRegistration(**agent_data)
                                for url, agent_data in data.get("agents", {}).items()}
                    self.metrics = data.get("metrics", {"start_time": time.time(),
                                                       "agent_requests": {},
                                                       "total_requests": 0})
                    logger.info(f"Loaded {len(self.agents)} agents from {self.file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {str(e)}")
            self.agents = {}
            self.metrics = {"start_time": time.time(), "agent_requests": {}, "total_requests": 0}

    async def _save_to_file(self):
        try:
            current_time = time.time()
            if current_time - self.last_save < DB_PERSIST_INTERVAL:
                return  # Don't save too frequently

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)

            with open(self.file_path, 'w') as f:
                agent_dict = {url: agent.dict() for url, agent in self.agents.items()}
                json.dump({"agents": agent_dict, "metrics": self.metrics}, f, indent=2)

            self.last_save = current_time
            logger.debug(f"Saved {len(self.agents)} agents to {self.file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {self.file_path}: {str(e)}")

    async def save_agent(self, agent: AgentRegistration):
        self.agents[agent.url] = agent
        await self._save_to_file()

    async def get_agent(self, url: str) -> Optional[AgentRegistration]:
        return self.agents.get(url)

    async def list_agents(self) -> List[AgentRegistration]:
        return list(self.agents.values())

    async def delete_agent(self, url: str):
        if url in self.agents:
            del self.agents[url]
            await self._save_to_file()

    async def update_agent_heartbeat(self, url: str, timestamp: float):
        if url in self.agents:
            self.agents[url].last_seen = timestamp
            # Only persist to disk occasionally to avoid frequent writes
            if time.time() - self.last_save > DB_PERSIST_INTERVAL:
                await self._save_to_file()

# Factory for storage backends
def get_storage_backend(backend_type: StorageBackend, **kwargs) -> BaseStorage:
    if backend_type == StorageBackend.MEMORY:
        return MemoryStorage()
    elif backend_type == StorageBackend.FILE:
        file_path = kwargs.get("file_path", "aira_db.json")
        return FileStorage(file_path)
    elif backend_type == StorageBackend.REDIS:
        # Placeholder for Redis backend
        raise NotImplementedError("Redis storage backend not implemented yet")
    elif backend_type == StorageBackend.MONGODB:
        # Placeholder for MongoDB backend
        raise NotImplementedError("MongoDB storage backend not implemented yet")
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")

# ==================== API Key Management ====================

class ApiKeyManager:
    """Manage API keys for hub access"""

    def __init__(self):
        self.api_keys: Dict[str, ApiKey] = {}

        # Add a default admin key if configured
        admin_key = os.environ.get("AIRA_ADMIN_KEY")
        if admin_key:
            self.api_keys[admin_key] = ApiKey(
                key=admin_key,
                scopes=["admin"],
                expires_at=None  # Never expires
            )

    def validate_key(self, api_key: str) -> bool:
        """Validate if an API key is valid"""
        if api_key not in self.api_keys:
            return False

        key_data = self.api_keys[api_key]

        # Check if expired
        if key_data.expires_at and time.time() > key_data.expires_at:
            return False

        return True

    def get_key_scopes(self, api_key: str) -> List[str]:
        """Get scopes for an API key"""
        if not self.validate_key(api_key):
            return []

        return self.api_keys[api_key].scopes

    def create_key(self, scopes: List[str] = [], expires_in_days: Optional[int] = None) -> str:
        """Create a new API key"""
        new_key = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)

        self.api_keys[new_key] = ApiKey(
            key=new_key,
            scopes=scopes,
            expires_at=expires_at
        )

        return new_key

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False

# ==================== Security & Auth ====================

def verify_api_key(x_api_key: str = Header(None)) -> bool:
    """Dependency for verifying API key"""
    # For now, no API key is required by default
    # This is a placeholder for future auth implementation
    return True

def admin_required(x_api_key: str = Header(None)) -> bool:
    """Dependency for admin-only endpoints"""
    # Check if the API key has admin scope
    # Placeholder for future auth implementation
    return True

# ==================== Application Setup ====================

# Create lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize storage and background tasks
    app.state.storage = get_storage_backend(
        StorageBackend(os.environ.get("AIRA_STORAGE_BACKEND", "file")),
        file_path=os.environ.get("AIRA_DB_FILE", "aira_db.json")
    )
    await app.state.storage.init()

    app.state.api_key_manager = ApiKeyManager()
    app.state.start_time = time.time()

    # Start background tasks
    app.state.cleanup_task = asyncio.create_task(cleanup_inactive_agents(app))
    app.state.persist_task = asyncio.create_task(periodic_persist(app))

    logger.info("AIRA Hub started")
    yield

    # Shutdown: Clean up resources
    app.state.cleanup_task.cancel()
    app.state.persist_task.cancel()
    try:
        await app.state.cleanup_task
        await app.state.persist_task
    except asyncio.CancelledError:
        pass

    await app.state.storage.close()
    logger.info("AIRA Hub stopped")

# Background task for cleaning up inactive agents
async def cleanup_inactive_agents(app: FastAPI):
    """Cleanup inactive agents periodically"""
    try:
        while True:
            await asyncio.sleep(60)  # Run every minute
            now = time.time()

            agents = await app.state.storage.list_agents()
            inactive_count = 0

            for agent in agents:
                heartbeat_timeout = DEFAULT_HEARTBEAT_TIMEOUT

                # If inactive for too long, mark as offline
                if (now - agent.last_seen) > heartbeat_timeout:
                    if agent.status != AgentStatus.OFFLINE:
                        agent.status = AgentStatus.OFFLINE
                        await app.state.storage.save_agent(agent)
                        inactive_count += 1

            if inactive_count > 0:
                logger.info(f"Marked {inactive_count} agents as offline")
    except asyncio.CancelledError:
        logger.info("Cleanup task cancelled")
    except Exception as e:
        logger.error(f"Error in cleanup task: {str(e)}")

# Background task for persisting data
async def periodic_persist(app: FastAPI):
    """Persist data periodically"""
    try:
        while True:
            await asyncio.sleep(DB_PERSIST_INTERVAL)
            if hasattr(app.state.storage, "_save_to_file"):
                await app.state.storage._save_to_file()
    except asyncio.CancelledError:
        logger.info("Persist task cancelled")
    except Exception as e:
        logger.error(f"Error in persist task: {str(e)}")

# Create the FastAPI app
app = FastAPI(
    title="AIRA Hub",
    description="A central registry for AI agents to discover and communicate with each other",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request counting middleware
@app.middleware("http")
async def count_requests(request: Request, call_next):
    start_time = time.time()

    # Update request metrics
    if hasattr(app.state, "storage"):
        app.state.storage.metrics["total_requests"] = \
            app.state.storage.metrics.get("total_requests", 0) + 1

    response = await call_next(request)

    # Log request duration for monitoring
    duration = time.time() - start_time
    logger.debug(f"{request.method} {request.url.path} completed in {duration:.4f}s")

    return response

# ==================== API Endpoints ====================

@app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Agents"])
async def register_agent(
        request: Request,
        agent: AgentRegistration,
        api_key_valid: bool = Depends(verify_api_key)
):
    """
    Register an agent with the AIRA hub

    This endpoint allows agents to register themselves with the hub,
    making them discoverable by other agents.
    """
    # Update timestamp and store
    agent.last_seen = time.time()

    # Initialize metrics if not present
    if not agent.metrics:
        agent.metrics = AgentMetrics(uptime=0)

    # Set initial status
    agent.status = AgentStatus.ONLINE

    # Store the agent
    await request.app.state.storage.save_agent(agent)

    logger.info(f"Agent registered: {agent.name} ({agent.url})")

    return {"status": "registered", "agent_url": agent.url}


@app.put("/agents/{agent_url}", tags=["Agents"])
async def update_agent(
        request: Request,
        agent_url: str,
        agent_update: AgentRegistration,
        api_key_valid: bool = Depends(verify_api_key)
):
    """
    Update an existing agent registration

    This endpoint allows agents to update their registration information.
    """
    # URL in path must match the agent URL
    if agent_url != agent_update.url:
        raise HTTPException(status_code=400, detail="URL in path does not match agent URL")

    # Check if agent exists
    existing_agent = await request.app.state.storage.get_agent(agent_url)
    if not existing_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update timestamps
    agent_update.last_seen = time.time()

    # Preserve metrics from existing agent
    if existing_agent.metrics:
        agent_update.metrics = existing_agent.metrics

    # Store updated agent
    await request.app.state.storage.save_agent(agent_update)

    logger.info(f"Agent updated: {agent_update.name} ({agent_update.url})")

    return {"status": "updated", "agent_url": agent_update.url}


# Fix for heartbeat function
@app.post("/heartbeat/{agent_url}", tags=["Agents"])
async def heartbeat(
        request: Request,
        agent_url: str,
        api_key_valid: bool = Depends(verify_api_key)
):
    """
    Update agent heartbeat

    This endpoint allows agents to send heartbeats to indicate they are still active.
    """
    # URL decode the agent_url (could contain special characters)
    agent_url = agent_url.replace("%3A", ":").replace("%2F", "/")

    # Get existing agent
    existing_agent = await request.app.state.storage.get_agent(agent_url)
    if not existing_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update heartbeat timestamp
    now = time.time()
    await request.app.state.storage.update_agent_heartbeat(agent_url, now)

    # Update agent status if needed
    if existing_agent.status != AgentStatus.ONLINE:
        existing_agent.status = AgentStatus.ONLINE
        await request.app.state.storage.save_agent(existing_agent)

    return {"status": "ok"}

@app.get("/agents", tags=["Discovery"])
async def list_agents(request: Request):
    """
    List all registered agents

    This endpoint returns a list of all registered agents.
    """
    agents = await request.app.state.storage.list_agents()

    # Filter out sensitive information
    for agent in agents:
        if "credentials" in agent.auth:
            del agent.auth["credentials"]

    return agents


@app.post("/discover", tags=["Discovery"])
async def discover_agents(
        request: Request,
        query: DiscoverQuery
):
    """
    Discover agents based on query parameters

    This endpoint allows clients to discover agents based on various criteria.
    """
    agents = await request.app.state.storage.list_agents()

    # Apply filters
    if query.skill_id:
        agents = [a for a in agents if any(s.get("id") == query.skill_id for s in a.skills)]

    if query.skill_tags:
        agents = [a for a in agents if any(
            set(query.skill_tags).intersection(set(s.get("tags", [])))
            for s in a.skills
        )]

    if query.resource_type:
        resource_type = query.resource_type.value if isinstance(query.resource_type,
                                                                ResourceType) else query.resource_type
        agents = [a for a in agents if any(r.type == resource_type for r in a.shared_resources)]

    if query.agent_tags:
        agents = [a for a in agents if set(query.agent_tags).intersection(set(a.tags))]

    if query.category:
        agents = [a for a in agents if a.category == query.category]

    if query.status:
        agents = [a for a in agents if a.status == query.status]

    # Apply pagination
    total = len(agents)
    agents = agents[query.offset:query.offset + query.limit]

    # Filter out sensitive information
    for agent in agents:
        if "credentials" in agent.auth:
            del agent.auth["credentials"]

    return {
        "total": total,
        "offset": query.offset,
        "limit": query.limit,
        "agents": agents
    }

@app.get("/agents/{agent_url}", tags=["Agents"])
async def get_agent(
    agent_url: str,
    request: Request
):
    """
    Get details for a specific agent

    This endpoint returns details for a specific agent.
    """
    agent = await request.app.state.storage.get_agent(agent_url)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Filter out sensitive information
    if "credentials" in agent.auth:
        del agent.auth["credentials"]

    return agent


@app.delete("/agents/{agent_url}", tags=["Agents"])
async def unregister_agent(
        request: Request,
        agent_url: str,
        admin: bool = Depends(admin_required)
):
    """
    Unregister an agent

    This endpoint allows admins to unregister an agent from the hub.
    """
    agent = await request.app.state.storage.get_agent(agent_url)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    await request.app.state.storage.delete_agent(agent_url)

    logger.info(f"Agent unregistered: {agent.name} ({agent.url})")

    return {"status": "unregistered", "agent_url": agent_url}

@app.get("/status", tags=["System"])
async def system_status(request: Request):
    """
    Get system status

    This endpoint returns the current status of the AIRA hub.
    """
    agents = await request.app.state.storage.list_agents()

    # Count active agents (seen in last 5 minutes)
    now = time.time()
    active_agents = [a for a in agents if (now - a.last_seen) < DEFAULT_HEARTBEAT_TIMEOUT]

    # Count total resources and skills
    total_resources = sum(len(a.shared_resources) for a in agents)
    total_skills = sum(len(a.skills) for a in agents)

    return HubHealth(
        status="healthy",
        uptime=now - app.state.start_time,
        registered_agents=len(agents),
        active_agents=len(active_agents),
        total_resources=total_resources,
        total_skills=total_skills,
        version="1.0.0"
    )

@app.post("/admin/api-keys", tags=["Admin"])
async def create_api_key(
    request: Request,
    scopes: List[str],
    expires_in_days: Optional[int] = None,
    admin: bool = Depends(admin_required)
):
    """
    Create a new API key

    This endpoint allows admins to create new API keys.
    """
    api_key = request.app.state.api_key_manager.create_key(scopes, expires_in_days)

    return {"key": api_key}


@app.delete("/admin/api-keys/{api_key}", tags=["Admin"])
async def revoke_api_key(
        request: Request,
        api_key: str,
        admin: bool = Depends(admin_required)
):
    """
    Revoke an API key

    This endpoint allows admins to revoke API keys.
    """
    success = request.app.state.api_key_manager.revoke_key(api_key)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"status": "revoked"}

@app.get("/resources/types", tags=["Resources"])
async def list_resource_types():
    """
    List available resource types

    This endpoint returns a list of available resource types.
    """
    return [t.value for t in ResourceType]

@app.get("/resources", tags=["Resources"])
async def list_resources(
    request: Request,
    resource_type: Optional[str] = None
):
    """
    List all shared resources

    This endpoint returns a list of all shared resources across all agents.
    """
    agents = await request.app.state.storage.list_agents()

    resources = []
    for agent in agents:
        for resource in agent.shared_resources:
            if not resource_type or str(resource.type) == resource_type:
                resources.append({
                    "resource": resource,
                    "agent": {
                        "url": agent.url,
                        "name": agent.name
                    }
                })

    return resources

@app.get("/categories", tags=["Discovery"])
async def list_categories(request: Request):
    """
    List all agent categories

    This endpoint returns a list of all unique agent categories.
    """
    agents = await request.app.state.storage.list_agents()

    categories = set()
    for agent in agents:
        if agent.category:
            categories.add(agent.category)

    return sorted(list(categories))

@app.get("/tags", tags=["Discovery"])
async def list_tags(
    request: Request,
    type: str = "agent"  # "agent" or "skill"
):
    """
    List all tags

    This endpoint returns a list of all unique tags used by agents or skills.
    """
    agents = await request.app.state.storage.list_agents()

    tags = set()
    if type == "agent":
        for agent in agents:
            tags.update(agent.tags)
    elif type == "skill":
        for agent in agents:
            for skill in agent.skills:
                tags.update(skill.get("tags", []))
    else:
        raise HTTPException(status_code=400, detail="Invalid tag type. Must be 'agent' or 'skill'")

    return sorted(list(tags))

# ==================== Run Server ====================

def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the AIRA Hub server"""
    import uvicorn

    # Configure logging for uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Run server
    uvicorn.run(
        "AiraHub:app",  # Use string reference for reload to work
        host=host,
        port=port,
        reload=reload,
        log_config=log_config
    )

@app.get("/docs/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    """Get OpenAPI schema"""
    return app.openapi()

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to documentation"""
    return {"message": "Welcome to AIRA Hub. See /docs for API documentation."}

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIRA Hub Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--db-file", default="aira_db.json", help="Database file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    # Set database file path from arguments
    os.environ["AIRA_DB_FILE"] = args.db_file

    # Set logging level
    logging.getLogger("aira_hub").setLevel(getattr(logging, args.log_level))

    # Run server
    run(host=args.host, port=args.port, reload=args.reload)
