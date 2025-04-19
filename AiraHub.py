#!/usr/bin/env python3
# AiraHub.py - Hub for MCP Tools and A2A Agent Skills with Tool Invocation and OAuth 2.1 Auth

import json
import uuid
import logging
import asyncio
import httpx
import time
import secrets
import jwt
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum

from urllib.parse import urlencode
from fastapi import FastAPI, Request, HTTPException, Depends, Header, BackgroundTasks, Form, Cookie, Response
from fastapi import status as http_status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field, EmailStr, field_validator, RootModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airahub")

app = FastAPI(title="AIRA Hub", description="Hub for MCP Tools and A2A Agent Skills")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Setup templates for auth pages
    templates = Jinja2Templates(directory="templates")
    # Add static files for CSS, etc.
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Templates or static files directory not found: {e}")
    # Continue without templates - they're only needed for the auth UI

# =====================================================================
# OAuth 2.1 Authentication
# =====================================================================

# Secret key for JWT signing
# In production, this should be loaded from a secure environment variable
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# OAuth token URL for the token endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: bool = False
    role: str = "user"  # user, agent, admin


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: int  # Timestamp


class TokenData(BaseModel):
    username: str
    role: str
    exp: int  # Expiration time


class RefreshToken(BaseModel):
    token: str
    user_id: str
    expires_at: datetime


# In-memory user database (replace with a real database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "hashed_password": "password123",  # In production, use proper hashing
        "disabled": False,
        "role": "admin"
    },
    "agent1": {
        "username": "agent1",
        "email": "agent1@example.com",
        "full_name": "Agent One",
        "hashed_password": "password123",
        "disabled": False,
        "role": "agent"
    }
}

# In-memory token database
active_refresh_tokens = {}


def verify_password(plain_password, hashed_password):
    # In production, replace with proper password hashing
    return plain_password == hashed_password


def get_user(username: str):
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, int(expire.timestamp())


def create_refresh_token(username: str):
    expires = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    token = secrets.token_urlsafe(32)
    active_refresh_tokens[token] = {
        "user_id": username,
        "expires_at": expires
    }
    return token, expires


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=http_status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        exp: int = payload.get("exp")

        if username is None or exp is None:
            raise credentials_exception

        token_data = TokenData(username=username, role=role, exp=exp)
    except jwt.PyJWTError:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


async def get_agent_or_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role not in ["agent", "admin"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


# =====================================================================
# Data Models
# =====================================================================

class AgentStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class Resource(BaseModel):
    uri: str
    description: Optional[str] = None
    type: Optional[str] = None
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# MCP-specific capabilities
class MCPCapabilities(BaseModel):
    resources: Optional[Dict[str, bool]] = None
    prompts: Optional[Dict[str, bool]] = None
    tools: Optional[Dict[str, bool]] = None
    logging: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None


# A2A-specific capabilities
class A2ACapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class MCPTool(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None


class A2ASkill(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None


class AgentRegistration(BaseModel):
    name: str
    url: str
    aira_capabilities: List[str]  # ["mcp", "a2a", "hybrid"]
    mcp_capabilities: Optional[MCPCapabilities] = None
    a2a_capabilities: Optional[A2ACapabilities] = None
    mcp_tools: List[MCPTool] = []
    a2a_skills: List[A2ASkill] = []
    shared_resources: List[Resource] = []
    tags: List[str] = []
    category: Optional[str] = None
    description: Optional[str] = None
    status: AgentStatus = AgentStatus.ONLINE
    created_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    last_seen: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    owner: Optional[str] = None  # Username of the user who registered this agent

    @field_validator('aira_capabilities')
    @classmethod
    def validate_capabilities(cls, v):
        valid_caps = ["mcp", "a2a", "hybrid"]
        for cap in v:
            if cap not in valid_caps:
                raise ValueError(f"Invalid capability: {cap}. Must be one of {valid_caps}")
        return v


# Tool Invocation Models
class ToolArguments(RootModel):
    root: Dict[str, Any]


class ToolCallRequest(BaseModel):
    agent_url: str
    tool_name: str
    arguments: Dict[str, Any]
    origin: Optional[str] = None  # Who is requesting this tool


class ToolResponse(BaseModel):
    success: bool
    message: str
    content: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ToolNotificationEvent(BaseModel):
    type: Literal["notification"] = "notification"
    method: str
    params: Dict[str, Any]


# MCP JSONRPC Messages
class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class JSONRPCNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None


# OAuth2 Dynamic Client Registration
class OAuthClient(BaseModel):
    client_id: str
    client_secret: str
    redirect_uris: List[str]
    grant_types: List[str]
    client_name: str
    owner: str  # Username of the user who registered this client


# =====================================================================
# Agent Store
# =====================================================================

class AgentStore:
    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}  # Track active sessions
        self._tool_calls: Dict[str, Dict[str, Any]] = {}  # Track tool call progress
        self._oauth_clients: Dict[str, OAuthClient] = {}  # OAuth clients

    async def save_agent(self, agent: AgentRegistration):
        # FIX: Improved merging logic for existing agent data
        existing = self._agents.get(agent.url)
        if existing:
            # Properly merge the agent data with existing data
            agent_dict = agent.model_dump()
            existing_dict = existing.model_dump()

            # Check mcp_tools and preserve existing if none provided in new agent
            if not agent_dict.get('mcp_tools') and existing_dict.get('mcp_tools'):
                agent_dict['mcp_tools'] = existing_dict['mcp_tools']

            # Check a2a_skills and preserve existing if none provided in new agent
            if not agent_dict.get('a2a_skills') and existing_dict.get('a2a_skills'):
                agent_dict['a2a_skills'] = existing_dict['a2a_skills']

            # For debugging
            logger.info(
                f"Saving agent {agent.url}: mcp_tools: {len(agent_dict['mcp_tools'])}, a2a_skills: {len(agent_dict['a2a_skills'])}")

            # Create a new agent with the merged data
            updated_agent = AgentRegistration(**agent_dict)
            self._agents[agent.url] = updated_agent
        else:
            # New agent, just save it directly
            logger.info(
                f"Creating new agent {agent.url}: mcp_tools: {len(agent.mcp_tools)}, a2a_skills: {len(agent.a2a_skills)}")
            self._agents[agent.url] = agent

    async def update_agent_status(self, url: str, status: AgentStatus):
        agent = await self.get_agent(url)
        if agent:
            agent.status = status
            agent.last_seen = datetime.utcnow().timestamp()
            await self.save_agent(agent)
            return True
        return False

    async def get_agent(self, url: str) -> Optional[AgentRegistration]:
        return self._agents.get(url)

    async def list_agents(self) -> List[AgentRegistration]:
        return list(self._agents.values())

    async def get_agents_by_owner(self, username: str) -> List[AgentRegistration]:
        return [a for a in self._agents.values() if a.owner == username]

    async def create_session(self, agent_url: str) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "agent_url": agent_url,
            "created_at": datetime.utcnow().timestamp(),
            "last_activity": datetime.utcnow().timestamp()
        }
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    async def track_tool_call(self, call_id: str, data: Dict[str, Any]):
        self._tool_calls[call_id] = {
            **data,
            "created_at": datetime.utcnow().timestamp(),
            "updated_at": datetime.utcnow().timestamp(),
            "status": "pending"
        }

    async def update_tool_call(self, call_id: str, data: Dict[str, Any]):
        if call_id in self._tool_calls:
            self._tool_calls[call_id].update({
                **data,
                "updated_at": datetime.utcnow().timestamp()
            })
            return True
        return False

    async def get_tool_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        return self._tool_calls.get(call_id)

    async def register_oauth_client(self, client: OAuthClient):
        self._oauth_clients[client.client_id] = client
        return client

    async def get_oauth_client(self, client_id: str) -> Optional[OAuthClient]:
        return self._oauth_clients.get(client_id)

    async def list_oauth_clients(self) -> List[OAuthClient]:
        return list(self._oauth_clients.values())

    async def list_oauth_clients_by_owner(self, username: str) -> List[OAuthClient]:
        return [c for c in self._oauth_clients.values() if c.owner == username]


# Initialize store
app.state.store = AgentStore()


# =====================================================================
# HTTP Client for Tool Invocation
# =====================================================================

async def get_http_client():
    client = httpx.AsyncClient(timeout=30.0)
    try:
        yield client
    finally:
        await client.aclose()


# =====================================================================
# Utility Functions
# =====================================================================

async def check_agent_heartbeats():
    """Background task to check agent heartbeats and update status"""
    while True:
        store = app.state.store
        agents = await store.list_agents()
        now = datetime.utcnow().timestamp()

        for agent in agents:
            # If no heartbeat in 30 seconds, mark as offline
            if now - agent.last_seen > 30:
                agent.status = AgentStatus.OFFLINE
                await store.save_agent(agent)

        await asyncio.sleep(10)  # Check every 10 seconds


# =====================================================================
# OAuth 2.1 Authentication Routes
# =====================================================================

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, expires_at = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )

    # Create refresh token
    refresh_token, _ = create_refresh_token(user.username)

    # Set refresh token as HttpOnly cookie
    response = JSONResponse({
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at
    })

    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,  # set to False if not using HTTPS
        samesite="lax",
        max_age=60 * 60 * 24 * REFRESH_TOKEN_EXPIRE_DAYS  # seconds
    )

    return response


@app.post("/refresh-token", response_model=Token)
async def refresh_access_token(refresh_token: str = Cookie(None)):
    if not refresh_token or refresh_token not in active_refresh_tokens:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = active_refresh_tokens[refresh_token]
    expires_at = token_data["expires_at"]

    if expires_at < datetime.utcnow():
        # Remove expired token
        active_refresh_tokens.pop(refresh_token, None)
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    username = token_data["user_id"]
    user = get_user(username)
    if not user:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, expires_at = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at
    }


@app.post("/logout")
async def logout(response: Response, refresh_token: str = Cookie(None)):
    if refresh_token and refresh_token in active_refresh_tokens:
        active_refresh_tokens.pop(refresh_token, None)

    response = JSONResponse({"detail": "Successfully logged out"})
    response.delete_cookie(key="refresh_token")
    return response


@app.get("/.well-known/oauth-authorization-server")
async def oauth_server_metadata():
    """OAuth 2.0 Authorization Server Metadata protocol"""
    base_url = "http://localhost:8015"  # Replace with actual base URL in production

    return {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "jwks_uri": f"{base_url}/.well-known/jwks.json",
        "registration_endpoint": f"{base_url}/register",
        "scopes_supported": ["profile", "email", "agent:read", "agent:write", "tools:invoke"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token", "client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "service_documentation": f"{base_url}/docs",
    }


@app.get("/authorize")
async def authorize(
        request: Request,
        response_type: str,
        client_id: str,
        redirect_uri: str,
        scope: str = None,
        state: str = None
):
    """OAuth 2.1 Authorization endpoint"""
    # Validate client
    store = request.app.state.store
    client = await store.get_oauth_client(client_id)

    if not client:
        raise HTTPException(status_code=400, detail="Invalid client_id")

    # Validate redirect URI
    if redirect_uri not in client.redirect_uris:
        raise HTTPException(status_code=400, detail="Invalid redirect_uri")

    # Validate response type
    if response_type != "code":
        raise HTTPException(status_code=400, detail="Unsupported response_type")

    try:
        # Render authorization page
        return templates.TemplateResponse(
            "authorize.html",
            {
                "request": request,
                "client_name": client.client_name,
                "scopes": scope.split() if scope else [],
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "state": state
            }
        )
    except Exception as e:
        # Fallback if templates aren't available
        return {
            "message": "Authorization required",
            "client_id": client_id,
            "scopes": scope.split() if scope else [],
            "redirect_uri": redirect_uri,
            "state": state
        }


@app.post("/authorize")
async def authorize_approve(
        request: Request,
        client_id: str = Form(...),
        redirect_uri: str = Form(...),
        approved: bool = Form(False),
        state: str = Form(None)
):
    """Handle authorization form submission"""
    if not approved:
        # User denied the authorization
        redirect_params = {"error": "access_denied"}
        if state:
            redirect_params["state"] = state

        redirect_url = f"{redirect_uri}?{urlencode(redirect_params)}"
        return RedirectResponse(url=redirect_url)

    # User approved - generate authorization code
    code = secrets.token_urlsafe(32)

    # Store code (in a real implementation, store in database with expiration)
    # For this example, we'll store in memory
    app.state.authorization_codes = getattr(app.state, "authorization_codes", {})
    app.state.authorization_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "expires_at": datetime.utcnow() + timedelta(minutes=10),
        # In a real implementation, associate with authenticated user
    }

    # Redirect back to client with code
    redirect_params = {"code": code}
    if state:
        redirect_params["state"] = state

    from urllib.parse import urlencode
    redirect_url = f"{redirect_uri}?{urlencode(redirect_params)}"
    return RedirectResponse(url=redirect_url)


@app.post("/register", response_model=OAuthClient)
async def register_oauth_client(
        request: Request,
        redirect_uris: List[str],
        client_name: str,
        current_user: User = Depends(get_current_active_user)
):
    """Dynamic Client Registration Protocol"""
    client_id = secrets.token_urlsafe(32)
    client_secret = secrets.token_urlsafe(32)

    client = OAuthClient(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uris=redirect_uris,
        grant_types=["authorization_code", "refresh_token"],
        client_name=client_name,
        owner=current_user.username
    )

    store = request.app.state.store
    await store.register_oauth_client(client)

    return client


# =====================================================================
# API Endpoints
# =====================================================================

@app.post("/connect/stream/init")
async def stream_init(
        request: Request,
        current_user: User = Depends(get_agent_or_admin_user)
):
    """Initialize or update agent metadata after connection"""
    try:
        body = await request.json()
        agent_url = body.get("url")

        logger.info(f"Stream init request received for {agent_url}")
        logger.info(f"Request body: {body}")

        agent = await request.app.state.store.get_agent(agent_url)
        if not agent:
            raise HTTPException(404, detail="Agent not registered via /connect/stream")

        # Verify ownership
        if agent.owner and agent.owner != current_user.username and current_user.role != "admin":
            raise HTTPException(403, detail="You do not have permission to update this agent")

        # FIX: Better handling of payload validation
        # Create a new agent dict from the existing agent
        agent_dict = agent.model_dump()

        # Update fields with provided values
        update_fields = [
            "description", "tags", "category"
        ]
        for field in update_fields:
            if field in body:
                agent_dict[field] = body[field]

        # Carefully handle capability-specific fields
        # MCP Capabilities
        if "mcp_capabilities" in body:
            try:
                mcp_caps = MCPCapabilities(**body["mcp_capabilities"])
                agent_dict["mcp_capabilities"] = mcp_caps.model_dump(exclude_none=True)
            except Exception as e:
                logger.error(f"Failed to parse mcp_capabilities: {e}")

        # A2A Capabilities
        if "a2a_capabilities" in body:
            try:
                a2a_caps = A2ACapabilities(**body["a2a_capabilities"])
                agent_dict["a2a_capabilities"] = a2a_caps.model_dump(exclude_none=True)
            except Exception as e:
                logger.error(f"Failed to parse a2a_capabilities: {e}")

        # FIX: Handle tools and skills with better validation and preserving existing data
        # Process MCP tools
        if "mcp_tools" in body and isinstance(body["mcp_tools"], list) and body["mcp_tools"]:
            try:
                # Validate each tool against the model
                mcp_tools = []
                for tool_data in body["mcp_tools"]:
                    tool = MCPTool(**tool_data)
                    mcp_tools.append(tool)

                agent_dict["mcp_tools"] = [t.model_dump(exclude_none=True) for t in mcp_tools]
                logger.info(f"Successfully processed {len(mcp_tools)} MCP tools")
            except Exception as e:
                logger.error(f"Failed to process MCP tools: {e}")
                # Keep existing tools if processing fails
                pass

        # Process A2A skills
        if "a2a_skills" in body and isinstance(body["a2a_skills"], list) and body["a2a_skills"]:
            try:
                # Validate each skill against the model
                a2a_skills = []
                for skill_data in body["a2a_skills"]:
                    skill = A2ASkill(**skill_data)
                    a2a_skills.append(skill)

                agent_dict["a2a_skills"] = [s.model_dump(exclude_none=True) for s in a2a_skills]
                logger.info(f"Successfully processed {len(a2a_skills)} A2A skills")
            except Exception as e:
                logger.error(f"Failed to process A2A skills: {e}")
                # Keep existing skills if processing fails
                pass

        # Process shared resources
        if "shared_resources" in body and isinstance(body["shared_resources"], list):
            try:
                resources = []
                for resource_data in body["shared_resources"]:
                    resource = Resource(**resource_data)
                    resources.append(resource)

                agent_dict["shared_resources"] = [r.model_dump(exclude_none=True) for r in resources]
            except Exception as e:
                logger.error(f"Failed to process shared resources: {e}")
                # Keep existing resources if processing fails
                pass

        # Create updated agent record
        try:
            new_agent = AgentRegistration(**agent_dict)
            await request.app.state.store.save_agent(new_agent)
            logger.info(f"✅ Updated agent {agent_url} with fields: {list(body.keys())}")
            logger.info(f"  - MCP Tools: {len(new_agent.mcp_tools)}")
            logger.info(f"  - A2A Skills: {len(new_agent.a2a_skills)}")

            return {"status": "updated", "details": {
                "mcp_tools_count": len(new_agent.mcp_tools),
                "a2a_skills_count": len(new_agent.a2a_skills)
            }}
        except Exception as e:
            logger.error(f"Failed to update agent: {e}")
            raise HTTPException(500, detail=f"Failed to update agent: {str(e)}")

    except Exception as e:
        logger.error(f"Error in stream_init: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, detail=f"Server error: {str(e)}")


@app.get("/connect/stream")
async def connect_stream(
        agent_url: str,
        name: str,
        aira_capabilities: str,
        current_user: User = Depends(get_agent_or_admin_user)
):
    """Establish SSE connection with an agent"""
    caps = aira_capabilities.split(",")

    # Validate capabilities
    valid_caps = ["mcp", "a2a", "hybrid"]
    for cap in caps:
        if cap not in valid_caps:
            raise HTTPException(400, detail=f"Invalid capability: {cap}")

    # Initialize agent registration with proper empty lists
    ag = AgentRegistration(
        url=agent_url,
        name=name,
        aira_capabilities=caps,
        shared_resources=[],
        mcp_tools=[],
        a2a_skills=[],
        tags=["streamed"],
        status=AgentStatus.ONLINE,
        owner=current_user.username
    )

    # Log the agent creation with details
    logger.info(f"Creating new agent: {name} at {agent_url} with capabilities {caps}")

    await app.state.store.save_agent(ag)
    session_id = await app.state.store.create_session(agent_url)

    async def event_generator():
        """Generate SSE events for the agent"""
        try:
            # Initial connection event
            yield json.dumps({
                "event": "connected",
                "data": {
                    "session_id": session_id,
                    "agent": ag.name,
                    "status": ag.status,
                    "ts": datetime.utcnow().isoformat() + "Z"
                }
            }) + "\n\n"

            # Heartbeat loop
            while True:
                await app.state.store.update_agent_status(agent_url, AgentStatus.ONLINE)
                yield json.dumps({
                    "event": "heartbeat",
                    "data": {
                        "agent": ag.name,
                        "status": ag.status,
                        "ts": datetime.utcnow().isoformat() + "Z"
                    }
                }) + "\n\n"

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            # Update agent status when connection is closed
            await app.state.store.update_agent_status(agent_url, AgentStatus.OFFLINE)
            logger.info(f"Agent {ag.name} disconnected")

    return EventSourceResponse(event_generator())


@app.get("/status")
async def status(current_user: User = Depends(get_current_active_user)):
    """Get status of all registered agents"""
    agents = await app.state.store.list_agents()
    now = datetime.utcnow().timestamp()
    result = []

    # Filter agents by permission
    if current_user.role == "admin":
        visible_agents = agents
    else:
        # Users can see their own agents and public agents
        visible_agents = [a for a in agents if a.owner == current_user.username or a.owner is None]

    for ag in visible_agents:
        heartbeat = round(now - ag.last_seen, 1)

        # FIX: Properly handle serialization of tools and skills
        try:
            # Convert tools and skills to dict safely
            mcp_tools_data = []
            for tool in ag.mcp_tools:
                # Handle both model instances and dicts
                if hasattr(tool, "model_dump"):
                    mcp_tools_data.append(tool.model_dump())
                elif isinstance(tool, dict):
                    mcp_tools_data.append(tool)

            a2a_skills_data = []
            for skill in ag.a2a_skills:
                # Handle both model instances and dicts
                if hasattr(skill, "model_dump"):
                    a2a_skills_data.append(skill.model_dump())
                elif isinstance(skill, dict):
                    a2a_skills_data.append(skill)

            # Log tools and skills for debugging
            logger.info(f"Agent {ag.name} has {len(mcp_tools_data)} MCP tools and {len(a2a_skills_data)} A2A skills")

            # Handle resources similarly
            resources_data = []
            for resource in ag.shared_resources:
                if hasattr(resource, "model_dump"):
                    resources_data.append(resource.model_dump())
                elif isinstance(resource, dict):
                    resources_data.append(resource)

            result.append({
                "name": ag.name,
                "url": ag.url,
                "status": ag.status,
                "source": "sse" if "streamed" in ag.tags else "manual",
                "aira_capabilities": ag.aira_capabilities,
                "heartbeat_seconds_ago": heartbeat,
                "tags": ag.tags,
                "mcp_tools": mcp_tools_data,
                "a2a_skills": a2a_skills_data,
                "shared_resources": resources_data,
                "owner": ag.owner
            })
        except Exception as e:
            logger.error(f"Error serializing agent {ag.name}: {e}")
            # Include a basic version of the agent without problematic fields
            result.append({
                "name": ag.name,
                "url": ag.url,
                "status": ag.status,
                "error": f"Error serializing agent data: {str(e)}",
                "aira_capabilities": ag.aira_capabilities,
                "owner": ag.owner
            })

    active_count = sum(1 for a in visible_agents if a.status == AgentStatus.ONLINE)
    return {
        "uptime": 99.9,  # Placeholder - implement real uptime tracking
        "registered": len(result),
        "active": active_count,
        "agents": result
    }


@app.get("/mcp/agents")
async def list_mcp(current_user: User = Depends(get_current_active_user)):
    """List all MCP-capable agents"""
    agents = await app.state.store.list_agents()

    # Filter by permission and capability
    if current_user.role == "admin":
        visible_agents = agents
    else:
        visible_agents = [a for a in agents if a.owner == current_user.username or a.owner is None]

    filtered = [a for a in visible_agents if "a2a" in a.aira_capabilities or "hybrid" in a.aira_capabilities]
    return {
        "total": len(filtered),
        "agents": [a.dict() for a in filtered]
    }


@app.get("/mcp/tools")
async def list_mcp_tools(current_user: User = Depends(get_current_active_user)):
    """List all available MCP tools from all agents"""
    agents = await app.state.store.list_agents()

    # Filter by permission
    if current_user.role == "admin":
        visible_agents = agents
    else:
        visible_agents = [a for a in agents if a.owner == current_user.username or a.owner is None]

    mcp_agents = [a for a in visible_agents if "mcp" in a.aira_capabilities or "hybrid" in a.aira_capabilities]

    all_tools = []
    for agent in mcp_agents:
        if agent.status == AgentStatus.ONLINE and agent.mcp_tools:
            # Log tools for debugging
            logger.info(f"Agent {agent.name} has {len(agent.mcp_tools)} MCP tools")

            for tool in agent.mcp_tools:
                try:
                    # Handle both model instances and dicts
                    tool_data = {}
                    if hasattr(tool, "model_dump"):
                        tool_data = tool.model_dump()
                    elif isinstance(tool, dict):
                        tool_data = tool
                    else:
                        logger.warning(f"Unknown tool type: {type(tool)}")
                        continue

                    all_tools.append({
                        "agent_name": agent.name,
                        "agent_url": agent.url,
                        "tool": tool_data,
                        "owner": agent.owner
                    })
                except Exception as e:
                    logger.error(f"Error processing tool from agent {agent.name}: {e}")

    # Log the total number of tools found
    logger.info(f"Found {len(all_tools)} MCP tools across all agents")

    return {
        "total": len(all_tools),
        "tools": all_tools
    }


@app.get("/a2a/skills")
async def list_a2a_skills(current_user: User = Depends(get_current_active_user)):
    """List all available A2A skills from all agents"""
    agents = await app.state.store.list_agents()

    # Filter by permission
    if current_user.role == "admin":
        visible_agents = agents
    else:
        visible_agents = [a for a in agents if a.owner == current_user.username or a.owner is None]

    a2a_agents = [a for a in visible_agents if "a2a" in a.aira_capabilities or "hybrid" in a.aira_capabilities]

    all_skills = []
    for agent in a2a_agents:
        if agent.status == AgentStatus.ONLINE and agent.a2a_skills:
            # Log skills for debugging
            logger.info(f"Agent {agent.name} has {len(agent.a2a_skills)} A2A skills")

            for skill in agent.a2a_skills:
                try:
                    # Handle both model instances and dicts
                    skill_data = {}
                    if hasattr(skill, "model_dump"):
                        skill_data = skill.model_dump()
                    elif isinstance(skill, dict):
                        skill_data = skill
                    else:
                        logger.warning(f"Unknown skill type: {type(skill)}")
                        continue

                    all_skills.append({
                        "agent_name": agent.name,
                        "agent_url": agent.url,
                        "skill": skill_data,
                        "owner": agent.owner
                    })
                except Exception as e:
                    logger.error(f"Error processing skill from agent {agent.name}: {e}")

    # Log the total number of skills found
    logger.info(f"Found {len(all_skills)} A2A skills across all agents")

    return {
        "total": len(all_skills),
        "skills": all_skills
    }


# =====================================================================
# Tool Invocation Endpoints
# =====================================================================

@app.post("/mcp/invoke-tool", response_model=ToolResponse)
async def invoke_mcp_tool(
        request: ToolCallRequest,
        client: httpx.AsyncClient = Depends(get_http_client),
        current_user: User = Depends(get_current_active_user),
        background_tasks: BackgroundTasks = None
):
    """Proxy an MCP tool call to the appropriate agent"""
    # Check if agent exists and is online
    agent = await app.state.store.get_agent(request.agent_url)
    if not agent:
        return ToolResponse(
            success=False,
            message=f"Agent not found: {request.agent_url}",
            error="AGENT_NOT_FOUND"
        )

    # Check permission to use the agent/tool
    if agent.owner and agent.owner != current_user.username and current_user.role != "admin":
        return ToolResponse(
            success=False,
            message="You do not have permission to use this agent",
            error="PERMISSION_DENIED"
        )

    if agent.status != AgentStatus.ONLINE:
        return ToolResponse(
            success=False,
            message=f"Agent is not online (status: {agent.status})",
            error="AGENT_OFFLINE"
        )

    # Verify agent supports MCP
    if "mcp" not in agent.aira_capabilities and "hybrid" not in agent.aira_capabilities:
        return ToolResponse(
            success=False,
            message=f"Agent does not support MCP",
            error="MCP_NOT_SUPPORTED"
        )

    # Check if tool exists on agent
    tool_exists = any(t.name == request.tool_name for t in agent.mcp_tools)
    if not tool_exists:
        return ToolResponse(
            success=False,
            message=f"Tool '{request.tool_name}' not found on agent",
            error="TOOL_NOT_FOUND"
        )

    # Generate unique ID for this tool call
    call_id = str(uuid.uuid4())
    await app.state.store.track_tool_call(call_id, {
        "agent_url": request.agent_url,
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "origin": request.origin,
        "type": "mcp",
        "requester": current_user.username
    })

    try:
        # Update agent status
        await app.state.store.update_agent_status(request.agent_url, AgentStatus.BUSY)

        # Prepare the MCP request (tools/call)
        mcp_request = JSONRPCRequest(
            id=call_id,
            method="tools/call",
            params={
                "name": request.tool_name,
                "arguments": request.arguments
            }
        )

        # Send request to agent
        response = await client.post(
            f"{request.agent_url}",
            json=mcp_request.dict(),
            headers={"Accept": "application/json"}
        )

        # Update agent status back to online
        await app.state.store.update_agent_status(request.agent_url, AgentStatus.ONLINE)

        # Process response
        if response.status_code == 200:
            json_response = response.json()
            if "result" in json_response:
                await app.state.store.update_tool_call(call_id, {
                    "status": "completed",
                    "response": json_response["result"]
                })

                return ToolResponse(
                    success=True,
                    message=f"Tool '{request.tool_name}' executed successfully",
                    content=json_response["result"].get("content", [])
                )
            elif "error" in json_response:
                await app.state.store.update_tool_call(call_id, {
                    "status": "error",
                    "error": json_response["error"]
                })

                return ToolResponse(
                    success=False,
                    message=f"Tool execution failed: {json_response['error'].get('message', 'Unknown error')}",
                    error=json_response["error"].get("message", "EXECUTION_FAILED")
                )

        # Handle non-200 responses
        await app.state.store.update_tool_call(call_id, {
            "status": "error",
            "error": f"HTTP {response.status_code}: {response.text}"
        })

        return ToolResponse(
            success=False,
            message=f"HTTP Error {response.status_code}",
            error=f"HTTP_{response.status_code}"
        )

    except Exception as e:
        await app.state.store.update_agent_status(request.agent_url, AgentStatus.ONLINE)
        await app.state.store.update_tool_call(call_id, {
            "status": "error",
            "error": str(e)
        })

        logger.error(f"Error invoking MCP tool: {str(e)}")
        return ToolResponse(
            success=False,
            message=f"Error invoking tool: {str(e)}",
            error="INVOCATION_ERROR"
        )


@app.post("/a2a/invoke-skill", response_model=ToolResponse)
async def invoke_a2a_skill(
        request: ToolCallRequest,
        client: httpx.AsyncClient = Depends(get_http_client),
        current_user: User = Depends(get_current_active_user)
):
    """Proxy an A2A skill call to the appropriate agent"""
    # Check if agent exists and is online
    agent = await app.state.store.get_agent(request.agent_url)
    if not agent:
        return ToolResponse(
            success=False,
            message=f"Agent not found: {request.agent_url}",
            error="AGENT_NOT_FOUND"
        )

    # Check permission to use the agent/skill
    if agent.owner and agent.owner != current_user.username and current_user.role != "admin":
        return ToolResponse(
            success=False,
            message="You do not have permission to use this agent",
            error="PERMISSION_DENIED"
        )

    if agent.status != AgentStatus.ONLINE:
        return ToolResponse(
            success=False,
            message=f"Agent is not online (status: {agent.status})",
            error="AGENT_OFFLINE"
        )

    # Verify agent supports A2A
    if "a2a" not in agent.aira_capabilities and "hybrid" not in agent.aira_capabilities:
        return ToolResponse(
            success=False,
            message=f"Agent does not support A2A",
            error="A2A_NOT_SUPPORTED"
        )

    # Check if skill exists on agent
    skill_exists = any(s.id == request.tool_name for s in agent.a2a_skills)
    if not skill_exists:
        return ToolResponse(
            success=False,
            message=f"Skill '{request.tool_name}' not found on agent",
            error="SKILL_NOT_FOUND"
        )

    # Generate unique ID for this tool call
    call_id = str(uuid.uuid4())
    await app.state.store.track_tool_call(call_id, {
        "agent_url": request.agent_url,
        "skill_id": request.tool_name,
        "arguments": request.arguments,
        "origin": request.origin,
        "type": "a2a",
        "requester": current_user.username
    })

    try:
        # Update agent status
        await app.state.store.update_agent_status(request.agent_url, AgentStatus.BUSY)

        # Prepare the A2A request (tasks/send)
        a2a_request = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "sessionId": str(uuid.uuid4()),  # Create a new session for this call
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": request.arguments.get("prompt", "Execute skill")
                        }
                    ]
                }
            }
        }

        # Send request to agent
        response = await client.post(
            f"{request.agent_url}",
            json=a2a_request,
            headers={"Accept": "application/json"}
        )

        # Update agent status back to online
        await app.state.store.update_agent_status(request.agent_url, AgentStatus.ONLINE)

        # Process response
        if response.status_code == 200:
            json_response = response.json()
            if "result" in json_response:
                await app.state.store.update_tool_call(call_id, {
                    "status": "completed",
                    "response": json_response["result"]
                })

                # Extract content from A2A response
                task_result = json_response["result"]
                content = []

                if task_result.get("artifacts"):
                    for artifact in task_result["artifacts"]:
                        for part in artifact.get("parts", []):
                            if part.get("type") == "text":
                                content.append({
                                    "type": "text",
                                    "text": part.get("text", "")
                                })

                return ToolResponse(
                    success=True,
                    message=f"Skill '{request.tool_name}' executed successfully",
                    content=content
                )
            elif "error" in json_response:
                await app.state.store.update_tool_call(call_id, {
                    "status": "error",
                    "error": json_response["error"]
                })

                return ToolResponse(
                    success=False,
                    message=f"Skill execution failed: {json_response['error'].get('message', 'Unknown error')}",
                    error=json_response["error"].get("message", "EXECUTION_FAILED")
                )

        # Handle non-200 responses
        await app.state.store.update_tool_call(call_id, {
            "status": "error",
            "error": f"HTTP {response.status_code}: {response.text}"
        })

        return ToolResponse(
            success=False,
            message=f"HTTP Error {response.status_code}",
            error=f"HTTP_{response.status_code}"
        )

    except Exception as e:
        await app.state.store.update_agent_status(request.agent_url, AgentStatus.ONLINE)
        await app.state.store.update_tool_call(call_id, {
            "status": "error",
            "error": str(e)
        })

        logger.error(f"Error invoking A2A skill: {str(e)}")
        return ToolResponse(
            success=False,
            message=f"Error invoking skill: {str(e)}",
            error="INVOCATION_ERROR"
        )


@app.get("/tools/calls/{call_id}")
async def get_tool_call_status(
        call_id: str,
        current_user: User = Depends(get_current_active_user)
):
    """Get the status of a tool call"""
    call_info = await app.state.store.get_tool_call(call_id)
    if not call_info:
        raise HTTPException(404, detail="Tool call not found")

    # Check permissions
    requester = call_info.get("requester")
    if requester != current_user.username and current_user.role != "admin":
        raise HTTPException(403, detail="You do not have permission to view this tool call")

    return {
        "call_id": call_id,
        "status": call_info.get("status", "unknown"),
        "created_at": call_info.get("created_at"),
        "updated_at": call_info.get("updated_at"),
        "type": call_info.get("type"),
        "agent_url": call_info.get("agent_url"),
        "tool_name": call_info.get("tool_name") or call_info.get("skill_id"),
        "response": call_info.get("response"),
        "requester": requester
    }


@app.get("/my/agents")
async def list_my_agents(current_user: User = Depends(get_current_active_user)):
    """List all agents owned by the current user"""
    agents = await app.state.store.get_agents_by_owner(current_user.username)
    return {
        "total": len(agents),
        "agents": [a.dict() for a in agents]
    }


@app.get("/my/oauth-clients")
async def list_my_oauth_clients(current_user: User = Depends(get_current_active_user)):
    """List all OAuth clients registered by the current user"""
    clients = await app.state.store.list_oauth_clients_by_owner(current_user.username)
    # Remove sensitive information
    client_data = []
    for client in clients:
        client_dict = client.dict()
        client_dict.pop("client_secret", None)
        client_data.append(client_dict)

    return {
        "total": len(clients),
        "clients": client_data
    }


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with basic information"""
    return {
        "msg": "AIRA Hub – A hub for MCP Tools and A2A Agent Skills",
        "endpoints": {
            "docs": "/docs",
            "auth": {
                "token": "/token",
                "register_client": "/register",
                "authorize": "/authorize"
            },
            "agents": {
                "status": "/status",
                "my_agents": "/my/agents",
                "mcp_agents": "/mcp/agents",
                "a2a_agents": "/a2a/agents"
            },
            "tools": {
                "mcp_tools": "/mcp/tools",
                "a2a_skills": "/a2a/skills",
                "invoke_mcp": "/mcp/invoke-tool",
                "invoke_a2a": "/a2a/invoke-skill"
            }
        },
        "version": "1.0.0",
        "oauth_metadata": "/.well-known/oauth-authorization-server"
    }


# =====================================================================
# Startup and Background Tasks
# =====================================================================

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the application starts"""
    # Start heartbeat checker
    asyncio.create_task(check_agent_heartbeats())
    logger.info("AIRA Hub started with OAuth 2.1 Authentication")
    logger.info("Added fixes for proper MCP tools and A2A skills handling")


@app.get("/a2a/agents")
async def list_a2a(current_user: User = Depends(get_current_active_user)):
    """List all A2A-capable agents"""
    agents = await app.state.store.list_agents()

    # Filter by permission and capability
    if current_user.role == "admin":
        visible_agents = agents
    else:
        visible_agents = [a for a in agents if a.owner == current_user.username or a.owner is None]

    filtered = [a for a in visible_agents if "a2a" in a.aira_capabilities or "hybrid" in a.aira_capabilities]
    return {
        "total": len(filtered),
        "agents": [a.dict() for a in filtered]
    }


# =====================================================================
# Main Entrypoint
# =====================================================================

def main():
    import argparse
    import uvicorn
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    cfg = parser.parse_args()

    # Set logging level based on debug flag
    if cfg.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled with extra logging")

    logger.info(f"Starting AIRA Hub on {cfg.host}:{cfg.port}")

    try:
        uvicorn.run("AiraHub:app", host=cfg.host, port=cfg.port, reload=cfg.reload)
    except Exception as e:
        logger.error(f"Error starting AIRA Hub: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())