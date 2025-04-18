# AuraHub.py – “vibe” edition
# ---------------------------------------------------------------------------
# – Keeps MCP‑only and A2A‑only worlds cleanly separated
# – Adds dedicated routes:
#       • /mcp/…     → MCP‑only view
#       • /a2a/…     → A2A‑only view
#       • /hybrid/…  → union view   (alias for canonical /… endpoints)
# – Auto‑detects client preference via:
#       • Explicit path  (/mcp , /a2a , /hybrid)
#       • Fallback query param  ?client_type=mcp|a2a|hybrid
#       • Ultimate default = hybrid
#
# Cognisphere?  Register it with both capabilities ["mcp","a2a"] and it will
# naturally show up in hybrid views only.  If you’d rather treat it as pure
# MCP, list just ["mcp"].
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio, json, logging, os, time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# basic config
# ---------------------------------------------------------------------------
DEFAULT_HEARTBEAT_TIMEOUT = 300
MAX_RESOURCES_PER_AGENT = 100
DB_FILE = os.getenv("AIRA_DB_FILE", "aira_db.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("aira_hub.log")],
)
log = logging.getLogger("AuraHub")

# ---------------------------------------------------------------------------
# enums / models
# ---------------------------------------------------------------------------
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

class StorageBackend(str, Enum):
    MEMORY = "memory"
    FILE = "file"

class AgentMetrics(BaseModel):
    request_count: int = 0
    error_count: int = 0
    uptime: float = 0.0
    last_response_time: Optional[float] = None
    avg_response_time: Optional[float] = None

class Resource(BaseModel):
    uri: str
    description: str
    type: Union[ResourceType, str]
    version: str = "1.0.0"
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = {}

    @validator("uri")
    def _minlen(cls, v):
        if len(v) < 3:
            raise ValueError("uri too short")
        return v

class AgentRegistration(BaseModel):
    url: str
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    skills: List[Dict[str, Any]] = []
    shared_resources: List[Resource] = []
    aira_capabilities: List[str] = []          # "mcp", "a2a" or both
    tags: List[str] = []
    category: Optional[str] = None

    status: AgentStatus = AgentStatus.ONLINE
    created_at: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    metrics: Optional[AgentMetrics] = None
    auth: Dict[str, Any] = {}

    @validator("url")
    def _http(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http/https")
        return v

    @validator("shared_resources")
    def _max(cls, v):
        if len(v) > MAX_RESOURCES_PER_AGENT:
            raise ValueError("too many resources")
        return v

class DiscoverQuery(BaseModel):
    skill_id: Optional[str] = None
    skill_tags: Optional[List[str]] = None
    resource_type: Optional[str] = None
    category: Optional[str] = None
    status: Optional[AgentStatus] = None
    offset: int = 0
    limit: int = 100

# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
class BaseStorage:
    async def init(self): ...
    async def close(self): ...
    async def save_agent(self, ag: AgentRegistration): ...
    async def get_agent(self, url: str) -> Optional[AgentRegistration]: ...
    async def list_agents(self) -> List[AgentRegistration]: ...
    async def delete_agent(self, url: str): ...

class MemoryStorage(BaseStorage):
    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}

    async def save_agent(self, ag: AgentRegistration):
        self._agents[ag.url] = ag

    async def get_agent(self, url, default=None):
        return self._agents.get(url)

    async def list_agents(self):
        return list(self._agents.values())

    async def delete_agent(self, url):
        self._agents.pop(url, None)

class FileStorage(MemoryStorage):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._dirty = False

    async def init(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            clean: Dict[str, AgentRegistration] = {}
            for url, blob in raw.items():
                try:
                    clean[url] = AgentRegistration(**blob)
                except ValidationError as e:
                    log.warning("skip malformed agent %s: %s", url, e)
            self._agents = clean
            log.info("loaded %d agents", len(self._agents))
        except Exception as e:
            log.error("error loading DB: %s", e)

    async def close(self):
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({u: a.dict() for u, a in self._agents.items()}, f, indent=2)
        self._dirty = False
        log.info("DB flushed")

    async def save_agent(self, ag):
        await super().save_agent(ag)
        self._dirty = True

    async def delete_agent(self, url):
        await super().delete_agent(url)
        self._dirty = True

def get_storage(kind=StorageBackend.FILE, **kw):
    return FileStorage(DB_FILE) if kind == StorageBackend.FILE else MemoryStorage()

# ---------------------------------------------------------------------------
# fastapi app
# ---------------------------------------------------------------------------
app = FastAPI(title="AIRA Hub", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# lifespan
@asynccontextmanager
async def lifespan(app_: FastAPI):
    app_.state.store = get_storage()
    await app_.state.store.init()

    # offline‑marking task
    async def cleaner():
        while True:
            await asyncio.sleep(60)
            now = time.time()
            for ag in await app_.state.store.list_agents():
                if now - ag.last_seen > DEFAULT_HEARTBEAT_TIMEOUT and ag.status != AgentStatus.OFFLINE:
                    ag.status = AgentStatus.OFFLINE
                    await app_.state.store.save_agent(ag)

    task = asyncio.create_task(cleaner())
    app_.state.started = time.time()
    try:
        yield
    finally:
        task.cancel()
        await app_.state.store.close()

app.router.lifespan_context = lifespan  # type: ignore

# ---------------------------------------------------------------------------
# helpers: capability filter + route factory
# ---------------------------------------------------------------------------
def _filter(agents: List[AgentRegistration], ctype: str) -> List[AgentRegistration]:
    if ctype == "mcp":
        return [a for a in agents if "mcp" in a.aira_capabilities and "a2a" not in a.aira_capabilities]
    if ctype == "a2a":
        return [a for a in agents if "a2a" in a.aira_capabilities and "mcp" not in a.aira_capabilities]
    return agents  # hybrid

def _ctype(req: Request, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return req.query_params.get("client_type", "hybrid").lower()

def _strip_auth(a: AgentRegistration) -> dict:
    d = a.dict()
    d.pop("auth", None)
    return d

# ---------------------------------------------------------------------------
# generic discovery endpoints generator
# ---------------------------------------------------------------------------
def register_views(prefix: str, ctype: str):
    tag = f"{ctype.upper()} Discovery"

    @app.get(f"{prefix}/agents", tags=[tag])
    async def list_agents(req: Request):
        agents = _filter(await req.app.state.store.list_agents(), ctype)
        return [_strip_auth(a) for a in agents]

    @app.post(f"{prefix}/discover", tags=[tag])
    async def discover(req: Request, body: DiscoverQuery):
        agents = _filter(await req.app.state.store.list_agents(), ctype)

        if body.skill_id:
            agents = [a for a in agents if any(s.get("id") == body.skill_id for s in a.skills)]
        if body.skill_tags:
            agents = [
                a for a in agents if any(set(body.skill_tags) & set(s.get("tags", [])) for s in a.skills)
            ]
        if body.resource_type:
            agents = [a for a in agents if any(str(r.type) == body.resource_type for r in a.shared_resources)]
        if body.category:
            agents = [a for a in agents if a.category == body.category]
        if body.status:
            agents = [a for a in agents if a.status == body.status]

        total = len(agents)
        agents = agents[body.offset : body.offset + body.limit]
        return {"total": total, "agents": [_strip_auth(a) for a in agents]}

# register trio of discovery namespaces
register_views("/mcp",  "mcp")
register_views("/a2a",  "a2a")
register_views("/hybrid", "hybrid")      # alias
# legacy paths map to hybrid
register_views("", "hybrid")

# ---------------------------------------------------------------------------
# agent registration / heartbeat always go through canonical root
# ---------------------------------------------------------------------------
@app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Agents"])
async def register(req: Request, ag: AgentRegistration):
    ag.last_seen = time.time()
    ag.status = AgentStatus.ONLINE
    if not ag.metrics:
        ag.metrics = AgentMetrics()
    await req.app.state.store.save_agent(ag)
    log.info("registered %s", ag.url)
    return {"status": "ok", "url": ag.url}

@app.post("/heartbeat/{agent_url:path}", tags=["Agents"])
async def beat(req: Request, agent_url: str):
    ag = await req.app.state.store.get_agent(agent_url)
    if not ag:
        raise HTTPException(404)
    ag.last_seen = time.time()
    ag.status = AgentStatus.ONLINE
    await req.app.state.store.save_agent(ag)
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------
@app.get("/events/stream", tags=["SSE"])
async def stream(agent_url: str, req: Request):
    ag = await req.app.state.store.get_agent(agent_url)
    if not ag:
        raise HTTPException(404)

    async def gen() -> AsyncGenerator[str, None]:
        while True:
            if await req.is_disconnected():
                break
            a = await req.app.state.store.get_agent(agent_url)
            yield (
                "data: "
                + json.dumps(
                    {
                        "agent": a.name if a else agent_url,
                        "status": a.status if a else "unknown",
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                )
                + "\n\n"
            )
            await asyncio.sleep(5)

    return EventSourceResponse(gen())

# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
@app.get("/status", tags=["System"])
async def status(req: Request):
    agents = await req.app.state.store.list_agents()
    now = time.time()
    return {
        "uptime": now - req.app.state.started,
        "registered": len(agents),
        "active": sum(1 for a in agents if now - a.last_seen < DEFAULT_HEARTBEAT_TIMEOUT),
    }

@app.get("/", include_in_schema=False)
async def root():
    return {"msg": "AIRA Hub – use /mcp , /a2a , /hybrid paths or query ?client_type="}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8015)
    p.add_argument("--reload", action="store_true")
    cfg = p.parse_args()

    uvicorn.run("AuraHub:app", host=cfg.host, port=cfg.port, reload=cfg.reload)
