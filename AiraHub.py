# AuraHub.py (PATCHED: Properly saves skills and tools from /connect/stream/init)
import json

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging

app = FastAPI()
log = logging.getLogger("uvicorn")

class AgentStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"

class Resource(BaseModel):
    uri: str
    description: Optional[str] = None
    type: Optional[str] = None
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentRegistration(BaseModel):
    name: str
    url: str
    aira_capabilities: List[str]
    skills: List[Dict[str, Any]] = []
    shared_resources: List[Resource] = []
    tags: List[str] = []
    category: Optional[str] = None
    description: Optional[str] = None
    status: AgentStatus = AgentStatus.ONLINE
    created_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    last_seen: float = Field(default_factory=lambda: datetime.utcnow().timestamp())

# Dummy store
class AgentStore:
    def __init__(self):
        self._agents = {}

    async def save_agent(self, agent: AgentRegistration):
        self._agents[agent.url] = agent

    async def get_agent(self, url: str):
        return self._agents.get(url)

    async def list_agents(self):
        return list(self._agents.values())

app.state.store = AgentStore()

@app.post("/connect/stream/init")
async def stream_init(request: Request):
    body = await request.json()
    agent_url = body.get("url")

    agent = await request.app.state.store.get_agent(agent_url)
    if not agent:
        raise HTTPException(404, detail="Agent not registered via /connect/stream")

    updated_fields = ["skills", "shared_resources", "tags", "category", "description"]

    agent_dict = agent.dict()
    for field in updated_fields:
        if field in body and isinstance(body[field], (list, str, dict)):
            agent_dict[field] = body[field]

    new_agent = AgentRegistration(**agent_dict)
    await request.app.state.store.save_agent(new_agent)
    log.info("✅ Updated agent fields: %s", list(body.keys()))
    return {"status": "updated"}

@app.get("/connect/stream")
async def connect_stream(agent_url: str, name: str, aira_capabilities: str):
    caps = aira_capabilities.split(",")
    ag = AgentRegistration(
        url=agent_url,
        name=name,
        aira_capabilities=caps,
        shared_resources=[],
        skills=[],
        tags=["streamed"],
        status=AgentStatus.ONLINE,
    )
    await app.state.store.save_agent(ag)

    async def event_generator():
        while True:
            ag.last_seen = datetime.utcnow().timestamp()
            await app.state.store.save_agent(ag)
            yield json.dumps({
                "agent": ag.name,
                "status": ag.status,
                "ts": datetime.utcnow().isoformat() + "Z"
            }) + "\n\n"
            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())

@app.get("/status")
async def status():
    agents = await app.state.store.list_agents()
    now = datetime.utcnow().timestamp()
    result = []
    for ag in agents:
        heartbeat = round(now - ag.last_seen, 1)
        result.append({
            "name": ag.name,
            "url": ag.url,
            "status": ag.status,
            "source": "sse" if "streamed" in ag.tags else "manual",
            "aira_capabilities": ag.aira_capabilities,
            "heartbeat_seconds_ago": heartbeat,
            "tags": ag.tags,
            "skills": ag.skills,
            "shared_resources": [r.dict() for r in ag.shared_resources],
        })
    return {"uptime": 99.9, "registered": len(result), "active": len(result), "agents": result}

@app.get("/mcp/agents")
async def list_mcp():
    agents = await app.state.store.list_agents()
    filtered = [a for a in agents if "mcp" in a.aira_capabilities]
    return {"total": len(filtered), "agents": [a.dict() for a in filtered]}

@app.get("/a2a/agents")
async def list_a2a():
    agents = await app.state.store.list_agents()
    filtered = [a for a in agents if "a2a" in a.aira_capabilities]
    return {"total": len(filtered), "agents": [a.dict() for a in filtered]}


@app.get("/", include_in_schema=False)
async def root():
    return {"msg": "AIRA Hub – use /mcp , /a2a , /hybrid paths or query ?client_type="}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8015)
    p.add_argument("--reload", action="store_true")
    cfg = p.parse_args()

    uvicorn.run("AiraHub:app", host=cfg.host, port=cfg.port, reload=cfg.reload)