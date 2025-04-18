# aira_web_panel.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import time
import json
import httpx

app = FastAPI()

# CORS for local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIRA_HUB = "https://aira-fl8f.onrender.com"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(
        """
        <html>
        <head>
            <title>AIRA Network Monitor</title>
            <style>
                body { background: black; color: #00FFAA; font-family: monospace; padding: 1em; }
                .agent { margin-bottom: 1em; border: 1px solid #00FFAA; padding: 1em; }
                .offline { opacity: 0.4; }
            </style>
        </head>
        <body>
            <h1>🌐 AIRA Network - Live Monitor</h1>
            <div id="agents">Loading...</div>
            <script>
                async function fetchAgents() {
                    const res = await fetch('/status');
                    const data = await res.json();
                    const div = document.getElementById('agents');
                    div.innerHTML = data.agents.map(agent => `
                        <div class="agent ${agent.status !== 'online' ? 'offline' : ''}">
                            <b>${agent.name}</b> (${agent.url})<br>
                            <i>${agent.status}</i> - ${agent.aira_capabilities.join(', ')}<br>
                            <small>Heartbeat: ${agent.heartbeat_seconds_ago}s ago</small><br>
                            Tags: ${agent.tags ? agent.tags.join(', ') : 'none'}
                        </div>
                    `).join('');
                }
                setInterval(fetchAgents, 5000);
                fetchAgents();
            </script>
        </body>
        </html>
        """
    )

@app.get("/status")
async def get_status():
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{AIRA_HUB}/status")
        return res.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("aira_web_panel:app", host="0.0.0.0", port=8030, reload=True)
