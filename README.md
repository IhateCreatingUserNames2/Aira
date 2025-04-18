# 🌐 AiraHub (AIRA Hub)

AiraHub is the central node of the AIRA Network — a decentralized discovery and interaction system for AI agents. It enables agents with different architectures to register, discover each other, and share tools or skills in real-time.

## 🌟 Key Features

- 🧠 Supports both MCP (Model Context Protocol) and A2A (Agent-to-Agent) architectures
- 🔍 Unified discovery interface for different agent types
- 🔧 Tool and skill registration capabilities
- 🌐 Automatic agent registration via SSE stream
- 🔁 Real-time heartbeat tracking
- 📦 Flexible storage backend (in-memory or file-based)

## ⚙️ Supported Capabilities

- Unified discovery of agents
- Filtering agents by:
  - Capability
  - Tags
  - Category
  - Status
- Registration of:
  - MCP Tools (`mcp_tool`)
  - A2A Skills (`a2a_skill`)

## 🚀 Quick Start

### Running AiraHub

```bash
python AiraHub.py --port 8015
```

Default configurations:
- Port: 8015
- Database file: `aira_db.json` (auto-created)

### Agent Registration

#### SSE Live Connection (Preferred Method)
Use `aira_sse_client.py` to connect:

```bash
GET /connect/stream?agent_url=...&name=...&aira_capabilities=mcp,a2a
```

#### Manual Registration
Send a POST request to `/register`:

```json
{
  "url": "http://localhost:8094/",
  "name": "MemoryAgent",
  "aira_capabilities": ["mcp", "a2a"],
  "skills": [...],
  "shared_resources": [...],
  "tags": ["streamed", "cognition"]
}
```

## 📡 Agent Discovery

Discovery Endpoints:
- `/mcp/agents`: MCP agents only
- `/a2a/agents`: A2A agents only
- `/hybrid/agents`: All agents (default)

Filtering options available via `/discover` endpoint:
- `skill_id`
- `skill_tags`
- `resource_type`
- `category`
- `status`

## 🛠 Tool and Skill Usage

Use `aira_tool_user.py` to:
- Discover agents and their capabilities
- Call MCP Tools directly
- Inspect available A2A Skills

## 🧠 Status Monitoring

Check system status via `/status` endpoint:
- Uptime
- Active vs. registered agents
- Last heartbeat
- Agent capabilities

## 🪐 Design Philosophy

AiraHub respects agent diversity:
- MCP Clients see only MCP Agents
- A2A Clients see only A2A Agents
- Hybrid Clients see all Agents

## 📂 Key Files

| File | Description |
|------|-------------|
| `AiraHub.py` | Main FastAPI app for AIRA hub |
| `aira_sse_client.py` | Example SSE-connected agent |
| `aira_tool_user.py` | Discovery and test client |

## 🔗 Architecture Overview

```
           +------------------------+
           |     AiraHub (API)     |
           |  FastAPI + SSE Layer  |
           +----------+-----------+
                      |
          +-----------+-----------+
          | Agent Registration    |
          | via REST or SSE       |
          +-----------------------+
          | Agent Discovery APIs  |
          | by capability/type    |
          +-----------------------+
          | Status & Heartbeat    |
          | Monitoring            |
          +-----------------------+
```

## 🚧 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Insert License Information Here]
